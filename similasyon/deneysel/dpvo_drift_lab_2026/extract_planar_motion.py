#!/usr/bin/env python3
"""Extract robust planar frame-to-frame motion from the 2026 7.5 FPS video.

The camera is predominantly nadir-facing and the ground occupies most of the
image.  A planar image-motion estimate is therefore a useful scale/course cue
that is independent from DPVO's internal translation gauge.  This tool does
not consume ground truth and can be run once for every downstream candidate.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class ExtractConfig:
    width: int = 640
    height: int = 360
    max_corners: int = 1600
    quality_level: float = 0.01
    min_distance: float = 7.0
    block_size: int = 7
    lk_window: int = 31
    lk_levels: int = 4
    fb_threshold: float = 1.5
    ransac_threshold: float = 2.0
    min_tracks: int = 30


FIELDS = (
    "sample_index",
    "track_count",
    "fb_track_count",
    "homography_ok",
    "homography_inliers",
    "homography_inlier_ratio",
    "homography_reprojection_rmse",
    "center_dx",
    "center_dy",
    "jacobian_00",
    "jacobian_01",
    "jacobian_10",
    "jacobian_11",
    "local_rotation_rad",
    "local_log_scale",
    "local_anisotropy",
    "perspective_x",
    "perspective_y",
    "homography_00",
    "homography_01",
    "homography_02",
    "homography_10",
    "homography_11",
    "homography_12",
    "homography_20",
    "homography_21",
    "homography_22",
    "affine_ok",
    "affine_inliers",
    "affine_dx",
    "affine_dy",
    "affine_rotation_rad",
    "affine_log_scale",
    "median_flow_x",
    "median_flow_y",
    "flow_mad",
)


def _gray(frame: np.ndarray, config: ExtractConfig) -> np.ndarray:
    if frame.shape[1] != config.width or frame.shape[0] != config.height:
        frame = cv2.resize(
            frame, (config.width, config.height), interpolation=cv2.INTER_AREA
        )
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def _project(H: np.ndarray, points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    homogeneous = np.column_stack([points, np.ones(len(points))])
    mapped = homogeneous @ np.asarray(H, dtype=np.float64).T
    return mapped[:, :2] / mapped[:, 2:3]


def _local_homography_geometry(
    H: np.ndarray, center: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float, float, float]:
    """Return center flow, local Jacobian, rotation, log-scale, anisotropy."""

    center = np.asarray(center, dtype=np.float64)
    basis = np.vstack([center, center + [1.0, 0.0], center + [0.0, 1.0]])
    mapped = _project(H, basis)
    flow = mapped[0] - center
    jacobian = np.column_stack([mapped[1] - mapped[0], mapped[2] - mapped[0]])
    u, singular, vt = np.linalg.svd(jacobian)
    rotation_matrix = u @ vt
    if np.linalg.det(rotation_matrix) < 0:
        u[:, -1] *= -1
        rotation_matrix = u @ vt
    rotation = math.atan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
    scale = max(float(np.sqrt(abs(np.linalg.det(jacobian)))), 1e-12)
    anisotropy = float(singular[0] / max(singular[-1], 1e-12))
    return flow, jacobian, float(rotation), float(math.log(scale)), anisotropy


def _empty_row(sample_index: int) -> dict[str, float | int]:
    row: dict[str, float | int] = {field: float("nan") for field in FIELDS}
    row.update(
        {
            "sample_index": int(sample_index),
            "track_count": 0,
            "fb_track_count": 0,
            "homography_ok": 0,
            "homography_inliers": 0,
            "homography_inlier_ratio": 0.0,
            "affine_ok": 0,
            "affine_inliers": 0,
        }
    )
    return row


def estimate_pair(
    previous: np.ndarray,
    current: np.ndarray,
    sample_index: int,
    config: ExtractConfig,
) -> dict[str, float | int]:
    row = _empty_row(sample_index)
    points0 = cv2.goodFeaturesToTrack(
        previous,
        maxCorners=config.max_corners,
        qualityLevel=config.quality_level,
        minDistance=config.min_distance,
        blockSize=config.block_size,
        useHarrisDetector=False,
    )
    if points0 is None or len(points0) < config.min_tracks:
        return row
    row["track_count"] = int(len(points0))
    lk = dict(
        winSize=(config.lk_window, config.lk_window),
        maxLevel=config.lk_levels,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )
    points1, status1, _ = cv2.calcOpticalFlowPyrLK(
        previous, current, points0, None, **lk
    )
    if points1 is None or status1 is None:
        return row
    points0_back, status0, _ = cv2.calcOpticalFlowPyrLK(
        current, previous, points1, None, **lk
    )
    if points0_back is None or status0 is None:
        return row
    p0 = points0.reshape(-1, 2)
    p1 = points1.reshape(-1, 2)
    p0_back = points0_back.reshape(-1, 2)
    valid = (
        status1.reshape(-1).astype(bool)
        & status0.reshape(-1).astype(bool)
        & np.all(np.isfinite(p1), axis=1)
        & (np.linalg.norm(p0_back - p0, axis=1) <= config.fb_threshold)
    )
    p0 = p0[valid]
    p1 = p1[valid]
    row["fb_track_count"] = int(len(p0))
    if len(p0) < config.min_tracks:
        return row

    flow = p1 - p0
    median_flow = np.median(flow, axis=0)
    flow_residual = np.linalg.norm(flow - median_flow, axis=1)
    row["median_flow_x"] = float(median_flow[0])
    row["median_flow_y"] = float(median_flow[1])
    row["flow_mad"] = float(np.median(np.abs(flow_residual - np.median(flow_residual))))

    H, mask_h = cv2.findHomography(
        p0, p1, cv2.RANSAC, ransacReprojThreshold=config.ransac_threshold
    )
    if H is not None and mask_h is not None and np.all(np.isfinite(H)):
        inliers = mask_h.reshape(-1).astype(bool)
        count = int(np.sum(inliers))
        if count >= config.min_tracks:
            predicted = _project(H, p0[inliers])
            residual = np.linalg.norm(predicted - p1[inliers], axis=1)
            center = np.array([config.width / 2.0, config.height / 2.0])
            center_flow, jacobian, rotation, log_scale, anisotropy = (
                _local_homography_geometry(H, center)
            )
            row.update(
                {
                    "homography_ok": 1,
                    "homography_inliers": count,
                    "homography_inlier_ratio": float(count / len(p0)),
                    "homography_reprojection_rmse": float(
                        np.sqrt(np.mean(residual**2))
                    ),
                    "center_dx": float(center_flow[0]),
                    "center_dy": float(center_flow[1]),
                    "jacobian_00": float(jacobian[0, 0]),
                    "jacobian_01": float(jacobian[0, 1]),
                    "jacobian_10": float(jacobian[1, 0]),
                    "jacobian_11": float(jacobian[1, 1]),
                    "local_rotation_rad": rotation,
                    "local_log_scale": log_scale,
                    "local_anisotropy": anisotropy,
                    "perspective_x": float(H[2, 0] / H[2, 2]),
                    "perspective_y": float(H[2, 1] / H[2, 2]),
                    **{
                        f"homography_{row_index}{column_index}": float(
                            H[row_index, column_index] / H[2, 2]
                        )
                        for row_index in range(3)
                        for column_index in range(3)
                    },
                }
            )

    affine, mask_a = cv2.estimateAffinePartial2D(
        p0,
        p1,
        method=cv2.RANSAC,
        ransacReprojThreshold=config.ransac_threshold,
        maxIters=3000,
        confidence=0.999,
        refineIters=20,
    )
    if affine is not None and mask_a is not None and np.all(np.isfinite(affine)):
        count = int(np.sum(mask_a))
        if count >= config.min_tracks:
            a, b = float(affine[0, 0]), float(affine[1, 0])
            row.update(
                {
                    "affine_ok": 1,
                    "affine_inliers": count,
                    "affine_dx": float(affine[0, 2]),
                    "affine_dy": float(affine[1, 2]),
                    "affine_rotation_rad": float(math.atan2(b, a)),
                    "affine_log_scale": float(math.log(max(math.hypot(a, b), 1e-12))),
                }
            )
    return row


def extract(video: Path, output: Path, config: ExtractConfig) -> dict:
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"Video açılamadı: {video}")
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    ok, frame = capture.read()
    if not ok:
        raise RuntimeError(f"İlk kare okunamadı: {video}")
    previous = _gray(frame, config)
    rows = [_empty_row(0)]
    started = time.monotonic()
    for sample_index in range(1, frame_count):
        ok, frame = capture.read()
        if not ok:
            raise RuntimeError(
                f"Video {sample_index}. karede bitti; beklenen={frame_count}"
            )
        current = _gray(frame, config)
        rows.append(estimate_pair(previous, current, sample_index, config))
        previous = current
        if sample_index % 100 == 0 or sample_index + 1 == frame_count:
            elapsed = time.monotonic() - started
            print(
                f"{sample_index + 1}/{frame_count} "
                f"({(sample_index + 1) / max(elapsed, 1e-9):.2f} fps)"
            )
    capture.release()

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    h_ok = np.array([int(row["homography_ok"]) for row in rows], dtype=bool)
    a_ok = np.array([int(row["affine_ok"]) for row in rows], dtype=bool)
    summary = {
        "video": str(video.resolve()),
        "output": str(output.resolve()),
        "reported_fps": fps,
        "frame_count": frame_count,
        "config": config.__dict__,
        "homography_success_count": int(np.sum(h_ok)),
        "homography_success_rate": float(np.mean(h_ok[1:])),
        "affine_success_count": int(np.sum(a_ok)),
        "affine_success_rate": float(np.mean(a_ok[1:])),
        "elapsed_seconds": time.monotonic() - started,
    }
    output.with_suffix(".json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    args = parser.parse_args()
    summary = extract(
        args.video,
        args.output,
        ExtractConfig(width=args.width, height=args.height),
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
