#!/usr/bin/env python3
"""Evaluate ORB through the legacy Formatter_for_yolo + Calculate_Direction path.

This does not use a standalone ORB fit. Instead it injects an ORB tracker adapter
into `Class.Formatter_for_yolo.tracker`, then calls `formatter()` exactly like the
old pipeline. The formatter/Calculate_Direction code performs the calibration and
2D->3D conversion.

GT mapping for 7.5 FPS images vs 30 FPS GT:

    image_0 -> gt_0
    image_1 -> gt_4
    image_2 -> gt_8
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
import math
import re
import sys
from pathlib import Path

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Class.ORB import CameraMovementTracker  # noqa: E402


def load_gt(path: Path) -> np.ndarray:
    rows: list[list[float]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                [
                    float(row["translation_x"]),
                    float(row["translation_y"]),
                    float(row["translation_z"]),
                ]
            )
    if not rows:
        raise ValueError(f"GT CSV is empty: {path}")
    return np.asarray(rows, dtype=np.float64)


def list_frames(frames_dir: Path) -> list[Path]:
    frames = sorted(
        p
        for p in frames_dir.iterdir()
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    )
    if not frames:
        raise ValueError(f"No image frames found in: {frames_dir}")
    return frames


def parse_calibration_text(path: Path) -> tuple[np.ndarray, np.ndarray]:
    text = path.read_text(encoding="utf-8", errors="replace")
    focal = re.search(r"FocalLength:\s*\[([^\]]+)\]", text)
    principal = re.search(r"PrincipalPoint:\s*\[([^\]]+)\]", text)
    radial = re.search(r"RadialDistortion:\s*\[([^\]]+)\]", text)
    tangential = re.search(r"TangentialDistortion:\s*\[([^\]]+)\]", text)

    if not focal or not principal:
        raise ValueError(f"Could not parse camera calibration text: {path}")

    fx, fy = [float(x) for x in focal.group(1).split()]
    cx, cy = [float(x) for x in principal.group(1).split()]
    k = [float(x) for x in radial.group(1).split()] if radial else [0.0, 0.0]
    p = [float(x) for x in tangential.group(1).split()] if tangential else [0.0, 0.0]

    camera_matrix = np.array(
        [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    dist_coeffs = np.array([k[0], k[1], p[0], p[1]], dtype=np.float64)
    return camera_matrix, dist_coeffs


class ORBTrackerAdapter:
    """Adapter matching DPVO_object.process_frames_from_list(idx, frame_path)."""

    def __init__(self, camera_matrix: np.ndarray, dist_coeffs: np.ndarray, max_width: int = 0):
        self.tracker = CameraMovementTracker(camera_matrix, dist_coeffs)
        self.max_width = int(max_width or 0)
        self.base_camera_matrix = camera_matrix.copy()
        self.dist_coeffs = dist_coeffs

    def process_frames_from_list(self, idx, frame_path):
        frame = cv2.imread(str(frame_path))
        if frame is None:
            raise ValueError(f"Frame could not be read: {frame_path}")
        if self.max_width > 0 and frame.shape[1] > self.max_width:
            scale = self.max_width / float(frame.shape[1])
            new_h = max(1, int(round(frame.shape[0] * scale)))
            frame = cv2.resize(frame, (self.max_width, new_h), interpolation=cv2.INTER_AREA)
            if idx == 0:
                scaled_matrix = self.base_camera_matrix.copy()
                scaled_matrix[0, 0] *= scale
                scaled_matrix[0, 2] *= scale
                scaled_matrix[1, 1] *= scale
                scaled_matrix[1, 2] *= scale
                self.tracker.camera_matrix = scaled_matrix
        self.tracker.process_frame(frame)
        x, y = self.tracker.get_positions()
        return float(x), float(y), 0.0


def reset_formatter_state(formatter_module, tracker):
    formatter_module.calibration_frames = []
    formatter_module.gt_data = []
    formatter_module.positions_data = []
    formatter_module.detected_objects = []
    formatter_module.scale_factor = None
    formatter_module.detected = None
    formatter_module.offset = None
    formatter_module.tracker = tracker


def finite_xyz(value) -> tuple[float, float, float] | None:
    try:
        x = float(value[0])
        y = float(value[1])
        z = float(value[2])
    except Exception:
        return None
    if not math.isfinite(x) or not math.isfinite(y) or not math.isfinite(z):
        return None
    return x, y, z


def main() -> int:
    parser = argparse.ArgumentParser(description="Legacy formatter + ORB RMSE")
    parser.add_argument("--frames", required=True, type=Path)
    parser.add_argument("--gt", required=True, type=Path)
    parser.add_argument("--calib-text", required=True, type=Path)
    parser.add_argument("--fps-ratio", default=4, type=int)
    parser.add_argument("--gt-offset", default=0, type=int)
    parser.add_argument("--calib-image-frames", default=450, type=int)
    parser.add_argument("--limit", default=0, type=int)
    parser.add_argument("--orb-max-width", default=0, type=int)
    parser.add_argument("--show-legacy-output", action="store_true")
    parser.add_argument(
        "--output-csv",
        default=REPO_ROOT / "similasyon" / "_debug" / "legacy_orb_formatter_rmse.csv",
        type=Path,
    )
    args = parser.parse_args()

    from Class import Formatter_for_yolo as legacy_formatter

    frames = list_frames(args.frames)
    gt = load_gt(args.gt)
    if args.limit and args.limit > 0:
        frames = frames[: args.limit]

    mapped: list[tuple[int, Path, int, np.ndarray]] = []
    for image_idx, frame_path in enumerate(frames):
        gt_idx = args.gt_offset + image_idx * args.fps_ratio
        if gt_idx >= len(gt):
            break
        mapped.append((image_idx, frame_path, gt_idx, gt[gt_idx]))

    if len(mapped) <= args.calib_image_frames:
        raise ValueError(
            f"Need more mapped frames ({len(mapped)}) than calib-image-frames ({args.calib_image_frames})"
        )

    camera_matrix, dist_coeffs = parse_calibration_text(args.calib_text)
    reset_formatter_state(
        legacy_formatter,
        ORBTrackerAdapter(camera_matrix, dist_coeffs, max_width=args.orb_max_width),
    )

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    rows: list[list[object]] = []
    errors: list[np.ndarray] = []
    nonfinite = 0

    print("=" * 72)
    print("Legacy Formatter + ORB RMSE Evaluation")
    print("=" * 72)
    print(f"Frames dir:       {args.frames}")
    print(f"GT CSV:           {args.gt}")
    print(f"Calibration text: {args.calib_text}")
    print(f"Mapped pairs:     {len(mapped)}")
    print(f"GT mapping:       gt_idx = {args.gt_offset} + image_idx * {args.fps_ratio}")
    print(f"Calibration imgs: first {args.calib_image_frames}")
    print(f"ORB max width:    {args.orb_max_width or 'original'}")
    print("-" * 72)

    for n, (image_idx, frame_path, gt_idx, gt_vec) in enumerate(mapped):
        health_status = "1" if n < args.calib_image_frames else "0"
        if args.show_legacy_output:
            pred = legacy_formatter.formatter(None, str(frame_path), image_idx, gt_vec, health_status)
        else:
            with contextlib.redirect_stdout(io.StringIO()):
                pred = legacy_formatter.formatter(None, str(frame_path), image_idx, gt_vec, health_status)

        xyz = finite_xyz(pred)
        if health_status == "0":
            if xyz is None:
                nonfinite += 1
                pred_x = pred_y = pred_z = float("nan")
                err_x = err_y = err_z = err_3d = float("nan")
            else:
                pred_x, pred_y, pred_z = xyz
                err = np.asarray(
                    [pred_x - gt_vec[0], pred_y - gt_vec[1], pred_z - gt_vec[2]],
                    dtype=np.float64,
                )
                err_x, err_y, err_z = float(err[0]), float(err[1]), float(err[2])
                err_3d = float(np.linalg.norm(err))
                errors.append(err)

            rows.append(
                [
                    image_idx,
                    frame_path.name,
                    gt_idx,
                    pred_x,
                    pred_y,
                    pred_z,
                    gt_vec[0],
                    gt_vec[1],
                    gt_vec[2],
                    err_x,
                    err_y,
                    err_z,
                    err_3d,
                ]
            )

        if (n + 1) % 50 == 0 or n == len(mapped) - 1:
            phase = "CALIB" if health_status == "1" else "EVAL"
            if xyz is None:
                xyz_text = "(nan,nan,nan)"
            else:
                xyz_text = f"({xyz[0]:.3f},{xyz[1]:.3f},{xyz[2]:.3f})"
            print(
                f"{n + 1:5d}/{len(mapped)} {phase} "
                f"frame={frame_path.name} gt_idx={gt_idx} pred_xyz={xyz_text} "
                f"gt_xyz=({gt_vec[0]:.3f},{gt_vec[1]:.3f},{gt_vec[2]:.3f})"
            )

    with args.output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "image_idx",
                "frame_name",
                "gt_idx",
                "pred_x",
                "pred_y",
                "pred_z",
                "gt_x",
                "gt_y",
                "gt_z",
                "err_x",
                "err_y",
                "err_z",
                "err_3d",
            ]
        )
        writer.writerows(rows)

    print("=" * 72)
    print("Legacy Formatter + ORB RMSE on evaluation section")
    print("=" * 72)
    print(f"Eval rows:        {len(rows)}")
    print(f"Finite rows:      {len(errors)}")
    print(f"Non-finite rows:  {nonfinite}")
    if errors:
        err_arr = np.vstack(errors)
        axis = np.sqrt(np.mean(np.square(err_arr), axis=0))
        err_3d = np.linalg.norm(err_arr, axis=1)
        total = float(np.sqrt(np.mean(np.sum(np.square(err_arr), axis=1))))
        print(f"RMSE X/Y/Z:       {axis[0]:.4f} / {axis[1]:.4f} / {axis[2]:.4f}")
        print(f"RMSE 3D:          {total:.4f}")
        print(f"Median 3D err:    {float(np.median(err_3d)):.4f}")
        print(f"P95 3D err:       {float(np.percentile(err_3d, 95)):.4f}")
    print(f"Predictions CSV:  {args.output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
