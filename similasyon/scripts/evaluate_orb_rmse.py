#!/usr/bin/env python3
"""Evaluate legacy Class/ORB.py trajectory by mapping 2D ORB motion to 3D GT.

GT mapping for 7.5 FPS images vs 30 FPS GT:

    image_0 -> gt_0
    image_1 -> gt_4
    image_2 -> gt_8

ORB produces a cumulative 2D image-plane motion. During the calibration window,
this script fits a linear model:

    [orb_x, orb_y] -> [gt_x, gt_y, gt_z]

Then it reports 3D RMSE on the evaluation window.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from pathlib import Path

import cv2
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge


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
        raise ValueError(f"Could not parse focal/principal point from: {path}")

    fx, fy = [float(x) for x in focal.group(1).split()]
    cx, cy = [float(x) for x in principal.group(1).split()]
    k = [float(x) for x in radial.group(1).split()] if radial else [0.0, 0.0]
    p = [float(x) for x in tangential.group(1).split()] if tangential else [0.0, 0.0]

    camera_matrix = np.array(
        [
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    dist_coeffs = np.array([k[0], k[1], p[0], p[1]], dtype=np.float64)
    return camera_matrix, dist_coeffs


def rmse(errors: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
    axis = np.sqrt(np.mean(np.square(errors), axis=0))
    err_3d = np.linalg.norm(errors, axis=1)
    total = float(np.sqrt(np.mean(np.sum(np.square(errors), axis=1))))
    return axis, total, err_3d


def main() -> int:
    parser = argparse.ArgumentParser(description="ORB 2D -> 3D RMSE evaluation")
    parser.add_argument("--frames", required=True, type=Path)
    parser.add_argument("--gt", required=True, type=Path)
    parser.add_argument("--calib-text", required=True, type=Path)
    parser.add_argument("--fps-ratio", default=4, type=int)
    parser.add_argument("--gt-offset", default=0, type=int)
    parser.add_argument("--calib-image-frames", default=450, type=int)
    parser.add_argument("--limit", default=0, type=int)
    parser.add_argument("--fit", choices=["linear", "ridge"], default="linear")
    parser.add_argument("--ridge-alpha", default=1.0, type=float)
    parser.add_argument(
        "--output-csv",
        default=REPO_ROOT / "similasyon" / "_debug" / "orb_rmse_predictions.csv",
        type=Path,
    )
    args = parser.parse_args()

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
    tracker = CameraMovementTracker(camera_matrix, dist_coeffs)

    orb_xy: list[np.ndarray] = []
    gt_xyz: list[np.ndarray] = []
    model = None
    rows: list[list[object]] = []
    errors: list[np.ndarray] = []
    nonfinite = 0

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("ORB 2D -> 3D RMSE Evaluation")
    print("=" * 72)
    print(f"Frames dir:       {args.frames}")
    print(f"GT CSV:           {args.gt}")
    print(f"Calibration text: {args.calib_text}")
    print(f"Mapped pairs:     {len(mapped)}")
    print(f"GT mapping:       gt_idx = {args.gt_offset} + image_idx * {args.fps_ratio}")
    print(f"Calibration imgs: first {args.calib_image_frames}")
    print(f"Fit:              {args.fit}")
    print("-" * 72)

    for n, (image_idx, frame_path, gt_idx, gt_vec) in enumerate(mapped):
        frame = cv2.imread(str(frame_path))
        if frame is None:
            raise ValueError(f"Frame could not be read: {frame_path}")

        tracker.process_frame(frame)
        pos2 = np.asarray(tracker.get_positions(), dtype=np.float64).reshape(2)

        if n < args.calib_image_frames:
            orb_xy.append(pos2.copy())
            gt_xyz.append(gt_vec.copy())
            pred = gt_vec.copy()
        else:
            if model is None:
                x_train = np.asarray(orb_xy, dtype=np.float64)
                y_train = np.asarray(gt_xyz, dtype=np.float64)
                if args.fit == "ridge":
                    model = Ridge(alpha=args.ridge_alpha, fit_intercept=True)
                else:
                    model = LinearRegression(fit_intercept=True)
                model.fit(x_train, y_train)

                train_pred = model.predict(x_train)
                train_err = train_pred - y_train
                train_axis, train_total, train_3d = rmse(train_err)
                print(
                    f"Calibration fit RMSE X/Y/Z={train_axis[0]:.4f}/"
                    f"{train_axis[1]:.4f}/{train_axis[2]:.4f}, "
                    f"3D={train_total:.4f}, median={float(np.median(train_3d)):.4f}"
                )

            pred = np.asarray(model.predict(pos2.reshape(1, -1))[0], dtype=np.float64)
            if not np.all(np.isfinite(pred)):
                nonfinite += 1
                err = np.array([np.nan, np.nan, np.nan], dtype=np.float64)
                err_3d = float("nan")
            else:
                err = pred - gt_vec
                err_3d = float(np.linalg.norm(err))
                errors.append(err)

            rows.append(
                [
                    image_idx,
                    frame_path.name,
                    gt_idx,
                    pos2[0],
                    pos2[1],
                    pred[0],
                    pred[1],
                    pred[2],
                    gt_vec[0],
                    gt_vec[1],
                    gt_vec[2],
                    err[0],
                    err[1],
                    err[2],
                    err_3d,
                ]
            )

        if (n + 1) % 50 == 0 or n == len(mapped) - 1:
            phase = "CALIB" if n < args.calib_image_frames else "EVAL"
            print(
                f"{n + 1:5d}/{len(mapped)} {phase} "
                f"frame={frame_path.name} gt_idx={gt_idx} "
                f"orb=({pos2[0]:.2f},{pos2[1]:.2f}) "
                f"pred=({pred[0]:.3f},{pred[1]:.3f},{pred[2]:.3f}) "
                f"gt=({gt_vec[0]:.3f},{gt_vec[1]:.3f},{gt_vec[2]:.3f})"
            )

    with args.output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "image_idx",
                "frame_name",
                "gt_idx",
                "orb_x",
                "orb_y",
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
    print("ORB RMSE on evaluation section")
    print("=" * 72)
    print(f"Eval rows:        {len(rows)}")
    print(f"Finite rows:      {len(errors)}")
    print(f"Non-finite rows:  {nonfinite}")
    if errors:
        err_arr = np.vstack(errors)
        axis, total, err_3d = rmse(err_arr)
        print(f"RMSE X/Y/Z:       {axis[0]:.4f} / {axis[1]:.4f} / {axis[2]:.4f}")
        print(f"RMSE 3D:          {total:.4f}")
        print(f"Median 3D err:    {float(np.median(err_3d)):.4f}")
        print(f"P95 3D err:       {float(np.percentile(err_3d, 95)):.4f}")
    print(f"Predictions CSV:  {args.output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
