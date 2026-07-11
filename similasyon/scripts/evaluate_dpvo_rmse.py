#!/usr/bin/env python3
"""Evaluate PositioningDPVO against 30 FPS GT and 7.5 FPS image frames.

This script exercises only the positioning pipeline:

- reads image frames from a directory,
- maps each image frame to a GT row by `gt_index = image_zero_based * fps_ratio`,
- uses the first calibration window as health_status=1,
- evaluates DPVO predictions after that window as health_status=0,
- reports RMSE.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.models.positioning_dpvo import PositioningDPVO  # noqa: E402


def load_settings(path: Path) -> dict:
    import yaml

    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_gt(path: Path) -> np.ndarray:
    rows: list[list[float]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        required = {"translation_x", "translation_y", "translation_z"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"GT CSV missing columns: {sorted(missing)}")
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


def rmse(errors: np.ndarray) -> tuple[np.ndarray, float]:
    axis = np.sqrt(np.mean(np.square(errors), axis=0))
    total = float(np.sqrt(np.mean(np.sum(np.square(errors), axis=1))))
    return axis, total


def main() -> int:
    parser = argparse.ArgumentParser(description="DPVO-only RMSE evaluation")
    parser.add_argument("--frames", required=True, type=Path, help="7.5 FPS image frame directory")
    parser.add_argument("--gt", required=True, type=Path, help="30 FPS translation CSV")
    parser.add_argument("--settings", default=ROOT / "config" / "settings.yaml", type=Path)
    parser.add_argument("--fps-ratio", default=4, type=int, help="30 FPS GT / 7.5 FPS image ratio")
    parser.add_argument(
        "--gt-offset",
        default=0,
        type=int,
        help="GT row offset for image frame 0. image_i maps to gt_offset + i*fps_ratio",
    )
    parser.add_argument(
        "--calib-gt-frames",
        default=450,
        type=int,
        help="Number of 30 FPS GT rows to use as healthy calibration window",
    )
    parser.add_argument(
        "--limit",
        default=0,
        type=int,
        help="Optional max number of image frames to process",
    )
    parser.add_argument(
        "--output-csv",
        default=ROOT / "_debug" / "dpvo_rmse_predictions.csv",
        type=Path,
        help="Where to save per-frame predictions",
    )
    parser.add_argument(
        "--min-allowed-scale",
        default=None,
        type=float,
        help="Optional temporary override for dpvo.min_allowed_scale",
    )
    parser.add_argument(
        "--max-allowed-scale",
        default=None,
        type=float,
        help="Optional temporary override for dpvo.max_allowed_scale",
    )
    parser.add_argument(
        "--disable-direction-guard",
        action="store_true",
        help="Temporarily disable DirectionGuard for this evaluation run",
    )
    parser.add_argument(
        "--camera-calib",
        default=None,
        type=Path,
        help="Optional temporary override for model_paths.camera_calib",
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

    if not mapped:
        raise ValueError("No frame/GT pairs after FPS mapping.")

    calib_image_frames = int(math.ceil(args.calib_gt_frames / float(args.fps_ratio)))
    calib_image_frames = max(calib_image_frames, 1)
    if len(mapped) <= calib_image_frames:
        raise ValueError(
            f"Not enough mapped frames ({len(mapped)}) for calibration window ({calib_image_frames})."
        )

    settings = load_settings(args.settings)
    settings.setdefault("dpvo", {})
    if args.min_allowed_scale is not None:
        settings["dpvo"]["min_allowed_scale"] = args.min_allowed_scale
    if args.max_allowed_scale is not None:
        settings["dpvo"]["max_allowed_scale"] = args.max_allowed_scale
    if args.disable_direction_guard:
        settings["dpvo"]["use_direction_guard"] = False
    if args.camera_calib is not None:
        settings.setdefault("model_paths", {})
        settings["model_paths"]["camera_calib"] = str(args.camera_calib)
    positioning = PositioningDPVO(settings)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)

    eval_errors: list[np.ndarray] = []
    eval_rows: list[list[object]] = []

    print("=" * 72)
    print("DPVO RMSE Evaluation")
    print("=" * 72)
    print(f"Frames dir:      {args.frames}")
    print(f"GT CSV:          {args.gt}")
    print(f"Image frames:    {len(frames)}")
    print(f"GT rows:         {len(gt)}")
    print(f"Mapped pairs:    {len(mapped)}")
    print(f"FPS ratio:       {args.fps_ratio} GT rows / image")
    print(f"GT offset:       {args.gt_offset}")
    print(f"Calibration:     first {calib_image_frames} image frames ({args.calib_gt_frames} GT rows)")
    print("-" * 72)

    for n, (image_idx, frame_path, gt_idx, gt_vec) in enumerate(mapped):
        health_status = "1" if n < calib_image_frames else "0"
        pred = np.asarray(
            positioning.process_frame(
                frame_idx=image_idx,
                frame_path=str(frame_path),
                health_status=health_status,
                gt_x=float(gt_vec[0]),
                gt_y=float(gt_vec[1]),
                gt_z=float(gt_vec[2]),
            ),
            dtype=np.float64,
        )

        err = pred - gt_vec
        if health_status == "0":
            eval_errors.append(err)
            eval_rows.append(
                [
                    image_idx,
                    frame_path.name,
                    gt_idx,
                    pred[0],
                    pred[1],
                    pred[2],
                    gt_vec[0],
                    gt_vec[1],
                    gt_vec[2],
                    err[0],
                    err[1],
                    err[2],
                    float(np.linalg.norm(err)),
                ]
            )

        if (n + 1) % 50 == 0 or n == len(mapped) - 1:
            phase = "CALIB" if health_status == "1" else "EVAL"
            print(
                f"{n + 1:5d}/{len(mapped)} {phase} "
                f"frame={frame_path.name} gt_idx={gt_idx} "
                f"pred=({pred[0]:.3f},{pred[1]:.3f},{pred[2]:.3f}) "
                f"gt=({gt_vec[0]:.3f},{gt_vec[1]:.3f},{gt_vec[2]:.3f})"
            )

    if not eval_errors:
        raise ValueError("No evaluation frames; calibration window consumed all frames.")

    errors = np.vstack(eval_errors)
    axis_rmse, total_rmse = rmse(errors)
    mean_abs = np.mean(np.abs(errors), axis=0)
    median_3d = float(np.median(np.linalg.norm(errors, axis=1)))
    p95_3d = float(np.percentile(np.linalg.norm(errors, axis=1), 95))

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
        writer.writerows(eval_rows)

    print("=" * 72)
    print("RMSE on health_status=0 evaluation section")
    print("=" * 72)
    print(f"Eval frames:     {len(errors)}")
    print(f"RMSE X/Y/Z:      {axis_rmse[0]:.4f} / {axis_rmse[1]:.4f} / {axis_rmse[2]:.4f}")
    print(f"RMSE 3D:         {total_rmse:.4f}")
    print(f"MeanAbs X/Y/Z:   {mean_abs[0]:.4f} / {mean_abs[1]:.4f} / {mean_abs[2]:.4f}")
    print(f"Median 3D err:   {median_3d:.4f}")
    print(f"P95 3D err:      {p95_3d:.4f}")
    print(f"Predictions CSV: {args.output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
