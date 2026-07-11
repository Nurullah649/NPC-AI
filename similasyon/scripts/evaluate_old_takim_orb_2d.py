#!/usr/bin/env python3
"""Evaluate old TAKIM_BAGLANTI_ARAYUZU ORB positioning logic in 2D.

This mirrors the positioning part in:

    TAKIM_BAGLANTI_ARAYUZU/src/object_detection_model.py

It intentionally:
- uses TAKIM_BAGLANTI_ARAYUZU/Class/CameraMovementTracker.py,
- uses Class.Calculate_Direction,
- keeps the old 2D-only translation logic,
- bypasses YOLO/detection,
- calls the tracker twice on the first frame as the old code does.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import cv2
import numpy as np
from sklearn.linear_model import LinearRegression


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Class.Calculate_Direction import Calculate_Direction  # noqa: E402
from TAKIM_BAGLANTI_ARAYUZU.Class.CameraMovementTracker import CameraMovementTracker  # noqa: E402


def load_gt(path: Path) -> np.ndarray:
    rows: list[list[float]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append([float(row["translation_x"]), float(row["translation_y"])])
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


def old_read_calibration_file() -> tuple[np.ndarray, np.ndarray]:
    camera_matrix = np.array(
        [
            [1.4133e03, 0.0, 950.0639],
            [0.0, 1.4188e03, 543.3796],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    dist_coeffs = np.array([-0.0091, 0.0666, 0.0, 0.0], dtype=np.float64)
    return camera_matrix, dist_coeffs


def resize_frame_if_needed(frame: np.ndarray, max_width: int) -> np.ndarray:
    if max_width <= 0 or frame.shape[1] <= max_width:
        return frame
    scale = max_width / float(frame.shape[1])
    new_h = max(1, int(round(frame.shape[0] * scale)))
    return cv2.resize(frame, (max_width, new_h), interpolation=cv2.INTER_AREA)


def finite_xy(v) -> tuple[float, float] | None:
    try:
        x = float(v[0])
        y = float(v[1])
    except Exception:
        return None
    if not math.isfinite(x) or not math.isfinite(y):
        return None
    return x, y


def main() -> int:
    parser = argparse.ArgumentParser(description="Old TAKIM_BAGLANTI_ARAYUZU ORB 2D evaluator")
    parser.add_argument("--frames", required=True, type=Path)
    parser.add_argument("--gt", required=True, type=Path)
    parser.add_argument("--fps-ratio", default=1, type=int)
    parser.add_argument("--gt-offset", default=0, type=int)
    parser.add_argument("--calib-image-frames", default=450, type=int)
    parser.add_argument("--limit", default=0, type=int)
    parser.add_argument("--max-width", default=0, type=int, help="Optional frame resize for speed; 0 keeps original")
    parser.add_argument(
        "--output-csv",
        default=REPO_ROOT / "similasyon" / "_debug" / "old_takim_orb_2d.csv",
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

    camera_matrix, dist_coeffs = old_read_calibration_file()
    first_gt = mapped[0][3]
    tracker = CameraMovementTracker(camera_matrix, dist_coeffs, np.array([first_gt[0], first_gt[1]]))

    is_first_frame = True
    calibration_frames = []
    positions_data = []
    gt_data = []
    scale_factor = None
    offset = None
    detected = None
    detected2 = None

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    rows: list[list[object]] = []
    errors: list[np.ndarray] = []

    print("=" * 72)
    print("Old TAKIM_BAGLANTI_ARAYUZU ORB 2D Evaluation")
    print("=" * 72)
    print(f"Frames dir:       {args.frames}")
    print(f"GT CSV:           {args.gt}")
    print(f"Mapped pairs:     {len(mapped)}")
    print(f"GT mapping:       gt_idx = {args.gt_offset} + image_idx * {args.fps_ratio}")
    print(f"Calibration imgs: first {args.calib_image_frames}")
    print(f"Max width:        {args.max_width or 'original'}")
    print("-" * 72)

    for n, (image_idx, frame_path, gt_idx, gt_vec) in enumerate(mapped):
        frame = cv2.imread(str(frame_path))
        if frame is None:
            raise ValueError(f"Frame could not be read: {frame_path}")
        frame = resize_frame_if_needed(frame, args.max_width)

        # Match old object_detection_model.py:
        # if self.tracker.is_first_frame: process old_frame
        # then always process new_frame again.
        if is_first_frame:
            is_first_frame = False
        if tracker.is_first_frame:
            tracker.process_frame(frame)
        tracker.process_frame(frame)
        positions = tracker.get_positions().copy()

        health_status = "1" if n < args.calib_image_frames else "0"
        if health_status == "1":
            calibration_frames.append((gt_vec[0], gt_vec[1]))
            gt_data.append([float(gt_vec[0]), float(gt_vec[1])])
            positions_data.append([float(positions[0]), float(positions[1])])
            pred = np.array([gt_vec[0], gt_vec[1]], dtype=np.float64)
        else:
            if scale_factor is None:
                detected = Calculate_Direction(gt_data=gt_data, alg_data=positions_data)
                if detected.calculate_direction_change():
                    gt_positions = np.array(gt_data)
                    alg_positions = np.array(positions_data)
                    model = LinearRegression(fit_intercept=False, positive=False)
                    model.fit(alg_positions, gt_positions)
                    scale_factor = model.coef_
                    offset = model.intercept_
                    scaled_positions = np.dot(positions, scale_factor.T) + offset
                    positions[0] = scaled_positions[0]
                    positions[1] = scaled_positions[1]
            elif detected.calculate_direction_change():
                scaled_positions = np.dot(positions, scale_factor.T) + offset
                positions[0] = scaled_positions[0]
                positions[1] = scaled_positions[1]
            else:
                positions[0] = positions[0] / detected.get_scale_factor()
                positions[1] = positions[1] / detected.get_scale_factor()
                match detected.compare_total_directions():
                    case 0:
                        ters_dizi = list(map(lambda pair: (pair[1], pair[0]), positions_data))
                        detected2 = Calculate_Direction(gt_data=gt_data, alg_data=ters_dizi)
                        match detected2.compare_total_directions():
                            case 1:
                                positions[0] = (positions[0] * -1) / detected.get_scale_factor()
                                positions[1] = (positions[1]) / detected.get_scale_factor()
                            case 2:
                                positions[0] = (positions[0]) / detected.get_scale_factor()
                                positions[1] = (positions[1] * -1)
                            case 3:
                                positions[0] = (positions[0] * -1) / detected.get_scale_factor()
                                positions[1] = (positions[1] * -1) / detected.get_scale_factor()
                            case 4:
                                positions[0] = (positions[0]) / detected.get_scale_factor()
                                positions[1] = (positions[1]) / detected.get_scale_factor()
                    case 1:
                        positions[0] = (positions[0] * -1) / detected.get_scale_factor()
                        positions[1] = (positions[1]) / detected.get_scale_factor()
                    case 2:
                        positions[0] = (positions[0]) / detected.get_scale_factor()
                        positions[1] = (positions[1] * -1) / detected.get_scale_factor()
                    case 3:
                        positions[0] = (positions[0] * -1) / detected.get_scale_factor()
                        positions[1] = (positions[1] * -1) / detected.get_scale_factor()
                    case 4:
                        positions[0] = (positions[0]) / detected.get_scale_factor()
                        positions[1] = (positions[1]) / detected.get_scale_factor()

            pred = np.array([float(positions[0]), float(positions[1])], dtype=np.float64)
            xy = finite_xy(pred)
            if xy is not None:
                err = pred - gt_vec
                errors.append(err)
                err_x, err_y = float(err[0]), float(err[1])
                err_2d = float(np.linalg.norm(err))
            else:
                err_x = err_y = err_2d = float("nan")

            rows.append(
                [
                    image_idx,
                    frame_path.name,
                    gt_idx,
                    pred[0],
                    pred[1],
                    gt_vec[0],
                    gt_vec[1],
                    err_x,
                    err_y,
                    err_2d,
                ]
            )

        if (n + 1) % 50 == 0 or n == len(mapped) - 1:
            phase = "CALIB" if health_status == "1" else "EVAL"
            print(
                f"{n + 1:5d}/{len(mapped)} {phase} "
                f"frame={frame_path.name} gt_idx={gt_idx} "
                f"pred=({pred[0]:.3f},{pred[1]:.3f}) "
                f"gt=({gt_vec[0]:.3f},{gt_vec[1]:.3f})"
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
                "gt_x",
                "gt_y",
                "err_x",
                "err_y",
                "err_2d",
            ]
        )
        writer.writerows(rows)

    print("=" * 72)
    print("Old ORB 2D E on evaluation section")
    print("=" * 72)
    print(f"Eval rows:        {len(rows)}")
    print(f"Finite rows:      {len(errors)}")
    if errors:
        err_arr = np.vstack(errors)
        e2 = np.linalg.norm(err_arr, axis=1)
        rmse2 = float(np.sqrt(np.mean(np.sum(np.square(err_arr), axis=1))))
        print(f"E 2D mean:        {float(np.mean(e2)):.4f}")
        print(f"RMSE 2D:          {rmse2:.4f}")
        print(f"Median 2D err:    {float(np.median(e2)):.4f}")
        print(f"P95 2D err:       {float(np.percentile(e2, 95)):.4f}")
    print(f"Predictions CSV:  {args.output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
