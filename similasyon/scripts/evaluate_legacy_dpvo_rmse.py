#!/usr/bin/env python3
"""Evaluate legacy Formatter_for_yolo + DPVO_Obejct + Calculate_Direction path.

This intentionally uses the old classes instead of the new PositioningDPVO code.
GT mapping for 7.5 FPS images vs 30 FPS GT is:

    image_0 -> gt_0
    image_1 -> gt_4
    image_2 -> gt_8

The legacy formatter returns x/y/z, so this script reports 3D RMSE.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
import math
import tempfile
import sys
import types
from pathlib import Path

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def stub_detection_only_modules():
    """Avoid loading unrelated GPU detection/landing models in positioning-only tests."""
    stub = types.ModuleType("Class.Does_it_intersect")
    stub.does_other_center_intersect_and_predict = lambda *args, **kwargs: False
    sys.modules["Class.Does_it_intersect"] = stub


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


def reset_formatter_state(formatter_module, calib_path: Path | None):
    from Class.DPVO_Obejct import DPVO_object

    formatter_module.calibration_frames = []
    formatter_module.gt_data = []
    formatter_module.positions_data = []
    formatter_module.detected_objects = []
    formatter_module.scale_factor = None
    formatter_module.detected = None
    formatter_module.offset = None
    formatter_module.tracker = DPVO_object()
    if calib_path is not None:
        formatter_module.tracker.calib = str(calib_path)


def make_resized_frame(src: Path, dst_dir: Path, max_height: int) -> Path:
    if max_height <= 0:
        return src
    image = cv2.imread(str(src))
    if image is None:
        raise ValueError(f"Frame could not be read: {src}")
    h, w = image.shape[:2]
    if h <= max_height:
        return src
    scale = max_height / float(h)
    new_w = max(1, int(round(w * scale)))
    resized = cv2.resize(image, (new_w, max_height), interpolation=cv2.INTER_AREA)
    dst = dst_dir / f"{src.stem}_h{max_height}.jpg"
    cv2.imwrite(str(dst), resized, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    return dst


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
    parser = argparse.ArgumentParser(description="Legacy DPVO formatter RMSE")
    parser.add_argument("--frames", required=True, type=Path)
    parser.add_argument("--gt", required=True, type=Path)
    parser.add_argument("--calib", default=None, type=Path)
    parser.add_argument("--dpvo-config", default=REPO_ROOT / "Class" / "DPVO" / "config" / "npc.yaml", type=Path)
    parser.add_argument(
        "--no-merge-dpvo-config",
        action="store_true",
        help="Keep Class/DPVO/dpvo/config.py defaults, matching DPVO_Obejct.py's original behavior.",
    )
    parser.add_argument("--fps-ratio", default=4, type=int)
    parser.add_argument("--gt-offset", default=0, type=int)
    parser.add_argument("--calib-image-frames", default=450, type=int)
    parser.add_argument("--limit", default=0, type=int)
    parser.add_argument("--resize-max-height", default=0, type=int)
    parser.add_argument(
        "--output-csv",
        default=REPO_ROOT / "similasyon" / "_debug" / "legacy_dpvo_rmse_predictions.csv",
        type=Path,
    )
    parser.add_argument("--show-legacy-output", action="store_true")
    args = parser.parse_args()

    from Class.DPVO.dpvo.config import cfg
    stub_detection_only_modules()
    from Class import Formatter_for_yolo as legacy_formatter

    if not args.no_merge_dpvo_config:
        cfg.merge_from_file(str(args.dpvo_config))
    reset_formatter_state(legacy_formatter, args.calib)

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

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    rows: list[list[object]] = []
    errors: list[np.ndarray] = []
    nonfinite = 0

    print("=" * 72)
    print("Legacy DPVO RMSE Evaluation")
    print("=" * 72)
    print(f"Frames dir:       {args.frames}")
    print(f"GT CSV:           {args.gt}")
    print(f"Camera calib:     {args.calib}")
    print(f"DPVO config:      {'config.py defaults' if args.no_merge_dpvo_config else args.dpvo_config}")
    print(f"Mapped pairs:     {len(mapped)}")
    print(f"GT mapping:       gt_idx = {args.gt_offset} + image_idx * {args.fps_ratio}")
    print(f"Calibration imgs: first {args.calib_image_frames}")
    print(f"Resize max h:     {args.resize_max_height or 'original'}")
    print("-" * 72)

    with tempfile.TemporaryDirectory(prefix="legacy_dpvo_resized_") as tmp:
        tmp_dir = Path(tmp)
        for n, (image_idx, frame_path, gt_idx, gt_vec) in enumerate(mapped):
            health_status = "1" if n < args.calib_image_frames else "0"
            proc_frame_path = make_resized_frame(frame_path, tmp_dir, args.resize_max_height)
            if args.show_legacy_output:
                pred = legacy_formatter.formatter(
                    results=None,
                    path=str(proc_frame_path),
                    idx=image_idx,
                    gt_data_=gt_vec,
                    health_status=health_status,
                )
            else:
                with contextlib.redirect_stdout(io.StringIO()):
                    pred = legacy_formatter.formatter(
                        results=None,
                        path=str(proc_frame_path),
                        idx=image_idx,
                        gt_data_=gt_vec,
                        health_status=health_status,
                    )
            xyz = finite_xyz(pred)
            if health_status == "0":
                if xyz is None:
                    nonfinite += 1
                    pred_x = pred_y = pred_z = float("nan")
                    err_x = err_y = err_z = err_3d = float("nan")
                else:
                    pred_x, pred_y, pred_z = xyz
                    err = np.asarray([pred_x - gt_vec[0], pred_y - gt_vec[1], pred_z - gt_vec[2]], dtype=np.float64)
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
    print("Legacy 3D RMSE on health_status=0 evaluation section")
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
