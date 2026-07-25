#!/usr/bin/env python3
"""Align a frozen MASt3R trajectory with health=1 only and score health=0."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from trajectory_tools import fit_affine, load_ground_truth, relative_drift, trajectory_metrics


HERE = Path(__file__).resolve().parent


def _read_rows(path: Path, *, require_contiguous: bool = True) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"CSV is empty: {path}")
    indices = [int(row["sample_index"]) for row in rows]
    if len(indices) != len(set(indices)):
        raise ValueError(f"CSV contains duplicate sample_index values: {path}")
    if require_contiguous and indices != list(range(len(rows))):
        raise ValueError("Raw trajectory sample_index must be contiguous and start at zero")
    return rows


def _read_anchors(path: Path, calib_frames: int) -> tuple[np.ndarray, np.ndarray]:
    rows = _read_rows(path, require_contiguous=False)
    indices = np.asarray([int(row["sample_index"]) for row in rows], dtype=np.int64)
    if np.any(indices >= calib_frames):
        raise ValueError("Calibration anchor CSV contains a health=0 frame")
    points = np.asarray(
        [[row["raw_x"], row["raw_y"], row["raw_z"]] for row in rows],
        dtype=np.float64,
    )
    valid = np.all(np.isfinite(points), axis=1)
    indices, points = indices[valid], points[valid]
    order = np.argsort(indices)
    return indices[order], points[order]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", required=True, type=Path)
    parser.add_argument("--anchors", required=True, type=Path)
    parser.add_argument("--gt", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--runtime", default=None, type=Path)
    parser.add_argument("--calib-frames", default=450, type=int)
    parser.add_argument("--ridge-alpha", default=0.1, type=float)
    parser.add_argument("--condition-max", default=1.0e6, type=float)
    args = parser.parse_args()

    raw_rows = _read_rows(args.raw)
    gt_by_index = load_ground_truth(args.gt)
    missing = [index for index in range(len(raw_rows)) if index not in gt_by_index]
    if missing:
        raise ValueError(f"GT is missing {len(missing)} trajectory frame(s), first={missing[:5]}")
    if len(raw_rows) <= args.calib_frames:
        raise ValueError("Trajectory has no health=0 evaluation section")
    gt = np.asarray([gt_by_index[index] for index in range(len(raw_rows))])

    anchor_indices, anchor_raw = _read_anchors(args.anchors, args.calib_frames)
    if len(anchor_indices) < 4:
        raise ValueError(f"Need >=4 finite health=1 anchors, got {len(anchor_indices)}")
    anchor_gt = gt[anchor_indices]
    alignment = fit_affine(
        anchor_raw,
        anchor_gt,
        ridge_alpha=args.ridge_alpha,
        condition_max=args.condition_max,
        anchor_index=-1,
    )

    gauge_raw = np.asarray(
        [[row["gauge_x"], row["gauge_y"], row["gauge_z"]] for row in raw_rows],
        dtype=np.float64,
    )
    prediction = np.full_like(gt, np.nan)
    prediction[: args.calib_frames] = gt[: args.calib_frames]
    fallback_count = 0
    previous = prediction[args.calib_frames - 1].copy()
    for index in range(args.calib_frames, len(raw_rows)):
        if np.all(np.isfinite(gauge_raw[index])):
            candidate = alignment.apply(gauge_raw[index : index + 1])[0]
            if np.all(np.isfinite(candidate)):
                previous = candidate
            else:
                fallback_count += 1
        else:
            fallback_count += 1
        prediction[index] = previous

    evaluation = slice(args.calib_frames, len(raw_rows))
    summary = trajectory_metrics(prediction[evaluation], gt[evaluation])
    summary.update(
        {
            "algorithm": "MASt3R-SLAM causal competition replay",
            "raw_trajectory": str(args.raw.resolve()),
            "ground_truth": str(args.gt.resolve()),
            "sampled_frame_count": len(raw_rows),
            "calibration_sampled_frames": args.calib_frames,
            "evaluation_sampled_frames": len(raw_rows) - args.calib_frames,
            "health0_gt_visible_to_slam": False,
            "health0_gt_used_for_alignment": False,
            "health0_gt_used_only_for_offline_scoring": True,
            "causal_pose_snapshots": True,
            "calibration_anchor_count": len(anchor_indices),
            "first_calibration_anchor": int(anchor_indices[0]),
            "last_calibration_anchor": int(anchor_indices[-1]),
            "fallback_prediction_count": fallback_count,
            "relative_drift": relative_drift(prediction[evaluation], gt[evaluation]),
        }
    )
    if args.runtime is not None:
        with args.runtime.open("r", encoding="utf-8") as handle:
            runtime = json.load(handle)
        if runtime.get("health0_gt_visible_to_slam") is not False:
            raise ValueError("Runtime provenance does not prove GT isolation")
        summary["runtime"] = runtime

    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = args.output_dir / "predictions.csv"
    metrics_path = args.output_dir / "metrics.json"
    alignment_path = args.output_dir / "alignment.json"
    with predictions_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "sample_index", "health_status", "raw_x", "raw_y", "raw_z",
                "pred_x", "pred_y", "pred_z", "gt_x", "gt_y", "gt_z",
                "err_x", "err_y", "err_z", "err_3d",
            ]
        )
        for index, row in enumerate(raw_rows):
            error = prediction[index] - gt[index]
            writer.writerow(
                [
                    index,
                    "1" if index < args.calib_frames else "0",
                    *gauge_raw[index].tolist(),
                    *prediction[index].tolist(),
                    *gt[index].tolist(),
                    *error.tolist(),
                    float(np.linalg.norm(error)),
                ]
            )
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    matrix_singular = np.linalg.svd(alignment.matrix, compute_uv=False)
    with alignment_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "fit_method": "centered_affine_with_conditioned_ridge",
                "fit_uses_health1_only": True,
                "sample_count": len(anchor_indices),
                "matrix": alignment.matrix.tolist(),
                "translation": alignment.translation.tolist(),
                "anchor_offset": alignment.anchor_offset.tolist(),
                "source_rank": alignment.rank,
                "source_condition_number": alignment.condition_number,
                "source_singular_values": alignment.singular_values.tolist(),
                "used_ridge": alignment.used_ridge,
                "matrix_singular_values": matrix_singular.tolist(),
                "matrix_determinant": float(np.linalg.det(alignment.matrix)),
                "matrix_condition_number": float(np.linalg.cond(alignment.matrix)),
                "ned_y_row": alignment.matrix[1].tolist(),
            },
            handle,
            indent=2,
        )

    print("=" * 72)
    print("MASt3R-SLAM health=0 competition result")
    print("=" * 72)
    print(f"E 3D:       {summary['E_3d']:.6f} m")
    print(f"RMSE 3D:    {summary['RMSE_3d']:.6f} m")
    print("RMSE X/Y/Z: " + " / ".join(f"{x:.6f}" for x in summary["RMSE_xyz"]) + " m")
    print(f"Median/P95/Max: {summary['median_3d']:.6f} / {summary['p95_3d']:.6f} / {summary['max_3d']:.6f} m")
    print(f"Anchors:    {len(anchor_indices)} health=1 keyframes; ridge={alignment.used_ridge}")
    print(f"Outputs:    {predictions_path} / {metrics_path} / {alignment_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
