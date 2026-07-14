#!/usr/bin/env python3
"""Reproduce the conservative 2026 offline DPVO fusion result.

The prediction path never reads health=0 ground truth.  Health=0 is consumed
only after every prediction has been frozen, to produce the evaluation report.
This is an offline SLAM result because the terminal PGO trajectory can revise
past poses after the video has ended; it must not be reported as a live result.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from offline_sweep import metrics, relative_drift
from plane_odometry_sweep import (
    decompose_plane_motion,
    load_data,
    load_homographies,
    predict as predict_plane,
)


CALIBRATION_FRAMES = 450
PLANE_XY_BLEND = 0.1
RELOCALIZATION_SCORE = 0.03
RELOCALIZATION_INLIERS = 50
RELOCALIZATION_INLIER_RATIO = 0.3
RELOCALIZATION_CONFIRMATIONS = 5
RELOCALIZATION_MIN_INNOVATION_M = 15.0
RELOCALIZATION_MAX_INNOVATION_M = 50.0
RELOCALIZATION_GAIN = 0.2
RELOCALIZATION_COOLDOWN_FRAMES = 100


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_terminal_trajectory(path: Path, expected_length: int) -> np.ndarray:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != expected_length:
        raise ValueError(
            f"Terminal trajectory length {len(rows)} != expected {expected_length}"
        )
    timestamps = np.asarray([float(row["input_timestamp"]) for row in rows])
    expected = np.arange(expected_length, dtype=np.float64)
    if not np.array_equal(timestamps, expected):
        raise ValueError("Terminal trajectory timestamps are not contiguous input frames")
    raw = np.asarray(
        [[float(row[key]) for key in ("raw_x", "raw_y", "raw_z")] for row in rows],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(raw)):
        raise ValueError("Terminal trajectory contains NaN/Inf")
    return raw


def load_relocalization(path: Path, expected_length: int) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != expected_length:
        raise ValueError(
            f"Relocalization length {len(rows)} != expected {expected_length}"
        )
    if [int(row["frame"]) for row in rows] != list(range(expected_length)):
        raise ValueError("Relocalization frames are not contiguous")
    return rows


def terminal_affine(
    raw: np.ndarray, calibration_gt: np.ndarray
) -> tuple[np.ndarray, np.ndarray, dict]:
    design = np.column_stack([raw[:CALIBRATION_FRAMES], np.ones(CALIBRATION_FRAMES)])
    coefficients, _, rank, singular = np.linalg.lstsq(
        design, calibration_gt, rcond=None
    )
    prediction = np.column_stack([raw, np.ones(len(raw))]) @ coefficients
    matrix = coefficients[:3].T
    matrix_singular = np.linalg.svd(matrix, compute_uv=False)
    return prediction, coefficients, {
        "design_rank": int(rank),
        "design_singular_values": singular.astype(float).tolist(),
        "matrix": matrix.astype(float).tolist(),
        "translation": coefficients[3].astype(float).tolist(),
        "matrix_singular_values": matrix_singular.astype(float).tolist(),
        "matrix_determinant": float(np.linalg.det(matrix)),
        "matrix_condition": float(np.linalg.cond(matrix)),
        "ned_y_row": matrix[1].astype(float).tolist(),
    }


def is_relocalization_candidate(row: dict[str, str]) -> bool:
    candidate = int(row["candidate_frame"])
    return (
        0 <= candidate < CALIBRATION_FRAMES
        and float(row["dbow_score"]) >= RELOCALIZATION_SCORE
        and int(row["homography_inliers"]) >= RELOCALIZATION_INLIERS
        and float(row["homography_inlier_ratio"])
        >= RELOCALIZATION_INLIER_RATIO
        and int(row["fundamental_inliers"]) >= RELOCALIZATION_INLIERS
        and float(row["fundamental_inlier_ratio"])
        >= RELOCALIZATION_INLIER_RATIO
    )


def apply_relocalization(
    prediction: np.ndarray,
    calibration_gt: np.ndarray,
    candidates: list[dict[str, str]],
) -> tuple[np.ndarray, list[dict]]:
    output = prediction.copy()
    correction = np.zeros(3, dtype=np.float64)
    run: list[tuple[int, int, np.ndarray]] = []
    last_event = -10**9
    events: list[dict] = []
    for frame in range(CALIBRATION_FRAMES, len(output)):
        output[frame] = prediction[frame] + correction
        row = candidates[frame]
        if not is_relocalization_candidate(row):
            run = []
            continue
        if run and frame - run[-1][0] > 2:
            run = []
        candidate = int(row["candidate_frame"])
        # This indexing is intentionally bounded to health=1 by the candidate
        # predicate above.  No health=0 GT can enter the measurement.
        run.append((frame, candidate, calibration_gt[candidate].copy()))
        run = run[-RELOCALIZATION_CONFIRMATIONS:]
        if (
            len(run) < RELOCALIZATION_CONFIRMATIONS
            or frame - last_event < RELOCALIZATION_COOLDOWN_FRAMES
        ):
            continue

        measurement = np.median(np.vstack([item[2] for item in run]), axis=0)
        innovation = measurement - output[frame]
        innovation_norm = float(np.linalg.norm(innovation))
        if not (
            RELOCALIZATION_MIN_INNOVATION_M
            <= innovation_norm
            <= RELOCALIZATION_MAX_INNOVATION_M
        ):
            continue
        correction += RELOCALIZATION_GAIN * innovation
        output[frame] = prediction[frame] + correction
        events.append(
            {
                "frame": frame,
                "candidate_frame": candidate,
                "confirmed_candidate_frames": [item[1] for item in run],
                "measurement_ned": measurement.astype(float).tolist(),
                "innovation_m": innovation.astype(float).tolist(),
                "innovation_norm_m": innovation_norm,
                "gain": RELOCALIZATION_GAIN,
                "applied_correction_m": correction.astype(float).tolist(),
            }
        )
        last_event = frame
        run = []
    return output, events


def write_predictions(
    path: Path, prediction: np.ndarray, target: np.ndarray
) -> None:
    error = prediction - target
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "sample_index",
                "health_status",
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
        for frame in range(len(prediction)):
            writer.writerow(
                [
                    frame,
                    1 if frame < CALIBRATION_FRAMES else 0,
                    *prediction[frame].astype(float).tolist(),
                    *target[frame].astype(float).tolist(),
                    *error[frame].astype(float).tolist(),
                    float(np.linalg.norm(error[frame])),
                ]
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--planar-motion", required=True, type=Path)
    parser.add_argument("--terminal-trajectory", required=True, type=Path)
    parser.add_argument("--relocalization", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    data = load_data(args.predictions, args.planar_motion)
    if len(data.calibration) != CALIBRATION_FRAMES:
        raise ValueError(
            f"Expected {CALIBRATION_FRAMES} calibration frames, got {len(data.calibration)}"
        )
    terminal_raw = load_terminal_trajectory(args.terminal_trajectory, len(data.raw))
    candidates = load_relocalization(args.relocalization, len(data.raw))

    intrinsics = np.asarray(
        [[463.23333740234375, 0.0, 318.0023193359375],
         [0.0, 462.3666687011719, 186.2986602783203],
         [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    plane_delta, decomposition = decompose_plane_motion(
        load_homographies(args.planar_motion), intrinsics
    )
    plane_evaluation, plane_diagnostics = predict_plane(
        data,
        plane_delta,
        data.calibration,
        data.evaluation,
        PLANE_XY_BLEND,
    )
    terminal_prediction, _, affine_diagnostics = terminal_affine(
        terminal_raw, data.gt[:CALIBRATION_FRAMES]
    )

    # Output during health=1 is the supplied GT, exactly as in the live path.
    hybrid = data.gt.copy()
    hybrid[data.evaluation, 0] = plane_evaluation[:, 0]
    hybrid[data.evaluation, 1:] = terminal_prediction[data.evaluation, 1:]
    final_prediction, events = apply_relocalization(
        hybrid, data.gt[:CALIBRATION_FRAMES], candidates
    )

    final_metrics = metrics(
        final_prediction[data.evaluation], data.gt[data.evaluation]
    )
    report = {
        "schema_version": 1,
        "result_class": "offline_slam_route_development",
        "live_result": False,
        "protocol": {
            "health0_gt_used_for_prediction": False,
            "health0_gt_used_for_fit": False,
            "health0_gt_used_for_report_metrics_only": True,
            "independent_holdout": False,
            "warning": (
                "The method was developed on this 2026 route. Validate unchanged "
                "parameters on a separate route before a deployment claim."
            ),
        },
        "fixed_parameters": {
            "calibration_frames": CALIBRATION_FRAMES,
            "plane_xy_blend": PLANE_XY_BLEND,
            "hybrid_axis_sources": {
                "NED_X": "health1-selected ground-plane/DPVO blend",
                "NED_Y": "terminal classic-loop PGO trajectory",
                "NED_Z": "terminal classic-loop PGO trajectory",
            },
            "relocalization": {
                "dbow_score": RELOCALIZATION_SCORE,
                "minimum_inliers": RELOCALIZATION_INLIERS,
                "minimum_inlier_ratio": RELOCALIZATION_INLIER_RATIO,
                "confirmations": RELOCALIZATION_CONFIRMATIONS,
                "innovation_min_m": RELOCALIZATION_MIN_INNOVATION_M,
                "innovation_max_m": RELOCALIZATION_MAX_INNOVATION_M,
                "gain": RELOCALIZATION_GAIN,
                "cooldown_frames": RELOCALIZATION_COOLDOWN_FRAMES,
            },
        },
        "metrics": final_metrics,
        "relative_drift": relative_drift(
            final_prediction[data.evaluation], data.gt[data.evaluation]
        ),
        "component_metrics": {
            "terminal_affine": metrics(
                terminal_prediction[data.evaluation], data.gt[data.evaluation]
            ),
            "plane": metrics(plane_evaluation, data.gt[data.evaluation]),
            "hybrid_before_relocalization": metrics(
                hybrid[data.evaluation], data.gt[data.evaluation]
            ),
        },
        "relocalization_events": events,
        "terminal_affine_diagnostics": affine_diagnostics,
        "plane_decomposition": decomposition,
        "plane_fit_diagnostics": plane_diagnostics,
        "inputs": {
            key: {"path": str(path), "sha256": sha256(path)}
            for key, path in {
                "predictions": args.predictions,
                "planar_motion": args.planar_motion,
                "terminal_trajectory": args.terminal_trajectory,
                "relocalization": args.relocalization,
            }.items()
        },
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_predictions(
        args.output_dir / "predictions.csv", final_prediction, data.gt
    )
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        f"Offline fusion: E={final_metrics['E_3d']:.6f} "
        f"RMSE={final_metrics['RMSE_3d']:.6f} "
        f"p95={final_metrics['p95_3d']:.6f} max={final_metrics['max_3d']:.6f}"
    )
    print(f"Relocalization events: {len(events)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
