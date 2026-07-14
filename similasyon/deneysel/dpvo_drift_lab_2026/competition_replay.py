#!/usr/bin/env python3
"""Replay the 2026 fusion under the one-shot competition protocol.

Unlike :mod:`offline_fusion`, this evaluator never consumes the terminal PGO
trajectory.  A prediction is committed as soon as its frame is visited and is
never changed afterwards.  Health=0 ground truth is read only after both
candidate streams have been frozen, for reporting metrics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from offline_fusion import (
    CALIBRATION_FRAMES,
    PLANE_XY_BLEND,
    RELOCALIZATION_CONFIRMATIONS,
    RELOCALIZATION_COOLDOWN_FRAMES,
    RELOCALIZATION_INLIERS,
    RELOCALIZATION_INLIER_RATIO,
    RELOCALIZATION_MAX_INNOVATION_M,
    RELOCALIZATION_MIN_INNOVATION_M,
    RELOCALIZATION_SCORE,
    is_relocalization_candidate,
    load_relocalization,
    write_predictions,
)
from offline_sweep import metrics, relative_drift
from plane_odometry_sweep import (
    decompose_plane_motion,
    load_data,
    load_homographies,
    predict as predict_plane,
)


LOCKED_RELOCALIZATION_GAIN = 0.2
EXPLORATORY_RELOCALIZATION_GAIN = 0.9
SIMILARITY_TOLERANCE = 0.10


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def apply_causal_relocalization(
    prediction: np.ndarray,
    calibration_gt: np.ndarray,
    candidates: list[dict[str, str]],
    *,
    gain: float,
) -> tuple[np.ndarray, list[dict]]:
    """Apply map corrections without ever changing an emitted prefix."""

    if len(calibration_gt) != CALIBRATION_FRAMES:
        raise ValueError("Relocalization map must contain health=1 GT only")
    if len(prediction) != len(candidates):
        raise ValueError("Prediction/candidate lengths differ")
    if not 0.0 <= gain <= 1.0:
        raise ValueError("Relocalization gain must be in [0, 1]")

    committed: list[np.ndarray] = []
    correction = np.zeros(3, dtype=np.float64)
    run: list[tuple[int, int, np.ndarray]] = []
    last_event = -10**9
    events: list[dict] = []
    for frame in range(len(prediction)):
        current = np.asarray(prediction[frame], dtype=np.float64) + correction
        if frame < CALIBRATION_FRAMES:
            committed.append(current.copy())
            continue

        row = candidates[frame]
        if not is_relocalization_candidate(row):
            run = []
            committed.append(current.copy())
            continue
        if run and frame - run[-1][0] > 2:
            run = []
        candidate = int(row["candidate_frame"])
        # Candidate eligibility bounds this lookup to the health=1 map.
        run.append((frame, candidate, calibration_gt[candidate].copy()))
        run = run[-RELOCALIZATION_CONFIRMATIONS:]
        if (
            len(run) >= RELOCALIZATION_CONFIRMATIONS
            and frame - last_event >= RELOCALIZATION_COOLDOWN_FRAMES
        ):
            measurement = np.median(np.vstack([item[2] for item in run]), axis=0)
            innovation = measurement - current
            innovation_norm = float(np.linalg.norm(innovation))
            if (
                RELOCALIZATION_MIN_INNOVATION_M
                <= innovation_norm
                <= RELOCALIZATION_MAX_INNOVATION_M
            ):
                correction += gain * innovation
                current = np.asarray(prediction[frame], dtype=np.float64) + correction
                events.append(
                    {
                        "frame": frame,
                        "candidate_frame": candidate,
                        "confirmed_candidate_frames": [item[1] for item in run],
                        "measurement_ned": measurement.astype(float).tolist(),
                        "innovation_norm_m": innovation_norm,
                        "gain": float(gain),
                        "applied_correction_m": correction.astype(float).tolist(),
                    }
                )
                last_event = frame
                run = []
        committed.append(current.copy())
    return np.vstack(committed), events


def acceptance_gate(candidate: dict, offline: dict) -> dict:
    limits = {
        key: float(offline[key]) * (1.0 + SIMILARITY_TOLERANCE)
        for key in ("E_3d", "RMSE_3d", "p95_3d", "max_3d")
    }
    checks = {key: float(candidate[key]) <= limit for key, limit in limits.items()}
    return {
        "tolerance_fraction": SIMILARITY_TOLERANCE,
        "limits": limits,
        "checks": checks,
        "passed": bool(all(checks.values())),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--planar-motion", required=True, type=Path)
    parser.add_argument("--relocalization", required=True, type=Path)
    parser.add_argument("--offline-report", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    data = load_data(args.predictions, args.planar_motion)
    if len(data.calibration) != CALIBRATION_FRAMES:
        raise ValueError(
            f"Expected {CALIBRATION_FRAMES} calibration frames, got "
            f"{len(data.calibration)}"
        )
    candidates = load_relocalization(args.relocalization, len(data.raw))
    offline_report = json.loads(args.offline_report.read_text(encoding="utf-8"))
    offline_metrics = offline_report["metrics"]

    intrinsics = np.asarray(
        [
            [463.23333740234375, 0.0, 318.0023193359375],
            [0.0, 462.3666687011719, 186.2986602783203],
            [0.0, 0.0, 1.0],
        ],
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

    # Health=1 is an exact pass-through.  Every health=0 base pose depends on
    # calibration data, raw DPVO up to that frame and planar deltas up to that
    # frame only.  The terminal trajectory is deliberately absent.
    causal_base = np.empty_like(data.gt)
    causal_base[data.calibration] = data.gt[data.calibration]
    causal_base[data.evaluation] = plane_evaluation
    locked, locked_events = apply_causal_relocalization(
        causal_base,
        data.gt[:CALIBRATION_FRAMES],
        candidates,
        gain=LOCKED_RELOCALIZATION_GAIN,
    )
    exploratory, exploratory_events = apply_causal_relocalization(
        causal_base,
        data.gt[:CALIBRATION_FRAMES],
        candidates,
        gain=EXPLORATORY_RELOCALIZATION_GAIN,
    )

    # GT below this line is used for scoring only; both output streams are
    # already immutable numpy arrays.
    evaluation_gt = data.gt[data.evaluation]
    locked_metrics = metrics(locked[data.evaluation], evaluation_gt)
    exploratory_metrics = metrics(exploratory[data.evaluation], evaluation_gt)
    report = {
        "schema_version": 1,
        "result_class": "causal_competition_replay",
        "online_causal_result": True,
        "live_server_run": False,
        "production_integrated": False,
        "protocol": {
            "one_prediction_per_frame": True,
            "historical_prediction_revisions": 0,
            "terminal_pgo_used": False,
            "future_frames_used": False,
            "health0_gt_visible_to_positioner": False,
            "health0_gt_used_for_report_metrics_only": True,
            "independent_holdout": False,
            "warning": (
                "The exploratory gain was inspected on this route and is not "
                "eligible for production integration without an independent route."
            ),
        },
        "offline_reference": offline_metrics,
        "locked_candidate": {
            "gain": LOCKED_RELOCALIZATION_GAIN,
            "metrics": locked_metrics,
            "relative_drift": relative_drift(
                locked[data.evaluation], evaluation_gt
            ),
            "events": locked_events,
            "acceptance": acceptance_gate(locked_metrics, offline_metrics),
        },
        "exploratory_candidate": {
            "gain": EXPLORATORY_RELOCALIZATION_GAIN,
            "metrics": exploratory_metrics,
            "relative_drift": relative_drift(
                exploratory[data.evaluation], evaluation_gt
            ),
            "events": exploratory_events,
            "acceptance": acceptance_gate(exploratory_metrics, offline_metrics),
            "production_eligible": False,
        },
        "fixed_parameters": {
            "calibration_frames": CALIBRATION_FRAMES,
            "plane_xy_blend": PLANE_XY_BLEND,
            "dbow_score": RELOCALIZATION_SCORE,
            "minimum_inliers": RELOCALIZATION_INLIERS,
            "minimum_inlier_ratio": RELOCALIZATION_INLIER_RATIO,
        },
        "plane_decomposition": decomposition,
        "plane_fit_diagnostics": plane_diagnostics,
        "integration_decision": "no_go",
        "inputs": {
            key: {"path": str(path), "sha256": sha256(path)}
            for key, path in {
                "predictions": args.predictions,
                "planar_motion": args.planar_motion,
                "relocalization": args.relocalization,
                "offline_report": args.offline_report,
            }.items()
        },
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_predictions(
        args.output_dir / "locked_predictions.csv", locked, data.gt
    )
    write_predictions(
        args.output_dir / "exploratory_predictions.csv", exploratory, data.gt
    )
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        "Competition replay: "
        f"locked E={locked_metrics['E_3d']:.6f} "
        f"RMSE={locked_metrics['RMSE_3d']:.6f} "
        f"p95={locked_metrics['p95_3d']:.6f} "
        f"max={locked_metrics['max_3d']:.6f}; "
        f"exploratory E={exploratory_metrics['E_3d']:.6f}; "
        "decision=no_go"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
