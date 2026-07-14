#!/usr/bin/env python3
"""Leakage-safe ground-plane homography odometry experiment.

The homography is decomposed with the calibrated camera matrix.  A positive
near-optical-axis plane normal selects the physical solution without GT.  The
translation-over-plane-distance vector is metrically propagated from the
plane-distance recurrence, then mapped to NED using health=1 only.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np

from offline_sweep import (
    fit_linear,
    folds,
    load_data,
    metrics,
    pose_affine,
    relative_drift,
)


H_FIELDS = tuple(f"homography_{i}{j}" for i in range(3) for j in range(3))


def load_homographies(path: Path) -> np.ndarray:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    missing = set(H_FIELDS) - set(rows[0] if rows else ())
    if missing:
        raise ValueError(f"Homography CSV eksik sütunlar: {sorted(missing)}")
    result = np.repeat(np.eye(3)[None], len(rows), axis=0)
    for index, row in enumerate(rows[1:], start=1):
        result[index] = np.asarray(
            [[float(row[f"homography_{i}{j}"]) for j in range(3)] for i in range(3)],
            dtype=np.float64,
        )
    if not np.all(np.isfinite(result)):
        raise ValueError("Homography dizisi NaN veya Inf içeriyor.")
    return result


def decompose_plane_motion(
    homographies: np.ndarray, intrinsics: np.ndarray
) -> tuple[np.ndarray, dict]:
    """Return arbitrary-metric camera translation in the initial world frame."""
    camera_to_initial = np.eye(3, dtype=np.float64)
    plane_distance = 1.0
    delta = np.zeros((len(homographies), 3), dtype=np.float64)
    factors: list[float] = []
    normals: list[np.ndarray] = []
    solution_indices: list[int] = []
    for index in range(1, len(homographies)):
        count, rotations, translations, plane_normals = cv2.decomposeHomographyMat(
            homographies[index], intrinsics
        )
        candidates = []
        for solution in range(count):
            normal = np.asarray(plane_normals[solution]).reshape(3)
            if normal[2] < 0.0:
                continue
            candidates.append(
                (
                    -float(normal[2]),
                    solution,
                    np.asarray(rotations[solution], dtype=np.float64),
                    np.asarray(translations[solution], dtype=np.float64).reshape(3),
                    normal,
                )
            )
        if not candidates:
            raise RuntimeError(f"Frame {index}: pozitif düzlem normalli çözüm yok.")
        _, solution, rotation, translation_over_distance, normal = min(
            candidates, key=lambda item: item[0]
        )

        # OpenCV gives X_current = R X_previous + t.  The new camera centre in
        # previous coordinates is -R^T t; rotate it into the initial frame.
        delta[index] = -camera_to_initial @ rotation.T @ (
            plane_distance * translation_over_distance
        )
        camera_to_initial = camera_to_initial @ rotation.T

        # If n^T X_previous=d, after X_current=R X_previous+t the plane
        # distance is d_current=d+(R n)^T t.  t here is normalized by d.
        factor = 1.0 + float((rotation @ normal) @ translation_over_distance)
        # A single noisy decomposition must not destroy the remaining path.
        factor = float(np.clip(factor, 0.8, 1.2))
        plane_distance *= factor
        factors.append(factor)
        normals.append(normal)
        solution_indices.append(solution)

    normal_array = np.vstack(normals)
    return delta, {
        "solution_counts": np.bincount(solution_indices, minlength=4).tolist(),
        "factor_percentiles": np.percentile(factors, [1, 5, 50, 95, 99]).tolist(),
        "plane_distance_end_over_start": float(plane_distance),
        "median_plane_normal_camera": np.median(normal_array, axis=0).tolist(),
    }


def predict(data, plane_delta, train, future, blend):
    baseline, baseline_diag = pose_affine(
        data.raw[train], data.gt[train], data.raw[future], ridge=1e-3
    )
    gt_delta = np.vstack([np.zeros((1, 3)), np.diff(data.gt, axis=0)])
    weights, plane_diag = fit_linear(
        plane_delta[train[1:]],
        gt_delta[train[1:]],
        ridge=1e-6,
        robust=True,
    )
    plane_prediction = data.gt[train[-1]] + np.cumsum(
        plane_delta[future] @ weights, axis=0
    )
    prediction = baseline.copy()
    prediction[:, :2] = (
        (1.0 - blend) * baseline[:, :2]
        + blend * plane_prediction[:, :2]
    )
    return prediction, {
        "baseline": baseline_diag,
        "plane_fit": plane_diag,
        "plane_to_ned_weights": weights.tolist(),
        "xy_blend": float(blend),
        "z_source": "dpvo_pose_ridge_1e-3",
    }


def run(predictions: Path, planar: Path, intrinsics: np.ndarray) -> dict:
    data = load_data(predictions, planar)
    homographies = load_homographies(planar)
    if len(homographies) != len(data.raw):
        raise ValueError("Homography/prediction satır sayısı farklı.")
    plane_delta, decomposition = decompose_plane_motion(homographies, intrinsics)
    blends = np.linspace(0.0, 1.0, 11)
    cv_folds = folds(data.calibration)
    candidates = []
    for blend in blends:
        chunks = {"expanding": [], "rolling180": []}
        for fold in cv_folds:
            prediction, _ = predict(
                data, plane_delta, fold["train"], fold["validation"], float(blend)
            )
            chunks[fold["protocol"]].append(
                prediction - data.gt[fold["validation"]]
            )
        protocol_metrics = {
            name: metrics(np.vstack(value), np.zeros_like(np.vstack(value)))
            for name, value in chunks.items()
        }
        candidates.append(
            {
                "xy_blend": float(blend),
                "protocol_metrics": protocol_metrics,
                "selection_score": {
                    "worst_E_3d": max(v["E_3d"] for v in protocol_metrics.values()),
                    "worst_RMSE_3d": max(
                        v["RMSE_3d"] for v in protocol_metrics.values()
                    ),
                    "worst_p95_3d": max(
                        v["p95_3d"] for v in protocol_metrics.values()
                    ),
                },
            }
        )

    def selection_key(item):
        score = item["selection_score"]
        return (
            score["worst_E_3d"],
            score["worst_RMSE_3d"],
            score["worst_p95_3d"],
            item["xy_blend"],
        )

    selected = min(candidates, key=selection_key)
    evaluation = data.evaluation
    final_prediction, diagnostics = predict(
        data,
        plane_delta,
        data.calibration,
        evaluation,
        selected["xy_blend"],
    )
    final = {
        "metrics": metrics(final_prediction, data.gt[evaluation]),
        "relative_drift": relative_drift(final_prediction, data.gt[evaluation]),
    }

    # This block is explicitly diagnostic. It quantifies headroom but is not
    # eligible for a deployment/result claim because it consults health=0 GT.
    oracle = []
    for candidate in candidates:
        prediction, _ = predict(
            data,
            plane_delta,
            data.calibration,
            evaluation,
            candidate["xy_blend"],
        )
        oracle.append(
            {
                "xy_blend": candidate["xy_blend"],
                "metrics": metrics(prediction, data.gt[evaluation]),
            }
        )
    oracle_best = min(oracle, key=lambda item: item["metrics"]["E_3d"])
    return {
        "schema_version": 1,
        "protocol": {
            "candidate_selection": "health1 chronological expanding+rolling holdouts only",
            "health0_used_for_selected_fit_or_selection": False,
            "oracle_block_is_not_eligible": True,
        },
        "intrinsics": intrinsics.tolist(),
        "decomposition": decomposition,
        "candidates": candidates,
        "selected": selected,
        "final_health0": final,
        "final_fit_diagnostics": diagnostics,
        "research_only_health0_oracle": oracle_best,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--planar-motion", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--fx", type=float, default=463.23333740234375)
    parser.add_argument("--fy", type=float, default=462.3666687011719)
    parser.add_argument("--cx", type=float, default=318.0023193359375)
    parser.add_argument("--cy", type=float, default=186.2986602783203)
    args = parser.parse_args()
    intrinsics = np.asarray(
        [[args.fx, 0.0, args.cx], [0.0, args.fy, args.cy], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    report = run(args.predictions, args.planar_motion, intrinsics)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    score = report["final_health0"]["metrics"]
    oracle = report["research_only_health0_oracle"]
    print(
        f"Seçilen XY blend={report['selected']['xy_blend']:.2f}: "
        f"E={score['E_3d']:.6f} RMSE={score['RMSE_3d']:.6f} "
        f"p95={score['p95_3d']:.6f} max={score['max_3d']:.6f}"
    )
    print(
        "Yalnız headroom/oracle (sonuç değil): "
        f"blend={oracle['xy_blend']:.2f} E={oracle['metrics']['E_3d']:.6f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

