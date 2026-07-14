#!/usr/bin/env python3
"""Produce axis, heading, scale, cross-track and lag diagnostics."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from offline_sweep import metrics, relative_drift


AXES = "xyz"


def _slope(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    centered = x - np.mean(x)
    denominator = float(centered @ centered)
    return float(centered @ (y - np.mean(y)) / denominator) if denominator else 0.0


def _wrap_angle(value: np.ndarray) -> np.ndarray:
    return (value + np.pi) % (2.0 * np.pi) - np.pi


def load_predictions(path: Path) -> dict[str, np.ndarray]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {
        "sample_index",
        "health_status",
        *(f"pred_{axis}" for axis in AXES),
        *(f"gt_{axis}" for axis in AXES),
    }
    missing = required - set(rows[0] if rows else ())
    if missing:
        raise ValueError(f"Prediction CSV eksik sütunlar: {sorted(missing)}")
    return {
        "sample_index": np.asarray([int(row["sample_index"]) for row in rows]),
        "health": np.asarray([int(float(row["health_status"])) for row in rows]),
        "prediction": np.asarray(
            [[float(row[f"pred_{axis}"]) for axis in AXES] for row in rows]
        ),
        "gt": np.asarray(
            [[float(row[f"gt_{axis}"]) for axis in AXES] for row in rows]
        ),
    }


def rolling_motion(prediction: np.ndarray, gt: np.ndarray, window: int = 150) -> dict:
    scale = []
    heading = []
    end_indices = []
    for end in range(window, len(prediction)):
        pred_delta = prediction[end, :2] - prediction[end - window, :2]
        gt_delta = gt[end, :2] - gt[end - window, :2]
        gt_distance = float(np.linalg.norm(gt_delta))
        if gt_distance < 5.0:
            continue
        scale.append(float(np.linalg.norm(pred_delta) / gt_distance))
        heading.append(
            float(
                _wrap_angle(
                    np.arctan2(pred_delta[1], pred_delta[0])
                    - np.arctan2(gt_delta[1], gt_delta[0])
                )
            )
        )
        end_indices.append(end)
    scale_array = np.asarray(scale)
    heading_array = np.asarray(heading)
    edge_count = min(200, len(scale_array))
    return {
        "window_frames": window,
        "count": len(scale_array),
        "scale_ratio": {
            "median": float(np.median(scale_array)),
            "p05": float(np.percentile(scale_array, 5)),
            "p95": float(np.percentile(scale_array, 95)),
            "first_median": float(np.median(scale_array[:edge_count])),
            "last_median": float(np.median(scale_array[-edge_count:])),
            "slope_per_100_frames": 100.0 * _slope(
                np.asarray(end_indices), scale_array
            ),
        },
        "heading_error_degrees": {
            "signed_median": float(np.degrees(np.median(heading_array))),
            "mae": float(np.degrees(np.mean(np.abs(heading_array)))),
            "p95_abs": float(np.degrees(np.percentile(np.abs(heading_array), 95))),
            "first_signed_median": float(
                np.degrees(np.median(heading_array[:edge_count]))
            ),
            "last_signed_median": float(
                np.degrees(np.median(heading_array[-edge_count:]))
            ),
        },
    }


def cross_track(error: np.ndarray, gt: np.ndarray, tangent_half_window: int = 15) -> dict:
    tangent = np.zeros((len(gt), 2), dtype=np.float64)
    for index in range(len(gt)):
        start = max(0, index - tangent_half_window)
        end = min(len(gt) - 1, index + tangent_half_window)
        tangent[index] = gt[end, :2] - gt[start, :2]
    tangent /= np.maximum(np.linalg.norm(tangent, axis=1, keepdims=True), 1e-9)
    along = np.sum(error[:, :2] * tangent, axis=1)
    cross = error[:, 0] * (-tangent[:, 1]) + error[:, 1] * tangent[:, 0]
    return {
        "tangent_half_window": tangent_half_window,
        "along_track": {
            "mean_signed_m": float(np.mean(along)),
            "rmse_m": float(np.sqrt(np.mean(along**2))),
            "p95_abs_m": float(np.percentile(np.abs(along), 95)),
            "slope_m_per_100_frames": 100.0 * _slope(np.arange(len(along)), along),
        },
        "cross_track": {
            "mean_signed_m": float(np.mean(cross)),
            "rmse_m": float(np.sqrt(np.mean(cross**2))),
            "p95_abs_m": float(np.percentile(np.abs(cross), 95)),
            "slope_m_per_100_frames": 100.0 * _slope(np.arange(len(cross)), cross),
        },
    }


def lag_scan(prediction: np.ndarray, gt: np.ndarray, radius: int = 5) -> dict:
    scores = []
    for lag in range(-radius, radius + 1):
        if lag < 0:
            pred_view, gt_view = prediction[-lag:], gt[:lag]
        elif lag > 0:
            pred_view, gt_view = prediction[:-lag], gt[lag:]
        else:
            pred_view, gt_view = prediction, gt
        norm = np.linalg.norm(pred_view - gt_view, axis=1)
        delta_norm = np.linalg.norm(
            np.diff(pred_view, axis=0) - np.diff(gt_view, axis=0), axis=1
        )
        scores.append(
            {
                "lag_frames": lag,
                "absolute_E_3d": float(np.mean(norm)),
                "delta_E_3d": float(np.mean(delta_norm)),
            }
        )
    return {
        "scores": scores,
        "best_absolute": min(scores, key=lambda item: item["absolute_E_3d"]),
        "best_motion_delta": min(scores, key=lambda item: item["delta_E_3d"]),
        "note": "Absolute score is drift-confounded; motion-delta score is the lag decision.",
    }


def alignment_diagnostics(path: Path | None) -> dict | None:
    if path is None:
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    matrix = np.asarray(payload["matrix"], dtype=np.float64)
    singular = np.linalg.svd(matrix, compute_uv=False)
    horizontal = matrix[:2, :2]
    horizontal_singular = np.linalg.svd(horizontal, compute_uv=False)
    return {
        "matrix": matrix.tolist(),
        "singular_values": singular.tolist(),
        "determinant": float(np.linalg.det(matrix)),
        "condition_number": float(singular[0] / singular[-1]),
        "horizontal_2x2": {
            "matrix": horizontal.tolist(),
            "singular_values": horizontal_singular.tolist(),
            "determinant": float(np.linalg.det(horizontal)),
            "condition_number": float(
                horizontal_singular[0] / horizontal_singular[-1]
            ),
        },
        "ned_y_row": matrix[1].tolist(),
    }


def diagnose(predictions: Path, alignment: Path | None) -> dict:
    data = load_predictions(predictions)
    evaluation = np.flatnonzero(data["health"] == 0)
    prediction = data["prediction"][evaluation]
    gt = data["gt"][evaluation]
    sample_index = data["sample_index"][evaluation]
    error = prediction - gt
    path = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(gt, axis=0), axis=1))]
    axis = {}
    for column, name in enumerate(AXES):
        axis[name] = {
            "rmse_m": float(np.sqrt(np.mean(error[:, column] ** 2))),
            "mean_signed_m": float(np.mean(error[:, column])),
            "slope_m_per_frame": _slope(sample_index, error[:, column]),
            "slope_m_per_100m": 100.0 * _slope(path, error[:, column]),
        }
    return {
        "predictions": str(predictions.resolve()),
        "evaluation_count": len(evaluation),
        "metrics": metrics(prediction, gt),
        "axis": axis,
        "rolling_motion": rolling_motion(prediction, gt),
        "track_frame_error": cross_track(error, gt),
        "relative_drift": relative_drift(prediction, gt),
        "timestamp_lag_scan": lag_scan(prediction, gt),
        "alignment": alignment_diagnostics(alignment),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--alignment", type=Path, default=None)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = diagnose(args.predictions, args.alignment)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report["metrics"], indent=2, ensure_ascii=False))
    print(
        json.dumps(
            report["timestamp_lag_scan"]["best_motion_delta"], ensure_ascii=False
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
