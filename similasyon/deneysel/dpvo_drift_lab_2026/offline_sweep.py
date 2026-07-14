#!/usr/bin/env python3
"""Leakage-audited sequential sweep for DPVO and planar-motion fusion.

Every candidate is selected on chronological holdouts contained in the first
health=1 interval.  The selected specification is refitted on all health=1
samples and only then evaluated once on health=0.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np


AXES = "xyz"


@dataclass(frozen=True)
class Data:
    predictions_path: Path
    planar_path: Path
    sample_index: np.ndarray
    health: np.ndarray
    raw: np.ndarray
    gt: np.ndarray
    planar: dict[str, np.ndarray]

    @property
    def calibration(self) -> np.ndarray:
        return np.flatnonzero(self.health == 1)

    @property
    def evaluation(self) -> np.ndarray:
        return np.flatnonzero(self.health == 0)


@dataclass(frozen=True)
class Candidate:
    name: str
    family: str
    window: int = 1
    ridge: float = 0.0
    robust: bool = False
    time_degree: int = 0
    time_ridge_multiplier: float = 1.0
    planar_source: str | None = None
    planar_scale_power: float = 0.0
    planar_blend: float = 0.0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(path: Path, fields: tuple[str, ...]) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = set(fields) - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path} eksik sütunlar: {sorted(missing)}")
        return list(reader)


def load_data(predictions: Path, planar: Path) -> Data:
    pred_rows = _load(
        predictions,
        (
            "sample_index",
            "health_status",
            "raw_x",
            "raw_y",
            "raw_z",
            "gt_x",
            "gt_y",
            "gt_z",
        ),
    )
    planar_fields = (
        "sample_index",
        "center_dx",
        "center_dy",
        "median_flow_x",
        "median_flow_y",
        "affine_dx",
        "affine_dy",
        "local_rotation_rad",
        "local_log_scale",
    )
    planar_rows = _load(planar, planar_fields)
    if len(pred_rows) != len(planar_rows):
        raise ValueError(
            f"Predictions/planar satır sayısı farklı: {len(pred_rows)} / {len(planar_rows)}"
        )
    sample_index = np.asarray([int(row["sample_index"]) for row in pred_rows])
    planar_index = np.asarray([int(row["sample_index"]) for row in planar_rows])
    if not np.array_equal(sample_index, planar_index):
        raise ValueError("Predictions ve planar sample_index dizileri eşleşmiyor.")
    health = np.asarray(
        [int(float(row["health_status"])) for row in pred_rows], dtype=np.int8
    )
    first_zero = int(np.flatnonzero(health == 0)[0])
    if not np.all(health[:first_zero] == 1) or np.any(health[first_zero:] != 0):
        raise ValueError("Beklenen kesintisiz health=1 -> health=0 sırası bulunamadı.")
    raw = np.asarray(
        [[float(row[f"raw_{axis}"]) for axis in AXES] for row in pred_rows],
        dtype=np.float64,
    )
    gt = np.asarray(
        [[float(row[f"gt_{axis}"]) for axis in AXES] for row in pred_rows],
        dtype=np.float64,
    )
    planar_values = {
        field: np.nan_to_num(
            np.asarray([float(row[field]) for row in planar_rows], dtype=np.float64)
        )
        for field in planar_fields
        if field != "sample_index"
    }
    if not np.all(np.isfinite(raw)) or not np.all(np.isfinite(gt)):
        raise ValueError("Raw/GT NaN veya Inf içeriyor.")
    return Data(
        predictions_path=predictions.resolve(),
        planar_path=planar.resolve(),
        sample_index=sample_index,
        health=health,
        raw=raw,
        gt=gt,
        planar=planar_values,
    )


def causal_average(values: np.ndarray, window: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if window <= 1:
        return values.copy()
    cumulative = np.vstack([np.zeros((1, values.shape[1])), np.cumsum(values, axis=0)])
    index = np.arange(len(values))
    start = np.maximum(0, index - window + 1)
    return (cumulative[index + 1] - cumulative[start]) / (index - start + 1)[:, None]


def _weighted_ridge(
    design: np.ndarray,
    target: np.ndarray,
    sample_weight: np.ndarray,
    regularizer: np.ndarray,
) -> np.ndarray:
    root = np.sqrt(np.asarray(sample_weight, dtype=np.float64))
    x = design * root[:, None]
    y = target * root[:, None]
    return np.linalg.solve(x.T @ x + regularizer, x.T @ y)


def fit_linear(
    design: np.ndarray,
    target: np.ndarray,
    *,
    ridge: float,
    robust: bool,
    regularizer_multiplier: np.ndarray | None = None,
    max_iterations: int = 50,
) -> tuple[np.ndarray, dict]:
    design = np.asarray(design, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if design.ndim != 2 or target.ndim != 2 or len(design) != len(target):
        raise ValueError("Linear fit tasarım/hedef şekilleri geçersiz.")
    if regularizer_multiplier is None:
        multiplier = np.ones(design.shape[1], dtype=np.float64)
    else:
        multiplier = np.asarray(regularizer_multiplier, dtype=np.float64)
    regularizer = np.diag(np.maximum(0.0, ridge * multiplier))
    if ridge > 0:
        weights = _weighted_ridge(
            design, target, np.ones(len(design)), regularizer
        )
    else:
        weights = np.linalg.lstsq(design, target, rcond=None)[0]
    observation_weight = np.ones(len(design), dtype=np.float64)
    robust_scale = 0.0
    iterations = 0
    if robust:
        for iterations in range(1, max_iterations + 1):
            residual = target - design @ weights
            residual_norm = np.linalg.norm(residual, axis=1)
            center = float(np.median(residual_norm))
            robust_scale = float(
                1.4826 * np.median(np.abs(residual_norm - center))
            )
            if robust_scale <= 1e-12:
                break
            threshold = center + 1.345 * robust_scale
            new_weight = np.ones_like(residual_norm)
            outside = residual_norm > threshold
            new_weight[outside] = threshold / residual_norm[outside]
            new_weights = _weighted_ridge(
                design, target, new_weight, regularizer
            )
            if np.linalg.norm(new_weights - weights) <= 1e-10 * (
                1.0 + np.linalg.norm(weights)
            ):
                weights = new_weights
                observation_weight = new_weight
                break
            weights = new_weights
            observation_weight = new_weight
    return weights, {
        "rank": int(np.linalg.matrix_rank(design)),
        "condition": float(np.linalg.cond(design)),
        "robust": robust,
        "iterations": iterations,
        "robust_scale": robust_scale,
        "downweighted_count": int(np.sum(observation_weight < 1.0 - 1e-12)),
    }


def pose_affine(
    raw_train: np.ndarray,
    gt_train: np.ndarray,
    raw_future: np.ndarray,
    ridge: float,
) -> tuple[np.ndarray, dict]:
    src_mean = raw_train.mean(axis=0)
    dst_mean = gt_train.mean(axis=0)
    x = raw_train - src_mean
    y = gt_train - dst_mean
    regularizer = np.eye(3) * ridge
    if ridge > 0:
        weights = np.linalg.solve(x.T @ x + regularizer, x.T @ y)
    else:
        weights = np.linalg.lstsq(x, y, rcond=None)[0]
    matrix = weights.T
    translation = dst_mean - matrix @ src_mean
    base = raw_future @ matrix.T + translation
    train_last = raw_train[-1] @ matrix.T + translation
    anchor = gt_train[-1] - train_last
    return base + anchor, {
        "matrix": matrix.tolist(),
        "translation": translation.tolist(),
        "anchor": anchor.tolist(),
        "source_condition": float(np.linalg.cond(x)),
        "matrix_singular_values": np.linalg.svd(matrix, compute_uv=False).tolist(),
        "matrix_determinant": float(np.linalg.det(matrix)),
    }


def _global_raw_delta(data: Data) -> np.ndarray:
    return np.vstack([np.zeros((1, 3)), np.diff(data.raw, axis=0)])


def _delta_features(data: Data, candidate: Candidate) -> tuple[np.ndarray, np.ndarray]:
    delta = causal_average(_global_raw_delta(data), candidate.window)
    blocks = [delta]
    multiplier = [1.0] * 3
    if candidate.time_degree:
        time = data.sample_index.astype(np.float64) / 100.0
        for degree in range(1, candidate.time_degree + 1):
            blocks.append(delta * (time**degree)[:, None])
            multiplier.extend(
                [candidate.time_ridge_multiplier ** degree] * 3
            )
    return np.column_stack(blocks), np.asarray(multiplier, dtype=np.float64)


def delta_prediction(
    data: Data,
    candidate: Candidate,
    train: np.ndarray,
    future: np.ndarray,
) -> tuple[np.ndarray, dict]:
    feature, multiplier = _delta_features(data, candidate)
    gt_delta = np.vstack([np.zeros((1, 3)), np.diff(data.gt, axis=0)])
    fit_index = train[1:]
    weights, diagnostics = fit_linear(
        feature[fit_index],
        gt_delta[fit_index],
        ridge=candidate.ridge,
        robust=candidate.robust,
        regularizer_multiplier=multiplier,
    )
    predicted_delta = feature[future] @ weights
    prediction = data.gt[train[-1]] + np.cumsum(predicted_delta, axis=0)
    diagnostics.update(
        {
            "weights": weights.tolist(),
            "window": candidate.window,
            "time_degree": candidate.time_degree,
        }
    )
    return prediction, diagnostics


def planar_proxy(data: Data, source: str, scale_power: float) -> np.ndarray:
    if source == "center":
        flow = np.column_stack(
            [data.planar["center_dx"], data.planar["center_dy"]]
        )
    elif source == "median":
        flow = np.column_stack(
            [data.planar["median_flow_x"], data.planar["median_flow_y"]]
        )
    elif source == "affine":
        flow = np.column_stack(
            [data.planar["affine_dx"], data.planar["affine_dy"]]
        )
    else:
        raise ValueError(f"Bilinmeyen planar source: {source}")
    # H(prev->current) image rotation is approximately yaw_prev-yaw_current.
    # Rotate center flow back into the initial ground frame.  A remaining
    # constant orientation/sign is absorbed by the health=1 linear map.
    image_rotation = np.cumsum(data.planar["local_rotation_rad"])
    angle = -image_rotation
    cosine, sine = np.cos(angle), np.sin(angle)
    rotated = np.column_stack(
        [
            cosine * flow[:, 0] - sine * flow[:, 1],
            sine * flow[:, 0] + cosine * flow[:, 1],
        ]
    )
    relative_log_scale = np.cumsum(data.planar["local_log_scale"])
    scale = np.exp(np.clip(scale_power * relative_log_scale, -3.0, 3.0))
    return rotated * scale[:, None]


def planar_fusion_prediction(
    data: Data,
    candidate: Candidate,
    train: np.ndarray,
    future: np.ndarray,
) -> tuple[np.ndarray, dict]:
    baseline, baseline_diag = pose_affine(
        data.raw[train], data.gt[train], data.raw[future], candidate.ridge
    )
    proxy = planar_proxy(
        data, candidate.planar_source or "center", candidate.planar_scale_power
    )
    gt_delta_xy = np.vstack([np.zeros((1, 2)), np.diff(data.gt[:, :2], axis=0)])
    fit_index = train[1:]
    weights, planar_diag = fit_linear(
        proxy[fit_index],
        gt_delta_xy[fit_index],
        ridge=candidate.ridge,
        robust=candidate.robust,
    )
    planar_delta = proxy[future] @ weights
    planar_xy = data.gt[train[-1], :2] + np.cumsum(planar_delta, axis=0)
    prediction = baseline.copy()
    blend = float(candidate.planar_blend)
    prediction[:, :2] = (1.0 - blend) * baseline[:, :2] + blend * planar_xy
    return prediction, {
        "baseline": baseline_diag,
        "planar": planar_diag,
        "planar_weights": weights.tolist(),
        "planar_source": candidate.planar_source,
        "planar_scale_power": candidate.planar_scale_power,
        "planar_blend": blend,
    }


def planar_speed_prediction(
    data: Data,
    candidate: Candidate,
    train: np.ndarray,
    future: np.ndarray,
) -> tuple[np.ndarray, dict]:
    """Use planar motion only as a horizontal-speed observation.

    Directly integrating homography direction accumulates its own yaw error.
    This variant retains DPVO's horizontal direction and Z trajectory, and
    corrects only step magnitude with a metric learned on health=1.
    """
    baseline, baseline_diag = pose_affine(
        data.raw[train], data.gt[train], data.raw[future], candidate.ridge
    )
    proxy = planar_proxy(
        data, candidate.planar_source or "center", candidate.planar_scale_power
    )
    gt_delta_xy = np.vstack([np.zeros((1, 2)), np.diff(data.gt[:, :2], axis=0)])
    fit_index = train[1:]
    proxy_speed = np.linalg.norm(proxy[fit_index], axis=1, keepdims=True)
    gt_speed = np.linalg.norm(gt_delta_xy[fit_index], axis=1, keepdims=True)
    speed_weight, speed_diag = fit_linear(
        proxy_speed,
        gt_speed,
        ridge=candidate.ridge,
        robust=candidate.robust,
    )

    previous = data.gt[train[-1]]
    baseline_delta = np.diff(np.vstack([previous, baseline]), axis=0)
    baseline_speed = np.linalg.norm(baseline_delta[:, :2], axis=1)
    planar_speed = np.maximum(
        0.0,
        (np.linalg.norm(proxy[future], axis=1, keepdims=True) @ speed_weight).ravel(),
    )
    blend = float(candidate.planar_blend)
    fused_speed = (1.0 - blend) * baseline_speed + blend * planar_speed
    direction = baseline_delta[:, :2] / np.maximum(
        baseline_speed[:, None], 1e-9
    )
    prediction = baseline.copy()
    prediction[:, :2] = previous[:2] + np.cumsum(
        direction * fused_speed[:, None], axis=0
    )
    return prediction, {
        "baseline": baseline_diag,
        "speed_fit": speed_diag,
        "speed_weight": speed_weight.tolist(),
        "planar_source": candidate.planar_source,
        "planar_scale_power": candidate.planar_scale_power,
        "planar_blend": blend,
    }


def predict_candidate(
    data: Data,
    candidate: Candidate,
    train: np.ndarray,
    future: np.ndarray,
) -> tuple[np.ndarray, dict]:
    if len(train) < 4 or len(future) < 1 or train[-1] + 1 != future[0]:
        raise ValueError("Sequential fit train ve hemen sonraki future bloğunu bekliyor.")
    if candidate.family == "pose_affine":
        return pose_affine(
            data.raw[train], data.gt[train], data.raw[future], candidate.ridge
        )
    if candidate.family == "delta":
        return delta_prediction(data, candidate, train, future)
    if candidate.family == "planar_fusion":
        return planar_fusion_prediction(data, candidate, train, future)
    if candidate.family == "planar_speed":
        return planar_speed_prediction(data, candidate, train, future)
    raise ValueError(f"Bilinmeyen candidate family: {candidate.family}")


def metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float | list[float]]:
    error = np.asarray(prediction) - np.asarray(target)
    norm = np.linalg.norm(error, axis=1)
    return {
        "E_3d": float(np.mean(norm)),
        "RMSE_3d": float(np.sqrt(np.mean(norm**2))),
        "RMSE_xyz": np.sqrt(np.mean(error**2, axis=0)).tolist(),
        "median_3d": float(np.median(norm)),
        "p95_3d": float(np.percentile(norm, 95)),
        "max_3d": float(np.max(norm)),
    }


def relative_drift(
    prediction: np.ndarray,
    target: np.ndarray,
    lengths: tuple[float, ...] = (10.0, 50.0, 100.0, 250.0),
) -> dict:
    path = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(target, axis=0), axis=1))]
    result: dict[str, dict] = {}
    for requested in lengths:
        errors: list[float] = []
        percents: list[float] = []
        for start in range(len(path) - 1):
            end = int(np.searchsorted(path, path[start] + requested, side="left"))
            if end >= len(path):
                continue
            actual = float(path[end] - path[start])
            delta_error = float(
                np.linalg.norm(
                    (prediction[end] - prediction[start])
                    - (target[end] - target[start])
                )
            )
            errors.append(delta_error)
            percents.append(100.0 * delta_error / actual)
        result[f"{requested:g}"] = {
            "count": len(errors),
            "mean_error_m": float(np.mean(errors)) if errors else None,
            "p95_error_m": float(np.percentile(errors, 95)) if errors else None,
            "mean_drift_percent": float(np.mean(percents)) if percents else None,
        }
    return result


def folds(calibration: np.ndarray) -> list[dict]:
    result: list[dict] = []
    for end in range(120, len(calibration) - 44, 45):
        validation = calibration[end : end + 45]
        result.append(
            {
                "protocol": "expanding",
                "train": calibration[:end],
                "validation": validation,
            }
        )
        start = max(0, end - 180)
        result.append(
            {
                "protocol": "rolling180",
                "train": calibration[start:end],
                "validation": validation,
            }
        )
    return result


def candidates() -> list[Candidate]:
    result = [
        Candidate("pose_affine", "pose_affine"),
        Candidate("pose_ridge_1e-3", "pose_affine", ridge=1e-3),
        Candidate("pose_ridge_1e-1", "pose_affine", ridge=1e-1),
    ]
    for window in (1, 3, 5, 9):
        for robust in (False, True):
            label = "huber" if robust else "ls"
            result.append(
                Candidate(
                    f"delta_w{window}_{label}",
                    "delta",
                    window=window,
                    ridge=1e-6,
                    robust=robust,
                )
            )
    for multiplier in (10.0, 100.0, 1000.0):
        result.append(
            Candidate(
                f"delta_time1_huber_r{multiplier:g}",
                "delta",
                window=3,
                ridge=1e-4,
                robust=True,
                time_degree=1,
                time_ridge_multiplier=multiplier,
            )
        )
    for source in ("center", "median"):
        for power in (-1.0, -0.5, 0.0):
            for blend in (0.05, 0.1, 0.15, 0.2, 0.25, 0.3):
                result.append(
                    Candidate(
                        f"planar_{source}_p{power:+g}_b{blend:.2f}",
                        "planar_fusion",
                        ridge=1e-6,
                        robust=True,
                        planar_source=source,
                        planar_scale_power=power,
                        planar_blend=blend,
                    )
                )
    for power in (-1.0, -0.5, 0.0):
        for blend in (0.1, 0.25, 0.5, 0.75, 1.0):
            result.append(
                Candidate(
                    f"planar_speed_center_p{power:+g}_b{blend:.2f}",
                    "planar_speed",
                    ridge=1e-6,
                    robust=True,
                    planar_source="center",
                    planar_scale_power=power,
                    planar_blend=blend,
                )
            )
    return result


def run(data: Data) -> dict:
    calibration = data.calibration
    evaluation = data.evaluation
    cv_folds = folds(calibration)
    candidate_reports: list[dict] = []
    for priority, candidate in enumerate(candidates()):
        chunks: dict[str, list[np.ndarray]] = {"expanding": [], "rolling180": []}
        fold_reports: list[dict] = []
        failed: str | None = None
        for fold_index, fold in enumerate(cv_folds):
            try:
                prediction, _ = predict_candidate(
                    data, candidate, fold["train"], fold["validation"]
                )
            except (ValueError, np.linalg.LinAlgError) as exc:
                failed = f"fold {fold_index}: {exc}"
                break
            residual = prediction - data.gt[fold["validation"]]
            chunks[fold["protocol"]].append(residual)
            fold_reports.append(
                {
                    "fold": fold_index,
                    "protocol": fold["protocol"],
                    "train_start": int(fold["train"][0]),
                    "train_end": int(fold["train"][-1]),
                    "validation_start": int(fold["validation"][0]),
                    "validation_end": int(fold["validation"][-1]),
                    "metrics": metrics(
                        prediction, data.gt[fold["validation"]]
                    ),
                }
            )
        report = {
            "priority": priority,
            "spec": asdict(candidate),
            "status": "failed" if failed else "ok",
            "folds": fold_reports,
        }
        if failed:
            report["reason"] = failed
        else:
            protocol_metrics = {
                name: metrics(
                    np.vstack(values), np.zeros_like(np.vstack(values))
                )
                for name, values in chunks.items()
            }
            report["protocol_metrics"] = protocol_metrics
            report["selection_score"] = {
                "worst_E_3d": max(v["E_3d"] for v in protocol_metrics.values()),
                "worst_RMSE_3d": max(
                    v["RMSE_3d"] for v in protocol_metrics.values()
                ),
                "worst_p95_3d": max(
                    v["p95_3d"] for v in protocol_metrics.values()
                ),
            }
        candidate_reports.append(report)

    successful = [r for r in candidate_reports if r["status"] == "ok"]
    if not successful:
        raise RuntimeError("Hiçbir candidate CV tamamlayamadı.")

    def key(item: dict) -> tuple:
        score = item["selection_score"]
        return (
            round(score["worst_E_3d"], 9),
            round(score["worst_RMSE_3d"], 9),
            round(score["worst_p95_3d"], 9),
            item["priority"],
        )

    selected_report = min(successful, key=key)
    selected = candidates()[selected_report["priority"]]
    prediction, diagnostics = predict_candidate(
        data, selected, calibration, evaluation
    )
    final_metrics = metrics(prediction, data.gt[evaluation])
    return {
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "predictions": str(data.predictions_path),
            "predictions_sha256": _sha256(data.predictions_path),
            "planar_motion": str(data.planar_path),
            "planar_motion_sha256": _sha256(data.planar_path),
        },
        "dataset": {
            "row_count": len(data.raw),
            "health1_count": len(calibration),
            "health0_count": len(evaluation),
        },
        "protocol": {
            "selection_scope": "health1 chronological expanding+rolling holdouts only",
            "health0_scope": "one frozen-winner final offline score only",
            "selection_objective": "lexicographic worst-protocol E, RMSE, p95",
            "fold_count": len(cv_folds),
        },
        "candidates": candidate_reports,
        "selection": {
            "selected_spec": asdict(selected),
            "selected_cv": selected_report,
            "frozen_before_health0": True,
            "final_fit_diagnostics": diagnostics,
        },
        "final_health0": {
            "metrics": final_metrics,
            "relative_drift": relative_drift(
                prediction, data.gt[evaluation]
            ),
            "prediction_start": prediction[0].tolist(),
            "prediction_end": prediction[-1].tolist(),
        },
        "leakage_audit": {
            "health0_used_for_fit": False,
            "health0_used_for_candidate_or_hyperparameter_selection": False,
            "health0_used_for_final_frozen_score_only": True,
        },
    }


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--planar-motion", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    report = run(load_data(args.predictions, args.planar_motion))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    selected = report["selection"]["selected_spec"]
    score = report["final_health0"]["metrics"]
    print(f"Seçilen: {selected['name']}")
    print(
        f"health=0 E={score['E_3d']:.6f} RMSE={score['RMSE_3d']:.6f} "
        f"p95={score['p95_3d']:.6f} max={score['max_3d']:.6f}"
    )
    print(f"Rapor: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
