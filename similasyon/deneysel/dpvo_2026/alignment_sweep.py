#!/usr/bin/env python3
"""Leakage-safe offline alignment sweep for cached DPVO trajectories.

The competition positioner may use ground truth only while ``health_status``
is 1.  This tool mirrors that contract:

* every method and axis choice is selected with chronological holdouts drawn
  only from the initial health=1 calibration interval;
* the winning choice is then frozen and fitted once on all usable health=1
  samples;
* health=0 ground truth is consumed only for the final, one-shot offline score.

The input is the exact ``predictions.csv`` schema emitted by
``evaluate_dpvo.py``.  In particular, the alignment source is ``raw_x/y/z``;
``pred_x/y/z`` is deliberately never used as a fitting feature.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


AXIS_NAMES = ("x", "y", "z")
REQUIRED_COLUMNS = {
    "sample_index",
    "native_frame_index",
    "health_status",
    "raw_x",
    "raw_y",
    "raw_z",
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
}


@dataclass(frozen=True)
class TrajectoryData:
    path: Path
    sample_index: np.ndarray
    native_frame_index: np.ndarray
    health: np.ndarray
    raw: np.ndarray
    gt: np.ndarray
    columns: tuple[str, ...]

    @property
    def calibration_indices(self) -> np.ndarray:
        return np.flatnonzero(self.health == 1)

    @property
    def evaluation_indices(self) -> np.ndarray:
        return np.flatnonzero(self.health == 0)


@dataclass(frozen=True)
class ModelSpec:
    name: str
    family: str
    anchor: bool
    horizontal_axes: tuple[int, int] | None = None
    horizontal_signs: tuple[int, int] | None = None
    vertical_axis: int | None = None
    z_fit: str | None = None
    handedness: str | None = None

    def public(self) -> dict:
        result = asdict(self)
        if self.horizontal_axes is not None:
            result["horizontal_axis_names"] = [
                AXIS_NAMES[index] for index in self.horizontal_axes
            ]
        if self.vertical_axis is not None:
            result["vertical_axis_name"] = AXIS_NAMES[self.vertical_axis]
        return result


@dataclass
class AlignmentModel:
    spec: ModelSpec
    matrix: np.ndarray
    translation: np.ndarray
    anchor_offset: np.ndarray
    diagnostics: dict

    def predict(self, raw: np.ndarray) -> np.ndarray:
        raw = _points(raw, "raw")
        return raw @ self.matrix.T + self.translation + self.anchor_offset

    def parameters(self) -> dict:
        effective_translation = self.translation + self.anchor_offset
        return {
            "matrix": self.matrix.tolist(),
            "translation_before_anchor": self.translation.tolist(),
            "anchor_offset": self.anchor_offset.tolist(),
            "effective_translation": effective_translation.tolist(),
            "matrix_determinant": float(np.linalg.det(self.matrix)),
            "matrix_singular_values": np.linalg.svd(
                self.matrix, compute_uv=False
            ).tolist(),
            "diagnostics": _jsonable(self.diagnostics),
        }


@dataclass(frozen=True)
class SweepConfig:
    min_train: int = 120
    holdout: int = 45
    step: int = 45
    rolling_window: int = 180
    warmup_tolerance: float = 1e-12
    filter_leading_repeats: bool = True
    huber_k: float = 1.345
    huber_max_iterations: int = 50
    relative_segment_lengths_m: tuple[float, ...] = (10.0, 50.0, 100.0, 250.0)

    def validate(self) -> None:
        if self.min_train < 4:
            raise ValueError("min_train en az 4 olmalı.")
        if self.holdout < 1 or self.step < 1:
            raise ValueError("holdout ve step pozitif olmalı.")
        if self.step < self.holdout:
            raise ValueError(
                "Bağımsız CV validation blokları için step, holdout'tan "
                "küçük olamaz."
            )
        if self.rolling_window < self.min_train:
            raise ValueError("rolling_window, min_train değerinden küçük olamaz.")
        if self.warmup_tolerance < 0:
            raise ValueError("warmup_tolerance negatif olamaz.")
        if self.huber_k <= 0 or self.huber_max_iterations < 1:
            raise ValueError("Huber parametreleri pozitif olmalı.")
        if not self.relative_segment_lengths_m:
            raise ValueError("En az bir relative drift segment uzunluğu gerekli.")
        lengths = np.asarray(self.relative_segment_lengths_m, dtype=np.float64)
        if not np.all(np.isfinite(lengths)) or np.any(lengths <= 0):
            raise ValueError("Relative drift segment uzunlukları sonlu ve pozitif olmalı.")
        if len(np.unique(lengths)) != len(lengths):
            raise ValueError("Relative drift segment uzunlukları benzersiz olmalı.")


def _points(values: np.ndarray, label: str) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if result.ndim != 2 or result.shape[1] != 3:
        raise ValueError(f"{label} Nx3 olmalı; gelen={result.shape}")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{label} NaN/Inf içeriyor.")
    return result


def _health(value: str, line_number: int) -> int:
    text = str(value).strip()
    if text in {"0", "0.0"}:
        return 0
    if text in {"1", "1.0"}:
        return 1
    raise ValueError(
        f"health_status yalnız 0/1 olabilir (satır {line_number}): {value!r}"
    )


def load_predictions(path: Path | str) -> TrajectoryData:
    """Load and strictly validate an evaluator ``predictions.csv`` file."""

    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)

    sample_indices: list[int] = []
    native_indices: list[int] = []
    health: list[int] = []
    raw: list[list[float]] = []
    gt: list[list[float]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = tuple(reader.fieldnames or ())
        missing = REQUIRED_COLUMNS - set(columns)
        if missing:
            raise ValueError(f"predictions.csv sütunları eksik: {sorted(missing)}")
        for line_number, row in enumerate(reader, start=2):
            try:
                sample_indices.append(int(row["sample_index"]))
                native_indices.append(int(row["native_frame_index"]))
                health.append(_health(row["health_status"], line_number))
                raw.append([float(row[f"raw_{axis}"]) for axis in AXIS_NAMES])
                gt.append([float(row[f"gt_{axis}"]) for axis in AXIS_NAMES])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Geçersiz predictions.csv değeri, satır {line_number}: {exc}"
                ) from exc

    if not sample_indices:
        raise ValueError("predictions.csv boş.")
    sample_array = np.asarray(sample_indices, dtype=np.int64)
    native_array = np.asarray(native_indices, dtype=np.int64)
    health_array = np.asarray(health, dtype=np.int8)
    raw_array = _points(np.asarray(raw), "raw")
    gt_array = _points(np.asarray(gt), "gt")
    if len(np.unique(sample_array)) != len(sample_array):
        raise ValueError("sample_index değerleri benzersiz olmalı.")
    if np.any(np.diff(sample_array) <= 0):
        raise ValueError("sample_index kesin artan olmalı.")
    if len(np.unique(native_array)) != len(native_array):
        raise ValueError("native_frame_index değerleri benzersiz olmalı.")
    if np.any(np.diff(native_array) <= 0):
        raise ValueError("native_frame_index kesin artan olmalı.")
    if not np.any(health_array == 1) or not np.any(health_array == 0):
        raise ValueError("Hem health=1 kalibrasyon hem health=0 değerlendirme gerekli.")
    first_zero = int(np.flatnonzero(health_array == 0)[0])
    if np.any(health_array[first_zero:] == 1):
        raise ValueError(
            "Sızıntısız protokol ilk kesintisiz health=1 bloğunu bekliyor; "
            "health=0 sonrasında health=1 bulundu."
        )
    if not np.all(health_array[:first_zero] == 1):
        raise ValueError("Health sırası ilk health=1 bloğu, ardından health=0 olmalı.")

    return TrajectoryData(
        path=path,
        sample_index=sample_array,
        native_frame_index=native_array,
        health=health_array,
        raw=raw_array,
        gt=gt_array,
        columns=columns,
    )


def leading_repeat_filter(raw: np.ndarray, tolerance: float = 1e-12) -> tuple[np.ndarray, dict]:
    """Remove a stationary leading plateau produced while DPVO warms up.

    If two or more leading poses are indistinguishable from pose zero, the
    complete plateau (including pose zero) is excluded.  A single first pose
    is not considered a warm-up run and is retained.
    """

    raw = _points(raw, "raw calibration")
    distances = np.linalg.norm(raw - raw[0], axis=1)
    moving = np.flatnonzero(distances > tolerance)
    plateau_length = int(moving[0]) if len(moving) else len(raw)
    if plateau_length >= 2:
        keep = np.arange(plateau_length, len(raw), dtype=np.int64)
    else:
        keep = np.arange(len(raw), dtype=np.int64)
        plateau_length = 0
    if len(keep) < 4:
        raise ValueError(
            "Warm-up filtresinden sonra hizalama için yeterli örnek kalmadı."
        )
    return keep, {
        "policy": "drop_complete_leading_repeated_pose_plateau",
        "tolerance": float(tolerance),
        "dropped_count": int(plateau_length),
        "first_kept_calibration_offset": int(keep[0]),
    }


def _fit_sim2(src: np.ndarray, dst: np.ndarray) -> tuple[float, np.ndarray, np.ndarray, dict]:
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    if src.shape != dst.shape or src.ndim != 2 or src.shape[1] != 2:
        raise ValueError(f"Sim2 eş boyutlu Nx2 veri bekliyor: {src.shape}, {dst.shape}")
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src_centered = src - src_mean
    dst_centered = dst - dst_mean
    variance = float(np.mean(np.sum(src_centered**2, axis=1)))
    if variance <= 1e-16:
        raise ValueError("Sim2 kaynak varyansı sıfıra yakın.")
    covariance = dst_centered.T @ src_centered / len(src)
    u, singular, vt = np.linalg.svd(covariance)
    correction = np.eye(2)
    if np.linalg.det(u @ vt) < 0:
        correction[-1, -1] = -1
    rotation = u @ correction @ vt
    scale = float(np.trace(np.diag(singular) @ correction) / variance)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError(f"Geçersiz Sim2 scale: {scale}")
    translation = dst_mean - scale * (rotation @ src_mean)
    return scale, rotation, translation, {
        "scale": scale,
        "rotation": rotation.tolist(),
        "rotation_determinant": float(np.linalg.det(rotation)),
        "covariance_singular_values": singular.tolist(),
        "source_variance": variance,
    }


def _fit_sim3(src: np.ndarray, dst: np.ndarray) -> tuple[float, np.ndarray, np.ndarray, dict]:
    src = _points(src, "Sim3 src")
    dst = _points(dst, "Sim3 dst")
    if src.shape != dst.shape or len(src) < 3:
        raise ValueError("Sim3 eş boyutlu ve en az 3 Nx3 nokta bekliyor.")
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src_centered = src - src_mean
    dst_centered = dst - dst_mean
    variance = float(np.mean(np.sum(src_centered**2, axis=1)))
    if variance <= 1e-16:
        raise ValueError("Sim3 kaynak varyansı sıfıra yakın.")
    covariance = dst_centered.T @ src_centered / len(src)
    u, singular, vt = np.linalg.svd(covariance)
    correction = np.eye(3)
    if np.linalg.det(u @ vt) < 0:
        correction[-1, -1] = -1
    rotation = u @ correction @ vt
    scale = float(np.trace(np.diag(singular) @ correction) / variance)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError(f"Geçersiz Sim3 scale: {scale}")
    translation = dst_mean - scale * (rotation @ src_mean)
    return scale, rotation, translation, {
        "scale": scale,
        "rotation": rotation.tolist(),
        "rotation_determinant": float(np.linalg.det(rotation)),
        "covariance_singular_values": singular.tolist(),
        "source_variance": variance,
        "reflection_correction": correction.tolist(),
    }


def _fit_z_linear(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, dict]:
    source = np.asarray(source, dtype=np.float64).reshape(-1)
    target = np.asarray(target, dtype=np.float64).reshape(-1)
    if len(source) != len(target) or len(source) < 2:
        raise ValueError("Z fit eş boyutlu ve en az 2 örnek bekliyor.")
    design = np.column_stack([source, np.ones(len(source))])
    if np.linalg.matrix_rank(design) < 2:
        raise ValueError("Z fit kaynak ekseni sabit.")
    coefficients = np.linalg.lstsq(design, target, rcond=None)[0]
    residual = target - design @ coefficients
    return coefficients, {
        "method": "linear",
        "slope": float(coefficients[0]),
        "intercept": float(coefficients[1]),
        "residual_median_absolute": float(np.median(np.abs(residual))),
    }


def _fit_z_huber(
    source: np.ndarray,
    target: np.ndarray,
    huber_k: float,
    max_iterations: int,
) -> tuple[np.ndarray, dict]:
    coefficients, _ = _fit_z_linear(source, target)
    source = np.asarray(source, dtype=np.float64).reshape(-1)
    target = np.asarray(target, dtype=np.float64).reshape(-1)
    design = np.column_stack([source, np.ones(len(source))])
    weights = np.ones(len(source), dtype=np.float64)
    iterations = 0
    robust_scale = 0.0
    for iterations in range(1, max_iterations + 1):
        residual = target - design @ coefficients
        centered = residual - np.median(residual)
        robust_scale = float(1.4826 * np.median(np.abs(centered)))
        if robust_scale <= 1e-12:
            break
        threshold = huber_k * robust_scale
        absolute = np.abs(centered)
        new_weights = np.ones_like(absolute)
        outliers = absolute > threshold
        new_weights[outliers] = threshold / absolute[outliers]
        root_weight = np.sqrt(new_weights)
        new_coefficients = np.linalg.lstsq(
            design * root_weight[:, None], target * root_weight, rcond=None
        )[0]
        delta = np.linalg.norm(new_coefficients - coefficients)
        scale = 1.0 + np.linalg.norm(coefficients)
        coefficients = new_coefficients
        weights = new_weights
        if delta <= 1e-12 * scale:
            break
    residual = target - design @ coefficients
    return coefficients, {
        "method": "huber_irls",
        "slope": float(coefficients[0]),
        "intercept": float(coefficients[1]),
        "huber_k": float(huber_k),
        "iterations": int(iterations),
        "robust_scale": float(robust_scale),
        "downweighted_count": int(np.sum(weights < 1.0 - 1e-12)),
        "min_weight": float(np.min(weights)),
        "residual_median_absolute": float(np.median(np.abs(residual))),
    }


def fit_model(
    spec: ModelSpec,
    raw: np.ndarray,
    gt: np.ndarray,
    *,
    huber_k: float = 1.345,
    huber_max_iterations: int = 50,
) -> AlignmentModel:
    """Fit one candidate without consulting any data beyond the arguments."""

    raw = _points(raw, "fit raw")
    gt = _points(gt, "fit gt")
    if raw.shape != gt.shape or len(raw) < 4:
        raise ValueError("Fit eş boyutlu ve en az 4 Nx3 örnek bekliyor.")

    diagnostics: dict
    if spec.family == "centered_affine":
        src_mean = raw.mean(axis=0)
        dst_mean = gt.mean(axis=0)
        src_centered = raw - src_mean
        dst_centered = gt - dst_mean
        weights = np.linalg.lstsq(src_centered, dst_centered, rcond=None)[0]
        matrix = weights.T
        translation = dst_mean - matrix @ src_mean
        diagnostics = {
            "source_rank": int(np.linalg.matrix_rank(src_centered)),
            "source_condition": float(np.linalg.cond(src_centered)),
            "source_singular_values": np.linalg.svd(
                src_centered, compute_uv=False
            ).tolist(),
            "implementation": "centered_numpy_lstsq_matching_live_linear_fit",
        }
    elif spec.family == "proper_sim3":
        scale, rotation, translation, diagnostics = _fit_sim3(raw, gt)
        matrix = scale * rotation
    elif spec.family == "signed_sim2_z":
        if (
            spec.horizontal_axes is None
            or spec.horizontal_signs is None
            or spec.vertical_axis is None
            or spec.z_fit not in {"linear", "robust"}
        ):
            raise ValueError(f"Eksik signed Sim2+Z spec: {spec}")
        axes = list(spec.horizontal_axes)
        signs = np.asarray(spec.horizontal_signs, dtype=np.float64)
        signed_source_xy = raw[:, axes] * signs
        scale, rotation, translation_xy, sim2_diagnostics = _fit_sim2(
            signed_source_xy, gt[:, :2]
        )
        if spec.z_fit == "linear":
            z_coefficients, z_diagnostics = _fit_z_linear(
                raw[:, spec.vertical_axis], gt[:, 2]
            )
        else:
            z_coefficients, z_diagnostics = _fit_z_huber(
                raw[:, spec.vertical_axis],
                gt[:, 2],
                huber_k=huber_k,
                max_iterations=huber_max_iterations,
            )
        matrix = np.zeros((3, 3), dtype=np.float64)
        matrix[:2, axes] = scale * rotation @ np.diag(signs)
        matrix[2, spec.vertical_axis] = z_coefficients[0]
        translation = np.array(
            [translation_xy[0], translation_xy[1], z_coefficients[1]],
            dtype=np.float64,
        )
        diagnostics = {
            "sim2": sim2_diagnostics,
            "z": z_diagnostics,
            "signed_axis_permutation": [
                f"{sign:+d}{AXIS_NAMES[axis]}"
                for axis, sign in zip(spec.horizontal_axes, spec.horizontal_signs)
            ]
            + [AXIS_NAMES[spec.vertical_axis]],
            "full_linear_handedness": (
                "reflection" if np.linalg.det(matrix) < 0 else "proper"
            ),
        }
    else:
        raise ValueError(f"Bilinmeyen model ailesi: {spec.family}")

    matrix = np.asarray(matrix, dtype=np.float64)
    translation = np.asarray(translation, dtype=np.float64)
    if matrix.shape != (3, 3) or translation.shape != (3,):
        raise RuntimeError("Alignment model internal shape error.")
    if not np.all(np.isfinite(matrix)) or not np.all(np.isfinite(translation)):
        raise ValueError("Alignment fit NaN/Inf parametre üretti.")
    base_last = matrix @ raw[-1] + translation
    anchor_offset = gt[-1] - base_last if spec.anchor else np.zeros(3)
    return AlignmentModel(
        spec=spec,
        matrix=matrix,
        translation=translation,
        anchor_offset=np.asarray(anchor_offset, dtype=np.float64),
        diagnostics=diagnostics,
    )


def candidate_specs() -> list[ModelSpec]:
    """Return the complete, mathematically non-redundant candidate space.

    A 2D similarity already absorbs axis swaps, rotations, and simultaneous
    sign flips.  For each of the three possible source planes, one proper and
    one reflected signed basis spans both horizontal handedness classes.
    """

    specs = [
        ModelSpec("centered_affine_anchor_off", "centered_affine", False),
        ModelSpec("centered_affine_anchor_on", "centered_affine", True),
        ModelSpec("proper_sim3_anchor_on", "proper_sim3", True),
    ]
    for horizontal_axes in itertools.combinations(range(3), 2):
        vertical_axis = next(axis for axis in range(3) if axis not in horizontal_axes)
        for signs, handedness in (((1, 1), "proper"), ((-1, 1), "reflected")):
            axis_label = "".join(AXIS_NAMES[index] for index in horizontal_axes)
            for z_fit in ("linear", "robust"):
                specs.append(
                    ModelSpec(
                        name=(
                            f"signed_sim2_{axis_label}_{handedness}_"
                            f"{AXIS_NAMES[vertical_axis]}_{z_fit}_anchor_on"
                        ),
                        family="signed_sim2_z",
                        anchor=True,
                        horizontal_axes=tuple(horizontal_axes),
                        horizontal_signs=signs,
                        vertical_axis=vertical_axis,
                        z_fit=z_fit,
                        handedness=handedness,
                    )
                )
    return specs


def make_cv_folds(
    count: int,
    *,
    min_train: int,
    holdout: int,
    step: int,
    rolling_window: int,
) -> list[dict]:
    """Build chronological expanding and rolling calibration-only folds."""

    if count < min_train + holdout:
        raise ValueError(
            f"CV için en az {min_train + holdout} kullanılabilir health=1 örneği "
            f"gerekli; gelen={count}"
        )
    if step < holdout:
        raise ValueError("Çakışmasız validation için step >= holdout olmalı.")
    train_ends = list(range(min_train, count - holdout + 1, step))
    final_train_end = count - holdout
    if train_ends[-1] != final_train_end:
        # Keep a full tail holdout without double-weighting samples already in
        # the previous validation block.
        if final_train_end - train_ends[-1] < holdout:
            train_ends[-1] = final_train_end
        else:
            train_ends.append(final_train_end)
    folds: list[dict] = []
    for protocol in ("expanding", "rolling"):
        for train_end in train_ends:
            train_start = 0 if protocol == "expanding" else max(0, train_end - rolling_window)
            folds.append(
                {
                    "protocol": protocol,
                    "train": np.arange(train_start, train_end, dtype=np.int64),
                    "validation": np.arange(
                        train_end, train_end + holdout, dtype=np.int64
                    ),
                }
            )
    return folds


def error_metrics(prediction: np.ndarray, target: np.ndarray) -> dict:
    prediction = _points(prediction, "prediction")
    target = _points(target, "target")
    if prediction.shape != target.shape or len(prediction) == 0:
        raise ValueError("Metric eş boyutlu ve boş olmayan Nx3 veri bekliyor.")
    residual = prediction - target
    error = np.linalg.norm(residual, axis=1)
    return {
        "count": int(len(error)),
        "E_3d": float(np.mean(error)),
        "RMSE_3d": float(np.sqrt(np.mean(error**2))),
        "median_3d": float(np.median(error)),
        "p95_3d": float(np.percentile(error, 95)),
        "max_3d": float(np.max(error)),
        "RMSE_xyz": np.sqrt(np.mean(residual**2, axis=0)).tolist(),
        "mean_signed_error_xyz": np.mean(residual, axis=0).tolist(),
    }


def _scalar_summary(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if not np.all(np.isfinite(values)):
        raise ValueError("Scalar metric NaN/Inf içeriyor.")
    if len(values) == 0:
        return {
            "count": 0,
            "min": None,
            "mean": None,
            "RMSE": None,
            "median": None,
            "p95": None,
            "max": None,
        }
    return {
        "count": int(len(values)),
        "min": float(np.min(values)),
        "mean": float(np.mean(values)),
        "RMSE": float(np.sqrt(np.mean(values**2))),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
    }


def relative_translation_drift(
    prediction: np.ndarray,
    target: np.ndarray,
    segment_lengths_m: Sequence[float] = (10.0, 50.0, 100.0, 250.0),
) -> dict:
    """Compute position-only, RPG-inspired translational segment drift.

    GT cumulative path length defines each segment.  For every possible start
    sample and requested length L, the endpoint is the first later matched
    sample whose cumulative GT distance reaches or exceeds L.  The relative
    translation error is ``||(pred[j]-pred[i]) - (gt[j]-gt[i])||`` and drift
    percent is that error divided by the *actual* sampled GT path length from
    i to j.  Using the actual length avoids a small sampling-overshoot bias.

    No orientation is present in ``predictions.csv``.  Consequently this is
    deliberately not advertised as full RPG relative pose error: rotational
    RPE and body-frame relative translation cannot be computed honestly.
    """

    prediction = _points(prediction, "relative drift prediction")
    target = _points(target, "relative drift target")
    if prediction.shape != target.shape or len(prediction) < 2:
        raise ValueError(
            "Relative drift eş boyutlu ve en az iki Nx3 örnek bekliyor."
        )
    lengths = np.asarray(tuple(segment_lengths_m), dtype=np.float64)
    if len(lengths) == 0 or not np.all(np.isfinite(lengths)) or np.any(lengths <= 0):
        raise ValueError("Segment uzunlukları sonlu, pozitif ve boş olmayan olmalı.")
    if len(np.unique(lengths)) != len(lengths):
        raise ValueError("Segment uzunlukları benzersiz olmalı.")

    cumulative_gt = np.concatenate(
        [
            np.zeros(1, dtype=np.float64),
            np.cumsum(np.linalg.norm(np.diff(target, axis=0), axis=1)),
        ]
    )
    starts = np.arange(len(target), dtype=np.int64)
    by_length: dict[str, dict] = {}
    all_errors: list[np.ndarray] = []
    all_drift_percent: list[np.ndarray] = []
    all_actual_lengths: list[np.ndarray] = []

    for requested_length in lengths:
        endpoints = np.searchsorted(
            cumulative_gt,
            cumulative_gt + requested_length,
            side="left",
        )
        valid = (endpoints < len(target)) & (endpoints > starts)
        valid_starts = starts[valid]
        valid_endpoints = endpoints[valid]
        actual_lengths = (
            cumulative_gt[valid_endpoints] - cumulative_gt[valid_starts]
        )
        if len(valid_starts):
            gt_delta = target[valid_endpoints] - target[valid_starts]
            prediction_delta = (
                prediction[valid_endpoints] - prediction[valid_starts]
            )
            errors = np.linalg.norm(prediction_delta - gt_delta, axis=1)
            drift_percent = 100.0 * errors / actual_lengths
            endpoint_spans = valid_endpoints - valid_starts
            all_errors.append(errors)
            all_drift_percent.append(drift_percent)
            all_actual_lengths.append(actual_lengths)
        else:
            errors = np.empty(0, dtype=np.float64)
            drift_percent = np.empty(0, dtype=np.float64)
            endpoint_spans = np.empty(0, dtype=np.int64)

        key = f"{float(requested_length):g}"
        by_length[key] = {
            "requested_gt_path_length_m": float(requested_length),
            "eligible_start_count": int(len(valid_starts)),
            "translation_delta_error_m": _scalar_summary(errors),
            "drift_percent": _scalar_summary(drift_percent),
            "actual_sampled_gt_path_length_m": _scalar_summary(actual_lengths),
            "sampling_overshoot_m": _scalar_summary(
                actual_lengths - requested_length
            ),
            "endpoint_frame_span": _scalar_summary(endpoint_spans),
        }

    if all_errors:
        aggregate_errors = np.concatenate(all_errors)
        aggregate_drift_percent = np.concatenate(all_drift_percent)
        aggregate_actual_lengths = np.concatenate(all_actual_lengths)
    else:
        aggregate_errors = np.empty(0, dtype=np.float64)
        aggregate_drift_percent = np.empty(0, dtype=np.float64)
        aggregate_actual_lengths = np.empty(0, dtype=np.float64)

    return {
        "metadata": {
            "metric_family": "position_only_rpg_inspired_translational_segment_drift",
            "reference_tool": "uzh-rpg/rpg_trajectory_evaluation",
            "reference_repository": (
                "https://github.com/uzh-rpg/rpg_trajectory_evaluation"
            ),
            "full_rpg_metric_equivalence": False,
            "endpoint_rule": (
                "for each start, first later matched sample whose cumulative "
                "GT path distance reaches or exceeds requested length"
            ),
            "translation_error_formula": (
                "norm((pred_j - pred_i) - (gt_j - gt_i))"
            ),
            "drift_percent_formula": (
                "100 * translation_delta_error_m / actual sampled cumulative "
                "GT path length"
            ),
            "coordinate_frame_note": (
                "world-frame position deltas after frozen DPVO-to-GT alignment"
            ),
            "orientation_available": False,
            "rotational_rpe_computed": False,
            "rotational_rpe_omission_reason": (
                "predictions.csv contains positions but no orientation quaternions; "
                "rotational RPE was not invented"
            ),
        },
        "requested_segment_lengths_m": lengths.tolist(),
        "health0_gt_path_length_m": float(cumulative_gt[-1]),
        "by_requested_length_m": by_length,
        "aggregate_all_lengths": {
            "translation_delta_error_m": _scalar_summary(aggregate_errors),
            "drift_percent": _scalar_summary(aggregate_drift_percent),
            "actual_sampled_gt_path_length_m": _scalar_summary(
                aggregate_actual_lengths
            ),
        },
    }


def _cross_validate(
    spec: ModelSpec,
    raw: np.ndarray,
    gt: np.ndarray,
    folds: Sequence[dict],
    config: SweepConfig,
) -> dict:
    residuals_by_protocol: dict[str, list[np.ndarray]] = {
        "expanding": [],
        "rolling": [],
    }
    fold_summaries: list[dict] = []
    for fold_index, fold in enumerate(folds):
        train = fold["train"]
        validation = fold["validation"]
        model = fit_model(
            spec,
            raw[train],
            gt[train],
            huber_k=config.huber_k,
            huber_max_iterations=config.huber_max_iterations,
        )
        prediction = model.predict(raw[validation])
        residual = prediction - gt[validation]
        residuals_by_protocol[fold["protocol"]].append(residual)
        fold_summaries.append(
            {
                "fold": int(fold_index),
                "protocol": fold["protocol"],
                "train_offset_start": int(train[0]),
                "train_offset_end_inclusive": int(train[-1]),
                "validation_offset_start": int(validation[0]),
                "validation_offset_end_inclusive": int(validation[-1]),
                "metrics": error_metrics(prediction, gt[validation]),
            }
        )

    protocol_metrics: dict[str, dict] = {}
    all_residuals: list[np.ndarray] = []
    for protocol, chunks in residuals_by_protocol.items():
        residual = np.vstack(chunks)
        all_residuals.append(residual)
        protocol_metrics[protocol] = error_metrics(
            residual, np.zeros_like(residual)
        )
    combined = np.vstack(all_residuals)
    combined_metrics = error_metrics(combined, np.zeros_like(combined))
    worst_e = max(metrics["E_3d"] for metrics in protocol_metrics.values())
    worst_rmse = max(metrics["RMSE_3d"] for metrics in protocol_metrics.values())
    worst_p95 = max(metrics["p95_3d"] for metrics in protocol_metrics.values())
    return {
        "status": "ok",
        "protocol_metrics": protocol_metrics,
        "combined_metrics": combined_metrics,
        "selection_score": {
            "worst_protocol_E_3d": float(worst_e),
            "worst_protocol_RMSE_3d": float(worst_rmse),
            "worst_protocol_p95_3d": float(worst_p95),
        },
        "folds": fold_summaries,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def model_from_report(report: dict) -> AlignmentModel:
    """Reconstruct the frozen winner from an in-memory sweep report.

    This is the reusable apply path for experiments: callers do not need to
    refit or repeat model selection, and therefore cannot accidentally consult
    health=0 GT a second time.  The provenance/leakage flags are validated
    before parameters are accepted.
    """

    try:
        audit = report["leakage_audit"]
        selection = report["selection"]
        if audit["health0_used_for_candidate_fit"] is not False:
            raise ValueError("Artifact health=0 fit sızıntısızlık şartını sağlamıyor.")
        if audit["health0_used_for_axis_or_method_selection"] is not False:
            raise ValueError("Artifact health=0 seçim sızıntısızlık şartını sağlamıyor.")
        if selection["chosen_before_health0_evaluation"] is not True:
            raise ValueError("Winner'ın health=0 öncesinde dondurulduğu doğrulanamıyor.")
        public_spec = selection["selected_spec"]
        spec = ModelSpec(
            name=str(public_spec["name"]),
            family=str(public_spec["family"]),
            anchor=bool(public_spec["anchor"]),
            horizontal_axes=(
                None
                if public_spec.get("horizontal_axes") is None
                else tuple(int(value) for value in public_spec["horizontal_axes"])
            ),
            horizontal_signs=(
                None
                if public_spec.get("horizontal_signs") is None
                else tuple(int(value) for value in public_spec["horizontal_signs"])
            ),
            vertical_axis=(
                None
                if public_spec.get("vertical_axis") is None
                else int(public_spec["vertical_axis"])
            ),
            z_fit=public_spec.get("z_fit"),
            handedness=public_spec.get("handedness"),
        )
        parameters = selection["final_parameters"]
        matrix = np.asarray(parameters["matrix"], dtype=np.float64)
        translation = np.asarray(
            parameters["translation_before_anchor"], dtype=np.float64
        )
        anchor_offset = np.asarray(parameters["anchor_offset"], dtype=np.float64)
        if matrix.shape != (3, 3):
            raise ValueError(f"Artifact matrix shape geçersiz: {matrix.shape}")
        if translation.shape != (3,) or anchor_offset.shape != (3,):
            raise ValueError("Artifact translation/anchor shape geçersiz.")
        if not all(
            np.all(np.isfinite(values))
            for values in (matrix, translation, anchor_offset)
        ):
            raise ValueError("Artifact alignment parametreleri NaN/Inf içeriyor.")
    except (KeyError, TypeError) as exc:
        raise ValueError(f"Geçersiz alignment sweep artifact'i: {exc}") from exc
    return AlignmentModel(
        spec=spec,
        matrix=matrix,
        translation=translation,
        anchor_offset=anchor_offset,
        diagnostics=dict(parameters.get("diagnostics", {})),
    )


def load_frozen_model(path: Path | str) -> AlignmentModel:
    """Load a leakage-audited frozen alignment model from sweep JSON."""

    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        report = json.load(handle)
    return model_from_report(report)


def run_sweep(data: TrajectoryData, config: SweepConfig = SweepConfig()) -> dict:
    """Run selection on health=1 and a single final score on health=0."""

    config.validate()
    calibration_indices = data.calibration_indices
    evaluation_indices = data.evaluation_indices
    raw_calibration_all = data.raw[calibration_indices]
    gt_calibration_all = data.gt[calibration_indices]

    if config.filter_leading_repeats:
        kept_offsets, warmup = leading_repeat_filter(
            raw_calibration_all, config.warmup_tolerance
        )
    else:
        kept_offsets = np.arange(len(calibration_indices), dtype=np.int64)
        warmup = {
            "policy": "disabled",
            "tolerance": float(config.warmup_tolerance),
            "dropped_count": 0,
            "first_kept_calibration_offset": 0,
        }
    usable_indices = calibration_indices[kept_offsets]
    raw_calibration = data.raw[usable_indices]
    gt_calibration = data.gt[usable_indices]
    folds = make_cv_folds(
        len(usable_indices),
        min_train=config.min_train,
        holdout=config.holdout,
        step=config.step,
        rolling_window=config.rolling_window,
    )

    candidates: list[dict] = []
    specs = candidate_specs()
    for priority, spec in enumerate(specs):
        candidate: dict = {
            "priority": priority,
            "spec": spec.public(),
        }
        try:
            candidate["cv"] = _cross_validate(
                spec, raw_calibration, gt_calibration, folds, config
            )
            full_model = fit_model(
                spec,
                raw_calibration,
                gt_calibration,
                huber_k=config.huber_k,
                huber_max_iterations=config.huber_max_iterations,
            )
            candidate["health1_refit"] = {
                "metrics_usable_health1": error_metrics(
                    full_model.predict(raw_calibration), gt_calibration
                ),
                "metrics_all_health1_including_warmup": error_metrics(
                    full_model.predict(raw_calibration_all), gt_calibration_all
                ),
                "parameters": full_model.parameters(),
            }
            candidate["status"] = "ok"
        except (ValueError, np.linalg.LinAlgError) as exc:
            candidate["status"] = "failed"
            candidate["reason"] = str(exc)
        candidates.append(candidate)

    successful = [candidate for candidate in candidates if candidate["status"] == "ok"]
    if not successful:
        raise RuntimeError("Hiçbir alignment adayı CV fitini tamamlayamadı.")

    # Rounding at sub-nanometre precision prevents numerically equivalent
    # signed bases from winning because of an irrelevant 1e-14 SVD difference.
    def selection_key(candidate: dict) -> tuple:
        score = candidate["cv"]["selection_score"]
        return (
            round(score["worst_protocol_E_3d"], 9),
            round(score["worst_protocol_RMSE_3d"], 9),
            round(score["worst_protocol_p95_3d"], 9),
            candidate["priority"],
        )

    selected_candidate = min(successful, key=selection_key)
    selected_spec = specs[selected_candidate["priority"]]
    final_model = fit_model(
        selected_spec,
        raw_calibration,
        gt_calibration,
        huber_k=config.huber_k,
        huber_max_iterations=config.huber_max_iterations,
    )

    # This is intentionally the first and only point where health=0 GT is
    # consumed.  Candidate reports above contain health=1 data only.
    final_prediction = final_model.predict(data.raw[evaluation_indices])
    final_offline_metrics = error_metrics(
        final_prediction, data.gt[evaluation_indices]
    )
    final_relative_drift = relative_translation_drift(
        final_prediction,
        data.gt[evaluation_indices],
        config.relative_segment_lengths_m,
    )
    first_eval_prediction = final_prediction[0]
    last_health1_gt = gt_calibration_all[-1]

    report = {
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "input": {
            "predictions_csv": str(data.path),
            "sha256": _sha256(data.path),
            "columns": list(data.columns),
        },
        "dataset": {
            "row_count": int(len(data.raw)),
            "health1_count": int(len(calibration_indices)),
            "health0_count": int(len(evaluation_indices)),
            "first_sample_index": int(data.sample_index[0]),
            "last_sample_index": int(data.sample_index[-1]),
            "first_native_frame_index": int(data.native_frame_index[0]),
            "last_native_frame_index": int(data.native_frame_index[-1]),
        },
        "protocol": {
            "selection_gt_scope": "health1_only",
            "selection_feature_columns": ["raw_x", "raw_y", "raw_z"],
            "ignored_feature_columns": ["pred_x", "pred_y", "pred_z"],
            "final_offline_gt_scope": "health0_after_model_selection_and_freeze",
            "selection_objective": (
                "lexicographic minimum of worst(expanding,rolling) E_3d, "
                "RMSE_3d, p95_3d; deterministic candidate priority tie-break"
            ),
            "config": asdict(config),
            "cv_fold_count": int(len(folds)),
            "cv_fold_count_by_protocol": {
                protocol: int(sum(fold["protocol"] == protocol for fold in folds))
                for protocol in ("expanding", "rolling")
            },
            "candidate_space_note": (
                "For each raw 2-axis plane, proper Sim2 rotation absorbs swaps, "
                "rotations and double sign flips; proper/reflected signed bases "
                "therefore cover both non-redundant horizontal handedness classes."
            ),
        },
        "warmup_filter": {
            **warmup,
            "dropped_sample_indices": data.sample_index[
                calibration_indices[: warmup["dropped_count"]]
            ].tolist(),
            "usable_health1_count": int(len(usable_indices)),
            "first_usable_sample_index": int(data.sample_index[usable_indices[0]]),
        },
        "candidates": candidates,
        "selection": {
            "selected_name": selected_spec.name,
            "selected_spec": selected_spec.public(),
            "chosen_before_health0_evaluation": True,
            "choice_used_health1_cv_only": True,
            "selected_cv": selected_candidate["cv"],
            "final_parameters": final_model.parameters(),
            "health1_refit_metrics": selected_candidate["health1_refit"],
        },
        "final_offline_evaluation": {
            "scope": "health0_only",
            "model_was_frozen_before_health0_gt_scoring": True,
            "health0_gt_values_loaded_with_csv_before_selection": True,
            "metrics": final_offline_metrics,
            "relative_translation_drift": final_relative_drift,
            "first_health0_sample_index": int(
                data.sample_index[evaluation_indices[0]]
            ),
            "last_health0_sample_index": int(
                data.sample_index[evaluation_indices[-1]]
            ),
            "first_health0_prediction": first_eval_prediction.tolist(),
            "last_health1_gt": last_health1_gt.tolist(),
            "transition_distance_from_last_health1_gt": float(
                np.linalg.norm(first_eval_prediction - last_health1_gt)
            ),
        },
        "leakage_audit": {
            "health0_used_for_candidate_fit": False,
            "health0_used_for_axis_or_method_selection": False,
            "health0_used_for_final_offline_absolute_and_relative_metrics_only": True,
        },
    }
    return _jsonable(report)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Select a DPVO->GT alignment using only health=1 chronological "
            "holdouts, then score the frozen winner once on health=0."
        )
    )
    parser.add_argument("predictions", type=Path, help="Evaluator predictions.csv")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "results" / "alignment_sweep.json",
    )
    parser.add_argument("--min-train", type=int, default=120)
    parser.add_argument("--holdout", type=int, default=45)
    parser.add_argument("--step", type=int, default=45)
    parser.add_argument("--rolling-window", type=int, default=180)
    parser.add_argument("--warmup-tolerance", type=float, default=1e-12)
    parser.add_argument("--no-warmup-filter", action="store_true")
    parser.add_argument("--huber-k", type=float, default=1.345)
    parser.add_argument("--huber-max-iterations", type=int, default=50)
    parser.add_argument(
        "--relative-segment-lengths-m",
        default="10,50,100,250",
        help="Comma-separated GT path lengths for position-only relative drift.",
    )
    return parser


def _parse_segment_lengths(value: str) -> tuple[float, ...]:
    try:
        lengths = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise ValueError(
            f"Geçersiz --relative-segment-lengths-m: {value!r}"
        ) from exc
    if not lengths:
        raise ValueError("--relative-segment-lengths-m boş olamaz.")
    return lengths


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = SweepConfig(
        min_train=args.min_train,
        holdout=args.holdout,
        step=args.step,
        rolling_window=args.rolling_window,
        warmup_tolerance=args.warmup_tolerance,
        filter_leading_repeats=not args.no_warmup_filter,
        huber_k=args.huber_k,
        huber_max_iterations=args.huber_max_iterations,
        relative_segment_lengths_m=_parse_segment_lengths(
            args.relative_segment_lengths_m
        ),
    )
    data = load_predictions(args.predictions)
    report = run_sweep(data, config)
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    metrics = report["final_offline_evaluation"]["metrics"]
    print(f"Seçilen: {report['selection']['selected_name']}")
    print(
        "health=0: "
        f"E={metrics['E_3d']:.6f} m, RMSE={metrics['RMSE_3d']:.6f} m, "
        f"p95={metrics['p95_3d']:.6f} m, max={metrics['max_3d']:.6f} m"
    )
    relative = report["final_offline_evaluation"]["relative_translation_drift"]
    for length, summary in relative["by_requested_length_m"].items():
        error = summary["translation_delta_error_m"]
        drift = summary["drift_percent"]
        if summary["eligible_start_count"]:
            print(
                f"relative {length} m: n={summary['eligible_start_count']}, "
                f"mean_error={error['mean']:.6f} m, "
                f"mean_drift={drift['mean']:.6f}%"
            )
        else:
            print(f"relative {length} m: uygun health=0 alt-segment yok")
    print(f"Rapor: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
