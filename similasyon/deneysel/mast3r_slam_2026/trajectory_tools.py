#!/usr/bin/env python3
"""Pure NumPy trajectory/alignment helpers for the MASt3R experiment."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class AffineAlignment:
    matrix: np.ndarray
    translation: np.ndarray
    anchor_offset: np.ndarray
    rank: int
    condition_number: float
    used_ridge: bool
    singular_values: np.ndarray

    def apply(self, points: np.ndarray) -> np.ndarray:
        values = np.asarray(points, dtype=np.float64)
        return values @ self.matrix.T + self.translation + self.anchor_offset


@dataclass(frozen=True)
class SimilarityAlignment:
    scale: float
    rotation: np.ndarray
    translation: np.ndarray
    rmse: float

    def apply(self, points: np.ndarray) -> np.ndarray:
        values = np.asarray(points, dtype=np.float64)
        return self.scale * (values @ self.rotation.T) + self.translation


def parse_frame_number(value: str) -> int:
    prefix, separator, suffix = str(value).rpartition("_")
    if prefix != "frame" or not separator or not suffix.isdigit():
        raise ValueError(f"Invalid frame_numbers value: {value!r}")
    return int(suffix)


def load_ground_truth(path: Path) -> dict[int, np.ndarray]:
    required = {"translation_x", "translation_y", "translation_z", "frame_numbers"}
    rows: dict[int, np.ndarray] = {}
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"GT CSV missing columns: {sorted(missing)}")
        for csv_row, row in enumerate(reader, start=2):
            index = parse_frame_number(row["frame_numbers"])
            if index in rows:
                raise ValueError(f"Duplicate frame {index} at CSV row {csv_row}")
            value = np.asarray(
                [row["translation_x"], row["translation_y"], row["translation_z"]],
                dtype=np.float64,
            )
            if not np.all(np.isfinite(value)):
                raise ValueError(f"Non-finite GT at CSV row {csv_row}")
            rows[index] = value
    if not rows:
        raise ValueError(f"GT CSV is empty: {path}")
    return rows


def fit_similarity(source: np.ndarray, target: np.ndarray) -> SimilarityAlignment:
    """Least-squares Sim(3), including scale, mapping source to target."""
    src = np.asarray(source, dtype=np.float64)
    dst = np.asarray(target, dtype=np.float64)
    if src.shape != dst.shape or src.ndim != 2 or src.shape[1] != 3:
        raise ValueError(f"Expected matching Nx3 points, got {src.shape} and {dst.shape}")
    if len(src) < 3:
        raise ValueError("Similarity fit needs at least three point pairs")
    src_mean = np.mean(src, axis=0)
    dst_mean = np.mean(dst, axis=0)
    src_centered = src - src_mean
    dst_centered = dst - dst_mean
    variance = float(np.mean(np.sum(src_centered**2, axis=1)))
    if variance <= np.finfo(np.float64).eps:
        raise ValueError("Similarity source has no usable motion")
    covariance = dst_centered.T @ src_centered / len(src)
    u, singular, vt = np.linalg.svd(covariance)
    correction = np.eye(3, dtype=np.float64)
    if np.linalg.det(u @ vt) < 0:
        correction[-1, -1] = -1.0
    rotation = u @ correction @ vt
    scale = float(np.sum(singular * np.diag(correction)) / variance)
    translation = dst_mean - scale * (rotation @ src_mean)
    predicted = scale * (src @ rotation.T) + translation
    rmse = float(np.sqrt(np.mean(np.sum((predicted - dst) ** 2, axis=1))))
    return SimilarityAlignment(scale, rotation, translation, rmse)


def fit_affine(
    source: np.ndarray,
    target: np.ndarray,
    *,
    ridge_alpha: float = 0.1,
    condition_max: float = 1.0e6,
    anchor_index: int = -1,
) -> AffineAlignment:
    """Match the production DPVO centered affine/ridge/anchor convention."""
    src = np.asarray(source, dtype=np.float64)
    dst = np.asarray(target, dtype=np.float64)
    if src.shape != dst.shape or src.ndim != 2 or src.shape[1] != 3:
        raise ValueError(f"Expected matching Nx3 points, got {src.shape} and {dst.shape}")
    if len(src) < 4:
        raise ValueError("Affine calibration needs at least four point pairs")
    src_mean = np.mean(src, axis=0)
    dst_mean = np.mean(dst, axis=0)
    src_centered = src - src_mean
    dst_centered = dst - dst_mean
    singular = np.linalg.svd(src_centered, compute_uv=False)
    rank = int(np.linalg.matrix_rank(src_centered))
    condition = float(np.linalg.cond(src_centered))
    use_ridge = rank < 3 or not np.isfinite(condition) or condition > condition_max
    if use_ridge:
        lhs = src_centered.T @ src_centered + float(ridge_alpha) * np.eye(3)
        weights = np.linalg.solve(lhs, src_centered.T @ dst_centered)
    else:
        weights = np.linalg.lstsq(src_centered, dst_centered, rcond=None)[0]
    matrix = weights.T
    translation = dst_mean - matrix @ src_mean
    candidate = matrix @ src[anchor_index] + translation
    anchor_offset = dst[anchor_index] - candidate
    return AffineAlignment(
        matrix=matrix,
        translation=translation,
        anchor_offset=anchor_offset,
        rank=rank,
        condition_number=condition,
        used_ridge=use_ridge,
        singular_values=singular,
    )


def trajectory_metrics(prediction: np.ndarray, target: np.ndarray) -> dict:
    error = np.asarray(prediction, dtype=np.float64) - np.asarray(target, dtype=np.float64)
    if error.ndim != 2 or error.shape[1] != 3 or len(error) == 0:
        raise ValueError(f"Expected non-empty Nx3 errors, got {error.shape}")
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
    pred = np.asarray(prediction, dtype=np.float64)
    gt = np.asarray(target, dtype=np.float64)
    path = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(gt, axis=0), axis=1))]
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
                np.linalg.norm((pred[end] - pred[start]) - (gt[end] - gt[start]))
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

