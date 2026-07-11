#!/usr/bin/env python3
"""Fit legacy regression and Sim3 on cached raw DPVO CSV, evaluate remaining frames."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from sklearn.linear_model import LinearRegression


def load_csv(path: Path) -> tuple[list[int], np.ndarray, np.ndarray]:
    image_idx: list[int] = []
    raw: list[list[float]] = []
    gt: list[list[float]] = []
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            image_idx.append(int(row["image_idx"]))
            raw.append([float(row["raw_dpvo_x"]), float(row["raw_dpvo_y"]), float(row["raw_dpvo_z"])])
            gt.append([float(row["gt_x"]), float(row["gt_y"]), float(row["gt_z"])])
    return image_idx, np.asarray(raw, dtype=np.float64), np.asarray(gt, dtype=np.float64)


def metrics(pred: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    err = np.linalg.norm(pred - gt, axis=1)
    return {
        "E": float(err.mean()),
        "RMSE": float(math.sqrt(float(np.mean(err * err)))),
        "median": float(np.median(err)),
        "p95": float(np.percentile(err, 95)),
        "max": float(err.max()),
    }


def fit_legacy_regression(raw_calib: np.ndarray, gt_calib: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    model = LinearRegression(fit_intercept=False, positive=False)
    model.fit(raw_calib, gt_calib)
    coef = np.asarray(model.coef_, dtype=np.float64)
    intercept = np.asarray(model.intercept_, dtype=np.float64)
    return coef, intercept


def apply_legacy_regression(raw: np.ndarray, coef: np.ndarray, intercept: np.ndarray) -> np.ndarray:
    return raw @ coef.T + intercept


def fit_umeyama_sim3(src: np.ndarray, dst: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Return scale, rotation, translation mapping src -> dst.

    Uses Umeyama least-squares similarity transform:
        dst ~= scale * R @ src + t
    """
    if src.shape != dst.shape or src.ndim != 2 or src.shape[1] != 3:
        raise ValueError(f"Expected Nx3 src/dst, got {src.shape}, {dst.shape}")

    n = src.shape[0]
    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)
    src_c = src - mu_src
    dst_c = dst - mu_dst

    cov = (dst_c.T @ src_c) / n
    u, d, vt = np.linalg.svd(cov)
    s = np.eye(3)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        s[-1, -1] = -1
    r = u @ s @ vt
    var_src = float(np.sum(src_c * src_c) / n)
    scale = float(np.trace(np.diag(d) @ s) / var_src)
    t = mu_dst - scale * (r @ mu_src)
    return scale, r, t


def apply_sim3(raw: np.ndarray, scale: float, r: np.ndarray, t: np.ndarray) -> np.ndarray:
    return (scale * (r @ raw.T)).T + t


def write_predictions(
    path: Path,
    image_idx: list[int],
    raw: np.ndarray,
    gt: np.ndarray,
    pred_reg: np.ndarray,
    pred_sim3: np.ndarray,
    calib_frames: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "image_idx",
                "phase",
                "raw_x",
                "raw_y",
                "raw_z",
                "gt_x",
                "gt_y",
                "gt_z",
                "legacy_reg_x",
                "legacy_reg_y",
                "legacy_reg_z",
                "legacy_reg_err_3d",
                "sim3_x",
                "sim3_y",
                "sim3_z",
                "sim3_err_3d",
            ]
        )
        for i, idx in enumerate(image_idx):
            reg_err = float(np.linalg.norm(pred_reg[i] - gt[i]))
            sim3_err = float(np.linalg.norm(pred_sim3[i] - gt[i]))
            w.writerow(
                [
                    idx,
                    "calib" if i < calib_frames else "eval",
                    *raw[i].tolist(),
                    *gt[i].tolist(),
                    *pred_reg[i].tolist(),
                    reg_err,
                    *pred_sim3[i].tolist(),
                    sim3_err,
                ]
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("similasyon/_debug/raw_dpvo_scaled_only_npcyaml_gc/raw_dpvo_times_scale.csv"),
    )
    parser.add_argument("--calib-frames", type=int, default=450)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("similasyon/_debug/raw_dpvo_fit_from_csv_npcyaml_gc"),
    )
    args = parser.parse_args()

    image_idx, raw, gt = load_csv(args.csv)
    c = args.calib_frames
    if len(raw) <= c:
        raise ValueError(f"Need more than {c} rows, got {len(raw)}")

    coef, intercept = fit_legacy_regression(raw[:c], gt[:c])
    pred_reg = apply_legacy_regression(raw, coef, intercept)

    sim3_scale, sim3_r, sim3_t = fit_umeyama_sim3(raw[:c], gt[:c])
    pred_sim3 = apply_sim3(raw, sim3_scale, sim3_r, sim3_t)

    results = {
        "source_csv": str(args.csv),
        "calib_frames": c,
        "eval_frames": len(raw) - c,
        "legacy_regression": {
            "coef": coef.tolist(),
            "intercept": intercept.tolist(),
            "calib": metrics(pred_reg[:c], gt[:c]),
            "eval": metrics(pred_reg[c:], gt[c:]),
            "all": metrics(pred_reg, gt),
        },
        "sim3": {
            "scale": sim3_scale,
            "rotation": sim3_r.tolist(),
            "translation": sim3_t.tolist(),
            "calib": metrics(pred_sim3[:c], gt[:c]),
            "eval": metrics(pred_sim3[c:], gt[c:]),
            "all": metrics(pred_sim3, gt),
        },
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = args.output_dir / "fit_metrics.json"
    summary.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    write_predictions(
        args.output_dir / "fit_predictions.csv",
        image_idx,
        raw,
        gt,
        pred_reg,
        pred_sim3,
        c,
    )

    print("=" * 72)
    print("Raw DPVO cached fit evaluation")
    print("=" * 72)
    print(f"Source CSV:   {args.csv}")
    print(f"Calib frames: {c}")
    print(f"Eval frames:  {len(raw) - c}")
    print("-" * 72)
    print("Legacy regression eval:")
    for k, v in results["legacy_regression"]["eval"].items():
        print(f"  {k}: {v:.6f}")
    print("Sim3 eval:")
    for k, v in results["sim3"]["eval"].items():
        print(f"  {k}: {v:.6f}")
    print("-" * 72)
    print(f"Saved metrics:     {summary}")
    print(f"Saved predictions: {args.output_dir / 'fit_predictions.csv'}")


if __name__ == "__main__":
    main()
