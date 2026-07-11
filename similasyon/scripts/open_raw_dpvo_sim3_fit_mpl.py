#!/usr/bin/env python3
"""Open GT vs Sim3-fitted raw DPVO trajectory."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def set_axes_equal(ax) -> None:
    xs, ys, zs = ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()
    mids = [np.mean(xs), np.mean(ys), np.mean(zs)]
    radius = 0.5 * max(abs(xs[1] - xs[0]), abs(ys[1] - ys[0]), abs(zs[1] - zs[0]))
    ax.set_xlim3d(mids[0] - radius, mids[0] + radius)
    ax.set_ylim3d(mids[1] - radius, mids[1] + radius)
    ax.set_zlim3d(mids[2] - radius, mids[2] + radius)


def load(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    gt = []
    pred = []
    phase = []
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            gt.append([float(row["gt_x"]), float(row["gt_y"]), float(row["gt_z"])])
            pred.append([float(row["sim3_x"]), float(row["sim3_y"]), float(row["sim3_z"])])
            phase.append(row["phase"])
    return np.asarray(gt), np.asarray(pred), np.asarray(phase)


def mean_rmse(pred: np.ndarray, gt: np.ndarray) -> tuple[float, float]:
    err = np.linalg.norm(pred - gt, axis=1)
    return float(err.mean()), float(math.sqrt(np.mean(err * err)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("similasyon/_debug/raw_dpvo_fit_from_csv_npcyaml_gc/fit_predictions.csv"),
    )
    parser.add_argument("--eval-only", action="store_true")
    args = parser.parse_args()

    gt, pred, phase = load(args.csv)
    mask = phase == "eval" if args.eval_only else np.ones(len(phase), dtype=bool)
    gt_plot = gt[mask]
    pred_plot = pred[mask]
    e, rmse = mean_rmse(pred_plot, gt_plot)

    fig = plt.figure(figsize=(13, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(gt_plot[:, 0], gt_plot[:, 1], gt_plot[:, 2], color="green", linewidth=2.5, label="GT")
    ax.plot(
        pred_plot[:, 0],
        pred_plot[:, 1],
        pred_plot[:, 2],
        color="blue",
        linewidth=1.8,
        label=f"DPVO + Sim3 | E={e:.1f}, RMSE={rmse:.1f}",
    )
    ax.scatter(gt_plot[0, 0], gt_plot[0, 1], gt_plot[0, 2], color="cyan", s=60, label="start")
    ax.scatter(gt_plot[-1, 0], gt_plot[-1, 1], gt_plot[-1, 2], color="lime", s=60, label="GT end")
    ax.scatter(pred_plot[0, 0], pred_plot[0, 1], pred_plot[0, 2], color="red", marker="o", s=70, label="DPVO start")
    ax.scatter(pred_plot[-1, 0], pred_plot[-1, 1], pred_plot[-1, 2], color="black", marker="X", s=90, label="DPVO end")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    title_scope = "eval-only" if args.eval_only else "all frames"
    ax.set_title(f"GT vs DPVO Sim3 ({title_scope})")
    ax.legend()
    ax.grid(True)
    set_axes_equal(ax)
    plt.show()


if __name__ == "__main__":
    main()
