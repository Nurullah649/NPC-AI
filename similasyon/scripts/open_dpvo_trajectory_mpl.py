#!/usr/bin/env python3
"""Open DPVO vs GT trajectory as an interactive Matplotlib 3D plot."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def set_axes_equal(ax):
    """Set equal scale for 3D axes."""
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    y_range = abs(y_limits[1] - y_limits[0])
    z_range = abs(z_limits[1] - z_limits[0])

    x_middle = np.mean(x_limits)
    y_middle = np.mean(y_limits)
    z_middle = np.mean(z_limits)

    plot_radius = 0.5 * max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


def load_csv(path: Path):
    pred = []
    gt = []
    idx = []
    err = []
    with path.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            idx.append(int(row["image_idx"]))
            pred.append([float(row["pred_x"]), float(row["pred_y"]), float(row["pred_z"])])
            gt.append([float(row["gt_x"]), float(row["gt_y"]), float(row["gt_z"])])
            err.append(float(row["err_3d"]))
    return np.asarray(idx), np.asarray(pred), np.asarray(gt), np.asarray(err)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        default=Path(
            "similasyon/_debug/trajectory_dpvo_oturum3_1080p_defaultcfg/"
            "dpvo_vs_gt_eval_trajectory.csv"
        ),
        type=Path,
    )
    args = parser.parse_args()

    idx, pred, gt, err = load_csv(args.csv)

    e = float(np.mean(err))
    rmse = float(math.sqrt(np.mean(err * err)))

    fig = plt.figure(figsize=(13, 9))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(gt[:, 0], gt[:, 1], gt[:, 2], color="green", linewidth=2.0, label="GT")
    ax.plot(pred[:, 0], pred[:, 1], pred[:, 2], color="red", linewidth=2.0, label="DPVO pred")

    step = max(1, len(gt) // 60)
    for i in range(0, len(gt), step):
        ax.plot(
            [gt[i, 0], pred[i, 0]],
            [gt[i, 1], pred[i, 1]],
            [gt[i, 2], pred[i, 2]],
            color="purple",
            alpha=0.18,
            linewidth=0.8,
        )

    ax.scatter(gt[0, 0], gt[0, 1], gt[0, 2], color="blue", s=60, label=f"Eval start {idx[0]}")
    ax.scatter(gt[-1, 0], gt[-1, 1], gt[-1, 2], color="black", s=60, label=f"GT end {idx[-1]}")
    ax.scatter(pred[-1, 0], pred[-1, 1], pred[-1, 2], color="orange", s=60, label="Pred end")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(f"DPVO vs GT trajectory | E={e:.2f}m | RMSE={rmse:.2f}m")
    ax.legend()
    ax.grid(True)
    set_axes_equal(ax)

    print("Matplotlib interactive 3D trajectory opened.")
    print("Controls: left drag rotate, right drag/pan depending backend, wheel zoom.")
    print(f"CSV: {args.csv.resolve()}")
    print(f"E={e:.4f}, RMSE={rmse:.4f}")

    plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
