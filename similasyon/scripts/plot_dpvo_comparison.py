#!/usr/bin/env python3
"""Create a compact GT-versus-DPVO diagnostic plot from an evaluator CSV."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


COMPONENTS = (("x", "X (m)"), ("y", "Y (m)"), ("z", "Z (m)"))
GT_COLOR = "#0072B2"
DPVO_COLOR = "#D55E00"
HEALTHY_COLOR = "#009E73"


def load_csv(path: Path) -> np.ndarray:
    rows = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding="utf-8")
    if rows.size == 0:
        raise ValueError(f"Prediction CSV is empty: {path}")
    return np.atleast_1d(rows)


def metric_text(metrics_path: Path | None) -> str:
    if metrics_path is None or not metrics_path.is_file():
        return ""
    values = json.loads(metrics_path.read_text(encoding="utf-8"))
    return "E = {E_3d:.3f} m   |   RMSE = {RMSE_3d:.3f} m".format(**values)


def add_calibration_markers(axis: plt.Axes, calibration_seconds: float) -> None:
    axis.axvspan(0, calibration_seconds, color=HEALTHY_COLOR, alpha=0.08, lw=0)
    axis.axvline(calibration_seconds, color=HEALTHY_COLOR, ls="--", lw=1.4)


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot GT and DPVO predictions from evaluator output.")
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--fps", default=7.5, type=float)
    parser.add_argument("--metrics", type=Path, default=None)
    args = parser.parse_args()

    if args.fps <= 0:
        raise ValueError("--fps must be positive")

    rows = load_csv(args.predictions)
    required = {
        "sample_index",
        "health_status",
        "pred_x",
        "pred_y",
        "pred_z",
        "gt_x",
        "gt_y",
        "gt_z",
        "err_3d",
    }
    available = set(rows.dtype.names or ())
    missing = required - available
    if missing:
        raise ValueError(f"Prediction CSV lacks columns: {sorted(missing)}")

    seconds = rows["sample_index"].astype(float) / args.fps
    healthy = rows["health_status"].astype(str) == "1"
    if not np.any(healthy):
        raise ValueError("Prediction CSV contains no health=1 calibration rows.")
    calibration_seconds = (float(np.max(rows["sample_index"][healthy])) + 1.0) / args.fps

    fig = plt.figure(figsize=(21, 11.5), dpi=150)
    grid = fig.add_gridspec(2, 3, hspace=0.34, wspace=0.29)
    component_axes = [fig.add_subplot(grid[0, index]) for index in range(3)]
    trajectory_3d = fig.add_subplot(grid[1, 0], projection="3d")
    trajectory_xy = fig.add_subplot(grid[1, 1])
    error_axis = fig.add_subplot(grid[1, 2])

    for axis, (component, label) in zip(component_axes, COMPONENTS):
        gt = rows[f"gt_{component}"].astype(float)
        pred = rows[f"pred_{component}"].astype(float)
        axis.plot(seconds, gt, color=GT_COLOR, lw=1.55, label="GT")
        axis.plot(seconds, pred, color=DPVO_COLOR, lw=1.15, alpha=0.9, label="DPVO")
        add_calibration_markers(axis, calibration_seconds)
        axis.set_title(label, weight="bold")
        axis.set_xlabel("Time (s)")
        axis.set_ylabel(label)
        axis.grid(alpha=0.22)

    component_axes[0].legend(loc="best", frameon=True)
    component_axes[0].text(
        0.02,
        0.04,
        "Green region: health=1 calibration\nDashed line: health=0 starts",
        transform=component_axes[0].transAxes,
        fontsize=8.5,
        color="#33665a",
        va="bottom",
        bbox={"facecolor": "white", "edgecolor": "#009E73", "alpha": 0.82, "pad": 3},
    )

    gt_x, gt_y, gt_z = (rows[f"gt_{axis}"].astype(float) for axis, _ in COMPONENTS)
    pred_x, pred_y, pred_z = (rows[f"pred_{axis}"].astype(float) for axis, _ in COMPONENTS)
    calibration_end = int(np.flatnonzero(healthy)[-1])

    trajectory_3d.plot(gt_x, gt_y, gt_z, color=GT_COLOR, lw=1.45, label="GT")
    trajectory_3d.plot(pred_x, pred_y, pred_z, color=DPVO_COLOR, lw=1.1, alpha=0.9, label="DPVO")
    trajectory_3d.scatter(*[array[0] for array in (gt_x, gt_y, gt_z)], color="black", s=19, label="Start")
    trajectory_3d.scatter(
        *[array[calibration_end] for array in (gt_x, gt_y, gt_z)],
        color=HEALTHY_COLOR,
        marker="s",
        s=30,
        label="Health=0 start",
    )
    trajectory_3d.set_title("3D trajectory", weight="bold")
    trajectory_3d.set_xlabel("X (m)")
    trajectory_3d.set_ylabel("Y (m)")
    trajectory_3d.set_zlabel("Z (m)")
    trajectory_3d.view_init(elev=24, azim=-56)
    trajectory_3d.legend(loc="upper left", fontsize=8)

    trajectory_xy.plot(gt_x, gt_y, color=GT_COLOR, lw=1.55, label="GT")
    trajectory_xy.plot(pred_x, pred_y, color=DPVO_COLOR, lw=1.15, alpha=0.9, label="DPVO")
    trajectory_xy.scatter(gt_x[0], gt_y[0], color="black", s=22, zorder=4, label="Start")
    trajectory_xy.scatter(
        gt_x[calibration_end],
        gt_y[calibration_end],
        color=HEALTHY_COLOR,
        marker="s",
        s=36,
        zorder=4,
        label="Health=0 start",
    )
    trajectory_xy.set_title("Top view (X–Y)", weight="bold")
    trajectory_xy.set_xlabel("X (m)")
    trajectory_xy.set_ylabel("Y (m)")
    trajectory_xy.axis("equal")
    trajectory_xy.grid(alpha=0.22)
    trajectory_xy.legend(loc="best", fontsize=8)

    error = rows["err_3d"].astype(float)
    error_axis.plot(seconds, error, color="#6A3D9A", lw=1.35, label="3D position error")
    add_calibration_markers(error_axis, calibration_seconds)
    error_axis.set_title("3D error", weight="bold")
    error_axis.set_xlabel("Time (s)")
    error_axis.set_ylabel("Error (m)")
    error_axis.grid(alpha=0.22)
    error_axis.legend(loc="upper left")

    text = metric_text(args.metrics)
    fig.suptitle(
        "GT vs DPVO — exact 7.5 FPS sampling" + (f"\n{text}" if text else ""),
        fontsize=16,
        weight="bold",
        y=0.985,
    )
    fig.text(
        0.5,
        0.015,
        f"{len(rows)} samples · health=1: first {calibration_seconds:.0f} s · health=0: remaining evaluation section",
        ha="center",
        fontsize=10,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight")
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
