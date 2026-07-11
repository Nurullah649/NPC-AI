#!/usr/bin/env python3
"""Open raw/scaled DPVO variants against GT as interactive Matplotlib 3D plot."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def set_axes_equal(ax):
    xs, ys, zs = ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()
    mids = [np.mean(xs), np.mean(ys), np.mean(zs)]
    radius = 0.5 * max(abs(xs[1] - xs[0]), abs(ys[1] - ys[0]), abs(zs[1] - zs[0]))
    ax.set_xlim3d(mids[0] - radius, mids[0] + radius)
    ax.set_ylim3d(mids[1] - radius, mids[1] + radius)
    ax.set_zlim3d(mids[2] - radius, mids[2] + radius)


def load(path: Path):
    cols = {k: [] for k in [
        "gt_x", "gt_y", "gt_z",
        "raw_x", "raw_y", "raw_z",
        "origin_x", "origin_y", "origin_z",
        "centered_x", "centered_y", "centered_z",
        "axis_x", "axis_y", "axis_z",
        "affine_x", "affine_y", "affine_z",
    ]}
    with path.open() as f:
        for r in csv.DictReader(f):
            for k in cols:
                cols[k].append(float(r[k]))
    def arr(prefix):
        return np.c_[cols[f"{prefix}_x"], cols[f"{prefix}_y"], cols[f"{prefix}_z"]]
    return {
        "GT": arr("gt"),
        "Raw DPVO": arr("raw"),
        "Origin scale": arr("origin"),
        "Centered scale": arr("centered"),
        "Axis scale": arr("axis"),
        "Affine diagnostic": arr("affine"),
    }


def err(a, gt):
    e = np.linalg.norm(a - gt, axis=1)
    return float(e.mean()), float(math.sqrt(np.mean(e * e)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        default=Path("similasyon/_debug/trajectory_dpvo_oturum3_1080p_defaultcfg_scaled/dpvo_scaled_variants.csv"),
        type=Path,
    )
    args = parser.parse_args()
    d = load(args.csv)
    gt = d["GT"]

    fig = plt.figure(figsize=(13, 9))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(gt[:, 0], gt[:, 1], gt[:, 2], color="green", linewidth=2.5, label="GT")

    styles = [
        ("Raw DPVO", "red", 1.5),
        ("Origin scale", "orange", 1.2),
        ("Centered scale", "blue", 2.0),
        ("Axis scale", "purple", 1.6),
        ("Affine diagnostic", "black", 1.2),
    ]
    for name, color, lw in styles:
        a = d[name]
        e, r = err(a, gt)
        ax.plot(a[:, 0], a[:, 1], a[:, 2], color=color, linewidth=lw, label=f"{name} E={e:.1f}")

    ax.scatter(gt[0, 0], gt[0, 1], gt[0, 2], color="cyan", s=60, label="start")
    ax.scatter(gt[-1, 0], gt[-1, 1], gt[-1, 2], color="lime", s=60, label="GT end")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("DPVO scale variants vs GT")
    ax.legend()
    ax.grid(True)
    set_axes_equal(ax)
    plt.show()


if __name__ == "__main__":
    main()
