#!/usr/bin/env python3
"""Open GT vs DPVO multiplied by one scalar only."""

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


def mean_e(pred: np.ndarray, gt: np.ndarray) -> tuple[float, float]:
    err = np.linalg.norm(pred - gt, axis=1)
    return float(err.mean()), float(math.sqrt(np.mean(err * err)))


def load_raw(csv_path: Path) -> tuple[np.ndarray, np.ndarray, list[int]]:
    gt = []
    raw = []
    image_idx = []
    with csv_path.open() as f:
        for row in csv.DictReader(f):
            image_idx.append(int(row["image_idx"]))
            gt.append([float(row["gt_x"]), float(row["gt_y"]), float(row["gt_z"])])
            raw.append([float(row["raw_x"]), float(row["raw_y"]), float(row["raw_z"])])
    return np.asarray(gt, dtype=float), np.asarray(raw, dtype=float), image_idx


def write_scaled_csv(out_path: Path, image_idx: list[int], gt: np.ndarray, scaled: np.ndarray, scale: float) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image_idx", "scale", "gt_x", "gt_y", "gt_z", "dpvo_scaled_x", "dpvo_scaled_y", "dpvo_scaled_z"])
        for idx, g, p in zip(image_idx, gt, scaled):
            w.writerow([idx, scale, *g.tolist(), *p.tolist()])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("similasyon/_debug/trajectory_dpvo_oturum3_1080p_defaultcfg_scaled/dpvo_scaled_variants.csv"),
    )
    parser.add_argument("--scale", type=float, default=63.540449556921054)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("similasyon/_debug/trajectory_dpvo_oturum3_1080p_defaultcfg_scaled/dpvo_scale_only_chord.csv"),
    )
    args = parser.parse_args()

    gt, raw, image_idx = load_raw(args.csv)
    scaled = raw * args.scale
    write_scaled_csv(args.out, image_idx, gt, scaled, args.scale)

    raw_e, raw_rmse = mean_e(raw, gt)
    scaled_e, scaled_rmse = mean_e(scaled, gt)
    print(f"scale={args.scale}")
    print(f"raw_E={raw_e:.6f} raw_RMSE={raw_rmse:.6f}")
    print(f"scaled_E={scaled_e:.6f} scaled_RMSE={scaled_rmse:.6f}")
    print(f"saved={args.out}")

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(gt[:, 0], gt[:, 1], gt[:, 2], color="green", linewidth=2.5, label="GT")
    ax.plot(
        scaled[:, 0],
        scaled[:, 1],
        scaled[:, 2],
        color="orange",
        linewidth=1.8,
        label=f"DPVO * {args.scale:.6f} | E={scaled_e:.1f}",
    )
    ax.scatter(gt[0, 0], gt[0, 1], gt[0, 2], color="cyan", s=60, label="start")
    ax.scatter(gt[-1, 0], gt[-1, 1], gt[-1, 2], color="lime", s=60, label="GT end")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("GT vs scalar-multiplied DPVO only")
    ax.legend()
    ax.grid(True)
    set_axes_equal(ax)
    plt.show()


if __name__ == "__main__":
    main()
