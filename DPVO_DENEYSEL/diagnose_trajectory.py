#!/usr/bin/env python3
"""Report where raw DPVO loses translational motion after scale normalization."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def load(path: Path) -> tuple[np.ndarray, np.ndarray]:
    raw: list[list[float]] = []
    gt: list[list[float]] = []
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            raw.append([float(row["raw_dpvo_x"]), float(row["raw_dpvo_y"]), float(row["raw_dpvo_z"])])
            gt.append([float(row["gt_x"]), float(row["gt_y"]), float(row["gt_z"])])
    return np.asarray(raw, dtype=np.float64), np.asarray(gt, dtype=np.float64)


def path_length(points: np.ndarray) -> float:
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--calib-frames", type=int, default=450)
    parser.add_argument("--window", type=int, default=100)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    raw, gt = load(args.csv)
    if len(raw) <= args.calib_frames:
        raise ValueError("Trajectory kalibrasyon bölümünden uzun olmalı.")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    raw_cal_path = path_length(raw[: args.calib_frames])
    gt_cal_path = path_length(gt[: args.calib_frames])
    scale = gt_cal_path / max(raw_cal_path, 1e-12)
    calibration_step = np.linalg.norm(np.diff(raw[: args.calib_frames], axis=0), axis=1)
    near_zero_threshold = max(float(np.median(calibration_step)) * 0.05, 1e-8)

    rows: list[dict] = []
    for start in range(0, len(raw) - 1, args.window):
        end = min(start + args.window, len(raw) - 1)
        raw_segment = raw[start : end + 1]
        gt_segment = gt[start : end + 1]
        raw_steps = np.linalg.norm(np.diff(raw_segment, axis=0), axis=1)
        raw_scaled_path = float(raw_steps.sum() * scale)
        gt_path = path_length(gt_segment)
        ratio = raw_scaled_path / max(gt_path, 1e-12)
        rows.append(
            {
                "start": start,
                "end": end,
                "raw_scaled_path": raw_scaled_path,
                "gt_path": gt_path,
                "motion_ratio": ratio,
                "near_zero_step_fraction": float(np.mean(raw_steps < near_zero_threshold)),
                "status": "LOST" if ratio < 0.20 else ("WEAK" if ratio < 0.50 else "OK"),
            }
        )

    report = {
        "frames": len(raw),
        "calibration_frames": args.calib_frames,
        "calibration_path_scale": scale,
        "near_zero_threshold_raw": near_zero_threshold,
        "lost_windows": sum(row["status"] == "LOST" for row in rows),
        "weak_windows": sum(row["status"] == "WEAK" for row in rows),
        "windows": rows,
    }
    (args.output_dir / "trajectory_health.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    )
    with (args.output_dir / "trajectory_health.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# Ham DPVO hareket sağlık raporu",
        "",
        f"İlk {args.calib_frames} kare yol oranıyla bulunan yalnızca-teşhis scale: `{scale:.6f}`.",
        "Bu scale hizalama/skor için kullanılmaz; DPVO'nun hareketi kaybettiği aralıkları işaretler.",
        "",
        "| Aralık | Ölçekli raw yol | GT yol | Oran | Sıfıra yakın adım | Durum |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['start']}–{row['end']} | {row['raw_scaled_path']:.3f} | "
            f"{row['gt_path']:.3f} | {row['motion_ratio']:.3f} | "
            f"{row['near_zero_step_fraction']:.1%} | {row['status']} |"
        )
    (args.output_dir / "trajectory_health.md").write_text("\n".join(lines) + "\n")
    print(
        f"windows={len(rows)} lost={report['lost_windows']} weak={report['weak_windows']} "
        f"scale={scale:.4f} -> {args.output_dir}"
    )


if __name__ == "__main__":
    main()
