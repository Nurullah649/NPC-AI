#!/usr/bin/env python3
"""Create a compact, reproducible summary of the 800-frame DPVO A/B runs."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent / "results" / "ab_800"


def load_trajectory(path: Path) -> tuple[np.ndarray, np.ndarray]:
    raw: list[list[float]] = []
    gt: list[list[float]] = []
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            raw.append([float(row["raw_dpvo_x"]), float(row["raw_dpvo_y"]), float(row["raw_dpvo_z"])])
            gt.append([float(row["gt_x"]), float(row["gt_y"]), float(row["gt_z"])])
    return np.asarray(raw, dtype=np.float64), np.asarray(gt, dtype=np.float64)


def path_diagnostics(raw: np.ndarray, gt: np.ndarray, calib_frames: int = 450) -> dict[str, float]:
    raw_step = np.linalg.norm(np.diff(raw, axis=0), axis=1)
    gt_step = np.linalg.norm(np.diff(gt, axis=0), axis=1)
    split = calib_frames - 1
    raw_cal_path = float(raw_step[:split].sum())
    gt_cal_path = float(gt_step[:split].sum())
    path_scale = gt_cal_path / max(raw_cal_path, 1e-12)
    raw_eval_path = float(raw_step[split:].sum())
    gt_eval_path = float(gt_step[split:].sum())
    threshold = max(float(np.median(raw_step[:split])) * 0.05, 1e-8)
    return {
        "calibration_path_scale": path_scale,
        "eval_motion_ratio": path_scale * raw_eval_path / max(gt_eval_path, 1e-12),
        "eval_near_zero_step_fraction": float(np.mean(raw_step[split:] < threshold)),
    }


def named_metric(metrics: list[dict], name: str) -> dict:
    return next(row for row in metrics if row["name"] == name)


def best_family(metrics: list[dict], family: str) -> dict:
    return min((row for row in metrics if row["family"] == family), key=lambda row: row["eval"]["E"])


def main() -> None:
    rows: list[dict] = []
    for run_dir in sorted(path for path in ROOT.iterdir() if path.is_dir()):
        metrics_path = run_dir / "alignment" / "corrected_metrics.json"
        summary_path = run_dir / "raw_summary.json"
        trajectory_path = run_dir / "raw_fixed_trajectory.csv"
        if not (metrics_path.exists() and summary_path.exists() and trajectory_path.exists()):
            continue
        metrics = json.loads(metrics_path.read_text())
        summary = json.loads(summary_path.read_text())
        raw, gt = load_trajectory(trajectory_path)
        best = metrics[0]
        sim3 = named_metric(metrics, "abs_sim3")
        absolute = min(
            (row for row in metrics if row["family"] in {"absolute regression", "Sim3"}),
            key=lambda row: row["eval"]["E"],
        )
        delta = min(
            (row for row in metrics if row["family"] in {"delta regression", "delta Sim3"}),
            key=lambda row: row["eval"]["E"],
        )
        config_text = (run_dir / "npc_used.yaml").read_text()
        rows.append(
            {
                "run": run_dir.name,
                "frames": summary["frames"],
                "minutes": summary["elapsed_seconds"] / 60.0,
                "fps": summary["frames"] / summary["elapsed_seconds"],
                "undistort": summary.get("undistort", False),
                "patch_strategy": "RANDOM" if "CENTROID_SEL_STRAT: 'RANDOM'" in config_text else "EDGE_BIAS",
                "motion_model": "DAMPED_LINEAR" if "MOTION_MODEL: 'DAMPED_LINEAR'" in config_text else "CONSTANT_VELOCITY",
                "mixed_precision": "MIXED_PRECISION: True" in config_text,
                "best_method": best["name"],
                "best_E": best["eval"]["E"],
                "best_RMSE": best["eval"]["RMSE"],
                "best_absolute_method": absolute["name"],
                "best_absolute_E": absolute["eval"]["E"],
                "best_delta_method": delta["name"],
                "best_delta_E": delta["eval"]["E"],
                "sim3_E": sim3["eval"]["E"],
                "sim3_RMSE": sim3["eval"]["RMSE"],
                **path_diagnostics(raw, gt),
            }
        )

    ranked = sorted(rows, key=lambda row: (row["best_E"], row["best_RMSE"]))
    (ROOT / "AB_OZET.json").write_text(json.dumps(ranked, indent=2, ensure_ascii=False) + "\n")

    csv_fields = list(ranked[0]) if ranked else []
    with (ROOT / "AB_OZET.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fields)
        writer.writeheader()
        writer.writerows(ranked)

    lines = [
        "# DPVO 800-frame kontrollü A/B özeti",
        "",
        "Her koşul ilk 450 kareyle hizalanmış, E/RMSE kalan 350 karede ölçülmüştür.",
        "",
        "| # | Koşul | Patch | Motion | AMP | Undistort | FPS | En iyi E | RMSE | Sim3 E | En iyi yöntem |",
        "|---:|---|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for rank, row in enumerate(ranked, 1):
        lines.append(
            f"| {rank} | `{row['run']}` | {row['patch_strategy']} | {row['motion_model']} | "
            f"{int(row['mixed_precision'])} | {int(row['undistort'])} | {row['fps']:.2f} | "
            f"{row['best_E']:.4f} | {row['best_RMSE']:.4f} | {row['sim3_E']:.4f} | "
            f"`{row['best_method']}` |"
        )
    (ROOT / "AB_OZET.md").write_text("\n".join(lines) + "\n")
    print(f"{len(ranked)} koşul özetlendi: {ROOT / 'AB_OZET.md'}")


if __name__ == "__main__":
    main()
