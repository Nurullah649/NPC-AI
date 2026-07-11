#!/usr/bin/env python3
"""Apply the production PositioningDPVO calibration code to a cached raw run."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SIM_ROOT = REPO_ROOT / "similasyon"
if str(SIM_ROOT) not in sys.path:
    sys.path.insert(0, str(SIM_ROOT))

from src.models.positioning_dpvo import PositioningDPVO


def load(path: Path) -> tuple[np.ndarray, np.ndarray]:
    raw: list[list[float]] = []
    gt: list[list[float]] = []
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            raw.append([float(row["raw_dpvo_x"]), float(row["raw_dpvo_y"]), float(row["raw_dpvo_z"])])
            gt.append([float(row["gt_x"]), float(row["gt_y"]), float(row["gt_z"])])
    return np.asarray(raw, dtype=np.float64), np.asarray(gt, dtype=np.float64)


def metrics(prediction: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    error = np.linalg.norm(prediction - gt, axis=1)
    return {
        "E": float(error.mean()),
        "RMSE": float(math.sqrt(float(np.mean(error * error)))),
        "median": float(np.median(error)),
        "p95": float(np.percentile(error, 95)),
        "max": float(error.max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--calib-frames", type=int, default=450)
    parser.add_argument("--fit-method", choices=("linear", "ridge", "sim3"), default="linear")
    parser.add_argument("--ridge-alpha", type=float, default=0.1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    raw, gt = load(args.csv)
    calibration = args.calib_frames
    positioning = PositioningDPVO(
        {
            "dpvo": {
                "fit_method": args.fit_method,
                "absolute_ridge_alpha": args.ridge_alpha,
                "min_calib_frames": calibration,
                "use_direction_guard": True,
                "direction_guard_min_points": 30,
                "min_allowed_scale": 0.02,
                "max_allowed_scale": 50.0,
                "min_direction_similarity": -0.2,
            }
        }
    )
    positioning.dpvo_buffer = raw[:calibration].tolist()
    positioning.gt_buffer = gt[:calibration].tolist()
    positioning._update_calibration()
    if not positioning.is_calibrated:
        raise RuntimeError("Üretim hizalaması kalibre olmadı.")

    prediction = np.asarray([positioning._align_dpvo_to_gt(point) for point in raw])
    report = {
        "fit_method": positioning.fit_method,
        "calibration_frames": calibration,
        "evaluation_frames": len(raw) - calibration,
        "coef": positioning.sim3_R.tolist(),
        "intercept": positioning.sim3_t.tolist(),
        "anchor_offset": positioning.alignment_anchor_offset.tolist(),
        "calibration": metrics(prediction[:calibration], gt[:calibration]),
        "evaluation": metrics(prediction[calibration:], gt[calibration:]),
        "all": metrics(prediction, gt),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report["evaluation"], indent=2))


if __name__ == "__main__":
    main()
