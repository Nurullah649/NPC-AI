#!/usr/bin/env python3
"""Locate the largest NED discontinuity in a DPVO loop-closure run."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def vector(row: dict[str, str], prefix: str) -> list[float]:
    return [float(row[f"{prefix}_{axis}"]) for axis in ("x", "y", "z")]


def main() -> int:
    parser = argparse.ArgumentParser(description="Report the largest DPVO NED pose jump.")
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    with args.predictions.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) < 2:
        raise ValueError("At least two prediction rows are required.")

    predicted = np.asarray([vector(row, "pred") for row in rows], dtype=np.float64)
    steps = np.linalg.norm(np.diff(predicted, axis=0), axis=1)
    index = int(np.argmax(steps) + 1)
    before, after = rows[index - 1], rows[index]
    report = {
        "largest_ned_step_m": float(steps[index - 1]),
        "before": {
            "sample_index": int(before["sample_index"]),
            "native_frame_index": int(before["native_frame_index"]),
            "raw": vector(before, "raw"),
            "pred": vector(before, "pred"),
            "gt": vector(before, "gt"),
            "error_3d": float(before["err_3d"]),
        },
        "after": {
            "sample_index": int(after["sample_index"]),
            "native_frame_index": int(after["native_frame_index"]),
            "raw": vector(after, "raw"),
            "pred": vector(after, "pred"),
            "gt": vector(after, "gt"),
            "error_3d": float(after["err_3d"]),
        },
    }
    output = args.output or args.predictions.with_name("closure_jump.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
