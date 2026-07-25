#!/usr/bin/env python3
"""Create a compact, machine-readable and Markdown SLAM comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load(spec: str) -> tuple[str, Path, dict]:
    if "=" not in spec:
        raise ValueError(f"Expected LABEL=metrics.json, got {spec!r}")
    label, raw_path = spec.split("=", 1)
    path = Path(raw_path).resolve()
    with path.open("r", encoding="utf-8") as handle:
        metrics = json.load(handle)
    for key in ("E_3d", "RMSE_3d", "p95_3d", "max_3d"):
        if key not in metrics:
            raise ValueError(f"{path} has no {key}")
    return label, path, metrics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mast3r", required=True, type=Path)
    parser.add_argument("--baseline", required=True, action="append", help="LABEL=metrics.json")
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    with args.mast3r.open("r", encoding="utf-8") as handle:
        mast3r = json.load(handle)
    entries = [("MASt3R-SLAM", args.mast3r.resolve(), mast3r)]
    entries.extend(_load(spec) for spec in args.baseline)
    expected_protocol = (
        mast3r.get("sampled_frame_count"),
        mast3r.get("calibration_sampled_frames"),
        mast3r.get("evaluation_sampled_frames"),
    )
    for label, path, values in entries[1:]:
        protocol = (
            values.get("sampled_frame_count"),
            values.get("calibration_sampled_frames"),
            values.get("evaluation_sampled_frames"),
        )
        if protocol != expected_protocol:
            raise ValueError(
                f"Refusing an unequal comparison for {label}: MASt3R protocol "
                f"{expected_protocol}, {path} protocol {protocol}"
            )
    mast_e = float(mast3r["E_3d"])
    rows = []
    for label, path, values in entries:
        baseline_e = float(values["E_3d"])
        rows.append(
            {
                "label": label,
                "metrics_path": str(path),
                "E_3d": baseline_e,
                "RMSE_3d": float(values["RMSE_3d"]),
                "median_3d": float(values["median_3d"]),
                "p95_3d": float(values["p95_3d"]),
                "max_3d": float(values["max_3d"]),
                "mast3r_E_change_percent": (
                    0.0 if label == "MASt3R-SLAM" else 100.0 * (mast_e - baseline_e) / baseline_e
                ),
            }
        )
    ranking = sorted(rows, key=lambda row: row["E_3d"])
    report = {
        "metric_scope": "health=0 only; first 450 sampled frames are calibration",
        "lower_is_better": True,
        "rows": rows,
        "ranking_by_E_3d": [row["label"] for row in ranking],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "comparison.json"
    md_path = args.output_dir / "comparison.md"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    lines = [
        "# 2026 competition replay comparison",
        "",
        "All errors are measured only on health=0. Lower is better.",
        "",
        "| Method | E (m) | RMSE (m) | Median (m) | P95 (m) | Max (m) | MASt3R E change |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['label']} | {row['E_3d']:.3f} | {row['RMSE_3d']:.3f} | "
            f"{row['median_3d']:.3f} | {row['p95_3d']:.3f} | {row['max_3d']:.3f} | "
            f"{row['mast3r_E_change_percent']:+.1f}% |"
        )
    lines.extend(["", "Ranking by E: " + " < ".join(report["ranking_by_E_3d"]), ""])
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"Outputs: {json_path} / {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
