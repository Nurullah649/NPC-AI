#!/usr/bin/env python3
"""Compare completed full-length DPVO runs across input resolutions."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


VRAM_PATTERN = re.compile(r"VRAM=([0-9.]+)/([0-9.]+)GiB")


def load_run(path: Path) -> dict:
    raw = json.loads((path / "raw_summary.json").read_text())
    metrics_path = path / "alignment_extended" / "corrected_metrics.json"
    if not metrics_path.exists():
        metrics_path = path / "alignment" / "corrected_metrics.json"
    metrics = json.loads(metrics_path.read_text())
    by_name = {row["name"]: row for row in metrics}
    production = by_name["abs_linear_intercept_True_anchor_at_calib_end"]
    matches = [tuple(map(float, match)) for match in VRAM_PATTERN.findall((path / "run.log").read_text())]
    elapsed = float(raw["elapsed_seconds"])
    frames = int(raw["frames"])
    return {
        "name": path.name,
        "path": str(path),
        "frames": frames,
        "resize_height": int(raw["resize_height"]),
        "minutes": elapsed / 60.0,
        "fps": frames / elapsed,
        "peak_vram_allocated_gib": max((item[0] for item in matches), default=0.0),
        "peak_vram_reserved_gib": max((item[1] for item in matches), default=0.0),
        "best_method": metrics[0]["name"],
        "E": metrics[0]["eval"]["E"],
        "RMSE": metrics[0]["eval"]["RMSE"],
        "median": metrics[0]["eval"]["median"],
        "p95": metrics[0]["eval"]["p95"],
        "max": metrics[0]["eval"]["max"],
        "production_method": production["name"],
        "production_E": production["eval"]["E"],
        "production_RMSE": production["eval"]["RMSE"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    rows = [load_run(path) for path in args.runs]
    rows.sort(key=lambda row: row["E"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "FINAL_KARSILASTIRMA.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False) + "\n"
    )
    lines = [
        "# Tam 2250-frame çözünürlük karşılaştırması",
        "",
        "E/RMSE ilk 450 kareyle kalibre edilen hizalamanın kalan 1800 kare sonucudur.",
        "",
        "| Giriş | Süre | FPS | Aktif VRAM tepe | Reserve VRAM tepe | En iyi E | Üretim E | Üretim RMSE | Yöntem |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['resize_height']}p | {row['minutes']:.2f} dk | {row['fps']:.2f} | "
            f"{row['peak_vram_allocated_gib']:.2f} GiB | {row['peak_vram_reserved_gib']:.2f} GiB | "
            f"{row['E']:.4f} | {row['production_E']:.4f} | {row['production_RMSE']:.4f} | "
            f"`{row['best_method']}` |"
        )
    (args.output_dir / "FINAL_KARSILASTIRMA.md").write_text("\n".join(lines) + "\n")
    print(args.output_dir / "FINAL_KARSILASTIRMA.md")


if __name__ == "__main__":
    main()
