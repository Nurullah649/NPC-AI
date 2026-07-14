"""Seçili karelerde termal/RGB büyük-karo referans seed taraması."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import yaml


HERE = Path(__file__).resolve().parent
SIM_ROOT = HERE.parent.parent
sys.path.insert(0, str(SIM_ROOT))

from deneysel.reference_roi_v1.roi_matcher import RoiReferenceMatcher  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--settings", type=Path, default=SIM_ROOT / "config/settings.yaml")
    parser.add_argument("--experiment-config", type=Path, default=HERE / "config.yaml")
    parser.add_argument(
        "--session-dir", type=Path,
        default=SIM_ROOT / "_images/THYZ_2026_Online_Yarisma_Test_Oturumu",
    )
    parser.add_argument("--reference-order", type=int, default=5)
    parser.add_argument("--frames", default="1955,1960,1970,1980,1990,2000,2020,2040,2050,2060,2066,2070")
    parser.add_argument("--output", type=Path, default=HERE / "results/reference_5_crossmodal_probe.json")
    parser.add_argument("--visuals", type=Path, default=HERE / "visuals_ref5_crossmodal")
    return parser.parse_args()


def draw(frame, frame_idx, result, path):
    canvas = frame.copy()
    for candidate in result["accepted"]:
        bbox = candidate["output_bbox"]
        x1, y1, x2, y2 = [int(round(v)) for v in bbox]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (255, 0, 255), 3)
        cv2.putText(
            canvas,
            f"I={candidate['inliers']}/{candidate['matches']} score={candidate['score']:.1f}",
            (x1, max(22, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
            (255, 0, 255), 2, cv2.LINE_AA,
        )
    cv2.putText(canvas, f"frame {frame_idx} accepted={len(result['accepted'])}",
                (18, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), canvas)


def main():
    args = parse_args()
    with args.settings.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    with args.experiment_config.open("r", encoding="utf-8") as handle:
        config.update(yaml.safe_load(handle))
    with (args.session_dir / "references.json").open("r", encoding="utf-8") as handle:
        references = json.load(handle)
    reference = next(r for r in references if int(r["order"]) == args.reference_order)
    ref_path = args.session_dir / "references" / Path(reference["image_url"]).name
    matcher = RoiReferenceMatcher(config)
    frames = [int(value) for value in args.frames.split(",") if value.strip()]
    records = []
    started = time.perf_counter()
    for frame_idx in frames:
        frame = cv2.imread(str(args.session_dir / f"frame_{frame_idx:06d}.webp"))
        result = matcher.match_crossmodal_tiles(reference["url"], str(ref_path), frame)
        records.append({
            "frame": frame_idx,
            "bbox": None if result["bbox"] is None else list(result["bbox"]),
            "accepted_count": len(result["accepted"]),
            "accepted": result["accepted"],
            "candidates": result["candidates"],
            "tile_count": result["tile_count"],
        })
        draw(frame, frame_idx, result, args.visuals / f"frame_{frame_idx:06d}.jpg")
        print(frame_idx, "accepted", len(result["accepted"]), "bbox", result["bbox"])
    elapsed = time.perf_counter() - started
    report = {
        "reference": reference,
        "summary": {
            "frame_count": len(records),
            "matched_frames": sum(item["bbox"] is not None for item in records),
            "elapsed_seconds": elapsed,
            "fps": len(records) / max(elapsed, 1e-6),
        },
        "frames": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
