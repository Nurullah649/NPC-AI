"""Ana simulasyona baglanan stateful referans pipeline'ini offline test et."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import cv2
import yaml


HERE = Path(__file__).resolve().parent
SIM_ROOT = HERE.parent.parent
sys.path.insert(0, str(SIM_ROOT))

from src.models.reference_pipeline import ReferenceDetectionPipeline  # noqa: E402


SESSION = SIM_ROOT / "_images/THYZ_2026_Online_Yarisma_Test_Oturumu"
REF1_DETECTIONS = (
    SIM_ROOT / "deneysel/detection_motion_v2/results"
    / "server_test_110_1110_detections.json"
)
REF5_DETECTIONS = HERE / "results/reference_5_1955_2070_detections.json"
OUTPUT = HERE / "results/reference_production_pipeline.json"
VISUALS = HERE / "visuals_production_pipeline"
SAVE_FRAMES = {
    161, 176, 177, 190, 203, 205, 300, 320, 344,
    521, 540, 600, 652, 1607, 1640, 1700, 1744,
    1898, 1910, 1930, 1954,
    1955, 1960, 1980, 2000, 2020, 2066, 2067, 2068, 2069, 2070,
}


def _load_detection_cache(path: Path) -> dict[int, list[dict]]:
    with path.open("r", encoding="utf-8") as handle:
        frames = json.load(handle)["frames"]
    return {int(frame): items for frame, items in frames.items()}


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", choices=("all", "core", "ref5"), default="all")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    return parser.parse_args()


def _draw(frame, frame_idx: int, result: dict, output: Path):
    canvas = frame.copy()
    bbox = result.get("bbox")
    if bbox:
        x1, y1, x2, y2 = [int(round(value)) for value in bbox]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (255, 0, 255), 3)
    label = f"{frame_idx} {result['source']} {result['reason']}"
    cv2.putText(canvas, label, (18, 36), cv2.FONT_HERSHEY_SIMPLEX,
                0.75, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(canvas, label, (18, 36), cv2.FONT_HERSHEY_SIMPLEX,
                0.75, (255, 255, 255), 2, cv2.LINE_AA)
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), canvas)


def _run_range(pipeline, reference, start, end, detections, label):
    ref_path = SESSION / "references" / Path(reference["image_url"]).name
    records = []
    started = time.perf_counter()
    for frame_idx in range(start, end + 1):
        frame = cv2.imread(str(SESSION / f"frame_{frame_idx:06d}.webp"))
        result = pipeline.match_reference(
            reference,
            str(ref_path),
            frame,
            detections.get(frame_idx, []),
            frame_idx,
        ).to_dict()
        result["frame"] = frame_idx
        records.append(result)
        if frame_idx in SAVE_FRAMES:
            _draw(frame, frame_idx, result, VISUALS / f"{label}_{frame_idx:06d}.jpg")
    elapsed = time.perf_counter() - started
    return {
        "summary": {
            "frame_count": len(records),
            "output_frames": sum(item["bbox"] is not None for item in records),
            "sources": dict(Counter(item["source"] for item in records)),
            "elapsed_seconds": elapsed,
            "fps": len(records) / max(elapsed, 1e-6),
        },
        "frames": records,
    }


def main():
    args = _parse_args()
    with (SIM_ROOT / "config/settings.yaml").open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    with (SESSION / "references.json").open("r", encoding="utf-8") as handle:
        references = json.load(handle)
    by_order = {int(item["order"]): item for item in references}
    paths = {
        item["url"]: str(SESSION / "references" / Path(item["image_url"]).name)
        for item in references
    }
    ref1_detections = _load_detection_cache(REF1_DETECTIONS)
    ref5_detections = _load_detection_cache(REF5_DETECTIONS)

    pipeline = ReferenceDetectionPipeline(config)
    pipeline.register_references(references, paths)
    report = {}
    if args.scope in ("all", "core"):
        report["ref1"] = _run_range(
            pipeline, by_order[1], 161, 205, ref1_detections, "ref1"
        )
        report["negative"] = _run_range(
            pipeline, by_order[1], 300, 344, ref1_detections, "negative"
        )
    if args.scope == "all":
        report["ref2"] = _run_range(
            pipeline, by_order[2], 521, 652, ref1_detections, "ref2"
        )
        report["ref3"] = _run_range(
            pipeline, by_order[3], 1607, 1744, {}, "ref3"
        )
        report["ref4"] = _run_range(
            pipeline, by_order[4], 1898, 1954, {}, "ref4"
        )
    if args.scope in ("all", "core", "ref5"):
        report["ref5"] = _run_range(
            pipeline, by_order[5], 1955, 2070, ref5_detections, "ref5"
        )
    report["feature_cache_count"] = len(pipeline.feature_cache)
    report["scene_keyframes"] = {
        str(order): len(memory.keyframes)
        for order, memory in pipeline.scene_memories.items()
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    summary = {
        "feature_cache_count": report["feature_cache_count"],
        "scene_keyframes": report["scene_keyframes"],
    }
    summary.update({
        name: value["summary"]
        for name, value in report.items()
        if isinstance(value, dict) and "summary" in value
    })
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
