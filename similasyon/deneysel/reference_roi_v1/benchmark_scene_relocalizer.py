"""Ref-1 doğrulanmış sahne hafızasıyla 1955-2070 biçerdöver testi."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import yaml


HERE = Path(__file__).resolve().parent
SIM_ROOT = HERE.parent.parent
sys.path.insert(0, str(SIM_ROOT))

from deneysel.reference_roi_v1.reference_scene_relocalizer import (  # noqa: E402
    ReferenceSceneRelocalizer,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--settings", type=Path, default=SIM_ROOT / "config/settings.yaml")
    parser.add_argument("--experiment-config", type=Path, default=HERE / "config.yaml")
    parser.add_argument(
        "--session-dir", type=Path,
        default=SIM_ROOT / "_images/THYZ_2026_Online_Yarisma_Test_Oturumu",
    )
    parser.add_argument("--tracker-report", type=Path, default=HERE / "results/reference_1_tracker.json")
    parser.add_argument("--start", type=int, default=1955)
    parser.add_argument("--end", type=int, default=2070)
    parser.add_argument("--keyframe-step", type=int, default=3)
    parser.add_argument("--output", type=Path, default=HERE / "results/reference_5_scene_relocalizer.json")
    parser.add_argument("--visuals", type=Path, default=HERE / "visuals_ref5_scene")
    parser.add_argument(
        "--save-frames",
        default=(
            "1955,1960,1970,1980,1990,2000,2020,2040,2050,2060,"
            "2066,2067,2068,2069,2070"
        ),
    )
    return parser.parse_args()


def draw(frame, frame_idx, result, path):
    canvas = frame.copy()
    if result["bbox"]:
        x1, y1, x2, y2 = [int(round(value)) for value in result["bbox"]]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (255, 0, 255), 3)
    label = (
        f"{frame_idx} accepted={int(result['accepted'])} key={result['keyframe']} "
        f"in={result['inliers']}/{result['matches']} conf={result['confidence']:.2f} "
        f"color={result['color_correlation']:.2f}"
    )
    cv2.putText(canvas, label, (18, 36), cv2.FONT_HERSHEY_SIMPLEX,
                0.68, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(canvas, label, (18, 36), cv2.FONT_HERSHEY_SIMPLEX,
                0.68, (255, 255, 255), 2, cv2.LINE_AA)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), canvas)


def main():
    args = parse_args()
    with args.settings.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    with args.experiment_config.open("r", encoding="utf-8") as handle:
        config.update(yaml.safe_load(handle))
    with args.tracker_report.open("r", encoding="utf-8") as handle:
        tracked = json.load(handle)["frames"]

    relocalizer = ReferenceSceneRelocalizer(config)
    candidates = [item for item in tracked if item.get("active") and item.get("polygon")]
    for item in candidates[::args.keyframe_step]:
        image = cv2.imread(str(args.session_dir / f"frame_{item['frame']:06d}.webp"))
        relocalizer.add_keyframe(item["frame"], image, item["polygon"])

    save_frames = {int(value) for value in args.save_frames.split(",") if value.strip()}
    records = []
    started = time.perf_counter()
    for frame_idx in range(args.start, args.end + 1):
        frame = cv2.imread(str(args.session_dir / f"frame_{frame_idx:06d}.webp"))
        result = relocalizer.relocalize(frame).to_dict()
        result["frame"] = frame_idx
        records.append(result)
        if frame_idx in save_frames:
            draw(frame, frame_idx, result, args.visuals / f"frame_{frame_idx:06d}.jpg")
        if frame_idx % 10 == 0:
            print(frame_idx, result["accepted"], result["keyframe"], result["inliers"])
    elapsed = time.perf_counter() - started
    report = {
        "summary": {
            "frame_count": len(records),
            "matched_frames": sum(item["accepted"] for item in records),
            "keyframe_count": len(relocalizer.keyframes),
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
