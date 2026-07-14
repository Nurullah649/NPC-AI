"""Frame 176'daki doğrulanmış referansı kamera homografisiyle takip et."""

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

from deneysel.reference_roi_v1.reference_tracker import (  # noqa: E402
    CameraCompensatedReferenceTracker,
)


DEFAULT_SESSION = SIM_ROOT / "_images" / "THYZ_2026_Online_Yarisma_Test_Oturumu"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--settings", type=Path, default=SIM_ROOT / "config" / "settings.yaml")
    parser.add_argument("--experiment-config", type=Path, default=HERE / "config.yaml")
    parser.add_argument("--session-dir", type=Path, default=DEFAULT_SESSION)
    parser.add_argument("--seed-frame", type=int, default=176)
    parser.add_argument("--end-frame", type=int, default=210)
    parser.add_argument("--seed-bbox", default="210.6755,64.2684,358.4221,152.6467")
    parser.add_argument("--output", type=Path, default=HERE / "results" / "reference_1_tracker.json")
    parser.add_argument("--visuals", type=Path, default=HERE / "visuals_tracker")
    parser.add_argument("--save-all", action="store_true")
    return parser.parse_args()


def load_config(settings: Path, experiment: Path) -> dict:
    with settings.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    with experiment.open("r", encoding="utf-8") as handle:
        experiment_config = yaml.safe_load(handle)
    config.update(experiment_config)
    return config


def draw(frame, frame_idx: int, result: dict, output: Path):
    canvas = frame.copy()
    polygon = result.get("polygon")
    if polygon:
        points = __import__("numpy").array(polygon, dtype="int32").reshape(-1, 1, 2)
        cv2.polylines(canvas, [points], True, (255, 0, 255), 3, cv2.LINE_AA)
    label = (
        f"{frame_idx} {result['reason']} {result['model']} "
        f"conf={result['transform_confidence']:.2f} "
        f"in={result['inliers']}/{result['matches']} "
        f"vis={result['visible_ratio']:.2f} color={result['color_correlation']:.2f}"
    )
    cv2.putText(canvas, label, (18, 36), cv2.FONT_HERSHEY_SIMPLEX,
                0.62, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(canvas, label, (18, 36), cv2.FONT_HERSHEY_SIMPLEX,
                0.62, (255, 255, 255), 2, cv2.LINE_AA)
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), canvas)


def main():
    args = parse_args()
    config = load_config(args.settings, args.experiment_config)
    bbox = tuple(float(value) for value in args.seed_bbox.split(","))
    tracker = CameraCompensatedReferenceTracker(config)
    seed_path = args.session_dir / f"frame_{args.seed_frame:06d}.webp"
    seed = cv2.imread(str(seed_path))
    if seed is None:
        raise FileNotFoundError(seed_path)
    tracked_seed_bbox = tracker.seed(seed, bbox)
    records = [{
        "frame": args.seed_frame,
        "active": True,
        "bbox": list(tracked_seed_bbox),
        "reason": "seeded_from_verified_roi",
    }]
    started = time.perf_counter()
    for frame_idx in range(args.seed_frame + 1, args.end_frame + 1):
        frame_path = args.session_dir / f"frame_{frame_idx:06d}.webp"
        frame = cv2.imread(str(frame_path))
        if frame is None:
            raise FileNotFoundError(frame_path)
        result = tracker.update(frame).to_dict()
        result["frame"] = frame_idx
        records.append(result)
        if args.save_all or frame_idx in {177, 180, 184, 188, 190, 193, 195, 198, 200, 202, 205, 208, 210}:
            draw(frame, frame_idx, result, args.visuals / f"frame_{frame_idx:06d}.jpg")

    elapsed = time.perf_counter() - started
    active = sum(bool(record.get("active")) for record in records)
    report = {
        "seed_frame": args.seed_frame,
        "end_frame": args.end_frame,
        "detected_seed_bbox": list(bbox),
        "expanded_seed_bbox": list(tracked_seed_bbox),
        "summary": {
            "frame_count": len(records),
            "active_frames": active,
            "inactive_frames": len(records) - active,
            "elapsed_seconds": elapsed,
            "fps": (len(records) - 1) / max(elapsed, 1e-6),
        },
        "frames": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report["summary"], indent=2))
    print(args.output)


if __name__ == "__main__":
    main()
