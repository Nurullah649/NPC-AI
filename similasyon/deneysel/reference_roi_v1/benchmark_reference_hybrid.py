"""ROI doğrulama + kamera-kompanzasyonlu tracker uçtan uca deneyi."""

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

from deneysel.reference_roi_v1.reference_tracker import (  # noqa: E402
    CameraCompensatedReferenceTracker,
)
from deneysel.reference_roi_v1.roi_matcher import RoiReferenceMatcher  # noqa: E402


DEFAULT_SESSION = SIM_ROOT / "_images" / "THYZ_2026_Online_Yarisma_Test_Oturumu"
DEFAULT_CACHE = (
    SIM_ROOT / "deneysel" / "detection_motion_v2" / "results"
    / "server_test_110_1110_detections.json"
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--settings", type=Path, default=SIM_ROOT / "config" / "settings.yaml")
    parser.add_argument("--experiment-config", type=Path, default=HERE / "config.yaml")
    parser.add_argument("--session-dir", type=Path, default=DEFAULT_SESSION)
    parser.add_argument("--detection-cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--positive-start", type=int, default=161)
    parser.add_argument("--positive-end", type=int, default=205)
    parser.add_argument("--negative-start", type=int, default=300)
    parser.add_argument("--negative-end", type=int, default=344)
    parser.add_argument("--output", type=Path, default=HERE / "results" / "reference_1_hybrid.json")
    parser.add_argument("--visuals", type=Path, default=HERE / "visuals_hybrid")
    parser.add_argument("--save-frames", default="161,170,176,177,184,190,195,200,202,203,205,300,320,344")
    return parser.parse_args()


def load_config(settings: Path, experiment: Path) -> dict:
    with settings.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    with experiment.open("r", encoding="utf-8") as handle:
        config.update(yaml.safe_load(handle))
    return config


def load_detections(path: Path) -> dict[int, list[dict]]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return {int(frame): detections for frame, detections in payload["frames"].items()}


def load_reference(session_dir: Path) -> dict:
    with (session_dir / "references.json").open("r", encoding="utf-8") as handle:
        references = json.load(handle)
    return next(item for item in references if int(item["order"]) == 1)


def serializable_bbox(bbox):
    return None if bbox is None else [float(value) for value in bbox]


def draw(frame, record: dict, output: Path):
    canvas = frame.copy()
    bbox = record.get("bbox")
    if bbox:
        x1, y1, x2, y2 = [int(round(value)) for value in bbox]
        color = (0, 255, 255) if record["source"] == "roi_seed" else (255, 0, 255)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 3)
    label = (
        f"{record['frame']} source={record['source']} reason={record['reason']} "
        f"active={int(record['active'])}"
    )
    cv2.putText(canvas, label, (18, 36), cv2.FONT_HERSHEY_SIMPLEX,
                0.68, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(canvas, label, (18, 36), cv2.FONT_HERSHEY_SIMPLEX,
                0.68, (255, 255, 255), 2, cv2.LINE_AA)
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), canvas)


def run_range(start: int, end: int, label: str, matcher, reference, ref_path,
              tracker, session_dir, detections, save_frames, visuals) -> tuple[list[dict], float]:
    records = []
    started = time.perf_counter()
    for frame_idx in range(start, end + 1):
        frame_path = session_dir / f"frame_{frame_idx:06d}.webp"
        frame = cv2.imread(str(frame_path))
        if frame is None:
            raise FileNotFoundError(frame_path)

        bbox = None
        source = "none"
        reason = "no_verified_match"
        details = {}
        if tracker.active:
            tracked = tracker.update(frame).to_dict()
            details["tracker"] = tracked
            if tracked["active"]:
                bbox = tracked["bbox"]
                source = "tracker"
                reason = tracked["reason"]

        # Tracker yoksa veya güvenli çıktı veremediyse yalnızca tam-kare ve
        # YOLO-adaylı doğrulamayı deneriz. Serbest renk penceresi yanlış pozitif
        # üretebildiği için bu güvenli hibrite bilerek dahil edilmez.
        if bbox is None and not tracker.active:
            full_bbox = matcher.match_full(reference["url"], str(ref_path), frame)
            roi_result = matcher.match_candidates(
                reference["url"], str(ref_path), frame, detections.get(frame_idx, [])
            )
            seed_bbox = full_bbox if full_bbox is not None else roi_result["bbox"]
            details["full_bbox"] = serializable_bbox(full_bbox)
            details["roi_bbox"] = serializable_bbox(roi_result["bbox"])
            details["roi_accepted_count"] = len(roi_result["accepted"])
            if seed_bbox is not None:
                details["detected_seed_bbox"] = serializable_bbox(seed_bbox)
                bbox = tracker.seed(frame, seed_bbox)
                details["expanded_seed_bbox"] = serializable_bbox(bbox)
                source = "full_seed" if full_bbox is not None else "roi_seed"
                reason = "verified_seed"

        record = {
            "frame": frame_idx,
            "active": bbox is not None,
            "bbox": serializable_bbox(bbox),
            "source": source,
            "reason": reason,
            **details,
        }
        records.append(record)
        if frame_idx in save_frames:
            draw(frame, record, visuals / f"{label}_{frame_idx:06d}.jpg")
    return records, time.perf_counter() - started


def summarize(records: list[dict], elapsed: float) -> dict:
    sources = Counter(item["source"] for item in records)
    return {
        "frame_count": len(records),
        "output_frames": sum(item["bbox"] is not None for item in records),
        "first_output_frame": next((item["frame"] for item in records if item["bbox"]), None),
        "last_output_frame": next((item["frame"] for item in reversed(records) if item["bbox"]), None),
        "sources": dict(sources),
        "elapsed_seconds": elapsed,
        "fps": len(records) / max(elapsed, 1e-6),
    }


def main():
    args = parse_args()
    config = load_config(args.settings, args.experiment_config)
    detections = load_detections(args.detection_cache)
    reference = load_reference(args.session_dir)
    ref_path = args.session_dir / "references" / Path(reference["image_url"]).name
    matcher = RoiReferenceMatcher(config)
    matcher.prepare_reference(reference["url"], str(ref_path))
    save_frames = {int(value) for value in args.save_frames.split(",") if value.strip()}

    positive, positive_elapsed = run_range(
        args.positive_start, args.positive_end, "positive", matcher, reference,
        ref_path, CameraCompensatedReferenceTracker(config), args.session_dir,
        detections, save_frames, args.visuals,
    )
    negative, negative_elapsed = run_range(
        args.negative_start, args.negative_end, "negative", matcher, reference,
        ref_path, CameraCompensatedReferenceTracker(config), args.session_dir,
        detections, save_frames, args.visuals,
    )
    report = {
        "positive": {
            "summary": summarize(positive, positive_elapsed),
            "frames": positive,
        },
        "negative": {
            "summary": summarize(negative, negative_elapsed),
            "frames": negative,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps({key: value["summary"] for key, value in report.items()}, indent=2))
    print(args.output)


if __name__ == "__main__":
    main()
