"""Reference-1 tam-kare ve ROI fallback karşılaştırması."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import Counter
from pathlib import Path

import cv2
import yaml


HERE = Path(__file__).resolve().parent
SIM_ROOT = HERE.parent.parent
REPO_ROOT = SIM_ROOT.parent
sys.path.insert(0, str(SIM_ROOT))

from deneysel.reference_roi_v1.roi_matcher import RoiReferenceMatcher  # noqa: E402


LOGGER = logging.getLogger("benchmark_reference_roi")
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
    parser.add_argument("--reference-order", type=int, default=1)
    parser.add_argument("--positive-start", type=int, default=161)
    parser.add_argument("--positive-end", type=int, default=205)
    parser.add_argument("--negative-start", type=int, default=300)
    parser.add_argument("--negative-end", type=int, default=344)
    parser.add_argument("--output", type=Path, default=HERE / "results" / "reference_1_benchmark.json")
    parser.add_argument("--visuals", type=Path, default=HERE / "visuals")
    parser.add_argument("--save-frames", default="161,170,176,184,195,205,300,320,344")
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


def load_reference(session_dir: Path, order: int) -> dict:
    with (session_dir / "references.json").open("r", encoding="utf-8") as handle:
        references = json.load(handle)
    return next(reference for reference in references if int(reference["order"]) == order)


def serializable_bbox(bbox):
    return None if bbox is None else [float(value) for value in bbox]


def draw_visual(frame, frame_idx: int, full_bbox, roi_result: dict, discovery_result: dict,
                output_path: Path):
    canvas = frame.copy()
    for candidate in roi_result["candidates"]:
        x1, y1, x2, y2 = [int(round(value)) for value in candidate["candidate_bbox"]]
        color = (0, 200, 0) if candidate["accepted"] else (150, 150, 150)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        label = (
            f"ROI#{candidate['candidate_index']} {candidate['reject_reason']} "
            f"I:{candidate['inliers']}/{candidate['matches']} "
            f"C:{candidate['projection_coverage']:.2f} IoU:{candidate['candidate_iou']:.2f} "
            f"H:{candidate['color_hist_correlation']:.2f}"
        )
        cv2.putText(canvas, label, (x1, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.42, color, 1, cv2.LINE_AA)
    if full_bbox is not None:
        x1, y1, x2, y2 = [int(round(value)) for value in full_bbox]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (255, 120, 0), 3)
        cv2.putText(canvas, "FULL", (x1, max(22, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (255, 120, 0), 2, cv2.LINE_AA)
    if roi_result["bbox"] is not None:
        x1, y1, x2, y2 = [int(round(value)) for value in roi_result["bbox"]]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 255), 3)
        cv2.putText(canvas, "ROI BEST", (x1, min(canvas.shape[0] - 8, y2 + 22)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)
    if discovery_result["bbox"] is not None:
        x1, y1, x2, y2 = [int(round(value)) for value in discovery_result["bbox"]]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (255, 0, 255), 3)
        cv2.putText(canvas, "DISCOVERY", (x1, min(canvas.shape[0] - 8, y2 + 22)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, f"frame {frame_idx}", (18, 35), cv2.FONT_HERSHEY_SIMPLEX,
                0.9, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(canvas, f"frame {frame_idx}", (18, 35), cv2.FONT_HERSHEY_SIMPLEX,
                0.9, (255, 255, 255), 2, cv2.LINE_AA)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), canvas)


def summarize(frames: list[dict]) -> dict:
    full = sum(item["full_bbox"] is not None for item in frames)
    roi = sum(item["roi_bbox"] is not None for item in frames)
    discovery = sum(item["discovery_bbox"] is not None for item in frames)
    hybrid = sum(item["hybrid_bbox"] is not None for item in frames)
    ambiguous = sum(item["accepted_count"] > 1 for item in frames)
    evaluated = sum(len(item["candidates"]) for item in frames)
    accepted_candidates = sum(item["accepted_count"] for item in frames)
    reasons = Counter(
        candidate["reject_reason"]
        for item in frames
        for candidate in item["candidates"]
    )
    return {
        "frame_count": len(frames),
        "full_match_frames": full,
        "full_match_rate": full / max(1, len(frames)),
        "roi_match_frames": roi,
        "roi_match_rate": roi / max(1, len(frames)),
        "discovery_match_frames": discovery,
        "discovery_match_rate": discovery / max(1, len(frames)),
        "hybrid_match_frames": hybrid,
        "hybrid_match_rate": hybrid / max(1, len(frames)),
        "ambiguous_frames": ambiguous,
        "evaluated_candidates": evaluated,
        "accepted_candidates": accepted_candidates,
        "reasons": dict(reasons),
    }


def run_range(
    label: str,
    start: int,
    end: int,
    matcher: RoiReferenceMatcher,
    reference: dict,
    ref_path: Path,
    session_dir: Path,
    detections: dict[int, list[dict]],
    save_frames: set[int],
    visuals_dir: Path,
) -> list[dict]:
    records = []
    started = time.perf_counter()
    for ordinal, frame_idx in enumerate(range(start, end + 1), start=1):
        frame_path = session_dir / f"frame_{frame_idx:06d}.webp"
        frame = cv2.imread(str(frame_path))
        if frame is None:
            raise FileNotFoundError(frame_path)
        frame_detections = detections.get(frame_idx, [])
        full_bbox = matcher.match_full(reference["url"], str(ref_path), frame)
        roi_result = matcher.match_candidates(
            reference["url"], str(ref_path), frame, frame_detections
        )
        discovery_result = matcher.match_discovery_windows(
            reference["url"], str(ref_path), frame
        )
        hybrid_bbox = full_bbox if full_bbox is not None else (
            roi_result["bbox"] if roi_result["bbox"] is not None else discovery_result["bbox"]
        )
        record = {
            "frame": frame_idx,
            "full_bbox": serializable_bbox(full_bbox),
            "roi_bbox": serializable_bbox(roi_result["bbox"]),
            "discovery_bbox": serializable_bbox(discovery_result["bbox"]),
            "hybrid_bbox": serializable_bbox(hybrid_bbox),
            "hybrid_source": "full" if full_bbox is not None else (
                "roi" if roi_result["bbox"] is not None else (
                    "discovery" if discovery_result["bbox"] is not None else "none"
                )
            ),
            "accepted_count": len(roi_result["accepted"]),
            "accepted": roi_result["accepted"],
            "candidates": roi_result["candidates"],
            "discovery_accepted": discovery_result["accepted"],
            "discovery_candidates": discovery_result["candidates"],
            "discovery_proposal_count": discovery_result["proposal_count"],
        }
        records.append(record)
        if frame_idx in save_frames:
            draw_visual(
                frame, frame_idx, full_bbox, roi_result, discovery_result,
                visuals_dir / f"{label}_{frame_idx:06d}.jpg",
            )
        if ordinal % 10 == 0 or frame_idx == end:
            elapsed = time.perf_counter() - started
            LOGGER.info("%s %d/%d (%.2f frame/s)", label, ordinal, end - start + 1,
                        ordinal / max(elapsed, 1e-6))
    return records


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    config = load_config(args.settings, args.experiment_config)
    detections = load_detections(args.detection_cache)
    reference = load_reference(args.session_dir, args.reference_order)
    ref_path = args.session_dir / "references" / Path(reference["image_url"]).name
    matcher = RoiReferenceMatcher(config)
    matcher.prepare_reference(reference["url"], str(ref_path))
    save_frames = {int(value) for value in args.save_frames.split(",") if value.strip()}

    positive = run_range(
        "positive", args.positive_start, args.positive_end, matcher, reference,
        ref_path, args.session_dir, detections, save_frames, args.visuals,
    )
    negative = run_range(
        "negative", args.negative_start, args.negative_end, matcher, reference,
        ref_path, args.session_dir, detections, save_frames, args.visuals,
    )
    report = {
        "reference": reference,
        "positive": {"summary": summarize(positive), "frames": positive},
        "negative": {"summary": summarize(negative), "frames": negative},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    LOGGER.info("POS %s", report["positive"]["summary"])
    LOGGER.info("NEG %s", report["negative"]["summary"])
    LOGGER.info("Rapor: %s", args.output)


if __name__ == "__main__":
    main()
