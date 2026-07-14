"""Gercek sunucu karelerinde uretim tracker'i ile Motion V2'yi karsilastir.

Kullanicinin verdigi zayif etiketler:
  - 110..1049: hareketli arac yok (false-positive olcumu)
  - 1050..1110: hareketli arac var (frame-level recall olcumu)

YOLO ciktilari JSON'a onbelleklenir. Boylece tracker/config deneyleri modeli tekrar
calistirmadan ayni detection dizisi uzerinde tekrarlanabilir.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import yaml


HERE = Path(__file__).resolve().parent
SIM_ROOT = HERE.parent.parent
REPO_ROOT = SIM_ROOT.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(SIM_ROOT))

from motion.camera_motion import CameraMotionEstimator  # noqa: E402
from motion.vehicle_tracker import VehicleTrackerV2  # noqa: E402
from person.detector_adapter import DetectorAdapter  # noqa: E402
from src.models.motion_classifier import MotionClassifier as ProductionMotionV2  # noqa: E402
from src.models.motion_classifier_legacy import MotionClassifier as LegacyMotionClassifier  # noqa: E402


LOGGER = logging.getLogger("benchmark_server_sequence")
DEFAULT_FRAMES = SIM_ROOT / "_images" / "THYZ_2026_Online_Yarisma_Test_Oturumu"
DEFAULT_CACHE = HERE / "results" / "server_test_110_1110_detections.json"
DEFAULT_OUTPUT = HERE / "results" / "server_test_110_1110_motion.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=HERE / "config_experiment.yaml")
    parser.add_argument("--production-config", type=Path, default=SIM_ROOT / "config" / "settings.yaml")
    parser.add_argument("--frames-dir", type=Path, default=DEFAULT_FRAMES)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--start", type=int, default=110)
    parser.add_argument("--end", type=int, default=1110)
    parser.add_argument("--negative-start", type=int, default=110)
    parser.add_argument("--negative-end", type=int, default=1049)
    parser.add_argument("--positive-start", type=int, default=1050)
    parser.add_argument("--positive-end", type=int, default=1110)
    parser.add_argument("--force-detect", action="store_true")
    parser.add_argument("--skip-production", action="store_true")
    parser.add_argument("--include-production-v2", action="store_true")
    parser.add_argument("--log-every", type=int, default=25)
    return parser.parse_args()


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def frame_path(frames_dir: Path, frame_idx: int) -> Path:
    return frames_dir / f"frame_{frame_idx:06d}.webp"


def serializable_detection(det: dict) -> dict:
    return {
        "cls": int(det["cls"]),
        "cls_name": det.get("cls_name"),
        "conf": float(det.get("conf", 0.0)),
        "bbox": [float(value) for value in det["bbox"]],
    }


def load_detection_cache(path: Path) -> dict[int, list[dict]]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return {int(key): value for key, value in payload.get("frames", {}).items()}


def write_detection_cache(path: Path, frames: dict[int, list[dict]], metadata: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata": metadata,
        "frames": {str(key): value for key, value in sorted(frames.items())},
    }
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def ensure_detection_cache(args: argparse.Namespace, config: dict) -> dict[int, list[dict]]:
    cached = {} if args.force_detect else load_detection_cache(args.cache)
    required = list(range(args.start, args.end + 1))
    missing = [idx for idx in required if idx not in cached]
    if not missing:
        LOGGER.info("Detection cache hazir: %s (%d kare)", args.cache, len(required))
        return cached

    LOGGER.info("Detection cache eksik: %d/%d kare YOLO ile islenecek", len(missing), len(required))
    detector = DetectorAdapter(config)
    started = time.perf_counter()
    for count, frame_idx in enumerate(missing, start=1):
        image_path = frame_path(args.frames_dir, frame_idx)
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"Frame okunamadi: {image_path}")
        cached[frame_idx] = [
            serializable_detection(det) for det in detector.detect_full_frame(image)
        ]
        if count % args.log_every == 0 or count == len(missing):
            elapsed = time.perf_counter() - started
            LOGGER.info("YOLO: %d/%d (%.2f kare/sn)", count, len(missing), count / max(elapsed, 1e-6))
            write_detection_cache(
                args.cache,
                cached,
                {
                    "frames_dir": str(args.frames_dir.resolve()),
                    "config": str(args.config.resolve()),
                    "range": [args.start, args.end],
                },
            )
    return cached


def camera_dict(transform) -> dict:
    return {
        "model": transform.model_type.name,
        "confidence": float(transform.confidence),
        "matches": int(transform.match_count),
        "inliers": int(transform.inlier_count),
        "inlier_ratio": float(transform.inlier_ratio),
        "reprojection_error": float(transform.reprojection_error),
    }


def tracker_detection(det: dict) -> dict:
    return {
        "bbox": [float(value) for value in det["bbox"]],
        "conf": float(det.get("conf", 0.0)),
        "track_id": int(det.get("_track_id", -1)),
        "score": float(det.get("_motion_score", 0.0)),
        "moving": str(det.get("moving_status", "0")),
    }


def run_trackers(
    args: argparse.Namespace,
    experiment_config: dict,
    production_config: dict,
    detections_by_frame: dict[int, list[dict]],
) -> dict[str, list[dict]]:
    # V2'de ekranda/kararda uretim ile ayni piksel S semantigini koru.
    raw_v2_config = copy.deepcopy(experiment_config)
    raw_v2_config.setdefault("motion", {}).update({
        "normalize_by_bbox_diagonal": False,
        "moving_start_threshold": float(production_config["motion"]["moving_start_threshold"]),
        "moving_stop_threshold": float(production_config["motion"]["moving_stop_threshold"]),
    })

    v2_tracker = VehicleTrackerV2(raw_v2_config)
    camera_estimator = CameraMotionEstimator(raw_v2_config)
    production_tracker = None if args.skip_production else LegacyMotionClassifier(production_config)
    production_v2 = ProductionMotionV2(production_config) if args.include_production_v2 else None

    output: dict[str, list[dict]] = {"v2_raw": []}
    if production_tracker is not None:
        output["production"] = []
    if production_v2 is not None:
        output["production_v2"] = []

    prev_gray = None
    prev_detections = None
    started = time.perf_counter()
    for ordinal, frame_idx in enumerate(range(args.start, args.end + 1), start=1):
        image = cv2.imread(str(frame_path(args.frames_dir, frame_idx)))
        if image is None:
            raise FileNotFoundError(f"Frame okunamadi: {frame_path(args.frames_dir, frame_idx)}")
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        source_detections = detections_by_frame[frame_idx]

        transform = camera_estimator.estimate(
            prev_gray,
            gray,
            source_detections,
            prev_detections,
        )
        v2_output = v2_tracker.update(copy.deepcopy(source_detections), gray, transform)
        output["v2_raw"].append({
            "frame": frame_idx,
            "camera": camera_dict(transform),
            "vehicles": [tracker_detection(det) for det in v2_output if det.get("cls") == 0],
        })

        if production_tracker is not None:
            production_output = production_tracker.update(copy.deepcopy(source_detections), gray)
            output["production"].append({
                "frame": frame_idx,
                "vehicles": [tracker_detection(det) for det in production_output if det.get("cls") == 0],
            })

        if production_v2 is not None:
            candidate_output = production_v2.update(copy.deepcopy(source_detections), gray)
            output["production_v2"].append({
                "frame": frame_idx,
                "vehicles": [tracker_detection(det) for det in candidate_output if det.get("cls") == 0],
            })

        prev_gray = gray
        prev_detections = source_detections
        if ordinal % args.log_every == 0 or frame_idx == args.end:
            elapsed = time.perf_counter() - started
            LOGGER.info("Tracker: %d/%d (%.2f kare/sn)", ordinal, args.end - args.start + 1,
                        ordinal / max(elapsed, 1e-6))
    return output


def longest_consecutive(values: list[int]) -> int:
    if not values:
        return 0
    longest = current = 1
    for previous, current_value in zip(values, values[1:]):
        if current_value == previous + 1:
            current += 1
            longest = max(longest, current)
        else:
            current = 1
    return longest


def summarize_range(frames: list[dict], start: int, end: int) -> dict:
    selected = [entry for entry in frames if start <= entry["frame"] <= end]
    observations = [vehicle for entry in selected for vehicle in entry["vehicles"]]
    frames_with_vehicle = [entry for entry in selected if entry["vehicles"]]
    frames_with_moving = [
        entry for entry in frames_with_vehicle
        if any(vehicle["moving"] == "1" for vehicle in entry["vehicles"])
    ]
    track_frames: dict[int, list[int]] = defaultdict(list)
    for entry in selected:
        for vehicle in entry["vehicles"]:
            track_frames[vehicle["track_id"]].append(entry["frame"])
    score_values = [float(vehicle["score"]) for vehicle in observations]
    zero_scores = sum(score <= 1e-9 for score in score_values)
    moving_observations = sum(vehicle["moving"] == "1" for vehicle in observations)
    return {
        "range": [start, end],
        "frame_count": len(selected),
        "frames_with_vehicle": len(frames_with_vehicle),
        "frames_with_moving": len(frames_with_moving),
        "moving_frame_rate": len(frames_with_moving) / max(1, len(frames_with_vehicle)),
        "vehicle_observations": len(observations),
        "moving_observations": moving_observations,
        "moving_observation_rate": moving_observations / max(1, len(observations)),
        "zero_score_observations": zero_scores,
        "zero_score_rate": zero_scores / max(1, len(observations)),
        "unique_track_ids": len(track_frames),
        "track_ids_per_100_observations": len(track_frames) * 100.0 / max(1, len(observations)),
        "longest_track_run": max((longest_consecutive(sorted(set(values)))
                                  for values in track_frames.values()), default=0),
        "score_median": float(__import__("numpy").median(score_values)) if score_values else 0.0,
        "score_max": max(score_values, default=0.0),
    }


def summarize_camera(frames: list[dict]) -> dict:
    cameras = [entry["camera"] for entry in frames if "camera" in entry]
    models = Counter(item["model"] for item in cameras)
    reliable = sum(item["model"] != "NONE" for item in cameras)
    return {
        "models": dict(models),
        "reliable_rate": reliable / max(1, len(cameras)),
        "mean_inlier_ratio": sum(item["inlier_ratio"] for item in cameras) / max(1, len(cameras)),
        "mean_reprojection_error": sum(item["reprojection_error"] for item in cameras) / max(1, len(cameras)),
    }


def build_report(args: argparse.Namespace, outputs: dict[str, list[dict]]) -> dict:
    report = {
        "labels": {
            "negative_no_moving_vehicle": [args.negative_start, args.negative_end],
            "positive_moving_vehicle_present": [args.positive_start, args.positive_end],
        },
        "trackers": {},
    }
    for name, frames in outputs.items():
        report["trackers"][name] = {
            "negative": summarize_range(frames, args.negative_start, args.negative_end),
            "positive": summarize_range(frames, args.positive_start, args.positive_end),
            "frames": frames,
        }
        if name == "v2_raw":
            report["trackers"][name]["camera"] = summarize_camera(frames)
    return report


def print_summary(report: dict) -> None:
    for name, result in report["trackers"].items():
        negative = result["negative"]
        positive = result["positive"]
        LOGGER.info(
            "%s | NEG M-frame=%.1f%% zero=%.1f%% ids/100=%.1f | "
            "POS M-frame=%.1f%% zero=%.1f%% ids/100=%.1f longest=%d",
            name,
            negative["moving_frame_rate"] * 100.0,
            negative["zero_score_rate"] * 100.0,
            negative["track_ids_per_100_observations"],
            positive["moving_frame_rate"] * 100.0,
            positive["zero_score_rate"] * 100.0,
            positive["track_ids_per_100_observations"],
            positive["longest_track_run"],
        )


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    experiment_config = load_yaml(args.config)
    production_config = load_yaml(args.production_config)
    detections = ensure_detection_cache(args, experiment_config)
    outputs = run_trackers(args, experiment_config, production_config, detections)
    report = build_report(args, outputs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print_summary(report)
    LOGGER.info("Rapor: %s", args.output)


if __name__ == "__main__":
    main()
