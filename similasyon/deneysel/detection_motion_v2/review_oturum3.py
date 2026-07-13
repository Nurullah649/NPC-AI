#!/usr/bin/env python3
"""THYZ 2025 Oturum 3 uzerinde deneysel gorsel inceleme.

Bu arac ana pipeline'i degistirmez ve sunucuya veri gondermez.

Urettigi kontroller:

1. Insan araligi:
   - XML ground-truth
   - Full-frame YOLO
   - 2x3 temporal micro-tile + PersonTrackerV2
   - TP/FP/FN ve secili yan-yana gorseller

2. Arac hareket araligi:
   - XML bbox girisli VehicleTrackerV2 (motion algoritmasini izole eder)
   - YOLO bbox girisli VehicleTrackerV2 (uctan uca deneysel yol)
   - Secili yan-yana gorseller ve iki MP4

XML dosyalarinda moving_status bulunmadigi icin motion sonucu otomatik GT
metrik degildir. Gorseller/MP4 manuel onay icindir.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np
import yaml


BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from motion.camera_motion import CameraMotionEstimator, CameraTransform  # noqa: E402
from motion.vehicle_tracker import VehicleTrackerV2  # noqa: E402
from person.detector_adapter import DetectorAdapter, _box_iou  # noqa: E402
from person.tile_scheduler import TileScheduler  # noqa: E402
from person.tracker import PersonTrackerV2  # noqa: E402


LOGGER = logging.getLogger("review_oturum3")

XML_TO_CLASS = {
    "Taşıt": 0,
    "İnsan": 1,
    "UAP": 2,
    "UAİ": 3,
}

CLASS_NAMES = {0: "Tasit", 1: "Insan", 2: "UAP", 3: "UAI"}


def parse_args():
    default_root = Path(
        "/home/nurullah/Downloads/TEKNOFEST HYZ 2025 Verileri/"
        "THYZ_2025_Oturum_3"
    )
    parser = argparse.ArgumentParser(description="Oturum 3 deneysel gorsel kontrol")
    parser.add_argument("--config", default=str(BASE_DIR / "config_experiment.yaml"))
    parser.add_argument("--frames-dir", default=str(default_root / "Frames"))
    parser.add_argument("--annotations-dir", default=str(default_root / "Annotations"))
    parser.add_argument("--output-dir", default=str(BASE_DIR / "results" / "oturum3_review"))
    parser.add_argument("--person-start", type=int, default=535)
    parser.add_argument("--person-end", type=int, default=707)
    parser.add_argument("--motion-start", type=int, default=2200)
    parser.add_argument("--motion-end", type=int, default=2249)
    parser.add_argument(
        "--person-save-frames",
        default="535,545,560,580,600,620,640,660,680,700,707",
    )
    parser.add_argument(
        "--motion-save-frames",
        default="2204,2210,2216,2222,2228,2234,2240,2249",
    )
    parser.add_argument("--video-fps", type=float, default=7.5)
    parser.add_argument("--skip-person", action="store_true")
    parser.add_argument("--skip-motion", action="store_true")
    parser.add_argument("--skip-yolo-motion", action="store_true")
    return parser.parse_args()


def parse_int_set(value: str) -> set[int]:
    return {int(x.strip()) for x in value.split(",") if x.strip()}


def load_xml_annotations(path: Path) -> list[dict]:
    if not path.exists():
        return []
    root = ET.parse(path).getroot()
    detections = []
    for obj in root.findall(".//object"):
        class_name_xml = obj.findtext("name")
        cls_id = XML_TO_CLASS.get(class_name_xml)
        bbox_node = obj.find("bndbox")
        if cls_id is None or bbox_node is None:
            continue
        try:
            bbox = tuple(
                float(bbox_node.findtext(key))
                for key in ("xmin", "ymin", "xmax", "ymax")
            )
        except (TypeError, ValueError):
            continue
        attributes = {}
        for attribute in obj.findall("./attributes/attribute"):
            name = attribute.findtext("name")
            if name:
                attributes[name] = attribute.findtext("value")
        detections.append(
            {
                "cls": cls_id,
                "cls_name": CLASS_NAMES[cls_id],
                "conf": 1.0,
                "bbox": bbox,
                "_source": "xml_gt",
                "_attributes": attributes,
            }
        )
    return detections


def frame_path(frames_dir: Path, frame_idx: int) -> Path:
    return frames_dir / f"frame_{frame_idx:06d}.webp"


def annotation_path(annotations_dir: Path, frame_idx: int) -> Path:
    return annotations_dir / f"frame_{frame_idx:06d}.xml"


def match_counts(gt: list[dict], pred: list[dict], cls_id: int, iou_threshold=0.5):
    gt_cls = [g for g in gt if g.get("cls") == cls_id]
    pred_cls = sorted(
        [p for p in pred if p.get("cls") == cls_id],
        key=lambda p: float(p.get("conf", 0.0)),
        reverse=True,
    )
    matched_gt = set()
    tp = 0
    for prediction in pred_cls:
        best_iou = 0.0
        best_idx = -1
        for idx, target in enumerate(gt_cls):
            if idx in matched_gt:
                continue
            iou = _box_iou(prediction["bbox"], target["bbox"])
            if iou > best_iou:
                best_iou = iou
                best_idx = idx
        if best_idx >= 0 and best_iou >= iou_threshold:
            matched_gt.add(best_idx)
            tp += 1
    fp = len(pred_cls) - tp
    fn = len(gt_cls) - tp
    return {"tp": tp, "fp": fp, "fn": fn, "gt": len(gt_cls), "pred": len(pred_cls)}


def add_counts(total: dict, frame_counts: dict):
    for key in ("tp", "fp", "fn", "gt", "pred"):
        total[key] = total.get(key, 0) + int(frame_counts.get(key, 0))


def finalize_counts(counts: dict) -> dict:
    result = dict(counts)
    tp, fp, fn = result.get("tp", 0), result.get("fp", 0), result.get("fn", 0)
    result["precision"] = tp / max(1, tp + fp)
    result["recall"] = tp / max(1, tp + fn)
    return result


def transform_dict(transform: CameraTransform | None) -> dict:
    if transform is None:
        return {"model": "NONE", "confidence": 0.0}
    return {
        "model": transform.model_type.name,
        "confidence": float(transform.confidence),
        "matches": int(transform.match_count),
        "inliers": int(transform.inlier_count),
        "inlier_ratio": float(transform.inlier_ratio),
        "reprojection_error": float(transform.reprojection_error),
    }


def serializable_detection(det: dict) -> dict:
    return {
        "cls": int(det.get("cls", -1)),
        "conf": float(det.get("conf", 0.0)),
        "bbox": [float(v) for v in det.get("bbox", (0, 0, 0, 0))],
        "track_id": det.get("_track_id"),
        "moving_status": det.get("moving_status"),
        "motion_score": (
            float(det["_motion_score"]) if det.get("_motion_score") is not None else None
        ),
        "source": det.get("_source", "detector"),
    }


def draw_boxes(image: np.ndarray, detections: list[dict], mode: str) -> np.ndarray:
    canvas = image.copy()
    for det in detections:
        x1, y1, x2, y2 = [int(round(v)) for v in det["bbox"]]
        cls_id = int(det.get("cls", -1))
        if mode == "gt":
            color = (0, 255, 0)
            label = f"GT {CLASS_NAMES.get(cls_id, cls_id)}"
        elif mode == "motion":
            moving = str(det.get("moving_status", "?"))
            color = (0, 0, 255) if moving == "1" else (255, 180, 0)
            track_id = det.get("_track_id", "-")
            score = det.get("_motion_score")
            score_text = f"{float(score):.3f}" if score is not None else "-"
            label = f"Tasit T#{track_id} M:{moving} S:{score_text}"
        else:
            color = (0, 200, 255) if det.get("_source") != "tracker_coast" else (255, 0, 255)
            track_text = f" T#{det['_track_id']}" if det.get("_track_id") is not None else ""
            source = det.get("_source", "yolo")
            label = f"{CLASS_NAMES.get(cls_id, cls_id)} {float(det.get('conf', 0)):.2f}{track_text} {source}"
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 5)
        text_y = max(35, y1 - 10)
        cv2.putText(
            canvas,
            label,
            (max(0, x1), text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            color,
            2,
            cv2.LINE_AA,
        )
    return canvas


def title_bar(image: np.ndarray, title: str, detail: str = "") -> np.ndarray:
    canvas = image.copy()
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 90), (20, 20, 20), -1)
    cv2.putText(canvas, title, (25, 38), cv2.FONT_HERSHEY_SIMPLEX, 1.05, (255, 255, 255), 2)
    if detail:
        cv2.putText(canvas, detail, (25, 76), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (230, 230, 230), 2)
    return canvas


def comparison_image(left: np.ndarray, right: np.ndarray, width_each=1280) -> np.ndarray:
    def resized(image):
        scale = width_each / image.shape[1]
        height = int(round(image.shape[0] * scale))
        return cv2.resize(image, (width_each, height), interpolation=cv2.INTER_AREA)

    l_img, r_img = resized(left), resized(right)
    height = min(l_img.shape[0], r_img.shape[0])
    return np.hstack([l_img[:height], r_img[:height]])


def video_writer(path: Path, fps: float, size=(1920, 1080)):
    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(str(path), fourcc, fps, size)


def write_video_frame(writer, image, size=(1920, 1080)):
    writer.write(cv2.resize(image, size, interpolation=cv2.INTER_AREA))


def run_person_review(
    detector: DetectorAdapter,
    config: dict,
    frames_dir: Path,
    annotations_dir: Path,
    output_dir: Path,
    start: int,
    end: int,
    save_frames: set[int],
    fps: float,
) -> dict:
    LOGGER.info("Person review basliyor: %d-%d", start, end)
    person_dir = output_dir / "person"
    person_dir.mkdir(parents=True, exist_ok=True)

    scheduler = TileScheduler(config)
    tracker = PersonTrackerV2(config)
    camera = CameraMotionEstimator(config)
    prev_gray = None

    totals = {
        "yolo": {"tp": 0, "fp": 0, "fn": 0, "gt": 0, "pred": 0},
        "microtile": {"tp": 0, "fp": 0, "fn": 0, "gt": 0, "pred": 0},
        "tracker": {"tp": 0, "fp": 0, "fn": 0, "gt": 0, "pred": 0},
    }
    frame_results = []
    writer = video_writer(person_dir / "person_microtile_tracker.mp4", fps)

    for frame_idx in range(start, end + 1):
        path = frame_path(frames_dir, frame_idx)
        image = cv2.imread(str(path))
        if image is None:
            LOGGER.warning("Frame okunamadi: %s", path)
            continue
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gt = load_xml_annotations(annotation_path(annotations_dir, frame_idx))

        t0 = time.perf_counter()
        yolo_dets = detector.detect_full_frame(image)
        transform = camera.estimate(prev_gray, gray, yolo_dets) if prev_gray is not None else CameraTransform()

        active = tracker.get_active_track_bboxes()
        lost = tracker.get_lost_rois()
        crops = scheduler.schedule(frame_idx - start, image.shape[1], image.shape[0], active, lost)
        crop_dets = detector.detect_crops_batched(
            image, [crop.bbox for crop in crops], person_only=True,
        )

        yolo_persons = [d for d in yolo_dets if d.get("cls") == 1]
        yolo_others = [d for d in yolo_dets if d.get("cls") != 1]
        merged_persons = detector.global_nms(yolo_persons + crop_dets, 0.45)
        tracked_all = tracker.update(yolo_others + merged_persons, frame_idx, transform)
        tracked_persons = [d for d in tracked_all if d.get("cls") == 1]
        latency = time.perf_counter() - t0

        counts_yolo = match_counts(gt, yolo_persons, 1)
        counts_tiled = match_counts(gt, merged_persons, 1)
        counts_tracker = match_counts(gt, tracked_persons, 1)
        add_counts(totals["yolo"], counts_yolo)
        add_counts(totals["microtile"], counts_tiled)
        add_counts(totals["tracker"], counts_tracker)

        frame_results.append(
            {
                "frame": frame_idx,
                "latency_sec": latency,
                "camera": transform_dict(transform),
                "crop_count": len(crops),
                "counts": {
                    "yolo": counts_yolo,
                    "microtile": counts_tiled,
                    "tracker": counts_tracker,
                },
                "tracker_detections": [serializable_detection(d) for d in tracked_persons],
            }
        )

        gt_persons = [d for d in gt if d.get("cls") == 1]
        video_canvas = draw_boxes(image, gt_persons, "gt")
        video_canvas = draw_boxes(video_canvas, tracked_persons, "person")
        detail = (
            f"GT:{len(gt_persons)} Pred:{len(tracked_persons)} "
            f"TP:{counts_tracker['tp']} FP:{counts_tracker['fp']} FN:{counts_tracker['fn']} "
            f"cam:{transform.model_type.name} crops:{len(crops)}"
        )
        video_canvas = title_bar(video_canvas, f"Person EXP - frame {frame_idx:06d}", detail)
        write_video_frame(writer, video_canvas)

        if frame_idx in save_frames:
            left = draw_boxes(image, gt_persons, "gt")
            left = draw_boxes(left, yolo_persons, "person")
            left = title_bar(
                left,
                f"GT + Full YOLO - frame {frame_idx:06d}",
                f"TP:{counts_yolo['tp']} FP:{counts_yolo['fp']} FN:{counts_yolo['fn']}",
            )
            right = draw_boxes(image, gt_persons, "gt")
            right = draw_boxes(right, tracked_persons, "person")
            right = title_bar(
                right,
                "GT + Microtile + Tracker",
                f"TP:{counts_tracker['tp']} FP:{counts_tracker['fp']} FN:{counts_tracker['fn']} crops:{len(crops)}",
            )
            cv2.imwrite(
                str(person_dir / f"frame_{frame_idx:06d}_gt_yolo_full.jpg"),
                left,
                [cv2.IMWRITE_JPEG_QUALITY, 92],
            )
            cv2.imwrite(
                str(person_dir / f"frame_{frame_idx:06d}_gt_experimental_full.jpg"),
                right,
                [cv2.IMWRITE_JPEG_QUALITY, 92],
            )
            cv2.imwrite(
                str(person_dir / f"frame_{frame_idx:06d}_compare.jpg"),
                comparison_image(left, right),
                [cv2.IMWRITE_JPEG_QUALITY, 90],
            )

        prev_gray = gray
        if (frame_idx - start + 1) % 20 == 0:
            LOGGER.info("Person: %d/%d", frame_idx - start + 1, end - start + 1)

    writer.release()
    summary = {
        key: finalize_counts(value) for key, value in totals.items()
    }
    summary["scheduler"] = scheduler.get_stats()
    summary["frames"] = frame_results
    return summary


def run_motion_review(
    detector: DetectorAdapter,
    config: dict,
    frames_dir: Path,
    annotations_dir: Path,
    output_dir: Path,
    start: int,
    end: int,
    save_frames: set[int],
    fps: float,
    use_yolo: bool,
) -> dict:
    LOGGER.info("Motion review basliyor: %d-%d", start, end)
    motion_dir = output_dir / "motion"
    motion_dir.mkdir(parents=True, exist_ok=True)

    gt_tracker = VehicleTrackerV2(config)
    gt_camera = CameraMotionEstimator(config)
    gt_prev_gray = None

    yolo_tracker = VehicleTrackerV2(config) if use_yolo else None
    yolo_camera = CameraMotionEstimator(config) if use_yolo else None
    yolo_prev_gray = None

    gt_writer = video_writer(motion_dir / "motion_gt_bbox_tracker.mp4", fps)
    yolo_writer = video_writer(motion_dir / "motion_yolo_tracker.mp4", fps) if use_yolo else None

    frames_json = []
    for frame_idx in range(start, end + 1):
        path = frame_path(frames_dir, frame_idx)
        image = cv2.imread(str(path))
        if image is None:
            LOGGER.warning("Frame okunamadi: %s", path)
            continue
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gt_all = load_xml_annotations(annotation_path(annotations_dir, frame_idx))
        gt_vehicles = [dict(d) for d in gt_all if d.get("cls") == 0]

        gt_transform = (
            gt_camera.estimate(gt_prev_gray, gray, gt_vehicles)
            if gt_prev_gray is not None else CameraTransform()
        )
        gt_out = gt_tracker.update(gt_vehicles, gray, gt_transform)

        yolo_out = []
        yolo_transform = CameraTransform()
        if use_yolo:
            yolo_dets = detector.detect_full_frame(image)
            yolo_transform = (
                yolo_camera.estimate(yolo_prev_gray, gray, yolo_dets)
                if yolo_prev_gray is not None else CameraTransform()
            )
            yolo_out = yolo_tracker.update(yolo_dets, gray, yolo_transform)
            yolo_out = [d for d in yolo_out if d.get("cls") == 0]

        gt_canvas = draw_boxes(image, gt_out, "motion")
        gt_canvas = title_bar(
            gt_canvas,
            f"Motion V2 / GT bbox input - frame {frame_idx:06d}",
            f"vehicles:{len(gt_out)} cam:{gt_transform.model_type.name} "
            f"conf:{gt_transform.confidence:.2f}",
        )
        write_video_frame(gt_writer, gt_canvas)

        yolo_canvas = None
        if use_yolo:
            yolo_canvas = draw_boxes(image, yolo_out, "motion")
            yolo_canvas = title_bar(
                yolo_canvas,
                f"Motion V2 / YOLO bbox input - frame {frame_idx:06d}",
                f"vehicles:{len(yolo_out)} cam:{yolo_transform.model_type.name} "
                f"conf:{yolo_transform.confidence:.2f}",
            )
            write_video_frame(yolo_writer, yolo_canvas)

        if frame_idx in save_frames:
            right = yolo_canvas if yolo_canvas is not None else draw_boxes(image, gt_vehicles, "gt")
            cv2.imwrite(
                str(motion_dir / f"frame_{frame_idx:06d}_gt_tracker_full.jpg"),
                gt_canvas,
                [cv2.IMWRITE_JPEG_QUALITY, 92],
            )
            if yolo_canvas is not None:
                cv2.imwrite(
                    str(motion_dir / f"frame_{frame_idx:06d}_yolo_tracker_full.jpg"),
                    yolo_canvas,
                    [cv2.IMWRITE_JPEG_QUALITY, 92],
                )
            cv2.imwrite(
                str(motion_dir / f"frame_{frame_idx:06d}_compare.jpg"),
                comparison_image(gt_canvas, right),
                [cv2.IMWRITE_JPEG_QUALITY, 90],
            )

        frames_json.append(
            {
                "frame": frame_idx,
                "gt_vehicle_count": len(gt_vehicles),
                "gt_camera": transform_dict(gt_transform),
                "gt_tracker": [serializable_detection(d) for d in gt_out],
                "yolo_camera": transform_dict(yolo_transform) if use_yolo else None,
                "yolo_tracker": [serializable_detection(d) for d in yolo_out],
            }
        )

        gt_prev_gray = gray
        yolo_prev_gray = gray
        if (frame_idx - start + 1) % 10 == 0:
            LOGGER.info("Motion: %d/%d", frame_idx - start + 1, end - start + 1)

    gt_writer.release()
    if yolo_writer is not None:
        yolo_writer.release()

    return {"frames": frames_json, "moving_ground_truth_available": False}


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    frames_dir = Path(args.frames_dir)
    annotations_dir = Path(args.annotations_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(args.config, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    detector = DetectorAdapter(config)
    results = {
        "frames_dir": str(frames_dir),
        "annotations_dir": str(annotations_dir),
        "annotation_has_moving_status": False,
    }

    if not args.skip_person:
        results["person"] = run_person_review(
            detector,
            config,
            frames_dir,
            annotations_dir,
            output_dir,
            args.person_start,
            args.person_end,
            parse_int_set(args.person_save_frames),
            args.video_fps,
        )

    if not args.skip_motion:
        results["motion"] = run_motion_review(
            detector,
            config,
            frames_dir,
            annotations_dir,
            output_dir,
            args.motion_start,
            args.motion_end,
            parse_int_set(args.motion_save_frames),
            args.video_fps,
            use_yolo=not args.skip_yolo_motion,
        )

    result_path = output_dir / "review_summary.json"
    with open(result_path, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2, ensure_ascii=False)
    LOGGER.info("Inceleme tamamlandi: %s", result_path)


if __name__ == "__main__":
    main()
