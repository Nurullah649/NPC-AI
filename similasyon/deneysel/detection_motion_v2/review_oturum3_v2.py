#!/usr/bin/env python3
"""THYZ 2025 Oturum 3 uzerinde deneysel gorsel inceleme — v2.

v2 farklari:
  - A/A2/D/E/F/G karsilastirma modlari
  - source_aware_merge (global_nms yerine)
  - Kamera transformu tracker modlarinda
  - Gelişmiş gorsel etiketler (track_id bazli renk, OBS/COAST/CROP)
  - Motion manual label template

Urettigi kontroller:

1. Insan araligi:
   - XML ground-truth
   - Full-frame YOLO (A, A2)
   - 1crop/2crop + source_aware_merge (D, E)
   - 1crop/2crop + source_aware_merge + tracker (F, G)
   - TP/FP/FN ve per-mode metrikler

2. Arac hareket araligi:
   - XML bbox girisli VehicleTrackerV2
   - YOLO bbox girisli VehicleTrackerV2
   - Manual label template

XML dosyalarinda moving_status bulunmadigi icin motion sonucu otomatik GT
metrik degildir. Gorseller/MP4 manuel onay icindir.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
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
from person.merge import source_aware_merge  # noqa: E402
from person.tile_scheduler import TileScheduler  # noqa: E402
from person.tracker import PersonTrackerV2  # noqa: E402


LOGGER = logging.getLogger("review_oturum3_v2")

XML_TO_CLASS = {
    "Taşıt": 0,
    "İnsan": 1,
    "UAP": 2,
    "UAİ": 3,
}

CLASS_NAMES = {0: "Tasit", 1: "Insan", 2: "UAP", 3: "UAI"}

# --- Mode tanimlari ---

PERSON_MODES = ["A", "A2", "D", "E", "F", "G"]

MODE_DESCRIPTIONS = {
    "A": "Full YOLO only (no crops, no NMS, no tracker)",
    "A2": "Full YOLO + global NMS (no crops, no tracker)",
    "D": "YOLO + 1 crop + source_aware_merge (no tracker)",
    "E": "YOLO + 2 crop + source_aware_merge (no tracker)",
    "F": "YOLO + 1 crop + source_aware_merge + tracker",
    "G": "YOLO + 2 crop + source_aware_merge + tracker",
}


def _track_color(track_id: int) -> tuple:
    """Track ID'den deterministik BGR renk uret."""
    if track_id is None:
        return (0, 200, 255)
    h = hashlib.md5(str(track_id).encode()).hexdigest()
    b = (int(h[0:2], 16) % 180) + 75
    g = (int(h[2:4], 16) % 180) + 75
    r = (int(h[4:6], 16) % 180) + 75
    return (b, g, r)


def _source_color_and_label(det: dict) -> tuple:
    """Kaynaga gore (BGR renk, kaynak kisaltmasi) dondur.

    Kaynaklar:
      - tracker_coast -> COAST (purple)
      - crop           -> CROP  (orange)
      - diger          -> OBS   (yellow)
    """
    source = det.get("_source", "")
    if source == "tracker_coast":
        return (255, 0, 255), "COAST"   # purple
    elif source == "crop":
        return (0, 165, 255), "CROP"     # orange
    else:
        return (0, 255, 255), "OBS"      # yellow


def parse_args():
    default_root = Path(
        "/home/nurullah/Downloads/TEKNOFEST HYZ 2025 Verileri/"
        "THYZ_2025_Oturum_3"
    )
    parser = argparse.ArgumentParser(description="Oturum 3 deneysel gorsel kontrol v2")
    parser.add_argument("--config", default=str(BASE_DIR / "config_experiment.yaml"))
    parser.add_argument("--frames-dir", default=str(default_root / "Frames"))
    parser.add_argument("--annotations-dir", default=str(default_root / "Annotations"))
    parser.add_argument("--output-dir", default=str(BASE_DIR / "results" / "oturum3_review_v2"))
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
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-9, precision + recall)
    result["precision"] = precision
    result["recall"] = recall
    result["f1"] = f1
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
        track_id = det.get("_track_id")

        if mode == "gt":
            color = (0, 255, 0)
            label = f"GT {CLASS_NAMES.get(cls_id, cls_id)}"

        elif mode == "motion":
            moving = str(det.get("moving_status", "?"))
            color = (0, 0, 255) if moving == "1" else (255, 180, 0)
            tid = det.get("_track_id", "-")
            score = det.get("_motion_score")
            score_text = f"{float(score):.3f}" if score is not None else "-"
            label = f"Tasit T#{tid} M:{moving} S:{score_text}"

        elif mode == "person":
            # v2 label format
            source_color, source_abbr = _source_color_and_label(det)
            if track_id is not None:
                color = _track_color(track_id)
                label = f"T#{track_id} {float(det.get('conf', 0)):.2f} {source_abbr}"
            else:
                color = source_color
                src = det.get("_source", "yolo")
                label = f"{float(det.get('conf', 0)):.2f} {src}"

        else:
            # Fallback: original drawing
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


def _count_unique_track_ids(per_frame_results: list[dict]) -> int:
    """Frame sonuclarinda gecen unique track ID sayisi."""
    ids = set()
    for f in per_frame_results:
        for d in f.get("detections", []):
            tid = d.get("track_id")
            if tid is not None:
                ids.add(tid)
    return len(ids)


# ---------------------------------------------------------------------------
# Person review — v2: A/A2/D/E/F/G karsilastirma modlari
# ---------------------------------------------------------------------------

def run_person_review_v2(
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
    """Person review v2 — tum modlari tek geciste calistir.

    Modlar:
      A   : Full YOLO only (no crops, no NMS, no tracker)
      A2  : Full YOLO + global NMS (no crops, no tracker)
      D   : YOLO + 1 crop + source_aware_merge (no tracker)
      E   : YOLO + 2 crop + source_aware_merge (no tracker)
      F   : YOLO + 1 crop + source_aware_merge + tracker
      G   : YOLO + 2 crop + source_aware_merge + tracker
    """
    LOGGER.info("Person review v2 basliyor: %d-%d", start, end)
    person_dir = output_dir / "person"
    person_dir.mkdir(parents=True, exist_ok=True)

    # --- Config'ler ---
    config_1crop = copy.deepcopy(config)
    config_1crop["person_detection"]["max_crops_per_frame"] = 1

    # --- Scheduler'lar ---
    # D/F (1 crop) icin ayri scheduler'lar — ic durum farkli olmali
    scheduler_d = TileScheduler(config_1crop)  # D: no tracker
    scheduler_e = TileScheduler(config)        # E: 2 crop, no tracker
    scheduler_f = TileScheduler(config_1crop)  # F: 1 crop + tracker
    scheduler_g = TileScheduler(config)        # G: 2 crop + tracker

    # --- Tracker'lar ---
    tracker_f = PersonTrackerV2(config_1crop)  # F: 1 crop config
    tracker_g = PersonTrackerV2(config)        # G: default config

    # --- Camera ---
    camera = CameraMotionEstimator(config)
    prev_gray = None

    # --- Mode state ---
    mode_totals = {m: {"tp": 0, "fp": 0, "fn": 0, "gt": 0, "pred": 0}
                   for m in PERSON_MODES}
    mode_frames = {m: [] for m in PERSON_MODES}
    mode_scheduler_stats = {}
    mode_track_id_count = {}

    # --- Video writer'lar (yalniz tracker modlari F, G) ---
    writer_f = video_writer(person_dir / "mode_F_tracker.mp4", fps)
    writer_g = video_writer(person_dir / "mode_G_tracker.mp4", fps)
    # Ortak karsilastirma videosu: GT + A (sol) vs GT + G (sag)
    writer_compare = video_writer(person_dir / "person_A_vs_G_comparison.mp4", fps)

    crop_conf_threshold = config.get("person_detection", {}).get("confidence_threshold", 0.15)

    total_frames = end - start + 1

    for frame_idx in range(start, end + 1):
        path = frame_path(frames_dir, frame_idx)
        image = cv2.imread(str(path))
        if image is None:
            LOGGER.warning("Frame okunamadi: %s", path)
            continue
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gt = load_xml_annotations(annotation_path(annotations_dir, frame_idx))

        t0 = time.perf_counter()

        # --- Shared: YOLO full frame ---
        yolo_dets = detector.detect_full_frame(image)
        yolo_persons = [d for d in yolo_dets if d.get("cls") == 1]
        yolo_others = [d for d in yolo_dets if d.get("cls") != 1]

        # --- Camera transform (shared by F/G) ---
        transform = camera.estimate(prev_gray, gray, yolo_dets) if prev_gray is not None else CameraTransform()

        # --- Mode A: raw YOLO persons (no NMS, no crops, no tracker) ---
        a_persons = list(yolo_persons)

        # --- Mode A2: YOLO persons + global NMS ---
        a2_persons = detector.global_nms(yolo_persons, 0.45) if yolo_persons else []

        # --- Crop modes: schedule ---
        # D: 1 crop, no tracker
        crops_d = scheduler_d.schedule(frame_idx - start, image.shape[1], image.shape[0], [], [])
        # E: 2 crops, no tracker
        crops_e = scheduler_e.schedule(frame_idx - start, image.shape[1], image.shape[0], [], [])

        # F: 1 crop + tracker
        active_f = tracker_f.get_active_track_bboxes()
        lost_f = tracker_f.get_lost_rois()
        crops_f = scheduler_f.schedule(frame_idx - start, image.shape[1], image.shape[0], active_f, lost_f)

        # G: 2 crops + tracker
        active_g = tracker_g.get_active_track_bboxes()
        lost_g = tracker_g.get_lost_rois()
        crops_g = scheduler_g.schedule(frame_idx - start, image.shape[1], image.shape[0], active_g, lost_g)

        # --- Crop detections ---
        crop_dets_d = detector.detect_crops_batched(
            image, [c.bbox for c in crops_d], person_only=True,
            crop_confidence_threshold=crop_conf_threshold,
        ) if crops_d else []

        crop_dets_e = detector.detect_crops_batched(
            image, [c.bbox for c in crops_e], person_only=True,
            crop_confidence_threshold=crop_conf_threshold,
        ) if crops_e else []

        crop_dets_f = detector.detect_crops_batched(
            image, [c.bbox for c in crops_f], person_only=True,
            crop_confidence_threshold=crop_conf_threshold,
        ) if crops_f else []

        crop_dets_g = detector.detect_crops_batched(
            image, [c.bbox for c in crops_g], person_only=True,
            crop_confidence_threshold=crop_conf_threshold,
        ) if crops_g else []

        # --- Mode D: merge 1 crop, no tracker ---
        d_persons = source_aware_merge(yolo_persons, crop_dets_d, 0.45)

        # --- Mode E: merge 2 crops, no tracker ---
        e_persons = source_aware_merge(yolo_persons, crop_dets_e, 0.45)

        # --- Mode F: merge 1 crop + tracker ---
        f_merged = source_aware_merge(yolo_persons, crop_dets_f, 0.45)
        f_tracked_all = tracker_f.update(yolo_others + f_merged, frame_idx, transform)
        f_persons = [d for d in f_tracked_all if d.get("cls") == 1]

        # --- Mode G: merge 2 crops + tracker ---
        g_merged = source_aware_merge(yolo_persons, crop_dets_g, 0.45)
        g_tracked_all = tracker_g.update(yolo_others + g_merged, frame_idx, transform)
        g_persons = [d for d in g_tracked_all if d.get("cls") == 1]

        latency = time.perf_counter() - t0

        # --- Metrics per mode ---
        mode_predictions = {
            "A": a_persons,
            "A2": a2_persons,
            "D": d_persons,
            "E": e_persons,
            "F": f_persons,
            "G": g_persons,
        }

        gt_persons = [d for d in gt if d.get("cls") == 1]
        frame_mode_counts = {}

        for mode_name in PERSON_MODES:
            preds = mode_predictions[mode_name]
            cnt = match_counts(gt, preds, 1)
            add_counts(mode_totals[mode_name], cnt)
            frame_mode_counts[mode_name] = cnt

        # --- Per-frame JSON ---
        frame_entry = {
            "frame": frame_idx,
            "latency_sec": latency,
            "camera": transform_dict(transform),
            "counts": {
                m: frame_mode_counts[m] for m in PERSON_MODES
            },
        }
        # Tracker modlari icin detection detaylari
        if f_persons:
            frame_entry["detections_F"] = [serializable_detection(d) for d in f_persons]
        if g_persons:
            frame_entry["detections_G"] = [serializable_detection(d) for d in g_persons]
        frame_entry["crop_counts"] = {
            "D": len(crops_d), "E": len(crops_e),
            "F": len(crops_f), "G": len(crops_g),
        }

        for mode_name in PERSON_MODES:
            mode_frames[mode_name].append(frame_entry)

        # --- Video frames ---
        # Mode F video
        f_canvas = draw_boxes(image, gt_persons, "gt")
        f_canvas = draw_boxes(f_canvas, f_persons, "person")
        f_detail = (
            f"F: GT:{len(gt_persons)} Pred:{len(f_persons)} "
            f"TP:{frame_mode_counts['F']['tp']} FP:{frame_mode_counts['F']['fp']} FN:{frame_mode_counts['F']['fn']} "
            f"cam:{transform.model_type.name} crops:{len(crops_f)}"
        )
        f_canvas = title_bar(f_canvas, f"Mode F (1crop+tracker) - frame {frame_idx:06d}", f_detail)
        write_video_frame(writer_f, f_canvas)

        # Mode G video
        g_canvas = draw_boxes(image, gt_persons, "gt")
        g_canvas = draw_boxes(g_canvas, g_persons, "person")
        g_detail = (
            f"G: GT:{len(gt_persons)} Pred:{len(g_persons)} "
            f"TP:{frame_mode_counts['G']['tp']} FP:{frame_mode_counts['G']['fp']} FN:{frame_mode_counts['G']['fn']} "
            f"cam:{transform.model_type.name} crops:{len(crops_g)}"
        )
        g_canvas = title_bar(g_canvas, f"Mode G (2crop+tracker) - frame {frame_idx:06d}", g_detail)
        write_video_frame(writer_g, g_canvas)

        # Karsilastirma: GT + A (sol) vs GT + G (sag)
        a_canvas_left = draw_boxes(image, gt_persons, "gt")
        a_canvas_left = draw_boxes(a_canvas_left, a_persons, "person")
        a_canvas_left = title_bar(
            a_canvas_left,
            f"GT + Mode A (Full YOLO) - frame {frame_idx:06d}",
            f"TP:{frame_mode_counts['A']['tp']} FP:{frame_mode_counts['A']['fp']} FN:{frame_mode_counts['A']['fn']}",
        )
        g_canvas_right = draw_boxes(image, gt_persons, "gt")
        g_canvas_right = draw_boxes(g_canvas_right, g_persons, "person")
        g_canvas_right = title_bar(
            g_canvas_right,
            f"GT + Mode G (2crop+tracker) - frame {frame_idx:06d}",
            f"TP:{frame_mode_counts['G']['tp']} FP:{frame_mode_counts['G']['fp']} FN:{frame_mode_counts['G']['fn']} crops:{len(crops_g)}",
        )
        write_video_frame(writer_compare, comparison_image(a_canvas_left, g_canvas_right))

        # --- Save selected frames ---
        if frame_idx in save_frames:
            # Mode A (YOLO only)
            img_a = draw_boxes(image, gt_persons, "gt")
            img_a = draw_boxes(img_a, a_persons, "person")
            img_a = title_bar(
                img_a,
                f"Mode A (Full YOLO) - frame {frame_idx:06d}",
                f"TP:{frame_mode_counts['A']['tp']} FP:{frame_mode_counts['A']['fp']} FN:{frame_mode_counts['A']['fn']}",
            )
            cv2.imwrite(
                str(person_dir / f"frame_{frame_idx:06d}_mode_A.jpg"),
                img_a,
                [cv2.IMWRITE_JPEG_QUALITY, 92],
            )

            # Mode G (best pipeline)
            img_g = draw_boxes(image, gt_persons, "gt")
            img_g = draw_boxes(img_g, g_persons, "person")
            img_g = title_bar(
                img_g,
                f"Mode G (2crop+tracker) - frame {frame_idx:06d}",
                f"TP:{frame_mode_counts['G']['tp']} FP:{frame_mode_counts['G']['fp']} FN:{frame_mode_counts['G']['fn']} crops:{len(crops_g)}",
            )
            cv2.imwrite(
                str(person_dir / f"frame_{frame_idx:06d}_mode_G.jpg"),
                img_g,
                [cv2.IMWRITE_JPEG_QUALITY, 92],
            )

            # Comparison image: A vs G
            cv2.imwrite(
                str(person_dir / f"frame_{frame_idx:06d}_A_vs_G.jpg"),
                comparison_image(img_a, img_g),
                [cv2.IMWRITE_JPEG_QUALITY, 90],
            )

        prev_gray = gray
        if (frame_idx - start + 1) % 20 == 0:
            LOGGER.info("Person v2: %d/%d", frame_idx - start + 1, total_frames)

    writer_f.release()
    writer_g.release()
    writer_compare.release()

    # --- Scheduler stats ---
    mode_scheduler_stats["D"] = scheduler_d.get_stats()
    mode_scheduler_stats["E"] = scheduler_e.get_stats()
    mode_scheduler_stats["F"] = scheduler_f.get_stats()
    mode_scheduler_stats["G"] = scheduler_g.get_stats()
    mode_scheduler_stats["A"] = None
    mode_scheduler_stats["A2"] = None

    # --- Track ID counts (F, G) ---
    mode_track_id_count["F"] = _count_unique_track_ids(mode_frames["F"])
    mode_track_id_count["G"] = _count_unique_track_ids(mode_frames["G"])
    for m in ("A", "A2", "D", "E"):
        mode_track_id_count[m] = 0

    # --- Build per-mode summary ---
    modes_summary = {}
    for m in PERSON_MODES:
        modes_summary[m] = {
            **finalize_counts(mode_totals[m]),
            "description": MODE_DESCRIPTIONS[m],
            "frames": mode_frames[m],
            "scheduler_stats": mode_scheduler_stats.get(m),
            "track_id_count": mode_track_id_count.get(m, 0),
        }

    return {
        "modes": modes_summary,
        "total_frames": total_frames,
    }


# ---------------------------------------------------------------------------
# Motion review — v2 (manual label template)
# ---------------------------------------------------------------------------

def run_motion_review_v2(
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
    """Motion review v2 — v1 ile ayni, sonunda manual label template ekler.

    `motion_manual_labels.json` olusturur: her frame'deki her arac tespiti
    icin bir giris. Kullanici `manual_status` alanini doldurur.
    """
    LOGGER.info("Motion review v2 basliyor: %d-%d", start, end)
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
    manual_labels = []

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

        frame_entry = {
            "frame": frame_idx,
            "gt_vehicle_count": len(gt_vehicles),
            "gt_camera": transform_dict(gt_transform),
            "gt_tracker": [serializable_detection(d) for d in gt_out],
            "yolo_camera": transform_dict(yolo_transform) if use_yolo else None,
            "yolo_tracker": [serializable_detection(d) for d in yolo_out],
        }
        frames_json.append(frame_entry)

        # Manual label template: her vehicle detection per frame
        # GT tracker ciktisini kullan
        for det in gt_out:
            tid = det.get("_track_id")
            bbox = det.get("bbox", (0, 0, 0, 0))
            predicted = str(det.get("moving_status", "0"))
            manual_labels.append({
                "frame": frame_idx,
                "track_id": tid,
                "bbox": [float(v) for v in bbox],
                "predicted_status": predicted,
                "manual_status": None,
            })

        gt_prev_gray = gray
        yolo_prev_gray = gray
        if (frame_idx - start + 1) % 10 == 0:
            LOGGER.info("Motion: %d/%d", frame_idx - start + 1, end - start + 1)

    gt_writer.release()
    if yolo_writer is not None:
        yolo_writer.release()

    # Write manual label template
    labels_path = motion_dir / "motion_manual_labels.json"
    with open(labels_path, "w", encoding="utf-8") as f:
        json.dump(manual_labels, f, indent=2, ensure_ascii=False)
    LOGGER.info("Manual label template yazildi: %s (%d girdi)", labels_path, len(manual_labels))

    return {
        "frames": frames_json,
        "moving_ground_truth_available": False,
        "manual_labels_path": str(labels_path),
        "manual_label_count": len(manual_labels),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

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
        "version": "v2",
    }

    if not args.skip_person:
        LOGGER.info("=== PERSON REVIEW V2 BASLIYOR ===")
        results["person"] = run_person_review_v2(
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
        LOGGER.info("=== MOTION REVIEW V2 BASLIYOR ===")
        results["motion"] = run_motion_review_v2(
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

    # old_vs_new: placeholder (karsilastirma icin v1 sonuclari ayri script ile karsilastirilir)
    results["old_vs_new"] = {
        "note": "v1 results comparison: run review_oturum3.py separately and compare review_summary.json files",
        "v1_output_dir": str(Path(args.output_dir).parent / "oturum3_review"),
        "v2_output_dir": str(output_dir),
    }

    result_path = output_dir / "review_summary.json"
    with open(result_path, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2, ensure_ascii=False)
    LOGGER.info("Inceleme v2 tamamlandi: %s", result_path)


if __name__ == "__main__":
    main()
