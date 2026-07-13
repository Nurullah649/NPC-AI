"""EXP-3/4: Insan tespiti benchmark matrisi (Faz P3/P4).

Benchmark matrisi:
| Deney | Full YOLO | Ek model/crop | Tracker |
|---|---|---|---|
| A | 1280 | Yok | Yok |
| B | 1536 | Yok | Yok |
| C | 1280 | Full SAHI | Yok |
| D | 1280 | 1 crop/frame | Yok |
| E | 1280 | 2 crop/frame | Yok |
| F | 1280 | 1 crop/frame | Acik |
| G | 1280 | 2 crop/frame | Acik |

Her deney icin:
- Insan AP@0.5, AP@0.5:0.95
- Kucuk/orta/buyuk recall
- Precision ve recall
- Goruntu basina gecikme
- VRAM tepe kullanimi

Kullanim:
    python -m person.benchmark_person --config config_experiment.yaml
    python -m person.benchmark_person --config config_experiment.yaml --limit 100 --modes A,D,E,F,G
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from person.detector_adapter import DetectorAdapter, _box_iou
from person.tile_scheduler import TileScheduler, CropType
from person.tracker import PersonTrackerV2
from person.merge import source_aware_merge
from motion.camera_motion import CameraMotionEstimator, CameraTransform
from eval_baseline import (
    load_config, build_manifest, load_yolo_label,
    compute_ap_per_class, compute_ap_50_95, size_categorized_recall,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def run_experiment_A(detector, manifest, limit=None):
    """A: Full YOLO 1280, crop yok, tracker yok."""
    gt_list, pred_list, latencies = _run_detection_only(detector, manifest, limit)
    return _compute_metrics(gt_list, pred_list, latencies, detector, "A_yolo_1280")


def run_experiment_A2(detector, manifest, limit=None):
    """A2: Full YOLO + global NMS on persons only, no crops, no tracker."""
    gt_list, pred_list, latencies = [], [], []
    items = manifest[:limit] if limit else manifest
    for i, entry in enumerate(items):
        image = cv2.imread(entry["image"])
        if image is None:
            continue
        H, W = image.shape[:2]
        t0 = time.time()
        dets = detector.detect_full_frame(image)
        persons = [d for d in dets if d["cls"] == 1]
        others = [d for d in dets if d["cls"] != 1]
        merged_persons = detector.global_nms(persons, 0.45)
        all_dets = others + merged_persons
        latencies.append(time.time() - t0)
        gt_list.append(load_yolo_label(entry["label"], W, H))
        preds = [[d["cls"], d["bbox"][0], d["bbox"][1], d["bbox"][2], d["bbox"][3], d["conf"]]
                 for d in all_dets]
        pred_list.append(preds)
        if (i + 1) % 50 == 0:
            logger.info("  A2: %d/%d", i + 1, len(items))
    return _compute_metrics(gt_list, pred_list, latencies, detector, "A2_yolo_nms")


def run_experiment_B(detector, manifest, config, limit=None):
    """B: Full YOLO 1536, crop yok, tracker yok."""
    detector.imgsz = 1536
    gt_list, pred_list, latencies = _run_detection_only(detector, manifest, limit)
    detector.imgsz = 1280
    return _compute_metrics(gt_list, pred_list, latencies, detector, "B_yolo_1536")


def run_experiment_C(detector, manifest, limit=None):
    """C: Full YOLO 1280 + Full SAHI, tracker yok."""
    gt_list, pred_list, latencies = [], [], []

    items = manifest[:limit] if limit else manifest
    for i, entry in enumerate(items):
        image = cv2.imread(entry["image"])
        if image is None:
            continue
        H, W = image.shape[:2]

        t0 = time.time()
        yolo_dets = detector.detect_full_frame(image)

        try:
            from sahi.predict import get_sliced_prediction
            from sahi import AutoDetectionModel
            sahi_model = AutoDetectionModel.from_pretrained(
                model_type="ultralytics",
                model_path=detector.model_path,
                confidence_threshold=0.15,
                device="cuda" if torch.cuda.is_available() else "cpu",
            )
            result = get_sliced_prediction(image, sahi_model,
                                            slice_height=960, slice_width=960,
                                            overlap_height_ratio=0.15,
                                            overlap_width_ratio=0.15)
            sahi_persons = []
            for obj in result.object_prediction_list:
                if int(obj.category.id) != 1:
                    continue
                bbox = (obj.bbox.minx, obj.bbox.miny, obj.bbox.maxx, obj.bbox.maxy)
                sahi_persons.append({"cls": 1, "cls_name": "Insan",
                                     "conf": float(obj.score.value), "bbox": bbox})
        except Exception as e:
            logger.warning("SAHI hatasi: %s", e)
            sahi_persons = []

        yolo_persons = [d for d in yolo_dets if d["cls"] == 1]
        yolo_others = [d for d in yolo_dets if d["cls"] != 1]
        merged = detector.global_nms(yolo_persons + sahi_persons, 0.45)
        all_dets = yolo_others + merged

        latency = time.time() - t0
        latencies.append(latency)

        gt_list.append(load_yolo_label(entry["label"], W, H))
        preds = [[d["cls"], d["bbox"][0], d["bbox"][1], d["bbox"][2], d["bbox"][3], d["conf"]]
                 for d in all_dets]
        pred_list.append(preds)

        if (i + 1) % 50 == 0:
            logger.info("  C: %d/%d", i + 1, len(items))

    return _compute_metrics(gt_list, pred_list, latencies, detector, "C_full_sahi")


def run_experiment_D_or_E(detector, manifest, config, max_crops, use_tracker, label, limit=None):
    """D/E/F/G: Temporal micro-tiling + optional tracker."""
    import copy
    config_copy = copy.deepcopy(config)
    config_copy.setdefault("person_detection", {})["max_crops_per_frame"] = max_crops
    scheduler = TileScheduler(config_copy)
    tracker = PersonTrackerV2(config) if use_tracker else None
    camera_estimator = CameraMotionEstimator(config) if use_tracker else None
    prev_gray = None

    gt_list, pred_list, latencies = [], [], []
    items = manifest[:limit] if limit else manifest

    for i, entry in enumerate(items):
        image = cv2.imread(entry["image"])
        if image is None:
            continue
        H, W = image.shape[:2]

        t0 = time.time()
        yolo_dets = detector.detect_full_frame(image)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        cam_transform = None
        if camera_estimator and prev_gray is not None:
            cam_transform = camera_estimator.estimate(prev_gray, gray, yolo_dets)
        prev_gray = gray

        active_tracks = tracker.get_active_track_bboxes() if tracker else []
        lost_rois = tracker.get_lost_rois() if tracker else []
        crops = scheduler.schedule(i, W, H, active_tracks, lost_rois)

        crop_dets = detector.detect_crops_batched(image, [c.bbox for c in crops], person_only=True)

        yolo_persons = [d for d in yolo_dets if d["cls"] == 1]
        yolo_others = [d for d in yolo_dets if d["cls"] != 1]
        all_persons = source_aware_merge(yolo_persons, crop_dets, 0.45)

        if tracker:
            all_dets = yolo_others + all_persons
            all_dets = tracker.update(all_dets, i, cam_transform)
        else:
            all_dets = yolo_others + all_persons

        latency = time.time() - t0
        latencies.append(latency)

        gt_list.append(load_yolo_label(entry["label"], W, H))
        preds = [[d["cls"], d["bbox"][0], d["bbox"][1], d["bbox"][2], d["bbox"][3], d["conf"]]
                 for d in all_dets if d.get("cls") is not None]
        pred_list.append(preds)

        if (i + 1) % 50 == 0:
            logger.info("  %s: %d/%d", label, i + 1, len(items))

    stats = scheduler.get_stats()
    result = _compute_metrics(gt_list, pred_list, latencies, detector, label)
    result["scheduler_stats"] = stats
    return result


def _run_detection_only(detector, manifest, limit=None):
    gt_list, pred_list, latencies = [], [], []
    items = manifest[:limit] if limit else manifest
    for i, entry in enumerate(items):
        image = cv2.imread(entry["image"])
        if image is None:
            continue
        H, W = image.shape[:2]
        t0 = time.time()
        dets = detector.detect_full_frame(image)
        latencies.append(time.time() - t0)
        gt_list.append(load_yolo_label(entry["label"], W, H))
        preds = [[d["cls"], d["bbox"][0], d["bbox"][1], d["bbox"][2], d["bbox"][3], d["conf"]]
                 for d in dets]
        pred_list.append(preds)
        if (i + 1) % 50 == 0:
            logger.info("  %d/%d", i + 1, len(items))
    return gt_list, pred_list, latencies


def _compute_metrics(gt_list, pred_list, latencies, detector, label):
    class_ids = [0, 1, 2, 3]
    class_names = {0: "Tasit", 1: "Insan", 2: "UAP", 3: "UAI"}
    ap50 = compute_ap_per_class(gt_list, pred_list, class_ids, 0.5)
    ap_5095 = compute_ap_50_95(gt_list, pred_list, class_ids)
    person_recall = size_categorized_recall(gt_list, pred_list, cls_id=1)

    result = {
        "label": label,
        "n_images": len(gt_list),
        "ap50": {class_names[k]: v for k, v in ap50.items()},
        "ap_5095": ap_5095,
        "person_size_recall": person_recall,
        "latency_mean": float(np.mean(latencies)) if latencies else 0,
        "latency_std": float(np.std(latencies)) if latencies else 0,
        "vram_mb": detector.get_vram_usage_mb(),
        "timestamp": datetime.now().isoformat(),
    }

    logger.info("--- %s ---", label)
    for cls_name in ["Insan"]:
        if cls_name in result["ap50"]:
            r = result["ap50"][cls_name]
            logger.info("  %s AP@0.5: %.4f  R: %.3f  TP:%d FP:%d FN:%d",
                        cls_name, r["ap"], r["recall"], r["tp"], r["fp"], r["fn"])
    logger.info("  Latency: %.4f sn, VRAM: %.1f MB",
                result["latency_mean"], result["vram_mb"])
    return result


def main():
    parser = argparse.ArgumentParser(description="EXP-3/4: Insan tespiti benchmark")
    parser.add_argument("--config", default="config_experiment.yaml")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--modes", default="A,D,E,F,G",
                        help="Calistirilacak deneyler (virgulle ayrilmis)")
    args = parser.parse_args()

    config = load_config(args.config)
    eval_cfg = config.get("eval", {})
    manifest = build_manifest(
        eval_cfg.get("images_dir", "/home/nurullah/Desktop/FINAL_YOLO_DATASET_V3/images/val"),
        eval_cfg.get("labels_dir", "/home/nurullah/Desktop/FINAL_YOLO_DATASET_V3/labels/val"),
        eval_cfg.get("manifest_path", "results/eval_manifest.json"),
    )

    detector = DetectorAdapter(config)
    modes = args.modes.split(",")
    all_results = {}

    mode_runners = {
        "A": lambda: run_experiment_A(detector, manifest, args.limit),
        "A2": lambda: run_experiment_A2(detector, manifest, args.limit),
        "B": lambda: run_experiment_B(detector, manifest, config, args.limit),
        "C": lambda: run_experiment_C(detector, manifest, args.limit),
        "D": lambda: run_experiment_D_or_E(detector, manifest, config, 1, False, "D_1crop_no_tracker", args.limit),
        "E": lambda: run_experiment_D_or_E(detector, manifest, config, 2, False, "E_2crop_no_tracker", args.limit),
        "F": lambda: run_experiment_D_or_E(detector, manifest, config, 1, True, "F_1crop_tracker", args.limit),
        "G": lambda: run_experiment_D_or_E(detector, manifest, config, 2, True, "G_2crop_tracker", args.limit),
    }

    for mode in modes:
        mode = mode.strip()
        if mode in mode_runners:
            logger.info("\n=== Deney %s ===", mode)
            all_results[f"exp_{mode}"] = mode_runners[mode]()

    os.makedirs(args.output_dir, exist_ok=True)
    path = os.path.join(args.output_dir, "exp3_4_person_benchmark.json")
    with open(path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    logger.info("Sonuclar kaydedildi: %s", path)


if __name__ == "__main__":
    main()
