"""EXP-1: Dondurulmus eval dogruluk tabani (Faz P0).

Bu script:
1. 770 val goruntusu icin dondurulmus manifest olusturur/yukler
2. Full-frame YOLO ile baseline olcum yapar
3. CLAHE acik/kapali A/B testi yapar
4. Confidence threshold sweep yapar (0.05-0.40)
5. COCO-style AP@0.5 ve AP@0.5:0.95 hesaplar
6. Kucuk/orta/buyuk insan recall olcer
7. Sonuclari JSON/CSV olarak kaydeder

Kullanim:
    python eval_baseline.py --config config_experiment.yaml
    python eval_baseline.py --config config_experiment.yaml --limit 100
    python eval_baseline.py --config config_experiment.yaml --threshold-sweep
"""

import argparse
import csv
import hashlib
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch

# Package imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from person.detector_adapter import DetectorAdapter, _box_iou

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    import yaml
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def build_manifest(images_dir: str, labels_dir: str, manifest_path: str) -> list:
    """Dondurulmus eval manifesti olustur/yukle.

    Manifest, goruntu dosya adlarinin ve etiket dosyalarinin deterministik listesidir.
    Video sizintisi icerdigi tum raporlarda isaretlenir.
    """
    if os.path.exists(manifest_path):
        logger.info("Mevcut manifest yukleniyor: %s", manifest_path)
        with open(manifest_path, "r") as f:
            return json.load(f)

    logger.info("Yeni manifest olusturuluyor: %s -> %s", images_dir, manifest_path)
    images = sorted(Path(images_dir).glob("*.jpg")) + \
             sorted(Path(images_dir).glob("*.png"))

    manifest = []
    for img_path in images:
        name = img_path.stem
        label_path = os.path.join(labels_dir, name + ".txt")
        entry = {
            "image": str(img_path),
            "label": label_path,
            "name": name,
            "has_label": os.path.exists(label_path),
        }
        manifest.append(entry)

    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info("Manifest: %d goruntu kaydedildi", len(manifest))
    return manifest


def load_yolo_label(label_path: str, img_w: int, img_h: int) -> list:
    """YOLO format etiket yukle: normalized [cls, cx, cy, w, h] -> [cls, x1, y1, x2, y2]."""
    if not os.path.exists(label_path):
        return []
    boxes = []
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls = int(parts[0])
            cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            x1 = (cx - w / 2) * img_w
            y1 = (cy - h / 2) * img_h
            x2 = (cx + w / 2) * img_w
            y2 = (cy + h / 2) * img_h
            boxes.append([cls, x1, y1, x2, y2])
    return boxes


def compute_ap_coco(gt_boxes_list, pred_boxes_list, iou_threshold=0.5):
    """COCO-style AP hesapla (all-point interpolation).

    Args:
        gt_boxes_list: Her goruntu icin [[cls, x1, y1, x2, y2], ...]
        pred_boxes_list: Her goruntu icin [[cls, x1, y1, x2, y2, conf], ...]
        iou_threshold: AP hesaplanan IoU esigi

    Returns:
        (ap, precision, recall, tp, fp, fn)
    """
    all_preds = []
    n_gt = 0

    for img_idx, (gts, preds) in enumerate(zip(gt_boxes_list, pred_boxes_list)):
        for pred in preds:
            cls, x1, y1, x2, y2, conf = pred[0], pred[1], pred[2], pred[3], pred[4], pred[5]
            all_preds.append((conf, cls, x1, y1, x2, y2, img_idx))
        n_gt += len(gts)

    if n_gt == 0:
        return 0.0, 0.0, 0.0, 0, 0, 0

    all_preds.sort(key=lambda x: -x[0])

    tp = np.zeros(len(all_preds))
    fp = np.zeros(len(all_preds))
    gt_matched = [set() for _ in gt_boxes_list]

    for i, (conf, cls, x1, y1, x2, y2, img_idx) in enumerate(all_preds):
        best_iou = 0
        best_gt_idx = -1
        gts = gt_boxes_list[img_idx]
        for j, gt in enumerate(gts):
            if gt[0] != cls:
                continue
            if j in gt_matched[img_idx]:
                continue
            iou = _box_iou((x1, y1, x2, y2), (gt[1], gt[2], gt[3], gt[4]))
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = j

        if best_iou >= iou_threshold and best_gt_idx >= 0:
            tp[i] = 1
            gt_matched[img_idx].add(best_gt_idx)
        else:
            fp[i] = 1

    cum_tp = np.cumsum(tp)
    cum_fp = np.cumsum(fp)
    recall = cum_tp / n_gt
    precision = cum_tp / (cum_tp + cum_fp + 1e-6)

    # All-point interpolation AP
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    indices = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[indices + 1] - mrec[indices]) * mpre[indices + 1])

    final_tp = int(cum_tp[-1])
    final_fp = int(cum_fp[-1])
    final_fn = n_gt - final_tp
    final_precision = final_tp / max(1, final_tp + final_fp)
    final_recall = final_tp / max(1, n_gt)

    return float(ap), final_precision, final_recall, final_tp, final_fp, final_fn


def compute_ap_per_class(gt_boxes_list, pred_boxes_list, class_ids, iou_threshold=0.5):
    """Sinif bazinda AP hesapla."""
    results = {}
    for cls in class_ids:
        gt_filtered = [[g for g in gts if g[0] == cls] for gts in gt_boxes_list]
        pred_filtered = [[p for p in preds if p[0] == cls] for preds in pred_boxes_list]
        ap, prec, rec, tp, fp, fn = compute_ap_coco(gt_filtered, pred_filtered, iou_threshold)
        results[cls] = {"ap": ap, "precision": prec, "recall": rec, "tp": tp, "fp": fp, "fn": fn}
    return results


def compute_ap_50_95(gt_boxes_list, pred_boxes_list, class_ids):
    """AP@0.5:0.95 hesapla (10 IoU threshold ortalamasi)."""
    iou_thresholds = np.arange(0.5, 1.0, 0.05)
    all_aps = []
    for iou_th in iou_thresholds:
        results = compute_ap_per_class(gt_boxes_list, pred_boxes_list, class_ids, iou_th)
        class_aps = [r["ap"] for r in results.values()]
        all_aps.append(np.mean(class_aps) if class_aps else 0.0)
    return float(np.mean(all_aps))


def size_categorized_recall(gt_boxes_list, pred_boxes_list, cls_id=1,
                             small_max=32, medium_max=96):
    """Kucuk/orta/buyuk insan recall hesapla (COCO kategorileri)."""
    sizes = {"small": [0, small_max], "medium": [small_max, medium_max], "large": [medium_max, float("inf")]}
    results = {}

    for size_name, (min_h, max_h) in sizes.items():
        gt_count = 0
        matched = 0
        for img_idx, (gts, preds) in enumerate(zip(gt_boxes_list, pred_boxes_list)):
            cls_preds = [p for p in preds if p[0] == cls_id]
            for gt in gts:
                if gt[0] != cls_id:
                    continue
                h = gt[4] - gt[2]
                if h < min_h or h >= max_h:
                    continue
                gt_count += 1
                best_iou = 0
                for pred in cls_preds:
                    iou = _box_iou((pred[1], pred[2], pred[3], pred[4]), (gt[1], gt[2], gt[3], gt[4]))
                    best_iou = max(best_iou, iou)
                if best_iou >= 0.5:
                    matched += 1
        results[size_name] = {
            "count": gt_count,
            "matched": matched,
            "recall": matched / max(1, gt_count),
        }
    return results


def apply_clahe(image_bgr, clip_limit=2.0, tile_grid=8, split_rows=2, split_cols=3,
                brightness_threshold=50):
    """Tiled CLAHE uygula (ImagePreprocessor ile ayni mantik)."""
    h, w = image_bgr.shape[:2]
    row_breaks = [0] + [h * i // split_rows for i in range(1, split_rows)] + [h]
    col_breaks = [0] + [w * j // split_cols for j in range(1, split_cols)] + [w]

    parts = []
    for ri in range(split_rows):
        for cj in range(split_cols):
            tile = image_bgr[row_breaks[ri]:row_breaks[ri+1],
                             col_breaks[cj]:col_breaks[cj+1]]
            gray = cv2.cvtColor(tile, cv2.COLOR_BGR2GRAY)
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
            brightness = float(np.argmax(hist))
            if brightness > brightness_threshold:
                parts.append(tile)
            else:
                lab = cv2.cvtColor(tile, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab)
                clahe = cv2.createCLAHE(clipLimit=clip_limit,
                                         tileGridSize=(tile_grid, tile_grid))
                l2 = clahe.apply(l)
                parts.append(cv2.cvtColor(cv2.merge((l2, a, b)), cv2.COLOR_LAB2BGR))

    rows_merged = [np.hstack(parts[ri*split_cols:(ri+1)*split_cols]) for ri in range(split_rows)]
    return np.vstack(rows_merged)


def run_eval(detector, manifest, config, use_clahe=False, limit=None):
    """Eval seti uzerinde detection calistir ve metrikleri hesapla."""
    pre_cfg = config.get("preprocessing", {})
    gt_boxes_list = []
    pred_boxes_list = []
    latencies = []

    items = manifest[:limit] if limit else manifest

    for i, entry in enumerate(items):
        img_path = entry["image"]
        image = cv2.imread(img_path)
        if image is None:
            continue
        H, W = image.shape[:2]

        if use_clahe and pre_cfg.get("use_clahe", False):
            image = apply_clahe(
                image,
                clip_limit=pre_cfg.get("clahe_clip_limit", 2.0),
                tile_grid=pre_cfg.get("clahe_tile_grid_size", 8),
                split_rows=pre_cfg.get("split_rows", 2),
                split_cols=pre_cfg.get("split_cols", 3),
                brightness_threshold=pre_cfg.get("brightness_threshold", 50),
            )

        gts = load_yolo_label(entry["label"], W, H)
        gt_boxes_list.append(gts)

        t0 = time.time()
        detections = detector.detect_full_frame(image)
        latency = time.time() - t0
        latencies.append(latency)

        preds = []
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            preds.append([det["cls"], x1, y1, x2, y2, det["conf"]])
        pred_boxes_list.append(preds)

        if (i + 1) % 50 == 0:
            logger.info("  %d/%d goruntu islendi", i + 1, len(items))

    class_ids = [0, 1, 2, 3]
    class_names = {0: "Tasit", 1: "Insan", 2: "UAP", 3: "UAI"}

    ap50 = compute_ap_per_class(gt_boxes_list, pred_boxes_list, class_ids, 0.5)
    ap_5095 = compute_ap_50_95(gt_boxes_list, pred_boxes_list, class_ids)
    person_recall = size_categorized_recall(gt_boxes_list, pred_boxes_list, cls_id=1)

    vram = detector.get_vram_usage_mb()

    results = {
        "timestamp": datetime.now().isoformat(),
        "n_images": len(items),
        "ap50": {class_names[k]: v for k, v in ap50.items()},
        "ap_5095": ap_5095,
        "person_size_recall": person_recall,
        "latency_mean": float(np.mean(latencies)) if latencies else 0,
        "latency_std": float(np.std(latencies)) if latencies else 0,
        "vram_mb": vram,
        "use_clahe": use_clahe,
        "imgsz": config.get("detector", {}).get("imgsz", 1280),
    }

    logger.info("--- Sonuclar ---")
    for cls_name in ["Tasit", "Insan", "UAP", "UAI"]:
        if cls_name in results["ap50"]:
            r = results["ap50"][cls_name]
            logger.info("  %s AP@0.5: %.4f  P: %.3f  R: %.3f  TP:%d FP:%d FN:%d",
                        cls_name, r["ap"], r["precision"], r["recall"],
                        r["tp"], r["fp"], r["fn"])
    logger.info("  AP@0.5:0.95: %.4f", results["ap_5095"])
    logger.info("  Kucuk insan recall: %.3f", person_recall["small"]["recall"])
    logger.info("  Orta insan recall: %.3f", person_recall["medium"]["recall"])
    logger.info("  Buyuk insan recall: %.3f", person_recall["large"]["recall"])
    logger.info("  Latency: %.4f ± %.4f sn", results["latency_mean"], results["latency_std"])
    logger.info("  VRAM: %.1f MB", vram)

    return results


def run_threshold_sweep(detector, manifest, config, limit=100):
    """Insan confidence threshold sweep (0.05-0.40)."""
    thresholds = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
    results = {}

    original_conf = dict(detector.conf_thresholds)

    for th in thresholds:
        detector.conf_thresholds["Insan"] = th
        logger.info("Threshold sweep: Insan conf=%.2f", th)
        r = run_eval(detector, manifest, config, use_clahe=False, limit=limit)
        results[f"conf_{th:.2f}"] = {
            "threshold": th,
            "insan_ap50": r["ap50"].get("Insan", {}).get("ap", 0),
            "insan_precision": r["ap50"].get("Insan", {}).get("precision", 0),
            "insan_recall": r["ap50"].get("Insan", {}).get("recall", 0),
            "insan_tp": r["ap50"].get("Insan", {}).get("tp", 0),
            "insan_fp": r["ap50"].get("Insan", {}).get("fp", 0),
            "insan_fn": r["ap50"].get("Insan", {}).get("fn", 0),
            "latency_mean": r["latency_mean"],
        }

    detector.conf_thresholds = original_conf
    return results


def save_results(results, output_dir, filename):
    """Sonuclari JSON olarak kaydet."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    with open(path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info("Sonuclar kaydedildi: %s", path)


def save_csv(results, output_dir, filename):
    """Sonuclari CSV olarak kaydet."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["experiment", "class", "ap50", "precision", "recall", "tp", "fp", "fn",
                         "latency_mean", "vram_mb"])
        for exp_name, exp_data in results.items():
            if isinstance(exp_data, dict) and "ap50" in exp_data:
                for cls_name, metrics in exp_data["ap50"].items():
                    writer.writerow([
                        exp_name, cls_name,
                        metrics.get("ap", 0), metrics.get("precision", 0),
                        metrics.get("recall", 0), metrics.get("tp", 0),
                        metrics.get("fp", 0), metrics.get("fn", 0),
                        exp_data.get("latency_mean", 0), exp_data.get("vram_mb", 0),
                    ])
    logger.info("CSV kaydedildi: %s", path)


def main():
    parser = argparse.ArgumentParser(description="EXP-1: Dondurulmus eval dogruluk tabani")
    parser.add_argument("--config", default="config_experiment.yaml")
    parser.add_argument("--limit", type=int, default=None, help="Goruntu sayisi limiti")
    parser.add_argument("--threshold-sweep", action="store_true", help="Threshold sweep yap")
    parser.add_argument("--output-dir", default="results")
    args = parser.parse_args()

    config = load_config(args.config)
    eval_cfg = config.get("eval", {})

    manifest = build_manifest(
        eval_cfg.get("images_dir", "/home/nurullah/Desktop/FINAL_YOLO_DATASET_V3/images/val"),
        eval_cfg.get("labels_dir", "/home/nurullah/Desktop/FINAL_YOLO_DATASET_V3/labels/val"),
        eval_cfg.get("manifest_path", "results/eval_manifest.json"),
    )

    logger.info("=== EXP-1: Dondurulmus eval dogruluk tabani ===")
    logger.info("UYARI: Bu eval seti video sizintisi icerebilir. Sonuclar")
    logger.info("yalnizca esit kosullu A/B kiyasi olarak yorumlanmalidir.")

    detector = DetectorAdapter(config)

    all_results = {}

    # A: YOLO baseline (CLAHE kapali)
    logger.info("\n--- A: YOLO baseline (CLAHE kapali) ---")
    all_results["A_yolo_baseline"] = run_eval(detector, manifest, config, use_clahe=False, limit=args.limit)

    # B: YOLO + CLAHE
    pre_cfg = config.get("preprocessing", {})
    if pre_cfg.get("use_clahe", False) or True:
        logger.info("\n--- B: YOLO + CLAHE ---")
        config_clahe = dict(config)
        config_clahe["preprocessing"] = {**pre_cfg, "use_clahe": True}
        all_results["B_yolo_clahe"] = run_eval(detector, manifest, config_clahe, use_clahe=True, limit=args.limit)

    # Threshold sweep
    if args.threshold_sweep:
        logger.info("\n--- Threshold Sweep (Insan 0.05-0.40) ---")
        sweep_results = run_threshold_sweep(detector, manifest, config, limit=args.limit or 100)
        all_results["threshold_sweep"] = sweep_results

    # Donanim bilgisi
    all_results["hardware"] = {
        "cuda_available": torch.cuda.is_available(),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "torch_version": torch.__version__,
    }

    # Model hash
    model_path = detector.model_path
    if os.path.exists(model_path):
        with open(model_path, "rb") as f:
            model_hash = hashlib.md5(f.read()).hexdigest()[:16]
        all_results["model_hash"] = model_hash

    save_results(all_results, args.output_dir, "exp1_eval_baseline.json")
    save_csv(all_results, args.output_dir, "exp1_eval_baseline.csv")


if __name__ == "__main__":
    main()
