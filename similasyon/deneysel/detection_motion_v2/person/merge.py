"""Kaynak-farkindalikli tespit birlestirme (Faz P4).

YOLO full-frame ve crop tespitlerini class-agnostic IoU ile birlestirir.
"""

import logging
import numpy as np

logger = logging.getLogger(__name__)


def _box_iou(box1, box2) -> float:
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    if x1 >= x2 or y1 >= y2:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    area1 = max(1e-6, (box1[2] - box1[0]) * (box1[3] - box1[1]))
    area2 = max(1e-6, (box2[2] - box2[0]) * (box2[3] - box2[1]))
    return float(inter / (area1 + area2 - inter))


def source_aware_merge(yolo_persons, crop_persons, iou_threshold=0.45):
    """Kaynak-farkindalikli YOLO ve crop tespit birlestirme.

    Rules:
    - If a YOLO and crop detection overlap (IoU > threshold), keep the YOLO bbox
      but use the higher confidence (confidence fusion). Tag source as "merged".
    - Crop-only detections (no YOLO match) are added as new detections.
    - YOLO-only detections are kept as-is.
    - Crop-crop duplicates are removed (keep highest confidence).

    This ensures YOLO TP detections are not lost by being suppressed by crop boxes.
    """
    if not yolo_persons:
        result = []
        for cdet in sorted(crop_persons, key=lambda d: d["conf"], reverse=True):
            is_dup = False
            for existing in result:
                if _box_iou(cdet["bbox"], existing["bbox"]) > iou_threshold:
                    is_dup = True
                    break
            if not is_dup:
                det = dict(cdet)
                det["_source"] = "crop"
                result.append(det)
        return result
    if not crop_persons:
        return list(yolo_persons)

    crop_dets = sorted(crop_persons, key=lambda d: d["conf"], reverse=True)
    yolo_dets = sorted(yolo_persons, key=lambda d: d["conf"], reverse=True)

    used_crop = [False] * len(crop_dets)
    used_yolo = [False] * len(yolo_dets)
    merged = []

    # YOLO-oncelikli birlestirme
    for yi, ydet in enumerate(yolo_dets):
        best_ci = -1
        best_iou = 0.0
        for ci, cdet in enumerate(crop_dets):
            if used_crop[ci]:
                continue
            iou = _box_iou(ydet["bbox"], cdet["bbox"])
            if iou > best_iou:
                best_iou = iou
                best_ci = ci

        if best_iou > iou_threshold and best_ci >= 0:
            # Confidence fusion: higher confidence wins
            yconf = ydet["conf"]
            cconf = crop_dets[best_ci]["conf"]
            merged.append({
                "cls": 1,
                "cls_name": "Insan",
                "bbox": ydet["bbox"],
                "conf": max(yconf, cconf),
                "_source": "merged",
            })
            used_yolo[yi] = True
            used_crop[best_ci] = True
        else:
            used_yolo[yi] = False

    # YOLO-only
    for yi, ydet in enumerate(yolo_dets):
        if not used_yolo[yi]:
            det = dict(ydet)
            det["_source"] = det.get("_source", "yolo")
            merged.append(det)

    # Crop-only (kullanilmamis crop detections)
    for ci, cdet in enumerate(crop_dets):
        if used_crop[ci]:
            continue
        # Crop-crop duplicate kontrolu
        is_dup = False
        for existing in merged:
            if _box_iou(cdet["bbox"], existing["bbox"]) > iou_threshold:
                if cdet["conf"] <= existing["conf"]:
                    is_dup = True
                    break
        if not is_dup:
            det = dict(cdet)
            det["_source"] = "crop"
            merged.append(det)

    return merged


def weighted_box_fusion(yolo_persons, crop_persons, iou_threshold=0.45):
    """Agirlikli ortalama kutu birlestirme.

    When YOLO and crop overlap, create a fused bbox using weighted average
    (by confidence). Keep the higher confidence value.
    """
    if not yolo_persons:
        return list(crop_persons)
    if not crop_persons:
        return list(yolo_persons)

    crop_dets = sorted(crop_persons, key=lambda d: d["conf"], reverse=True)
    yolo_dets = sorted(yolo_persons, key=lambda d: d["conf"], reverse=True)

    used_crop = [False] * len(crop_dets)
    used_yolo = [False] * len(yolo_dets)
    merged = []

    for yi, ydet in enumerate(yolo_dets):
        best_ci = -1
        best_iou = 0.0
        for ci, cdet in enumerate(crop_dets):
            if used_crop[ci]:
                continue
            iou = _box_iou(ydet["bbox"], cdet["bbox"])
            if iou > best_iou:
                best_iou = iou
                best_ci = ci

        if best_iou > iou_threshold and best_ci >= 0:
            cdet = crop_dets[best_ci]
            yconf = ydet["conf"]
            cconf = cdet["conf"]
            total = yconf + cconf
            yw = yconf / total
            cw = cconf / total
            yb = ydet["bbox"]
            cb = cdet["bbox"]
            fused_bbox = (
                yb[0] * yw + cb[0] * cw,
                yb[1] * yw + cb[1] * cw,
                yb[2] * yw + cb[2] * cw,
                yb[3] * yw + cb[3] * cw,
            )
            merged.append({
                "cls": 1,
                "cls_name": "Insan",
                "bbox": fused_bbox,
                "conf": max(yconf, cconf),
                "_source": "weighted_fusion",
            })
            used_yolo[yi] = True
            used_crop[best_ci] = True
        else:
            used_yolo[yi] = False

    for yi, ydet in enumerate(yolo_dets):
        if not used_yolo[yi]:
            det = dict(ydet)
            det["_source"] = det.get("_source", "yolo")
            merged.append(det)

    for ci, cdet in enumerate(crop_dets):
        if used_crop[ci]:
            continue
        is_dup = False
        for existing in merged:
            if _box_iou(cdet["bbox"], existing["bbox"]) > iou_threshold:
                if cdet["conf"] <= existing["conf"]:
                    is_dup = True
                    break
        if not is_dup:
            det = dict(cdet)
            det["_source"] = "crop"
            merged.append(det)

    return merged
