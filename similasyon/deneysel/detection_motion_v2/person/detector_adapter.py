"""Salt-okunur YOLO detector adapter for experimental work.

Bu modul mevcut weights/detector/best.pt modelini salt-okunur olarak sarmalar.
- Full-frame inference
- Batched crop inference (tile scheduler ile birlikte kullanilir)
- Per-class confidence threshold (conf=0.01 raw + post-filter pattern)
- Model degistirilmez, uzerine yazilmaz

Uretim DetectorYOLO ile ayni cikti formatini kullanir:
    {"cls": int, "cls_name": str, "conf": float, "bbox": (x1, y1, x2, y2)}
"""

import logging
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

logger = logging.getLogger(__name__)

CLASS_NAMES = {0: "Tasit", 1: "Insan", 2: "UAP", 3: "UAI"}


class DetectorAdapter:
    """Mevcut best.pt modeli icin salt-okunur adapter.

    Args:
        config: Deneysel config dict. Su anahtarlari okur:
            - model_paths.detector: model yolu
            - detector.imgsz: inference giris boyutu
            - detector.conf_thresholds: per-class confidence esikleri
            - detector.iou_threshold: NMS IoU esigi
            - detector.max_det: maksimum tespit sayisi
    """

    def __init__(self, config: dict):
        self.config = config
        model_path = config.get("model_paths", {}).get("detector", "../../weights/detector/best.pt")
        self.model_path = self._resolve_model_path(model_path)

        det_cfg = config.get("detector", {})
        self.imgsz = int(det_cfg.get("imgsz", 1280))
        self.iou_threshold = float(det_cfg.get("iou_threshold", 0.7))
        self.max_det = int(det_cfg.get("max_det", 300))
        self.conf_thresholds = det_cfg.get("conf_thresholds", {
            "Tasit": 0.25, "Insan": 0.20, "UAP": 0.20, "UAI": 0.20,
        })

        pd_cfg = config.get("person_detection", {})
        self.crop_confidence_threshold = float(pd_cfg.get("confidence_threshold", 0.15))

        self.model = YOLO(self.model_path)
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(
            "DetectorAdapter baslatildi: model=%s, imgsz=%d, device=%s",
            self.model_path, self.imgsz, self._device,
        )

    @staticmethod
    def _resolve_model_path(rel_path: str) -> str:
        """Model yolunu cozumle. Uzerine yazma yapmaz."""
        candidates = [
            Path(rel_path),
            Path(__file__).parent / rel_path,
            Path(__file__).parent.parent / rel_path,
            Path(__file__).parent.parent.parent / rel_path,
            Path(__file__).parent.parent.parent.parent / rel_path,
            Path.home() / "NPC-AI" / "similasyon" / rel_path,
        ]
        for cand in candidates:
            if cand.exists():
                return str(cand.resolve())
        if not Path(rel_path).exists():
            raise FileNotFoundError(f"Model bulunamadi: {rel_path}")
        return str(Path(rel_path).resolve())

    def _format_detection(self, box, conf, cls_id, W, H, offset=(0.0, 0.0),
                          conf_threshold: float = None):
        """Tek bir tespiti dict formatina cevir."""
        cls_id = int(cls_id)
        cls_name = CLASS_NAMES.get(cls_id, "Bilinmiyor")
        min_conf = conf_threshold if conf_threshold is not None else \
            self.conf_thresholds.get(cls_name, 0.25)
        if conf < min_conf:
            return None

        x1, y1, x2, y2 = box
        x1 = max(0.0, min(float(x1) + offset[0], W - 1))
        y1 = max(0.0, min(float(y1) + offset[1], H - 1))
        x2 = max(0.0, min(float(x2) + offset[0], W - 1))
        y2 = max(0.0, min(float(y2) + offset[1], H - 1))

        if x1 >= x2 or y1 >= y2:
            return None

        return {
            "cls": cls_id,
            "cls_name": cls_name,
            "conf": float(conf),
            "bbox": (x1, y1, x2, y2),
        }

    def detect_full_frame(self, image: np.ndarray, person_only: bool = False) -> list:
        """Full-frame YOLO inference.

        Args:
            image: BGR numpy array (H, W, 3)
            person_only: True ise yalniz Insan (cls=1) sonuclarini dondur

        Returns:
            Detection dict listesi (full image koordinatlarinda)
        """
        H, W = image.shape[:2]
        results = self.model.predict(
            source=image,
            imgsz=self.imgsz,
            conf=0.01,
            iou=self.iou_threshold,
            max_det=self.max_det,
            verbose=False,
        )

        detections = []
        for result in results:
            if result.boxes is None:
                continue
            boxes = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            clss = result.boxes.cls.cpu().numpy()

            for box, conf, cls_id in zip(boxes, confs, clss):
                if person_only and int(cls_id) != 1:
                    continue
                det = self._format_detection(box, conf, cls_id, W, H)
                if det is not None:
                    detections.append(det)

        return detections

    def detect_crops_batched(
        self,
        image: np.ndarray,
        crops: list,
        person_only: bool = True,
        crop_confidence_threshold: float = None,
    ) -> list:
        """Birden fazla crop'u tek batch call'da isle.

        Args:
            image: Tam BGR goruntu (H, W, 3)
            crops: (x1, y1, x2, y2) tuple listesi - orijinal goruntu koordinatlarinda
            person_only: True ise yalniz Insan sonuclarini dondur
            crop_confidence_threshold: Crop inference icin confidence esigi (None ise self.crop_confidence_threshold kullanilir)

        Returns:
            Detection dict listesi (orijinal goruntu koordinatlarinda remap edilmis)
        """
        if not crops:
            return []

        H, W = image.shape[:2]
        crop_imgs = []
        offsets = []

        for (cx1, cy1, cx2, cy2) in crops:
            cx1 = max(0, int(cx1))
            cy1 = max(0, int(cy1))
            cx2 = min(W, int(cx2))
            cy2 = min(H, int(cy2))
            if cx1 >= cx2 or cy1 >= cy2:
                continue
            crop = image[cy1:cy2, cx1:cx2]
            crop_imgs.append(crop)
            offsets.append((float(cx1), float(cy1)))

        if not crop_imgs:
            return []

        results = self.model.predict(
            source=crop_imgs,
            imgsz=self.config.get("person_detection", {}).get("imgsz", 640),
            conf=0.01,
            iou=self.iou_threshold,
            max_det=self.max_det,
            verbose=False,
        )

        effective_threshold = crop_confidence_threshold if crop_confidence_threshold is not None \
            else self.crop_confidence_threshold
        detections = []
        for result, (ox, oy) in zip(results, offsets):
            if result.boxes is None:
                continue
            crop_h, crop_w = result.orig_shape
            crop_W = crop_w
            crop_H = crop_h
            boxes = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            clss = result.boxes.cls.cpu().numpy()

            for box, conf, cls_id in zip(boxes, confs, clss):
                if person_only and int(cls_id) != 1:
                    continue
                det = self._format_detection(
                    box, conf, cls_id,
                    W, H,
                    offset=(ox, oy),
                    conf_threshold=effective_threshold,
                )
                if det is not None:
                    det["_source"] = "crop"
                    detections.append(det)

        return detections

    @staticmethod
    def global_nms(detections: list, iou_threshold: float = 0.45) -> list:
        """Tum detection'lar uzerinde class-aware global NMS.

        Ayni siniftan ovelapping kutulari bastirir.
        En yuksek confidence olan korunur.
        """
        if not detections:
            return []

        by_class = {}
        for det in detections:
            cls = det["cls"]
            if cls not in by_class:
                by_class[cls] = []
            by_class[cls].append(det)

        result = []
        for cls, dets in by_class.items():
            dets_sorted = sorted(dets, key=lambda d: d["conf"], reverse=True)
            kept = []
            for det in dets_sorted:
                is_dup = False
                for k in kept:
                    iou = _box_iou(det["bbox"], k["bbox"])
                    if iou > iou_threshold:
                        is_dup = True
                        break
                if not is_dup:
                    kept.append(det)
            result.extend(kept)

        return result

    def get_vram_usage_mb(self) -> float:
        """GPU VRAM kullanimini MB olarak dondur (CUDA varsa)."""
        if not torch.cuda.is_available():
            return 0.0
        try:
            allocated = torch.cuda.memory_allocated() / 1024 / 1024
            reserved = torch.cuda.memory_reserved() / 1024 / 1024
            return max(allocated, reserved)
        except Exception:
            return 0.0


def _box_iou(box1, box2) -> float:
    """Iki kutu arasi IoU hesapla."""
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
