import logging
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from ..constants import classes


class DetectorYOLO:
    """YOLO tabanlı nesne dedektörü - Görev 1."""

    def __init__(self, config: dict):
        """
        Args:
            config: settings.yaml'den gelen detector config dikt.
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.config = config
        model_path = config.get("model_paths", {}).get("detector", "weights/detector/best.pt")
        self._resolve_model_path(model_path)

        # Detector ayarları
        det_cfg = config.get("detector", {})
        self.imgsz = det_cfg.get("imgsz", 1280)
        self.iou_threshold = det_cfg.get("iou_threshold", 0.7)
        self.max_det = det_cfg.get("max_det", 300)
        self.conf_thresholds = det_cfg.get("conf_thresholds", {
            "Tasit": 0.25, "Insan": 0.20, "UAP": 0.20, "UAI": 0.20,
        })

        # Class ID mapping
        self.class_to_id = {v: k for k, v in classes.items()}
        # Reverse: str class name -> id
        self.name_to_id = classes

        self.logger.info(f"DetectorYOLO başlatıldı. Model: {self.model_path}")
        self.logger.info(f"Conf thresholds: {self.conf_thresholds}")

    def _resolve_model_path(self, rel_path: str):
        """Model yolunu çözümle. Önce similasyon/ içinde ara, yoksa kök dizinde ara."""
        candidates = [
            Path(rel_path),
            Path(".") / rel_path,
            Path("..") / rel_path,
            Path.home() / "NPC-AI" / rel_path,
        ]
        for cand in candidates:
            if cand.exists():
                self.model_path = str(cand.resolve())
                self.logger.info(f"Model bulundu: {self.model_path}")
                break
        else:
            # Fallback: config'deki yolu olduğu gibi kullan
            self.model_path = rel_path
            self.logger.warning(f"Model {rel_path} bulunamadı, olduğu gibi kullanılıyor.")

        # Modeli yükle
        if not Path(self.model_path).exists():
            raise FileNotFoundError(
                f"YOLO model ağırlığı bulunamadı: {self.model_path}\n"
                f"Lütfen best.pt dosyasını similasyon/weights/detector/ altına kopyalayın."
            )
        self.model = YOLO(self.model_path)

    def detect(self, image: np.ndarray) -> list:
        """Tek bir frame üzerinde nesne tespiti yapar.

        Args:
            image: BGR formatında numpy array (H, W, 3)

        Returns:
            Her biri dict olan detection listesi:
            [{
                'cls': int (0-3 arası class id),
                'cls_name': str,
                'conf': float,
                'bbox': (x1, y1, x2, y2) - pixel coordinates,
            }, ...]
        """
        H, W = image.shape[:2]
        results = self.model.predict(
            source=image,
            imgsz=self.imgsz,
            conf=0.01,  # Per-class threshold sonradan uygulanacak
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
                cls_id = int(cls_id)
                cls_name = self.class_to_id.get(cls_id, "Bilinmiyor")

                # Per-class confidence threshold
                min_conf = self.conf_thresholds.get(cls_name, 0.25)
                if conf < min_conf:
                    continue

                # Bbox clipping
                x1, y1, x2, y2 = box
                x1 = max(0, min(float(x1), W - 1))
                y1 = max(0, min(float(y1), H - 1))
                x2 = max(0, min(float(x2), W - 1))
                y2 = max(0, min(float(y2), H - 1))

                # Negatif/ters bbox kontrolü
                if x1 >= x2 or y1 >= y2:
                    continue

                detections.append({
                    'cls': cls_id,
                    'cls_name': cls_name,
                    'conf': float(conf),
                    'bbox': (x1, y1, x2, y2),
                })

        self.logger.debug(f"Tespit edilen nesne: {len(detections)}")
        return detections
