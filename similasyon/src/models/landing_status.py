"""UAP/UAI iniş durumu çözümleyicisi - Görev 1 ek.

Fusion: rule-based + optional ResNet50 classifier.

Kurallar:
- cls 0/1 (Tasit, Insan) → landing_status = "-1" (İniş Alanı Değil)
- cls 2/3 (UAP, UAI):
  1. Rule-based: center_inside + IoU ile engel kontrolü
  2. Obstacle varsa + hard_veto_obstacles → kesin "0"
  3. Obstacle yoksa + classifier varsa → classifier kararı
  4. Obstacle yoksa + classifier yoksa → rule fallback "1"
"""
import logging

import numpy as np

from .landing_classifier import LandingClassifier


class LandingStatusResolver:
    """UAP/UAI iniş durumu çözümleyicisi - rule-based + classifier fusion."""

    def __init__(self, config: dict):
        self.logger = logging.getLogger(self.__class__.__name__)
        landing_cfg = config.get("landing", {})
        self.overlap_threshold = landing_cfg.get("overlap_threshold", 0.2)
        self.near_buffer_px = landing_cfg.get("near_buffer_px", 20)
        self.hard_veto_obstacles = landing_cfg.get("hard_veto_obstacles", True)
        self.classifier_threshold = landing_cfg.get("classifier_threshold", 0.5)

        # Classifier (yüklenemezse available=False, crash yok)
        self.classifier = LandingClassifier(config)

        self.logger.info(
            f"LandingStatusResolver başlatıldı. "
            f"overlap={self.overlap_threshold}, buffer={self.near_buffer_px}px, "
            f"hard_veto={self.hard_veto_obstacles}, "
            f"classifier={'available' if self.classifier.available else 'unavailable'}"
        )

    @staticmethod
    def _iou(box1: tuple, box2: tuple) -> float:
        """İki bbox arasındaki IoU (Intersection over Union)."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        if x1 >= x2 or y1 >= y2:
            return 0.0

        intersection = (x2 - x1) * (y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0

    @staticmethod
    def _center_inside(inner_box: tuple, outer_box: tuple) -> bool:
        """inner_box merkezinin outer_box içinde olup olmadığı."""
        cx = (inner_box[0] + inner_box[2]) / 2.0
        cy = (inner_box[1] + inner_box[3]) / 2.0
        return (outer_box[0] <= cx <= outer_box[2] and
                outer_box[1] <= cy <= outer_box[3])

    @staticmethod
    def _expanded_bbox(bbox: tuple, buffer_px: float, img_hw: tuple) -> tuple:
        """Bbox'ı buffer kadar genişlet."""
        H, W = img_hw
        x1 = max(0, bbox[0] - buffer_px)
        y1 = max(0, bbox[1] - buffer_px)
        x2 = min(W - 1, bbox[2] + buffer_px)
        y2 = min(H - 1, bbox[3] + buffer_px)
        return (x1, y1, x2, y2)

    def resolve(self, detections: list, image_bgr_or_shape) -> list:
        """Tüm detection'lar için landing_status hesapla.

        Args:
            detections: Detection listesi (cls, bbox, moving_status vb.)
            image_bgr_or_shape: numpy array (H,W,3) veya (H, W) tuple/shape.

        Returns:
            Her detection'a 'landing_status' eklenmiş liste.
        """
        # Backward compatibility: tuple shape ise sadece rule-based
        use_classifier = False
        image_bgr = None
        if isinstance(image_bgr_or_shape, np.ndarray) and image_bgr_or_shape.ndim == 3:
            image_bgr = image_bgr_or_shape
            if self.classifier.available:
                use_classifier = True
            img_shape = image_bgr.shape[:2]
        else:
            # tuple/list shape
            img_shape = image_bgr_or_shape[:2]

        H, W = img_shape

        # Non-landing classes
        for det in detections:
            if det['cls'] not in (2, 3):
                det['landing_status'] = "-1"

        uap_uai = [d for d in detections if d['cls'] in (2, 3)]
        # Obstacle: cls 0/1 veya SAHI kaynaklı extra person
        obstacles = [
            d for d in detections
            if d['cls'] in (0, 1) or d.get('source') == 'sahi'
        ]

        if not uap_uai:
            return detections

        for uap in uap_uai:
            uap_bbox = uap['bbox']
            expanded = self._expanded_bbox(uap_bbox, self.near_buffer_px, (H, W))

            # Rule-based obstacle check (center_inside + IoU)
            is_blocked = False
            for obs in obstacles:
                obs_bbox = obs['bbox']
                if self._center_inside(obs_bbox, expanded):
                    is_blocked = True
                    self.logger.debug(
                        f"UAP/UAI blocked: center inside. "
                        f"UAP={uap_bbox}, obs={obs_bbox}"
                    )
                    break
                iou = self._iou(expanded, obs_bbox)
                if iou > self.overlap_threshold:
                    is_blocked = True
                    self.logger.debug(
                        f"UAP/UAI blocked: IoU={iou:.3f}. "
                        f"UAP={uap_bbox}, obs={obs_bbox}"
                    )
                    break

            # Hard veto: obstacle varsa classifier sonucu yok sayılır
            if is_blocked and self.hard_veto_obstacles:
                uap['landing_status'] = "0"  # Inilemez
                continue

            # Obstacle yoksa classifier kararı
            if use_classifier and image_bgr is not None:
                result = self.classifier.predict_crop(image_bgr, uap_bbox)
                if result["available"]:
                    uap['landing_status'] = result["status"]  # "1" veya "0"
                    continue

            # Rule-based fallback (obstacle var ama hard_veto kapalı, ya da classifier yok)
            uap['landing_status'] = "0" if is_blocked else "1"

        return detections
