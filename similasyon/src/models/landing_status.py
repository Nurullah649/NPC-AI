import logging

import numpy as np


class LandingStatusResolver:
    """UAP/UAI iniş durumu çözümleyicisi - Görev 1 ek.

    UAP ve UAI sınıfları için:
    - Alan üzerinde/çok yakınında insan, taşıt veya başka tespit varsa → Inilemez
    - Engel yoksa → Inilebilir
    - Taşıt/İnsan için → İniş Alanı Değil
    """

    def __init__(self, config: dict):
        self.logger = logging.getLogger(self.__class__.__name__)
        landing_cfg = config.get("landing", {})
        self.overlap_threshold = landing_cfg.get("overlap_threshold", 0.2)
        self.near_buffer_px = landing_cfg.get("near_buffer_px", 20)

        self.logger.info(
            f"LandingStatusResolver başlatıldı. "
            f"overlap_threshold={self.overlap_threshold}, "
            f"near_buffer={self.near_buffer_px}px"
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

    def resolve(self, detections: list, img_shape: tuple) -> list:
        """Tüm detection'lar için landing_status hesapla.

        Args:
            detections: DetectorYOLO çıktısı + moving_status eklenmiş liste.
            img_shape: (H, W)

        Returns:
            Her detection'a 'landing_status' eklenmiş liste.
        """
        H, W = img_shape[:2]

        # Önce UAP/UAI olmayanları işaretle
        for det in detections:
            if det['cls'] not in (2, 3):
                det['landing_status'] = "-1"  # İniş Alanı Değil

        # UAP/UAI için engel kontrolü
        uap_uai = [d for d in detections if d['cls'] in (2, 3)]
        obstacles = [d for d in detections if d['cls'] in (0, 1)]  # Taşıt + İnsan

        if not uap_uai:
            return detections

        for uap in uap_uai:
            uap_bbox = uap['bbox']
            # Buffer ekle
            expanded = self._expanded_bbox(uap_bbox, self.near_buffer_px, (H, W))

            is_blocked = False
            for obs in obstacles:
                obs_bbox = obs['bbox']

                # 1. Center-inside kontrolü
                if self._center_inside(obs_bbox, expanded):
                    is_blocked = True
                    self.logger.debug(
                        f"UAP/UAI engellendi: obstacle center inside. "
                        f"UAP bbox: {uap_bbox}, Obstacle: {obs_bbox}"
                    )
                    break

                # 2. IoU kontrolü
                iou = self._iou(expanded, obs_bbox)
                if iou > self.overlap_threshold:
                    is_blocked = True
                    self.logger.debug(
                        f"UAP/UAI engellendi: IoU={iou:.3f}. "
                        f"UAP bbox: {uap_bbox}, Obstacle: {obs_bbox}"
                    )
                    break

            uap['landing_status'] = "0" if is_blocked else "1"
            # "0" = Inilemez, "1" = Inilebilir

        return detections
