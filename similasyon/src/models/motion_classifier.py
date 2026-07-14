"""Uretim arac hareket siniflandiricisi.

Motion V2 deneyinden uretime alinan akis:
- detection-bolge maskeli homography (affine/shift fallback),
- kamera-relative arac hizi ve missed-frame propagation,
- dinamik association + iki asamali confidence eslestirme,
- bbox-ici optical-flow ile yerel homography/paralaks duzeltmesi,
- piksel cinsinden S skoru ve 45/35 histerzis esikleri.
"""

from __future__ import annotations

import logging

import numpy as np

from .motion_camera_v2 import CameraMotionEstimator
from .vehicle_tracker_v2 import VehicleTrackerV2


class MotionClassifier:
    """Ana orkestratorle uyumlu Motion V2 facade'i."""

    def __init__(self, config: dict):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.config = config
        self.camera_estimator = CameraMotionEstimator(config)
        self.tracker = VehicleTrackerV2(config)
        self.prev_gray = None
        self.prev_detections = None

        motion_cfg = config.get("motion", {})
        self.logger.info(
            "MotionClassifier V2 baslatildi. camera_model=%s, "
            "moving_start_threshold=%.1fpx, moving_stop_threshold=%.1fpx, "
            "local_flow=%s, relative_velocity=%s",
            motion_cfg.get("camera_model", "homography"),
            float(motion_cfg.get("moving_start_threshold", 45.0)),
            float(motion_cfg.get("moving_stop_threshold", 35.0)),
            bool(motion_cfg.get("use_local_optical_flow", True)),
            bool(motion_cfg.get("camera_relative_velocity_enabled", True)),
        )

    @property
    def tracks(self):
        """Eski debug/tanimlama kullanimlari icin aktif track'ler."""
        return self.tracker.tracks

    @property
    def frame_count(self) -> int:
        return self.tracker.frame_count

    @property
    def next_id(self) -> int:
        return self.tracker.next_id

    def update(self, detections: list, gray: np.ndarray) -> list:
        transform = self.camera_estimator.estimate(
            self.prev_gray,
            gray,
            detections,
            self.prev_detections,
        )
        output = self.tracker.update(detections, gray, transform)

        self.prev_gray = gray
        # Kamera maskesi icin yalniz bbox/sinif gerekir. Tracker'in ekledigi
        # gecici alanlari sonraki frame'e tasimamak icin yalın kopya tut.
        self.prev_detections = [
            {
                "cls": det.get("cls"),
                "bbox": tuple(det["bbox"]),
            }
            for det in output
            if det.get("bbox") is not None
        ]
        return output

    def reset(self) -> None:
        self.tracker.reset()
        self.prev_gray = None
        self.prev_detections = None

    def get_moving_status(self, cls_id: int, bbox: tuple, gray: np.ndarray) -> str:
        """Eski tek-kutu API'si korunur; toplu update asil API'dir."""
        return "0" if cls_id == 0 else "-1"
