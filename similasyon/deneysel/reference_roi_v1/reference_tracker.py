"""Doğrulanmış bir referans kutusunu kamera hareketiyle taşıyan deneysel tracker."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional

import cv2
import numpy as np

from src.models.motion_camera_v2 import CameraModel, CameraMotionEstimator


def _polygon_area(points: np.ndarray) -> float:
    return float(abs(cv2.contourArea(points.astype(np.float32).reshape(-1, 1, 2))))


def _clip_bbox(points: np.ndarray, width: int, height: int) -> tuple[float, float, float, float]:
    return (
        float(np.clip(points[:, 0].min(), 0, width - 1)),
        float(np.clip(points[:, 1].min(), 0, height - 1)),
        float(np.clip(points[:, 0].max(), 0, width - 1)),
        float(np.clip(points[:, 1].max(), 0, height - 1)),
    )


@dataclass
class ReferenceTrackDiagnostics:
    active: bool
    bbox: Optional[tuple[float, float, float, float]]
    polygon: Optional[list[list[float]]]
    reason: str
    model: str = "none"
    transform_confidence: float = 0.0
    matches: int = 0
    inliers: int = 0
    inlier_ratio: float = 0.0
    reprojection_error: float = 0.0
    area_ratio: float = 0.0
    visible_ratio: float = 0.0
    color_correlation: float = -1.0
    color_distance: float = 1.0

    def to_dict(self) -> dict:
        payload = asdict(self)
        if self.bbox is not None:
            payload["bbox"] = list(self.bbox)
        return payload


class CameraCompensatedReferenceTracker:
    """Yalnızca güçlü bir eşleşmeyle seed edildikten sonra çıktı üretir.

    Referans nesne bu oturumda dünyada sabit. Bu nedenle kutunun dört köşesi,
    ardışık kareler arasındaki sağlam arka-plan homografisiyle taşınır. Eksen
    hizalı kutuyu tekrar tekrar warp etmek yerine polygon taşımak rotasyonda
    kutunun yapay biçimde büyümesini engeller.
    """

    def __init__(self, config: dict):
        cfg = config.get("reference_roi_experiment", {}).get("tracker", {})
        motion_cfg = dict(config.get("motion", {}))
        motion_cfg.update({
            "camera_model": "homography",
            "mask_detection_regions": False,
            "feature_type": "orb",
            "max_features": int(cfg.get("max_features", 1600)),
            "min_inlier_ratio": float(cfg.get("min_inlier_ratio", 0.30)),
            "max_reprojection_error": float(cfg.get("max_reprojection_error", 4.0)),
            "affine_min_matches": int(cfg.get("affine_min_matches", 16)),
            "affine_ransac_threshold": float(cfg.get("ransac_threshold", 4.0)),
        })
        self.estimator = CameraMotionEstimator({"motion": motion_cfg})
        self.min_transform_confidence = float(cfg.get("min_transform_confidence", 0.12))
        self.min_step_area_ratio = float(cfg.get("min_step_area_ratio", 0.55))
        self.max_step_area_ratio = float(cfg.get("max_step_area_ratio", 1.80))
        self.min_visible_ratio = float(cfg.get("min_visible_ratio", 0.30))
        self.min_color_correlation = float(cfg.get("min_color_correlation", 0.05))
        self.max_color_distance = float(cfg.get("max_color_distance", 0.85))
        self.max_unreliable_frames = int(cfg.get("max_unreliable_frames", 2))
        self.seed_expand_left_ratio = float(cfg.get("seed_expand_left_ratio", 0.25))
        self.seed_expand_right_ratio = float(cfg.get("seed_expand_right_ratio", 0.55))
        self.seed_expand_top_ratio = float(cfg.get("seed_expand_top_ratio", 0.25))
        self.seed_expand_bottom_ratio = float(cfg.get("seed_expand_bottom_ratio", 0.25))
        self.reset()

    @staticmethod
    def _color_histogram(image: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        histogram = cv2.calcHist([hsv], [0, 1], None, [36, 16], [0, 180, 0, 256])
        return cv2.normalize(histogram, histogram).flatten()

    def reset(self):
        self.active = False
        self.polygon: Optional[np.ndarray] = None
        self.last_good_gray: Optional[np.ndarray] = None
        self.seed_histogram: Optional[np.ndarray] = None
        self.unreliable_frames = 0

    def _expanded_seed_bbox(
        self,
        bbox: tuple[float, float, float, float],
        width: int,
        height: int,
    ) -> tuple[float, float, float, float]:
        x1, y1, x2, y2 = [float(value) for value in bbox]
        box_width = max(0.0, x2 - x1)
        box_height = max(0.0, y2 - y1)
        return (
            float(np.clip(x1 - box_width * self.seed_expand_left_ratio, 0, width - 1)),
            float(np.clip(y1 - box_height * self.seed_expand_top_ratio, 0, height - 1)),
            float(np.clip(x2 + box_width * self.seed_expand_right_ratio, 1, width)),
            float(np.clip(y2 + box_height * self.seed_expand_bottom_ratio, 1, height)),
        )

    def seed(
        self,
        frame: np.ndarray,
        bbox: tuple[float, float, float, float],
        expand: bool = True,
    ) -> tuple[float, float, float, float]:
        height, width = frame.shape[:2]
        if expand:
            x1, y1, x2, y2 = self._expanded_seed_bbox(bbox, width, height)
        else:
            x1, y1, x2, y2 = [float(value) for value in bbox]
            x1 = float(np.clip(x1, 0, width - 1))
            y1 = float(np.clip(y1, 0, height - 1))
            x2 = float(np.clip(x2, 1, width))
            y2 = float(np.clip(y2, 1, height))
        x2 = max(x1 + 1.0, x2)
        y2 = max(y1 + 1.0, y2)
        self.polygon = np.array(
            [[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32
        )
        crop = frame[int(y1):int(np.ceil(y2)), int(x1):int(np.ceil(x2))]
        self.seed_histogram = self._color_histogram(crop)
        self.last_good_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.unreliable_frames = 0
        self.active = True
        return (x1, y1, x2, y2)

    @staticmethod
    def _warp_polygon(points: np.ndarray, transform) -> np.ndarray:
        if transform.model_type == CameraModel.HOMOGRAPHY:
            return cv2.perspectiveTransform(
                points.reshape(-1, 1, 2), transform.matrix
            ).reshape(-1, 2)
        if transform.model_type == CameraModel.AFFINE:
            return cv2.transform(
                points.reshape(-1, 1, 2), transform.matrix
            ).reshape(-1, 2)
        if transform.model_type == CameraModel.SHIFT:
            return points + transform.shift.reshape(1, 2)
        return points.copy()

    @staticmethod
    def _visible_ratio(points: np.ndarray, width: int, height: int) -> float:
        area = _polygon_area(points)
        if area <= 1.0:
            return 0.0
        frame_polygon = np.array(
            [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
            dtype=np.float32,
        )
        try:
            intersection, _ = cv2.intersectConvexConvex(
                cv2.convexHull(points.astype(np.float32)),
                cv2.convexHull(frame_polygon),
            )
        except cv2.error:
            return 0.0
        return float(intersection / area)

    def _inactive(self, reason: str, **kwargs) -> ReferenceTrackDiagnostics:
        return ReferenceTrackDiagnostics(False, None, None, reason, **kwargs)

    def update(self, frame: np.ndarray) -> ReferenceTrackDiagnostics:
        if not self.active or self.polygon is None or self.last_good_gray is None:
            return self._inactive("not_seeded")

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        transform = self.estimator.estimate(self.last_good_gray, gray)
        common = {
            "model": transform.model_type.name.lower(),
            "transform_confidence": float(transform.confidence),
            "matches": int(transform.match_count),
            "inliers": int(transform.inlier_count),
            "inlier_ratio": float(transform.inlier_ratio),
            "reprojection_error": float(transform.reprojection_error),
        }
        if not transform.is_reliable or transform.confidence < self.min_transform_confidence:
            self.unreliable_frames += 1
            if self.unreliable_frames > self.max_unreliable_frames:
                self.reset()
                return self._inactive("transform_lost", **common)
            return self._inactive("transform_unreliable", **common)

        warped = self._warp_polygon(self.polygon, transform)
        if not np.isfinite(warped).all():
            self.reset()
            return self._inactive("non_finite_polygon", **common)

        previous_area = _polygon_area(self.polygon)
        current_area = _polygon_area(warped)
        area_ratio = current_area / max(previous_area, 1.0)
        common["area_ratio"] = float(area_ratio)
        if not self.min_step_area_ratio <= area_ratio <= self.max_step_area_ratio:
            self.reset()
            return self._inactive("area_jump", **common)

        height, width = frame.shape[:2]
        visible_ratio = self._visible_ratio(warped, width, height)
        common["visible_ratio"] = visible_ratio
        if visible_ratio < self.min_visible_ratio:
            self.reset()
            return self._inactive("left_frame", **common)

        bbox = _clip_bbox(warped, width, height)
        x1, y1, x2, y2 = bbox
        crop = frame[
            max(0, int(np.floor(y1))):min(height, int(np.ceil(y2))),
            max(0, int(np.floor(x1))):min(width, int(np.ceil(x2))),
        ]
        if crop.size == 0:
            self.reset()
            return self._inactive("empty_crop", **common)
        histogram = self._color_histogram(crop)
        color_correlation = float(cv2.compareHist(
            self.seed_histogram, histogram, cv2.HISTCMP_CORREL
        ))
        color_distance = float(cv2.compareHist(
            self.seed_histogram, histogram, cv2.HISTCMP_BHATTACHARYYA
        ))
        common["color_correlation"] = color_correlation
        common["color_distance"] = color_distance
        if (
            color_correlation < self.min_color_correlation
            or color_distance > self.max_color_distance
        ):
            self.unreliable_frames += 1
            if self.unreliable_frames > self.max_unreliable_frames:
                self.reset()
                return self._inactive("appearance_lost", **common)
            return self._inactive("appearance_unreliable", **common)

        self.polygon = warped.astype(np.float32)
        self.last_good_gray = gray
        self.unreliable_frames = 0
        return ReferenceTrackDiagnostics(
            True,
            bbox,
            warped.astype(float).tolist(),
            "tracked",
            **common,
        )
