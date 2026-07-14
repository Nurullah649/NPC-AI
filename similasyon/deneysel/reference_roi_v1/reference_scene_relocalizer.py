"""Önceden doğrulanmış sahne anahtar-karelerinden sabit referansı yeniden bul."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import cv2
import numpy as np

from src.models.motion_camera_v2 import (
    CameraModel,
    CameraMotionEstimator,
    CameraTransform,
)


def _area(points: np.ndarray) -> float:
    return float(abs(cv2.contourArea(points.astype(np.float32).reshape(-1, 1, 2))))


@dataclass
class SceneRelocalizationResult:
    accepted: bool
    bbox: tuple[float, float, float, float] | None
    polygon: list[list[float]] | None
    keyframe: int | None
    confidence: float
    matches: int
    inliers: int
    inlier_ratio: float
    reprojection_error: float
    visible_ratio: float
    area_ratio: float
    color_correlation: float
    color_distance: float
    reason: str

    def to_dict(self):
        payload = asdict(self)
        if self.bbox is not None:
            payload["bbox"] = list(self.bbox)
        return payload


class ReferenceSceneRelocalizer:
    """Sabit dünya nesnesini RGB arka-plan homografisiyle relocalize eder."""

    def __init__(self, config: dict):
        cfg = config.get("reference_roi_experiment", {}).get("scene_relocalizer", {})
        motion = dict(config.get("motion", {}))
        motion.update({
            "camera_model": "homography",
            "affine_fallback": False,
            "median_shift_fallback": False,
            "mask_detection_regions": False,
            "feature_type": "orb",
            "max_features": int(cfg.get("max_features", 2000)),
            "min_inlier_ratio": float(cfg.get("min_inlier_ratio", 0.30)),
            "max_reprojection_error": float(cfg.get("max_reprojection_error", 4.0)),
            "affine_ransac_threshold": float(cfg.get("ransac_threshold", 4.0)),
        })
        self.estimator = CameraMotionEstimator({"motion": motion})
        self.min_confidence = float(cfg.get("min_confidence", 0.75))
        self.min_inliers = int(cfg.get("min_inliers", 30))
        self.min_inlier_ratio = float(cfg.get("min_inlier_ratio", 0.30))
        self.max_reprojection_error = float(cfg.get("max_reprojection_error", 4.0))
        self.min_visible_ratio = float(cfg.get("min_visible_ratio", 0.25))
        self.min_area_ratio = float(cfg.get("min_area_ratio", 0.20))
        self.max_area_ratio = float(cfg.get("max_area_ratio", 5.00))
        self.min_color_correlation = float(cfg.get("min_color_correlation", 0.65))
        self.max_color_distance = float(cfg.get("max_color_distance", 0.55))
        self.output_expand_left_ratio = float(cfg.get("output_expand_left_ratio", 0.20))
        self.output_expand_right_ratio = float(cfg.get("output_expand_right_ratio", 0.05))
        self.output_expand_top_ratio = float(cfg.get("output_expand_top_ratio", 0.15))
        self.output_expand_bottom_ratio = float(cfg.get("output_expand_bottom_ratio", 0.05))
        self.keyframes: list[dict] = []

    @staticmethod
    def _color_histogram(image: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        histogram = cv2.calcHist(
            [hsv], [0, 1], None, [36, 16], [0, 180, 0, 256]
        )
        return cv2.normalize(histogram, histogram).flatten()

    def add_keyframe(self, frame_idx: int, frame: np.ndarray, polygon):
        points = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mask = np.full(gray.shape, 255, dtype=np.uint8)
        keypoints, descriptors = self.estimator._extract_features(gray, mask)
        x1, y1 = np.floor(points.min(axis=0)).astype(int)
        x2, y2 = np.ceil(points.max(axis=0)).astype(int)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
        target_crop = frame[y1:y2, x1:x2]
        self.keyframes.append({
            "frame": int(frame_idx),
            "gray": gray,
            "keypoints": keypoints,
            "descriptors": descriptors,
            "target_histogram": self._color_histogram(target_crop),
            "polygon": points,
            "area": _area(points),
        })

    @staticmethod
    def _warp(points: np.ndarray, homography: np.ndarray) -> np.ndarray:
        return cv2.perspectiveTransform(
            points.reshape(-1, 1, 2), homography
        ).reshape(-1, 2)

    @staticmethod
    def _visible_ratio(points: np.ndarray, width: int, height: int) -> float:
        polygon_area = _area(points)
        if polygon_area <= 1.0:
            return 0.0
        frame_polygon = np.array(
            [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
            dtype=np.float32,
        )
        try:
            intersection, _ = cv2.intersectConvexConvex(
                cv2.convexHull(points), cv2.convexHull(frame_polygon)
            )
        except cv2.error:
            return 0.0
        return float(intersection / polygon_area)

    def relocalize(self, frame: np.ndarray) -> SceneRelocalizationResult:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape
        mask = np.full(gray.shape, 255, dtype=np.uint8)
        current_keypoints, current_descriptors = self.estimator._extract_features(gray, mask)
        candidates = []
        for keyframe in self.keyframes:
            if keyframe["descriptors"] is None or current_descriptors is None:
                continue
            matches = self.estimator._bf.match(
                keyframe["descriptors"], current_descriptors
            )
            matches = sorted(matches, key=lambda match: match.distance)[:160]
            if len(matches) < 8:
                continue
            source = np.float32([
                keyframe["keypoints"][match.queryIdx].pt for match in matches
            ]).reshape(-1, 1, 2)
            destination = np.float32([
                current_keypoints[match.trainIdx].pt for match in matches
            ]).reshape(-1, 1, 2)
            homography, inlier_mask = cv2.findHomography(
                source,
                destination,
                method=cv2.RANSAC,
                ransacReprojThreshold=self.estimator.affine_ransac_threshold,
            )
            if homography is None or inlier_mask is None:
                continue
            inlier_count = int(inlier_mask.sum())
            inlier_ratio = inlier_count / len(matches)
            reprojection_error = self.estimator._reprojection_error_homography(
                source, destination, homography, inlier_mask
            )
            confidence = self.estimator._compute_confidence(
                inlier_ratio, reprojection_error, len(matches)
            )
            transform = CameraTransform(
                model_type=CameraModel.HOMOGRAPHY,
                matrix=homography.astype(np.float32),
                match_count=len(matches),
                inlier_count=inlier_count,
                inlier_ratio=inlier_ratio,
                reprojection_error=reprojection_error,
                confidence=confidence,
            )
            if transform.confidence < self.min_confidence:
                continue
            if transform.inlier_count < self.min_inliers:
                continue
            if transform.inlier_ratio < self.min_inlier_ratio:
                continue
            if transform.reprojection_error > self.max_reprojection_error:
                continue
            polygon = self._warp(keyframe["polygon"], transform.matrix)
            if not np.isfinite(polygon).all():
                continue
            area_ratio = _area(polygon) / max(keyframe["area"], 1.0)
            if not self.min_area_ratio <= area_ratio <= self.max_area_ratio:
                continue
            visible_ratio = self._visible_ratio(polygon, width, height)
            if visible_ratio < self.min_visible_ratio:
                continue
            bbox = (
                float(np.clip(polygon[:, 0].min(), 0, width - 1)),
                float(np.clip(polygon[:, 1].min(), 0, height - 1)),
                float(np.clip(polygon[:, 0].max(), 0, width - 1)),
                float(np.clip(polygon[:, 1].max(), 0, height - 1)),
            )
            x1, y1, x2, y2 = bbox
            current_crop = frame[
                max(0, int(np.floor(y1))):min(height, int(np.ceil(y2))),
                max(0, int(np.floor(x1))):min(width, int(np.ceil(x2))),
            ]
            if current_crop.size == 0:
                continue
            current_histogram = self._color_histogram(current_crop)
            color_correlation = float(cv2.compareHist(
                keyframe["target_histogram"], current_histogram, cv2.HISTCMP_CORREL
            ))
            color_distance = float(cv2.compareHist(
                keyframe["target_histogram"], current_histogram,
                cv2.HISTCMP_BHATTACHARYYA,
            ))
            if color_correlation < self.min_color_correlation:
                continue
            if color_distance > self.max_color_distance:
                continue
            box_width = x2 - x1
            box_height = y2 - y1
            output_bbox = (
                float(np.clip(x1 - box_width * self.output_expand_left_ratio, 0, width - 1)),
                float(np.clip(y1 - box_height * self.output_expand_top_ratio, 0, height - 1)),
                float(np.clip(x2 + box_width * self.output_expand_right_ratio, 0, width - 1)),
                float(np.clip(y2 + box_height * self.output_expand_bottom_ratio, 0, height - 1)),
            )
            score = (
                float(transform.confidence)
                + min(1.0, transform.inlier_count / 100.0)
                + float(transform.inlier_ratio)
                - 0.05 * float(transform.reprojection_error)
            )
            candidates.append((score, keyframe, transform, polygon, output_bbox,
                               visible_ratio, area_ratio, color_correlation,
                               color_distance))

        if not candidates:
            return SceneRelocalizationResult(
                False, None, None, None, 0.0, 0, 0, 0.0, 0.0, 0.0, 0.0,
                -1.0, 1.0,
                "no_reliable_scene_homography",
            )
        (
            _, keyframe, transform, polygon, bbox, visible_ratio, area_ratio,
            color_correlation, color_distance,
        ) = max(candidates, key=lambda item: item[0])
        return SceneRelocalizationResult(
            True,
            bbox,
            polygon.astype(float).tolist(),
            keyframe["frame"],
            float(transform.confidence),
            int(transform.match_count),
            int(transform.inlier_count),
            float(transform.inlier_ratio),
            float(transform.reprojection_error),
            visible_ratio,
            area_ratio,
            color_correlation,
            color_distance,
            "accepted",
        )
