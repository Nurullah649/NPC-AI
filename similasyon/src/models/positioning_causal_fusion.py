"""Causal ground-plane and calibration-map fusion for live positioning.

The component is deliberately one-way: it can correct the current and future
predictions, but it never rewrites an already emitted frame.  Absolute map
measurements are sourced only from frames whose translation was healthy.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable

import cv2
import numpy as np


class CausalPositionFusion:
    """Fuse DPVO with planar odometry and conservative visual relocalization."""

    def __init__(
        self,
        config: dict,
        intrinsics: np.ndarray,
        *,
        retrieval_factory: Callable[[str, int], object] | None = None,
    ):
        self.logger = logging.getLogger(self.__class__.__name__)
        cfg = config.get("dpvo", {}).get("causal_fusion", {})
        self.enabled = bool(cfg.get("enabled", False))
        self.plane_enabled = bool(cfg.get("plane_enabled", True))
        self.relocalization_enabled = bool(
            cfg.get("relocalization_enabled", True)
        )
        self.require_relocalization = bool(
            cfg.get("require_relocalization", True)
        )
        self.width = max(1, int(cfg.get("width", 640)))
        self.height = max(1, int(cfg.get("height", 360)))
        self.plane_xy_blend = float(
            np.clip(float(cfg.get("plane_xy_blend", 0.1)), 0.0, 1.0)
        )
        self.min_calibration_pairs = max(
            3, int(cfg.get("min_calibration_pairs", 49))
        )
        self.plane_ridge = max(0.0, float(cfg.get("plane_ridge", 1e-6)))

        self.exclusion_radius = max(1, int(cfg.get("exclusion_radius", 50)))
        self.dbow_score = max(0.0, float(cfg.get("dbow_score", 0.03)))
        self.minimum_inliers = max(8, int(cfg.get("minimum_inliers", 50)))
        self.minimum_inlier_ratio = float(
            np.clip(float(cfg.get("minimum_inlier_ratio", 0.3)), 0.0, 1.0)
        )
        self.confirmations = max(1, int(cfg.get("confirmations", 5)))
        self.confirmation_max_gap = max(
            1, int(cfg.get("confirmation_max_gap", 2))
        )
        self.innovation_min_m = max(
            0.0, float(cfg.get("innovation_min_m", 15.0))
        )
        self.innovation_max_m = max(
            self.innovation_min_m,
            float(cfg.get("innovation_max_m", 50.0)),
        )
        self.relocalization_gain = float(
            np.clip(float(cfg.get("relocalization_gain", 0.2)), 0.0, 1.0)
        )
        self.cooldown_frames = max(0, int(cfg.get("cooldown_frames", 100)))
        self.vocabulary_path = self._resolve_vocabulary(
            str(cfg.get("vocabulary_path", "Class/DPVO/ORBvoc.txt"))
        )
        self.intrinsics = np.asarray(intrinsics, dtype=np.float64)
        if self.intrinsics.shape != (3, 3):
            raise ValueError(
                f"Causal fusion intrinsics must be 3x3, got {self.intrinsics.shape}"
            )

        self._retrieval_factory = retrieval_factory
        self._retrieval = None
        self._retrieval_init_attempted = False
        self.relocalization_available = False

        self._previous_gray: np.ndarray | None = None
        self._camera_to_initial = np.eye(3, dtype=np.float64)
        self._plane_distance = 1.0
        self._current_plane_delta = np.zeros(3, dtype=np.float64)
        self._plane_design: list[np.ndarray] = []
        self._plane_target: list[np.ndarray] = []
        self._plane_weights: np.ndarray | None = None
        self._plane_position: np.ndarray | None = None
        self._last_healthy_gt: np.ndarray | None = None
        self._previous_status: str | None = None

        self._visual_frame_index = 0
        self._healthy_map: dict[int, np.ndarray] = {}
        self._pending_candidate: dict | None = None
        self._confirmation_run: list[tuple[int, int, np.ndarray]] = []
        self._last_event_frame = -10**9
        self._correction = np.zeros(3, dtype=np.float64)
        self.last_applied_correction_delta = np.zeros(3, dtype=np.float64)
        self.events: list[dict] = []
        self.telemetry = {
            "enabled": self.enabled,
            "plane_calibrated": False,
            "plane_fit_error": None,
            "relocalization_available": False,
            "relocalization_error": None,
            "event_count": 0,
        }

    @staticmethod
    def _resolve_vocabulary(value: str) -> str:
        path = Path(value).expanduser()
        repository_root = Path(__file__).resolve().parents[3]
        candidates = (
            path,
            Path.cwd() / path,
            repository_root / path,
            Path(__file__).resolve().parents[2] / path,
        )
        for candidate in candidates:
            if candidate.is_file():
                return str(candidate.resolve())
        return str((repository_root / path).resolve())

    def _initialize_retrieval(self) -> None:
        if (
            not self.enabled
            or not self.relocalization_enabled
            or self._retrieval_init_attempted
        ):
            return
        self._retrieval_init_attempted = True
        try:
            if not Path(self.vocabulary_path).is_file():
                raise FileNotFoundError(
                    f"ORB vocabulary not found: {self.vocabulary_path}"
                )
            factory = self._retrieval_factory
            if factory is None:
                import dpretrieval

                factory = dpretrieval.DPRetrieval
            self._retrieval = factory(
                self.vocabulary_path, self.exclusion_radius
            )
            self.relocalization_available = True
            self.telemetry["relocalization_available"] = True
            self.logger.info(
                "Causal relocalization initialized: vocabulary=%s radius=%d",
                self.vocabulary_path,
                self.exclusion_radius,
            )
        except Exception as exc:
            self.relocalization_available = False
            self.telemetry["relocalization_error"] = str(exc)
            if self.require_relocalization:
                self.enabled = False
                self.telemetry["enabled"] = False
                self.logger.error(
                    "Causal fusion disabled because relocalization is required: %s",
                    exc,
                )
            else:
                self.logger.warning(
                    "Causal relocalization unavailable; plane-only fusion remains: %s",
                    exc,
                )

    def _resize(self, image: np.ndarray) -> np.ndarray:
        if image.shape[1] == self.width and image.shape[0] == self.height:
            return np.ascontiguousarray(image, dtype=np.uint8)
        return np.ascontiguousarray(
            cv2.resize(
                image,
                (self.width, self.height),
                interpolation=cv2.INTER_AREA,
            ),
            dtype=np.uint8,
        )

    @staticmethod
    def _estimate_homography(
        previous: np.ndarray, current: np.ndarray
    ) -> np.ndarray | None:
        points0 = cv2.goodFeaturesToTrack(
            previous,
            maxCorners=1600,
            qualityLevel=0.01,
            minDistance=7.0,
            blockSize=7,
            useHarrisDetector=False,
        )
        if points0 is None or len(points0) < 30:
            return None
        lk = {
            "winSize": (31, 31),
            "maxLevel": 4,
            "criteria": (
                cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                30,
                0.01,
            ),
        }
        points1, status1, _ = cv2.calcOpticalFlowPyrLK(
            previous, current, points0, None, **lk
        )
        if points1 is None or status1 is None:
            return None
        points0_back, status0, _ = cv2.calcOpticalFlowPyrLK(
            current, previous, points1, None, **lk
        )
        if points0_back is None or status0 is None:
            return None
        p0 = points0.reshape(-1, 2)
        p1 = points1.reshape(-1, 2)
        p0_back = points0_back.reshape(-1, 2)
        valid = (
            status1.reshape(-1).astype(bool)
            & status0.reshape(-1).astype(bool)
            & np.all(np.isfinite(p1), axis=1)
            & (np.linalg.norm(p0_back - p0, axis=1) <= 1.5)
        )
        p0 = p0[valid]
        p1 = p1[valid]
        if len(p0) < 30:
            return None
        homography, mask = cv2.findHomography(
            p0, p1, cv2.RANSAC, ransacReprojThreshold=2.0
        )
        if (
            homography is None
            or mask is None
            or int(np.count_nonzero(mask)) < 30
            or not np.all(np.isfinite(homography))
            or abs(float(homography[2, 2])) <= 1e-12
        ):
            return None
        return np.asarray(homography / homography[2, 2], dtype=np.float64)

    def _decompose_plane_delta(self, homography: np.ndarray | None) -> np.ndarray:
        if homography is None:
            return np.zeros(3, dtype=np.float64)
        try:
            count, rotations, translations, normals = cv2.decomposeHomographyMat(
                homography, self.intrinsics
            )
            candidates = []
            for solution in range(count):
                normal = np.asarray(normals[solution], dtype=np.float64).reshape(3)
                if normal[2] < 0.0:
                    continue
                candidates.append(
                    (
                        -float(normal[2]),
                        np.asarray(rotations[solution], dtype=np.float64),
                        np.asarray(translations[solution], dtype=np.float64).reshape(3),
                        normal,
                    )
                )
            if not candidates:
                return np.zeros(3, dtype=np.float64)
            _, rotation, translation_over_distance, normal = min(
                candidates, key=lambda item: item[0]
            )
            delta = -self._camera_to_initial @ rotation.T @ (
                self._plane_distance * translation_over_distance
            )
            self._camera_to_initial = self._camera_to_initial @ rotation.T
            factor = 1.0 + float(
                (rotation @ normal) @ translation_over_distance
            )
            self._plane_distance *= float(np.clip(factor, 0.8, 1.2))
            return np.asarray(delta, dtype=np.float64)
        except cv2.error:
            return np.zeros(3, dtype=np.float64)

    def _fit_plane_mapping(self) -> None:
        if len(self._plane_design) < self.min_calibration_pairs:
            self.telemetry["plane_fit_error"] = (
                f"only {len(self._plane_design)} calibration pairs"
            )
            return
        design = np.asarray(self._plane_design, dtype=np.float64)
        target = np.asarray(self._plane_target, dtype=np.float64)
        if design.shape != target.shape or design.shape[1:] != (3,):
            self.telemetry["plane_fit_error"] = "invalid calibration shapes"
            return
        regularizer = np.eye(3, dtype=np.float64) * self.plane_ridge
        weights = np.linalg.solve(
            design.T @ design + regularizer, design.T @ target
        )
        observation_weight = np.ones(len(design), dtype=np.float64)
        for _ in range(50):
            residual_norm = np.linalg.norm(target - design @ weights, axis=1)
            center = float(np.median(residual_norm))
            scale = float(
                1.4826 * np.median(np.abs(residual_norm - center))
            )
            if scale <= 1e-12:
                break
            threshold = center + 1.345 * scale
            updated_weight = np.ones_like(residual_norm)
            outside = residual_norm > threshold
            updated_weight[outside] = threshold / residual_norm[outside]
            root = np.sqrt(updated_weight)
            x = design * root[:, None]
            y = target * root[:, None]
            updated = np.linalg.solve(
                x.T @ x + regularizer, x.T @ y
            )
            observation_weight = updated_weight
            if np.linalg.norm(updated - weights) <= 1e-10 * (
                1.0 + np.linalg.norm(weights)
            ):
                weights = updated
                break
            weights = updated
        if not np.all(np.isfinite(weights)) or np.linalg.matrix_rank(design) < 3:
            self.telemetry["plane_fit_error"] = "rank-deficient/non-finite fit"
            return
        self._plane_weights = weights
        self.telemetry["plane_calibrated"] = True
        self.telemetry["plane_fit_error"] = None
        self.telemetry["plane_downweighted_count"] = int(
            np.count_nonzero(observation_weight < 1.0 - 1e-12)
        )
        self.telemetry["plane_weights"] = weights.astype(float).tolist()
        self.logger.info(
            "Causal plane mapping calibrated with %d pairs; downweighted=%d",
            len(design),
            self.telemetry["plane_downweighted_count"],
        )

    @staticmethod
    def _geometric_stats(matches) -> dict:
        count = len(matches)
        result = {
            "match_count": count,
            "homography_inliers": 0,
            "homography_inlier_ratio": 0.0,
            "fundamental_inliers": 0,
            "fundamental_inlier_ratio": 0.0,
        }
        if count < 8:
            return result
        values = np.asarray(matches, dtype=np.float64)
        reference = values[:, :2].astype(np.float32)
        query = values[:, 2:4].astype(np.float32)
        try:
            _, mask = cv2.findHomography(
                reference,
                query,
                method=cv2.RANSAC,
                ransacReprojThreshold=3.0,
            )
            if mask is not None:
                inliers = int(np.count_nonzero(mask))
                result["homography_inliers"] = inliers
                result["homography_inlier_ratio"] = float(inliers / count)
        except cv2.error:
            pass
        try:
            _, mask = cv2.findFundamentalMat(
                reference,
                query,
                method=cv2.FM_RANSAC,
                ransacReprojThreshold=1.5,
                confidence=0.99,
            )
            if mask is not None:
                inliers = int(np.count_nonzero(mask))
                result["fundamental_inliers"] = inliers
                result["fundamental_inlier_ratio"] = float(inliers / count)
        except cv2.error:
            pass
        return result

    def _query_relocalization(self, image: np.ndarray) -> dict | None:
        if not self.relocalization_available or self._retrieval is None:
            return None
        frame = self._visual_frame_index
        try:
            self._retrieval.insert_image(image)
            score, candidate, matches = self._retrieval.query(frame)
            return {
                "frame": frame,
                "candidate_frame": int(candidate),
                "dbow_score": float(score),
                **self._geometric_stats(matches),
            }
        except Exception as exc:
            self.telemetry["relocalization_error"] = str(exc)
            self.logger.warning(
                "Causal relocalization query failed at frame %d: %s", frame, exc
            )
            return None

    def observe_frame(
        self,
        image: np.ndarray,
        health_status: str | None,
        healthy_gt: np.ndarray | None,
    ) -> None:
        """Consume the current image before its base DPVO pose is fused."""

        if not self.enabled:
            return
        self._initialize_retrieval()
        if not self.enabled:
            return
        status = None if health_status is None else str(health_status)
        resized = self._resize(image)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        homography = None
        if self._previous_gray is not None and self.plane_enabled:
            homography = self._estimate_homography(self._previous_gray, gray)
        self._previous_gray = gray
        self._current_plane_delta = self._decompose_plane_delta(homography)
        self._pending_candidate = self._query_relocalization(resized)

        frame = self._visual_frame_index
        if status == "1" and healthy_gt is not None:
            gt = np.asarray(healthy_gt, dtype=np.float64)
            self._healthy_map[frame] = gt.copy()
            if self._previous_status == "1" and self._last_healthy_gt is not None:
                self._plane_design.append(self._current_plane_delta.copy())
                self._plane_target.append(gt - self._last_healthy_gt)
            self._last_healthy_gt = gt.copy()
            self._plane_position = gt.copy()
            self._correction.fill(0.0)
            self._confirmation_run = []
        elif status == "0":
            if self._previous_status == "1":
                self._fit_plane_mapping()
                if self._last_healthy_gt is not None:
                    self._plane_position = self._last_healthy_gt.copy()
            if self._plane_weights is not None and self._plane_position is not None:
                self._plane_position = self._plane_position + (
                    self._current_plane_delta @ self._plane_weights
                )
        self._previous_status = status
        self._visual_frame_index += 1

    def _candidate_is_safe(self, row: dict | None) -> bool:
        if row is None:
            return False
        candidate = int(row["candidate_frame"])
        return (
            candidate in self._healthy_map
            and float(row["dbow_score"]) >= self.dbow_score
            and int(row["homography_inliers"]) >= self.minimum_inliers
            and float(row["homography_inlier_ratio"])
            >= self.minimum_inlier_ratio
            and int(row["fundamental_inliers"]) >= self.minimum_inliers
            and float(row["fundamental_inlier_ratio"])
            >= self.minimum_inlier_ratio
        )

    def fuse_position(
        self, base_position: np.ndarray, health_status: str | None
    ) -> np.ndarray:
        """Return the irrevocable position for the current frame."""

        base = np.asarray(base_position, dtype=np.float64)
        self.last_applied_correction_delta.fill(0.0)
        status = None if health_status is None else str(health_status)
        if not self.enabled or status != "0":
            return base.copy()
        fused_base = base.copy()
        if self._plane_weights is not None and self._plane_position is not None:
            fused_base[:2] = (
                (1.0 - self.plane_xy_blend) * base[:2]
                + self.plane_xy_blend * self._plane_position[:2]
            )
        current = fused_base + self._correction
        row = self._pending_candidate
        frame = self._visual_frame_index - 1
        if not self._candidate_is_safe(row):
            self._confirmation_run = []
            return current
        if (
            self._confirmation_run
            and frame - self._confirmation_run[-1][0]
            > self.confirmation_max_gap
        ):
            self._confirmation_run = []
        candidate = int(row["candidate_frame"])
        self._confirmation_run.append(
            (frame, candidate, self._healthy_map[candidate].copy())
        )
        self._confirmation_run = self._confirmation_run[-self.confirmations :]
        if (
            len(self._confirmation_run) < self.confirmations
            or frame - self._last_event_frame < self.cooldown_frames
        ):
            return current
        measurement = np.median(
            np.vstack([item[2] for item in self._confirmation_run]), axis=0
        )
        innovation = measurement - current
        innovation_norm = float(np.linalg.norm(innovation))
        if not self.innovation_min_m <= innovation_norm <= self.innovation_max_m:
            return current
        correction_delta = self.relocalization_gain * innovation
        self._correction += correction_delta
        self.last_applied_correction_delta = correction_delta.copy()
        current = fused_base + self._correction
        event = {
            "frame": frame,
            "candidate_frame": candidate,
            "confirmed_candidate_frames": [
                item[1] for item in self._confirmation_run
            ],
            "measurement_ned": measurement.astype(float).tolist(),
            "innovation_norm_m": innovation_norm,
            "gain": self.relocalization_gain,
            "applied_correction_m": self._correction.astype(float).tolist(),
        }
        self.events.append(event)
        self.telemetry["event_count"] = len(self.events)
        self._last_event_frame = frame
        self._confirmation_run = []
        self.logger.warning(
            "Causal relocalization applied at frame %d: candidate=%d "
            "innovation=%.3fm correction=%s",
            frame,
            candidate,
            innovation_norm,
            self._correction.tolist(),
        )
        return current
