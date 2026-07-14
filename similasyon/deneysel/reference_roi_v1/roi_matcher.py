"""YOLO aday ROI'lerinde geometrik doğrulamalı LightGlue fallback."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import ceil
from typing import Optional

import cv2
import numpy as np

from src.models.reference_matcher import ReferenceMatcher


def _bbox_iou(box_a, box_b) -> float:
    x1 = max(float(box_a[0]), float(box_b[0]))
    y1 = max(float(box_a[1]), float(box_b[1]))
    x2 = min(float(box_a[2]), float(box_b[2]))
    y2 = min(float(box_a[3]), float(box_b[3]))
    if x1 >= x2 or y1 >= y2:
        return 0.0
    intersection = (x2 - x1) * (y2 - y1)
    area_a = max(1e-6, (box_a[2] - box_a[0]) * (box_a[3] - box_a[1]))
    area_b = max(1e-6, (box_b[2] - box_b[0]) * (box_b[3] - box_b[1]))
    return float(intersection / (area_a + area_b - intersection))


def _hull_coverage(points: np.ndarray, width: int, height: int) -> float:
    if len(points) < 3:
        return 0.0
    hull = cv2.convexHull(points.astype(np.float32).reshape(-1, 1, 2))
    return float(cv2.contourArea(hull) / max(1.0, float(width * height)))


@dataclass
class RoiMatchDiagnostics:
    candidate_index: int
    candidate_bbox: tuple[float, float, float, float]
    candidate_conf: float
    crop_bbox: tuple[int, int, int, int]
    scale: int
    matches: int = 0
    inliers: int = 0
    inlier_ratio: float = 0.0
    reprojection_error: float = float("inf")
    projected_bbox: Optional[tuple[float, float, float, float]] = None
    projection_coverage: float = 0.0
    projection_aspect: float = 0.0
    candidate_iou: float = 0.0
    reference_hull_coverage: float = 0.0
    roi_hull_coverage: float = 0.0
    color_hist_correlation: float = -1.0
    color_hist_bhattacharyya: float = 1.0
    accepted: bool = False
    reject_reason: str = "not_evaluated"
    score: float = 0.0
    candidate_source: str = "yolo"
    output_bbox: Optional[tuple[float, float, float, float]] = None

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["candidate_bbox"] = list(self.candidate_bbox)
        payload["crop_bbox"] = list(self.crop_bbox)
        if self.projected_bbox is not None:
            payload["projected_bbox"] = list(self.projected_bbox)
        if self.output_bbox is not None:
            payload["output_bbox"] = list(self.output_bbox)
        if not np.isfinite(payload["reprojection_error"]):
            payload["reprojection_error"] = None
        return payload


class RoiReferenceMatcher:
    """Production matcher'i değiştirmeden ROI fallback dener."""

    def __init__(self, config: dict, base: ReferenceMatcher | None = None):
        self.config = config
        # Üretim pipeline'ı aynı LightGlue modeli ve feature cache'ini paylaşır.
        # Deneysel benchmark'lar base vermediğinde eski davranış korunur.
        self.base = base if base is not None else ReferenceMatcher(config)
        cfg = config.get("reference_roi_experiment", {})
        ref_cfg = config.get("reference", {})

        self.expand_ratio = float(cfg.get("expand_ratio", 0.25))
        self.target_long_side = int(cfg.get("target_long_side", 360))
        self.max_scale = int(cfg.get("max_scale", 4))
        self.candidate_min_conf = float(cfg.get("candidate_min_conf", 0.25))
        self.duplicate_iou = float(cfg.get("duplicate_iou", 0.85))

        self.min_matches = int(cfg.get("min_matches", ref_cfg.get("min_matches", 10)))
        self.min_inliers = int(cfg.get("min_inliers", ref_cfg.get("min_inliers", 8)))
        self.min_inlier_ratio = float(
            cfg.get("min_inlier_ratio", ref_cfg.get("min_inlier_ratio", 0.25))
        )
        self.max_reprojection_error = float(cfg.get("max_reprojection_error", 5.0))
        self.min_projection_coverage = float(cfg.get("min_projection_coverage", 0.30))
        self.max_projection_coverage = float(cfg.get("max_projection_coverage", 1.00))
        self.min_projection_aspect = float(cfg.get("min_projection_aspect", 0.40))
        self.max_projection_aspect = float(cfg.get("max_projection_aspect", 4.00))
        self.min_candidate_iou = float(cfg.get("min_candidate_iou", 0.20))
        self.min_reference_hull_coverage = float(
            cfg.get("min_reference_hull_coverage", 0.002)
        )
        self.min_roi_hull_coverage = float(cfg.get("min_roi_hull_coverage", 0.002))
        self.min_color_hist_correlation = float(
            cfg.get("min_color_hist_correlation", 0.12)
        )
        self.max_color_hist_bhattacharyya = float(
            cfg.get("max_color_hist_bhattacharyya", 0.75)
        )
        self._reference_color_histograms: dict[str, np.ndarray] = {}
        self.discovery_window_sizes = [
            tuple(int(value) for value in size)
            for size in cfg.get("discovery_window_sizes", [[240, 160], [320, 220], [420, 280]])
        ]
        self.discovery_stride_ratio = float(cfg.get("discovery_stride_ratio", 0.50))
        self.discovery_top_k = int(cfg.get("discovery_top_k", 8))
        self.discovery_nms_iou = float(cfg.get("discovery_nms_iou", 0.45))
        self.discovery_min_color_correlation = float(
            cfg.get("discovery_min_color_correlation", 0.55)
        )
        self.discovery_max_color_bhattacharyya = float(
            cfg.get("discovery_max_color_bhattacharyya", 0.55)
        )
        crossmodal_cfg = cfg.get("crossmodal", {})
        self.crossmodal_window_sizes = [
            tuple(int(value) for value in size)
            for size in crossmodal_cfg.get("window_sizes", [[640, 540], [800, 640]])
        ]
        self.crossmodal_stride_ratio = float(crossmodal_cfg.get("stride_ratio", 0.50))
        self.crossmodal_min_inliers = int(crossmodal_cfg.get("min_inliers", 10))
        self.crossmodal_min_inlier_ratio = float(
            crossmodal_cfg.get("min_inlier_ratio", 0.18)
        )
        self.crossmodal_min_projection_coverage = float(
            crossmodal_cfg.get("min_projection_coverage", 0.01)
        )
        self.crossmodal_max_projection_coverage = float(
            crossmodal_cfg.get("max_projection_coverage", 0.85)
        )
        self.crossmodal_min_reference_hull_coverage = float(
            crossmodal_cfg.get("min_reference_hull_coverage", 0.01)
        )
        self.crossmodal_min_roi_hull_coverage = float(
            crossmodal_cfg.get("min_roi_hull_coverage", 0.002)
        )

    @staticmethod
    def _color_histogram(image: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        histogram = cv2.calcHist(
            [hsv], [0, 1], None, [36, 16], [0, 180, 0, 256]
        )
        return cv2.normalize(histogram, histogram).flatten()

    def prepare_reference(self, ref_url: str, ref_path: str) -> Optional[dict]:
        self.base.precompute_reference(ref_url, ref_path)
        entry = self.base.feature_cache.get(ref_url)
        if ref_url not in self._reference_color_histograms:
            image = cv2.imread(ref_path)
            if image is not None:
                self._reference_color_histograms[ref_url] = self._color_histogram(image)
        return entry["features"] if entry is not None else None

    def match_full(self, ref_url: str, ref_path: str, frame: np.ndarray):
        return self.base.match(ref_url, ref_path, frame)

    def _deduplicate_candidates(self, detections: list[dict]) -> list[tuple[int, dict]]:
        eligible = [
            (index, det)
            for index, det in enumerate(detections)
            if det.get("cls") == 0
            and float(det.get("conf", 0.0)) >= self.candidate_min_conf
            and det.get("bbox") is not None
        ]
        eligible.sort(key=lambda item: float(item[1].get("conf", 0.0)), reverse=True)
        kept: list[tuple[int, dict]] = []
        for index, detection in eligible:
            if any(
                _bbox_iou(detection["bbox"], other[1]["bbox"]) >= self.duplicate_iou
                for other in kept
            ):
                continue
            kept.append((index, detection))
        return kept

    def _crop_candidate(self, frame: np.ndarray, bbox: tuple) -> tuple:
        height, width = frame.shape[:2]
        x1, y1, x2, y2 = [float(value) for value in bbox]
        pad_x = (x2 - x1) * self.expand_ratio
        pad_y = (y2 - y1) * self.expand_ratio
        crop_x1 = max(0, int(np.floor(x1 - pad_x)))
        crop_y1 = max(0, int(np.floor(y1 - pad_y)))
        crop_x2 = min(width, int(np.ceil(x2 + pad_x)))
        crop_y2 = min(height, int(np.ceil(y2 + pad_y)))
        crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]
        long_side = max(crop.shape[:2]) if crop.size else 0
        scale = max(1, min(self.max_scale, int(ceil(self.target_long_side / max(1, long_side)))))
        if scale > 1:
            crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        return crop, (crop_x1, crop_y1, crop_x2, crop_y2), scale

    def _diagnose_lightglue(
        self,
        ref_features: dict,
        probe: np.ndarray,
        diagnostics: RoiMatchDiagnostics,
        expected_bbox: tuple[float, float, float, float],
        reference_histogram: np.ndarray,
        candidate_image: np.ndarray,
    ) -> RoiMatchDiagnostics:
        import torch

        if not self.base._lightglue_available or "lightglue" not in ref_features:
            diagnostics.reject_reason = "lightglue_unavailable"
            return diagnostics

        with torch.inference_mode():
            frame_features = self.base._extractor.extract(
                self.base._to_lightglue_tensor(probe)
            )
            output = self.base._matcher({
                "image0": ref_features["lightglue"],
                "image1": frame_features,
            })

        pairs = self.base._remove_batch(output.get("matches")) if isinstance(output, dict) else None
        if pairs is None or not torch.is_tensor(pairs) or pairs.ndim != 2 or pairs.shape[1] != 2:
            diagnostics.reject_reason = "invalid_match_output"
            return diagnostics
        diagnostics.matches = int(pairs.shape[0])
        if diagnostics.matches < self.min_matches:
            diagnostics.reject_reason = "too_few_matches"
            return diagnostics

        ref_keypoints = self.base._remove_batch(ref_features["lightglue"].get("keypoints"))
        roi_keypoints = self.base._remove_batch(frame_features.get("keypoints"))
        if ref_keypoints is None or roi_keypoints is None:
            diagnostics.reject_reason = "missing_keypoints"
            return diagnostics

        source = ref_keypoints[pairs[:, 0]].detach().cpu().numpy().astype(np.float32)
        destination = roi_keypoints[pairs[:, 1]].detach().cpu().numpy().astype(np.float32)
        homography, mask = cv2.findHomography(
            source.reshape(-1, 1, 2),
            destination.reshape(-1, 1, 2),
            cv2.RANSAC,
            5.0,
        )
        if homography is None or mask is None:
            diagnostics.reject_reason = "homography_failed"
            return diagnostics

        inlier_mask = mask.reshape(-1).astype(bool)
        diagnostics.inliers = int(inlier_mask.sum())
        diagnostics.inlier_ratio = diagnostics.inliers / max(1, diagnostics.matches)
        if diagnostics.inliers:
            projected_inliers = cv2.perspectiveTransform(
                source[inlier_mask].reshape(-1, 1, 2), homography
            ).reshape(-1, 2)
            errors = np.linalg.norm(projected_inliers - destination[inlier_mask], axis=1)
            diagnostics.reprojection_error = float(np.median(errors))

        ref_height, ref_width = ref_features.get("shape", probe.shape[:2])
        probe_height, probe_width = probe.shape[:2]
        diagnostics.reference_hull_coverage = _hull_coverage(
            source[inlier_mask], ref_width, ref_height
        )
        diagnostics.roi_hull_coverage = _hull_coverage(
            destination[inlier_mask], probe_width, probe_height
        )

        corners = np.array(
            [[0, 0], [ref_width - 1, 0], [ref_width - 1, ref_height - 1], [0, ref_height - 1]],
            dtype=np.float32,
        ).reshape(-1, 1, 2)
        transformed = cv2.perspectiveTransform(corners, homography).reshape(-1, 2)
        if not np.isfinite(transformed).all():
            diagnostics.reject_reason = "non_finite_projection"
            return diagnostics

        x1 = float(np.clip(transformed[:, 0].min(), 0, probe_width - 1))
        y1 = float(np.clip(transformed[:, 1].min(), 0, probe_height - 1))
        x2 = float(np.clip(transformed[:, 0].max(), 0, probe_width - 1))
        y2 = float(np.clip(transformed[:, 1].max(), 0, probe_height - 1))
        if x1 >= x2 or y1 >= y2:
            diagnostics.reject_reason = "empty_projection"
            return diagnostics

        diagnostics.projected_bbox = (x1, y1, x2, y2)
        projected_width = x2 - x1
        projected_height = y2 - y1
        diagnostics.projection_coverage = (
            projected_width * projected_height / max(1.0, float(probe_width * probe_height))
        )
        diagnostics.projection_aspect = projected_width / max(1e-6, projected_height)
        diagnostics.candidate_iou = _bbox_iou(diagnostics.projected_bbox, expected_bbox)
        candidate_histogram = self._color_histogram(candidate_image)
        diagnostics.color_hist_correlation = float(cv2.compareHist(
            reference_histogram, candidate_histogram, cv2.HISTCMP_CORREL
        ))
        diagnostics.color_hist_bhattacharyya = float(cv2.compareHist(
            reference_histogram, candidate_histogram, cv2.HISTCMP_BHATTACHARYYA
        ))

        checks = [
            (diagnostics.inliers >= self.min_inliers, "too_few_inliers"),
            (diagnostics.inlier_ratio >= self.min_inlier_ratio, "low_inlier_ratio"),
            (diagnostics.reprojection_error <= self.max_reprojection_error, "high_reprojection_error"),
            (diagnostics.projection_coverage >= self.min_projection_coverage, "low_projection_coverage"),
            (diagnostics.projection_coverage <= self.max_projection_coverage, "high_projection_coverage"),
            (diagnostics.projection_aspect >= self.min_projection_aspect, "projection_too_narrow"),
            (diagnostics.projection_aspect <= self.max_projection_aspect, "projection_too_wide"),
            (diagnostics.candidate_iou >= self.min_candidate_iou, "low_candidate_iou"),
            (
                diagnostics.reference_hull_coverage >= self.min_reference_hull_coverage,
                "low_reference_hull_coverage",
            ),
            (diagnostics.roi_hull_coverage >= self.min_roi_hull_coverage, "low_roi_hull_coverage"),
            (
                diagnostics.color_hist_correlation >= self.min_color_hist_correlation,
                "low_color_correlation",
            ),
            (
                diagnostics.color_hist_bhattacharyya <= self.max_color_hist_bhattacharyya,
                "high_color_distance",
            ),
        ]
        for passed, reason in checks:
            if not passed:
                diagnostics.reject_reason = reason
                return diagnostics

        diagnostics.accepted = True
        diagnostics.reject_reason = "accepted"
        diagnostics.score = float(
            diagnostics.inliers
            + 10.0 * diagnostics.inlier_ratio
            + 3.0 * diagnostics.candidate_iou
            + diagnostics.projection_coverage
            + min(1.0, diagnostics.reference_hull_coverage * 10.0)
            + min(1.0, diagnostics.roi_hull_coverage * 10.0)
            + max(0.0, diagnostics.color_hist_correlation) * 3.0
            + max(0.0, 1.0 - diagnostics.color_hist_bhattacharyya)
        )
        return diagnostics

    def match_candidates(
        self,
        ref_url: str,
        ref_path: str,
        frame: np.ndarray,
        detections: list[dict],
    ) -> dict:
        ref_features = self.prepare_reference(ref_url, ref_path)
        if ref_features is None:
            return {"bbox": None, "accepted": [], "candidates": []}
        reference_histogram = self._reference_color_histograms.get(ref_url)
        if reference_histogram is None:
            return {"bbox": None, "accepted": [], "candidates": []}

        candidate_results = []
        for candidate_index, detection in self._deduplicate_candidates(detections):
            candidate_bbox = tuple(float(value) for value in detection["bbox"])
            probe, crop_bbox, scale = self._crop_candidate(frame, candidate_bbox)
            frame_height, frame_width = frame.shape[:2]
            object_x1 = max(0, min(frame_width - 1, int(np.floor(candidate_bbox[0]))))
            object_y1 = max(0, min(frame_height - 1, int(np.floor(candidate_bbox[1]))))
            object_x2 = max(0, min(frame_width, int(np.ceil(candidate_bbox[2]))))
            object_y2 = max(0, min(frame_height, int(np.ceil(candidate_bbox[3]))))
            candidate_image = frame[object_y1:object_y2, object_x1:object_x2]
            diagnostics = RoiMatchDiagnostics(
                candidate_index=candidate_index,
                candidate_bbox=candidate_bbox,
                candidate_conf=float(detection.get("conf", 0.0)),
                crop_bbox=crop_bbox,
                scale=scale,
            )
            if probe.size == 0 or candidate_image.size == 0:
                diagnostics.reject_reason = "empty_crop"
                candidate_results.append(diagnostics)
                continue

            crop_x1, crop_y1, _, _ = crop_bbox
            expected_bbox = tuple(
                value
                for point in (
                    ((candidate_bbox[0] - crop_x1) * scale, (candidate_bbox[1] - crop_y1) * scale),
                    ((candidate_bbox[2] - crop_x1) * scale, (candidate_bbox[3] - crop_y1) * scale),
                )
                for value in point
            )
            try:
                diagnostics = self._diagnose_lightglue(
                    ref_features,
                    probe,
                    diagnostics,
                    expected_bbox,
                    reference_histogram,
                    candidate_image,
                )
                if diagnostics.accepted:
                    diagnostics.output_bbox = candidate_bbox
            except Exception as error:
                diagnostics.reject_reason = f"exception:{type(error).__name__}"
            candidate_results.append(diagnostics)

        accepted = sorted(
            (item for item in candidate_results if item.accepted),
            key=lambda item: item.score,
            reverse=True,
        )
        best_bbox = accepted[0].candidate_bbox if accepted else None
        return {
            "bbox": best_bbox,
            "accepted": [item.to_dict() for item in accepted],
            "candidates": [item.to_dict() for item in candidate_results],
        }

    @staticmethod
    def _axis_positions(length: int, window: int, stride: int) -> list[int]:
        if window >= length:
            return [0]
        positions = list(range(0, length - window + 1, max(1, stride)))
        last = length - window
        if positions[-1] != last:
            positions.append(last)
        return positions

    def _discovery_proposals(
        self,
        frame: np.ndarray,
        reference_histogram: np.ndarray,
    ) -> list[dict]:
        height, width = frame.shape[:2]
        proposals = []
        for window_width, window_height in self.discovery_window_sizes:
            window_width = min(width, window_width)
            window_height = min(height, window_height)
            stride_x = int(round(window_width * self.discovery_stride_ratio))
            stride_y = int(round(window_height * self.discovery_stride_ratio))
            for y1 in self._axis_positions(height, window_height, stride_y):
                for x1 in self._axis_positions(width, window_width, stride_x):
                    x2 = x1 + window_width
                    y2 = y1 + window_height
                    crop = frame[y1:y2, x1:x2]
                    histogram = self._color_histogram(crop)
                    correlation = float(cv2.compareHist(
                        reference_histogram, histogram, cv2.HISTCMP_CORREL
                    ))
                    distance = float(cv2.compareHist(
                        reference_histogram, histogram, cv2.HISTCMP_BHATTACHARYYA
                    ))
                    if correlation < self.discovery_min_color_correlation:
                        continue
                    if distance > self.discovery_max_color_bhattacharyya:
                        continue
                    proposals.append({
                        "bbox": (float(x1), float(y1), float(x2), float(y2)),
                        "correlation": correlation,
                        "distance": distance,
                        "score": correlation + (1.0 - distance),
                    })

        proposals.sort(key=lambda item: item["score"], reverse=True)
        kept = []
        for proposal in proposals:
            if any(
                _bbox_iou(proposal["bbox"], other["bbox"]) >= self.discovery_nms_iou
                for other in kept
            ):
                continue
            kept.append(proposal)
            if len(kept) >= self.discovery_top_k:
                break
        return kept

    def match_discovery_windows(
        self,
        ref_url: str,
        ref_path: str,
        frame: np.ndarray,
    ) -> dict:
        ref_features = self.prepare_reference(ref_url, ref_path)
        reference_histogram = self._reference_color_histograms.get(ref_url)
        if ref_features is None or reference_histogram is None:
            return {"bbox": None, "accepted": [], "candidates": [], "proposal_count": 0}

        proposals = self._discovery_proposals(frame, reference_histogram)
        candidate_results = []
        for proposal_index, proposal in enumerate(proposals):
            x1, y1, x2, y2 = proposal["bbox"]
            window = frame[int(y1):int(y2), int(x1):int(x2)]
            long_side = max(window.shape[:2])
            scale = max(
                1,
                min(self.max_scale, int(ceil(self.target_long_side / max(1, long_side)))),
            )
            probe = (
                cv2.resize(window, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
                if scale > 1 else window
            )
            diagnostics = RoiMatchDiagnostics(
                candidate_index=proposal_index,
                candidate_bbox=proposal["bbox"],
                candidate_conf=float(proposal["score"]),
                crop_bbox=(int(x1), int(y1), int(x2), int(y2)),
                scale=scale,
                candidate_source="discovery",
            )
            expected_bbox = (0.0, 0.0, float(probe.shape[1] - 1), float(probe.shape[0] - 1))
            try:
                diagnostics = self._diagnose_lightglue(
                    ref_features,
                    probe,
                    diagnostics,
                    expected_bbox,
                    reference_histogram,
                    window,
                )
                if diagnostics.accepted and diagnostics.projected_bbox is not None:
                    px1, py1, px2, py2 = diagnostics.projected_bbox
                    diagnostics.output_bbox = (
                        x1 + px1 / scale,
                        y1 + py1 / scale,
                        x1 + px2 / scale,
                        y1 + py2 / scale,
                    )
            except Exception as error:
                diagnostics.reject_reason = f"exception:{type(error).__name__}"
            candidate_results.append(diagnostics)

        accepted = sorted(
            (item for item in candidate_results if item.accepted and item.output_bbox is not None),
            key=lambda item: item.score,
            reverse=True,
        )
        return {
            "bbox": accepted[0].output_bbox if accepted else None,
            "accepted": [item.to_dict() for item in accepted],
            "candidates": [item.to_dict() for item in candidate_results],
            "proposal_count": len(proposals),
        }

    def match_crossmodal_tiles(
        self,
        ref_url: str,
        ref_path: str,
        frame: np.ndarray,
    ) -> dict:
        """Termal referans/RGB frame için renk kullanmadan büyük karo tara.

        Bu yol tek başına nihai çıktı üretmek için değil, tracker'a verilecek
        seed adayını bulmak içindir. Güvenli hibrit kullanımda adayın zamansal
        olarak ikinci bir karede doğrulanması gerekir.
        """
        ref_features = self.prepare_reference(ref_url, ref_path)
        reference_histogram = self._reference_color_histograms.get(ref_url)
        if ref_features is None or reference_histogram is None:
            return {"bbox": None, "accepted": [], "candidates": [], "tile_count": 0}

        height, width = frame.shape[:2]
        tiles = []
        for window_width, window_height in self.crossmodal_window_sizes:
            window_width = min(width, window_width)
            window_height = min(height, window_height)
            stride_x = max(1, int(round(window_width * self.crossmodal_stride_ratio)))
            stride_y = max(1, int(round(window_height * self.crossmodal_stride_ratio)))
            for y1 in self._axis_positions(height, window_height, stride_y):
                for x1 in self._axis_positions(width, window_width, stride_x):
                    tiles.append((x1, y1, x1 + window_width, y1 + window_height))

        # _diagnose_lightglue ortak metrikleri hesaplıyor. Termal/RGB profili
        # için yalnız eşikleri geçici olarak değiştir; nesne örneği tek iş
        # parçacığında kullanıldığı deneysel benchmark'a özeldir.
        threshold_names = (
            "min_inliers", "min_inlier_ratio", "min_projection_coverage",
            "max_projection_coverage", "min_candidate_iou",
            "min_reference_hull_coverage", "min_roi_hull_coverage",
            "min_color_hist_correlation", "max_color_hist_bhattacharyya",
        )
        saved = {name: getattr(self, name) for name in threshold_names}
        self.min_inliers = self.crossmodal_min_inliers
        self.min_inlier_ratio = self.crossmodal_min_inlier_ratio
        self.min_projection_coverage = self.crossmodal_min_projection_coverage
        self.max_projection_coverage = self.crossmodal_max_projection_coverage
        self.min_candidate_iou = self.crossmodal_min_projection_coverage
        self.min_reference_hull_coverage = self.crossmodal_min_reference_hull_coverage
        self.min_roi_hull_coverage = self.crossmodal_min_roi_hull_coverage
        self.min_color_hist_correlation = -1.0
        self.max_color_hist_bhattacharyya = 1.0

        candidate_results = []
        try:
            for tile_index, (x1, y1, x2, y2) in enumerate(tiles):
                tile = frame[y1:y2, x1:x2]
                diagnostics = RoiMatchDiagnostics(
                    candidate_index=tile_index,
                    candidate_bbox=(float(x1), float(y1), float(x2), float(y2)),
                    candidate_conf=0.0,
                    crop_bbox=(x1, y1, x2, y2),
                    scale=1,
                    candidate_source="crossmodal_tile",
                )
                expected_bbox = (0.0, 0.0, float(tile.shape[1] - 1), float(tile.shape[0] - 1))
                try:
                    diagnostics = self._diagnose_lightglue(
                        ref_features,
                        tile,
                        diagnostics,
                        expected_bbox,
                        reference_histogram,
                        tile,
                    )
                    if diagnostics.accepted and diagnostics.projected_bbox is not None:
                        px1, py1, px2, py2 = diagnostics.projected_bbox
                        diagnostics.output_bbox = (
                            x1 + px1,
                            y1 + py1,
                            x1 + px2,
                            y1 + py2,
                        )
                except Exception as error:
                    diagnostics.reject_reason = f"exception:{type(error).__name__}"
                candidate_results.append(diagnostics)
        finally:
            for name, value in saved.items():
                setattr(self, name, value)

        accepted = sorted(
            (item for item in candidate_results if item.accepted and item.output_bbox is not None),
            key=lambda item: item.score,
            reverse=True,
        )
        return {
            "bbox": accepted[0].output_bbox if accepted else None,
            "accepted": [item.to_dict() for item in accepted],
            "candidates": [item.to_dict() for item in candidate_results],
            "tile_count": len(tiles),
        }
