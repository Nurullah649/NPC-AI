"""Gorev 3 icin dogrulama, takip ve sahne hafizasi orkestrasyonu.

Pipeline yalniz guvenilir bir tam-kare/YOLO-ROI eslesmesiyle track baslatir.
Sabit bir referans daha sonra farkli modalitede (termal/RGB) yayinlanirsa,
config'de acikca tanimlanan ayni-nesne bagi uzerinden onceki dogrulanmis sahne
anahtar kareleriyle yeniden konumlandirma yapabilir.
"""

from __future__ import annotations

import logging
import os
import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from deneysel.reference_roi_v1.reference_scene_relocalizer import (
    ReferenceSceneRelocalizer,
)
from deneysel.reference_roi_v1.reference_tracker import (
    CameraCompensatedReferenceTracker,
)
from deneysel.reference_roi_v1.roi_matcher import RoiReferenceMatcher

from .reference_matcher import ReferenceMatcher


@dataclass
class ReferencePipelineResult:
    bbox: Optional[tuple[float, float, float, float]]
    source: str
    reason: str
    reference_order: Optional[int]
    details: dict

    def to_dict(self) -> dict:
        payload = asdict(self)
        if self.bbox is not None:
            payload["bbox"] = [float(value) for value in self.bbox]
        return payload


class ReferenceDetectionPipeline:
    """Stateful reference detector used by the main simulation loop."""

    def __init__(self, config: dict):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.config = config
        cfg = config.get("reference_roi_experiment", {})
        self.enabled = bool(cfg.get("enabled", True))
        self.precompute_all = bool(cfg.get("precompute_all_references", True))
        self.scene_keyframe_step = max(1, int(cfg.get("scene_keyframe_step", 3)))
        self.scene_max_keyframes = max(1, int(cfg.get("scene_max_keyframes", 9)))
        self.scene_min_keyframes = max(1, int(cfg.get("scene_min_keyframes", 3)))
        self.scene_track_min_visible_ratio = float(
            cfg.get("scene_track_min_visible_ratio", 0.50)
        )
        self.scene_alias_requires_validation = bool(
            cfg.get("scene_alias_requires_validation", True)
        )
        self.persist_scene_memory = bool(cfg.get("persist_scene_memory", True))
        self.scene_memory_filename = str(
            cfg.get("scene_memory_filename", "_reference_scene_memory.json")
        )
        self.full_min_frame_coverage = float(
            cfg.get("full_min_frame_coverage", 0.0005)
        )
        self.full_max_frame_coverage = float(
            cfg.get("full_max_frame_coverage", 0.85)
        )
        self.full_max_frame_coverage_by_order = {
            int(order): float(value)
            for order, value in (
                cfg.get("full_max_frame_coverage_by_order", {}) or {}
            ).items()
        }
        self.scene_aliases = {
            int(target): int(source)
            for target, source in (cfg.get("scene_aliases", {}) or {}).items()
        }
        self.scene_source_orders = set(self.scene_aliases.values())
        self.roi_seed_expansions = {
            int(order): {
                name: float(values.get(name, 0.0))
                for name in ("left", "right", "top", "bottom")
            }
            for order, values in (cfg.get("roi_seed_expansion_by_order", {}) or {}).items()
        }
        self.seed_confirmation_frames = max(
            1, int(cfg.get("seed_confirmation_frames", 2))
        )
        self.seed_confirmation_max_gap = max(
            1, int(cfg.get("seed_confirmation_max_gap", 5))
        )
        self.seed_confirmation_min_iou = float(
            cfg.get("seed_confirmation_min_iou", 0.20)
        )
        self.tracker_revalidate_every = max(
            0, int(cfg.get("tracker_revalidate_every", 10))
        )
        self.tracker_revalidate_min_iou = float(
            cfg.get("tracker_revalidate_min_iou", 0.15)
        )
        self.tracker_max_validation_failures = max(
            1, int(cfg.get("tracker_max_validation_failures", 2))
        )

        self.base = ReferenceMatcher(config)
        self.roi = RoiReferenceMatcher(config, base=self.base)
        self.trackers: dict[str, CameraCompensatedReferenceTracker] = {}
        self.scene_memories: dict[int, ReferenceSceneRelocalizer] = {}
        self.reference_orders: dict[str, int] = {}
        self.reference_paths: dict[str, str] = {}
        self._session_dir: Optional[Path] = None
        self._memory_restored = False
        self._memory_track_counts: dict[int, int] = defaultdict(int)
        self._last_processed_frame: dict[str, int] = {}
        self._last_results: dict[str, tuple[int, ReferencePipelineResult]] = {}
        self._pending_seeds: dict[str, dict] = {}
        self._last_tile_attempt: dict[str, int] = {}
        self._last_tracker_validation: dict[str, int] = {}
        self._tracker_validation_failures: dict[str, int] = defaultdict(int)
        self.last_diagnostics: dict[str, dict] = {}

        self.logger.info(
            "Reference pipeline: enabled=%s aliases=%s keyframes=%d/%d",
            self.enabled,
            self.scene_aliases,
            self.scene_min_keyframes,
            self.scene_max_keyframes,
        )

    @property
    def feature_cache(self):
        """Backward-compatible access used by diagnostics/tests."""
        return self.base.feature_cache

    @staticmethod
    def _order(reference: dict | None) -> Optional[int]:
        if not isinstance(reference, dict):
            return None
        try:
            return int(reference.get("order"))
        except (TypeError, ValueError):
            return None

    def precompute_reference(self, ref_url: str, ref_path: str):
        return self.roi.prepare_reference(ref_url, ref_path)

    def register_references(self, references: list[dict], ref_image_paths: dict[str, str]):
        """Register every published reference and optionally cache its features."""
        for reference in references or []:
            if not isinstance(reference, dict):
                continue
            ref_url = str(reference.get("url", ""))
            if not ref_url:
                continue
            order = self._order(reference)
            if order is not None:
                self.reference_orders[ref_url] = order
            ref_path = ref_image_paths.get(ref_url)
            if ref_path:
                self.reference_paths[ref_url] = ref_path
                if self._session_dir is None:
                    self._session_dir = Path(ref_path).resolve().parent.parent
            if not self.precompute_all or not ref_path or not os.path.exists(ref_path):
                continue
            try:
                self.precompute_reference(ref_url, ref_path)
            except Exception as error:
                self.logger.warning(
                    "Referans on-hazirlama basarisiz (order=%s): %s", order, error
                )
        self._restore_scene_memories()
        self.logger.info(
            "%d/%d referansin feature cache'i hazir.",
            len(self.feature_cache),
            len(references or []),
        )

    def _tracker(self, ref_url: str) -> CameraCompensatedReferenceTracker:
        tracker = self.trackers.get(ref_url)
        if tracker is None:
            tracker = CameraCompensatedReferenceTracker(self.config)
            self.trackers[ref_url] = tracker
        return tracker

    def _scene_memory(self, source_order: int) -> ReferenceSceneRelocalizer:
        memory = self.scene_memories.get(source_order)
        if memory is None:
            memory = ReferenceSceneRelocalizer(self.config)
            self.scene_memories[source_order] = memory
        return memory

    def _scene_memory_path(self) -> Optional[Path]:
        if not self.persist_scene_memory or self._session_dir is None:
            return None
        return self._session_dir / self.scene_memory_filename

    def _restore_scene_memories(self) -> None:
        if self._memory_restored:
            return
        self._memory_restored = True
        cache_path = self._scene_memory_path()
        if cache_path is None or not cache_path.exists():
            return
        try:
            with cache_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            restored = 0
            for order_text, items in (payload.get("memories", {}) or {}).items():
                order = int(order_text)
                if order not in self.scene_source_orders:
                    continue
                expected_url = next(
                    (url for url, value in self.reference_orders.items() if value == order),
                    None,
                )
                cached_url = (payload.get("reference_urls", {}) or {}).get(order_text)
                if expected_url and cached_url and expected_url != cached_url:
                    continue
                memory = self._scene_memory(order)
                for item in items[:self.scene_max_keyframes]:
                    image_path = self._session_dir / str(item.get("image", ""))
                    image = cv2.imread(str(image_path))
                    polygon = item.get("polygon")
                    if image is None or polygon is None:
                        continue
                    memory.add_keyframe(int(item["frame"]), image, polygon)
                    restored += 1
                self._memory_track_counts[order] = (
                    len(memory.keyframes) * self.scene_keyframe_step
                )
            if restored:
                self.logger.info(
                    "Kalici referans sahne hafizasindan %d keyframe geri yuklendi.",
                    restored,
                )
        except Exception as error:
            self.logger.warning("Referans sahne hafizasi geri yuklenemedi: %s", error)

    def _persist_scene_memories(self) -> None:
        cache_path = self._scene_memory_path()
        if cache_path is None:
            return
        memories = {}
        reference_urls = {}
        for order, memory in self.scene_memories.items():
            items = []
            for keyframe in memory.keyframes:
                frame_idx = int(keyframe["frame"])
                candidates = sorted(self._session_dir.glob(f"frame_{frame_idx:06d}.*"))
                if not candidates:
                    continue
                items.append({
                    "frame": frame_idx,
                    "image": candidates[0].name,
                    "polygon": keyframe["polygon"].astype(float).tolist(),
                })
            if items:
                memories[str(order)] = items
                source_url = next(
                    (url for url, value in self.reference_orders.items() if value == order),
                    None,
                )
                if source_url:
                    reference_urls[str(order)] = source_url
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = cache_path.with_suffix(cache_path.suffix + ".tmp")
            with temporary.open("w", encoding="utf-8") as handle:
                json.dump(
                    {"version": 1, "reference_urls": reference_urls, "memories": memories},
                    handle,
                    indent=2,
                )
            os.replace(temporary, cache_path)
        except Exception as error:
            self.logger.warning("Referans sahne hafizasi kaydedilemedi: %s", error)

    def _expand_roi_seed(self, bbox, frame: np.ndarray, order: Optional[int]):
        expansion = self.roi_seed_expansions.get(order)
        if not expansion:
            return tuple(float(value) for value in bbox)
        height, width = frame.shape[:2]
        x1, y1, x2, y2 = [float(value) for value in bbox]
        box_width = max(1.0, x2 - x1)
        box_height = max(1.0, y2 - y1)
        return (
            float(np.clip(x1 - box_width * expansion["left"], 0, width - 1)),
            float(np.clip(y1 - box_height * expansion["top"], 0, height - 1)),
            float(np.clip(x2 + box_width * expansion["right"], 0, width - 1)),
            float(np.clip(y2 + box_height * expansion["bottom"], 0, height - 1)),
        )

    def _full_bbox_is_sane(
        self, bbox, frame: np.ndarray, order: Optional[int]
    ) -> tuple[bool, float]:
        height, width = frame.shape[:2]
        x1, y1, x2, y2 = [float(value) for value in bbox]
        if not np.isfinite([x1, y1, x2, y2]).all() or x1 >= x2 or y1 >= y2:
            return False, 0.0
        coverage = (x2 - x1) * (y2 - y1) / max(1.0, float(width * height))
        max_coverage = self.full_max_frame_coverage_by_order.get(
            order, self.full_max_frame_coverage
        )
        return self.full_min_frame_coverage <= coverage <= max_coverage, float(coverage)

    @staticmethod
    def _bbox_iou(box_a, box_b) -> float:
        x1 = max(float(box_a[0]), float(box_b[0]))
        y1 = max(float(box_a[1]), float(box_b[1]))
        x2 = min(float(box_a[2]), float(box_b[2]))
        y2 = min(float(box_a[3]), float(box_b[3]))
        if x1 >= x2 or y1 >= y2:
            return 0.0
        intersection = (x2 - x1) * (y2 - y1)
        area_a = max(
            1e-6, (float(box_a[2]) - float(box_a[0]))
            * (float(box_a[3]) - float(box_a[1]))
        )
        area_b = max(
            1e-6, (float(box_b[2]) - float(box_b[0]))
            * (float(box_b[3]) - float(box_b[1]))
        )
        return float(intersection / (area_a + area_b - intersection))

    def _expire_pending_seed(self, ref_url: str, frame_idx: int) -> None:
        pending = self._pending_seeds.get(ref_url)
        if pending is None:
            return
        if frame_idx - int(pending["frame_idx"]) > self.seed_confirmation_max_gap:
            self._pending_seeds.pop(ref_url, None)

    def _confirm_and_seed(
        self,
        ref_url: str,
        frame_idx: int,
        frame: np.ndarray,
        tracker: CameraCompensatedReferenceTracker,
        bbox,
        source: str,
        order: Optional[int],
        details: dict,
    ) -> ReferencePipelineResult:
        """Tek karelik sahte homografinin tracker'ı başlatmasını engelle."""
        candidate = tuple(float(value) for value in bbox)
        if self.seed_confirmation_frames <= 1:
            seeded = tracker.seed(frame, candidate, expand=False)
            self._last_tracker_validation[ref_url] = frame_idx
            self._tracker_validation_failures[ref_url] = 0
            return self._finish(
                ref_url, frame_idx, seeded, f"{source}_seed", "verified_seed",
                order, details,
            )

        pending = self._pending_seeds.get(ref_url)
        confirmation_iou = 0.0
        hits = 1
        if pending is not None:
            gap = frame_idx - int(pending["frame_idx"])
            confirmation_iou = self._bbox_iou(pending["bbox"], candidate)
            if (
                1 <= gap <= self.seed_confirmation_max_gap
                and confirmation_iou >= self.seed_confirmation_min_iou
            ):
                hits = int(pending.get("hits", 1)) + 1

        details["seed_confirmation"] = {
            "hits": hits,
            "required": self.seed_confirmation_frames,
            "iou": confirmation_iou,
        }
        if hits >= self.seed_confirmation_frames:
            self._pending_seeds.pop(ref_url, None)
            seeded = tracker.seed(frame, candidate, expand=False)
            self._last_tracker_validation[ref_url] = frame_idx
            self._tracker_validation_failures[ref_url] = 0
            return self._finish(
                ref_url, frame_idx, seeded, f"{source}_seed", "verified_seed",
                order, details,
            )

        self._pending_seeds[ref_url] = {
            "frame_idx": frame_idx,
            "bbox": candidate,
            "hits": hits,
            "source": source,
        }
        details["candidate_bbox"] = [float(value) for value in candidate]
        return self._finish(
            ref_url,
            frame_idx,
            None,
            "none",
            "awaiting_seed_confirmation",
            order,
            details,
        )

    def _tracker_validation_due(self, ref_url: str, frame_idx: int) -> bool:
        if self.tracker_revalidate_every <= 0:
            return False
        last = self._last_tracker_validation.get(ref_url, frame_idx)
        return frame_idx - last >= self.tracker_revalidate_every

    def _remember_tracked_polygon(
        self,
        order: Optional[int],
        frame_idx: int,
        frame: np.ndarray,
        polygon,
    ) -> None:
        if order is None or order not in self.scene_source_orders or polygon is None:
            return
        memory = self._scene_memory(order)
        count = self._memory_track_counts[order]
        self._memory_track_counts[order] = count + 1
        if count % self.scene_keyframe_step != 0:
            return
        if len(memory.keyframes) >= self.scene_max_keyframes:
            return
        try:
            memory.add_keyframe(frame_idx, frame, polygon)
            self._persist_scene_memories()
            self.logger.info(
                "Ref order=%d sahne hafizasi: keyframe=%d frame=%d",
                order,
                len(memory.keyframes),
                frame_idx,
            )
        except Exception as error:
            self.logger.warning("Referans sahne keyframe'i eklenemedi: %s", error)

    def _finish(
        self,
        ref_url: str,
        frame_idx: int,
        bbox,
        source: str,
        reason: str,
        order: Optional[int],
        details: dict,
    ) -> ReferencePipelineResult:
        normalized = (
            None if bbox is None else tuple(float(value) for value in bbox)
        )
        result = ReferencePipelineResult(normalized, source, reason, order, details)
        self._last_processed_frame[ref_url] = frame_idx
        self._last_results[ref_url] = (frame_idx, result)
        self.last_diagnostics[ref_url] = result.to_dict()
        return result

    def _accept_scene_result(
        self,
        ref_url: str,
        frame_idx: int,
        frame: np.ndarray,
        tracker: CameraCompensatedReferenceTracker,
        order: Optional[int],
        scene,
        details: dict,
    ) -> Optional[ReferencePipelineResult]:
        details["scene"] = scene.to_dict()
        if not scene.accepted or scene.bbox is None:
            return None
        # Kadraj sinirindaki kismi gorunum dogru bir sonuc olabilir, fakat bu
        # kutuyla tracker seed edilirse nesne ciktiktan sonra zemini tasiyabilir.
        if scene.visible_ratio < self.scene_track_min_visible_ratio:
            tracker.reset()
            return self._finish(
                ref_url,
                frame_idx,
                scene.bbox,
                "scene_match",
                "verified_partial_scene",
                order,
                details,
            )
        bbox = tracker.seed(frame, scene.bbox, expand=False)
        return self._finish(
            ref_url,
            frame_idx,
            bbox,
            "scene_seed",
            "verified_scene_relocalization",
            order,
            details,
        )

    def match_reference(
        self,
        reference: dict,
        ref_path: str,
        frame: np.ndarray,
        detections: list[dict],
        frame_idx: int,
    ) -> ReferencePipelineResult:
        """Find one active reference; return no box unless a gate accepts it."""
        ref_url = str(reference.get("url", "")) if isinstance(reference, dict) else ""
        order = self._order(reference)
        if not ref_url or frame is None or not ref_path:
            return ReferencePipelineResult(None, "none", "invalid_input", order, {})

        cached = self._last_results.get(ref_url)
        if cached is not None and cached[0] == frame_idx:
            return cached[1]

        if not self.enabled:
            bbox = self.base.match(ref_url, ref_path, frame)
            return self._finish(
                ref_url, frame_idx, bbox, "full" if bbox else "none",
                "legacy_match" if bbox else "no_verified_match", order, {},
            )

        tracker = self._tracker(ref_url)
        previous_frame = self._last_processed_frame.get(ref_url)
        if (
            tracker.active
            and previous_frame is not None
            and frame_idx != previous_frame + 1
        ):
            tracker.reset()
            self._pending_seeds.pop(ref_url, None)
            self._tracker_validation_failures[ref_url] = 0

        self._expire_pending_seed(ref_url, frame_idx)

        if tracker.active:
            tracked = tracker.update(frame)
            tracked_dict = tracked.to_dict()
            if tracked.active:
                if self._tracker_validation_due(ref_url, frame_idx):
                    validation = self.roi.validate_bbox(
                        ref_url, ref_path, frame, tracked.bbox
                    )
                    validation_iou = (
                        self._bbox_iou(tracked.bbox, validation["bbox"])
                        if validation.get("bbox") is not None else 0.0
                    )
                    self._last_tracker_validation[ref_url] = frame_idx
                    if (
                        validation.get("accepted")
                        and validation_iou >= self.tracker_revalidate_min_iou
                    ):
                        self._tracker_validation_failures[ref_url] = 0
                        corrected = tracker.seed(
                            frame, validation["bbox"], expand=False
                        )
                        self._remember_tracked_polygon(
                            order, frame_idx, frame, tracker.polygon
                        )
                        return self._finish(
                            ref_url,
                            frame_idx,
                            corrected,
                            "tracker_revalidated",
                            "reference_revalidated",
                            order,
                            {
                                "tracker": tracked_dict,
                                "validation": validation,
                                "validation_iou": validation_iou,
                            },
                        )

                    failures = self._tracker_validation_failures[ref_url] + 1
                    self._tracker_validation_failures[ref_url] = failures
                    if failures >= self.tracker_max_validation_failures:
                        tracker.reset()
                        self._pending_seeds.pop(ref_url, None)
                        return self._finish(
                            ref_url,
                            frame_idx,
                            None,
                            "none",
                            "tracker_reference_validation_failed",
                            order,
                            {
                                "tracker": tracked_dict,
                                "validation": validation,
                                "validation_iou": validation_iou,
                                "validation_failures": failures,
                            },
                        )
                self._remember_tracked_polygon(
                    order, frame_idx, frame, tracked.polygon
                )
                return self._finish(
                    ref_url, frame_idx, tracked.bbox, "tracker", tracked.reason,
                    order, {
                        "tracker": tracked_dict,
                        "validation_failures": self._tracker_validation_failures[ref_url],
                    },
                )
            # Tracker iki gecici guvensiz kare boyunca aktif kalabilir. Bu
            # karelerde sonuc eklemeyiz ve zayif bir seed ile ustune yazmayiz.
            if tracker.active:
                return self._finish(
                    ref_url, frame_idx, None, "none", tracked.reason,
                    order, {"tracker": tracked_dict},
                )
            self._tracker_validation_failures[ref_url] = 0

        details: dict = {}
        source_order = self.scene_aliases.get(order) if order is not None else None
        memory = self.scene_memories.get(source_order) if source_order is not None else None
        scene_ready = memory is not None and len(memory.keyframes) >= self.scene_min_keyframes

        # Modaliteler arasi dogrudan LightGlue bu veri setinde yanlis pozitif
        # uretti. Ayni-nesne bagi tanimliysa sahne hafizasi asil dogrulamadir;
        # reddedilen karede daha zayif tam-kare/ROI sonucu ile seed yapilmaz.
        if source_order is not None and self.scene_alias_requires_validation:
            if not scene_ready:
                details["scene_keyframes"] = 0 if memory is None else len(memory.keyframes)
                return self._finish(
                    ref_url,
                    frame_idx,
                    None,
                    "none",
                    "scene_memory_unavailable",
                    order,
                    details,
                )
            accepted_scene = self._accept_scene_result(
                ref_url,
                frame_idx,
                frame,
                tracker,
                order,
                memory.relocalize(frame),
                details,
            )
            if accepted_scene is not None:
                return accepted_scene
            return self._finish(
                ref_url, frame_idx, None, "none", "scene_validation_failed", order, details
            )

        full_bbox = self.roi.match_full(ref_url, ref_path, frame)
        details["full_bbox"] = (
            None if full_bbox is None else [float(value) for value in full_bbox]
        )
        details["full_match"] = dict(
            getattr(self.base, "last_match_diagnostics", {}) or {}
        )
        details["full_match_attempts"] = list(
            getattr(self.base, "last_match_attempts", []) or []
        )
        if full_bbox is not None:
            sane, coverage = self._full_bbox_is_sane(full_bbox, frame, order)
            details["full_frame_coverage"] = coverage
            if sane:
                return self._confirm_and_seed(
                    ref_url,
                    frame_idx,
                    frame,
                    tracker,
                    full_bbox,
                    "full",
                    order,
                    details,
                )
            details["full_reject_reason"] = "implausible_frame_coverage"

        roi_result = self.roi.match_candidates(
            ref_url, ref_path, frame, detections or []
        )
        details["roi_accepted_count"] = len(roi_result.get("accepted", []))
        details["roi_candidate_count"] = len(roi_result.get("candidates", []))
        roi_bbox = roi_result.get("bbox")
        if roi_bbox is not None:
            expanded = self._expand_roi_seed(roi_bbox, frame, order)
            details["roi_bbox"] = [float(value) for value in roi_bbox]
            return self._confirm_and_seed(
                ref_url,
                frame_idx,
                frame,
                tracker,
                expanded,
                "roi",
                order,
                details,
            )

        last_tile_frame = self._last_tile_attempt.get(ref_url)
        tile_due = (
            last_tile_frame is None
            or frame_idx - last_tile_frame >= self.roi.tile_frame_step
        )
        if tile_due:
            self._last_tile_attempt[ref_url] = frame_idx
            tile_result = self.roi.match_tiles(
                ref_url, ref_path, frame
            )
            details["tile_accepted_count"] = len(
                tile_result.get("accepted", [])
            )
            details["tile_candidate_count"] = len(
                tile_result.get("candidates", [])
            )
            tile_bbox = tile_result.get("bbox")
            if tile_bbox is not None:
                details["tile_best"] = tile_result["accepted"][0]
                return self._confirm_and_seed(
                    ref_url,
                    frame_idx,
                    frame,
                    tracker,
                    tile_bbox,
                    "tile",
                    order,
                    details,
                )
        else:
            details["tile_skipped"] = "frame_step"

        if scene_ready:
            accepted_scene = self._accept_scene_result(
                ref_url,
                frame_idx,
                frame,
                tracker,
                order,
                memory.relocalize(frame),
                details,
            )
            if accepted_scene is not None:
                return accepted_scene
        elif source_order is not None:
            details["scene_keyframes"] = 0 if memory is None else len(memory.keyframes)

        return self._finish(
            ref_url, frame_idx, None, "none", "no_verified_match", order, details
        )

    def match(self, ref_url: str, ref_path: str, frame_img: np.ndarray):
        """Legacy matcher API; stateful production path uses match_reference."""
        return self.base.match(ref_url, ref_path, frame_img)
