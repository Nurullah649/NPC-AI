"""Deterministik temporal micro-tiling scheduler (Faz P3).

1920x1080 goruntuyu 2x3 mantiksal discovery bolgesine bol.
Her karede yalniz 1-2 native-resolution crop isle.
Tum goruntu 3-6 karelik dongude taranir.
Aktif track cevresindeki ROI'ler discovery bolgelerinden once islenir.

Scheduler istatistikleri:
- discovery crop sayisi
- track ROI crop sayisi
- toplam ek inference
"""

import logging
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class CropType(IntEnum):
    DISCOVERY = 0
    TRACK_ROI = 1


@dataclass
class CropInfo:
    """Tek bir crop'un bilgisi."""
    crop_type: CropType
    bbox: tuple  # (x1, y1, x2, y2) orijinal goruntu koordinatlarinda
    tile_index: int = -1
    track_id: int = -1
    frame_idx: int = 0


@dataclass
class SchedulerStats:
    """Scheduler istatistikleri."""
    total_frames: int = 0
    discovery_crops: int = 0
    track_roi_crops: int = 0
    total_crops: int = 0
    skipped_redundant: int = 0
    full_cycles_completed: int = 0

    def to_dict(self) -> dict:
        return {
            "total_frames": self.total_frames,
            "discovery_crops": self.discovery_crops,
            "track_roi_crops": self.track_roi_crops,
            "total_crops": self.total_crops,
            "skipped_redundant": self.skipped_redundant,
            "full_cycles_completed": self.full_cycles_completed,
        }


class TileScheduler:
    """Deterministik temporal micro-tiling scheduler.

    Args:
        config: person_detection section of config_experiment.yaml
    """

    def __init__(self, config: dict):
        pd_cfg = config.get("person_detection", {})
        self.grid_rows = int(pd_cfg.get("grid_rows", 2))
        self.grid_cols = int(pd_cfg.get("grid_cols", 3))
        self.crop_overlap_ratio = float(pd_cfg.get("crop_overlap_ratio", 0.10))
        self.max_crops_per_frame = int(pd_cfg.get("max_crops_per_frame", 2))
        self.min_discovery_slots = int(pd_cfg.get("min_discovery_slots", 1))
        self.full_cycle_max_frames = int(pd_cfg.get("full_cycle_max_frames", 3))
        self.track_roi_expand_ratio = float(pd_cfg.get("track_roi_expand_ratio", 1.8))
        self.discovery_interval_frames = int(pd_cfg.get("discovery_interval_frames", 1))

        self._current_tile_idx = 0
        self._total_tiles = self.grid_rows * self.grid_cols
        self._frame_count = 0
        self._cycle_start_frame = 0
        self._discovery_frame_counter = 0
        self.stats = SchedulerStats()

        logger.info(
            "TileScheduler: grid=%dx%d, overlap=%.2f, max_crops/frame=%d, cycle=%d frames, min_discovery=%d",
            self.grid_rows, self.grid_cols, self.crop_overlap_ratio,
            self.max_crops_per_frame, self.full_cycle_max_frames,
            self.min_discovery_slots,
        )

    def _compute_tile_bounds(self, width: int, height: int, tile_idx: int) -> tuple:
        """Tek bir tile'in sinirlarini hesapla (overlap dahil).

        Tile indexing: row-major (row 0: tiles 0..cols-1, row 1: tiles cols..2*cols-1, ...)
        """
        row = tile_idx // self.grid_cols
        col = tile_idx % self.grid_cols

        base_w = width / self.grid_cols
        base_h = height / self.grid_rows

        overlap_w = base_w * self.crop_overlap_ratio
        overlap_h = base_h * self.crop_overlap_ratio

        x1 = max(0, int(col * base_w - overlap_w / 2))
        y1 = max(0, int(row * base_h - overlap_h / 2))
        x2 = min(width, int((col + 1) * base_w + overlap_w / 2))
        y2 = min(height, int((row + 1) * base_h + overlap_h / 2))

        return (x1, y1, x2, y2)

    def get_all_tile_bounds(self, width: int, height: int) -> list:
        """Tum tile'larin sinirlarini dondur."""
        return [self._compute_tile_bounds(width, height, i)
                for i in range(self._total_tiles)]

    def _get_next_discovery_tiles(self, n: int) -> list:
        """Sonraki n discovery tile indexini dondur (deterministik dongu)."""
        tiles = []
        for _ in range(n):
            tiles.append(self._current_tile_idx)
            self._current_tile_idx = (self._current_tile_idx + 1) % self._total_tiles
            if self._current_tile_idx == 0:
                self.stats.full_cycles_completed += 1
        return tiles

    def _build_track_roi(self, track_bbox: tuple, width: int, height: int,
                         track_id: int) -> tuple:
        """Track cevresinde genisletilmis ROI olustur."""
        x1, y1, x2, y2 = track_bbox
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        w = (x2 - x1) * self.track_roi_expand_ratio
        h = (y2 - y1) * self.track_roi_expand_ratio

        roi_x1 = max(0, int(cx - w / 2))
        roi_y1 = max(0, int(cy - h / 2))
        roi_x2 = min(width, int(cx + w / 2))
        roi_y2 = min(height, int(cy + h / 2))

        return (roi_x1, roi_y1, roi_x2, roi_y2)

    @staticmethod
    def _iou(box1: tuple, box2: tuple) -> float:
        """Iki kutu arasi IoU."""
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

    def _is_redundant(self, crop_bbox: tuple, existing_crops: list,
                      iou_threshold: float = 0.6) -> bool:
        """Crop, mevcut crop'larla cok ovelap ediyorsa redundant."""
        for existing in existing_crops:
            if self._iou(crop_bbox, existing.bbox) > iou_threshold:
                return True
        return False

    def schedule(
        self,
        frame_idx: int,
        width: int,
        height: int,
        active_tracks: list = None,
        lost_track_rois: list = None,
    ) -> list:
        """Bir kare icin crop listesini planla.

        Oncelik:
        1. Lost track ROI'leri (en yuksek oncelik - track kurtarma)
        2. Active track ROI'leri
        3. Discovery tile'lari

        Args:
            frame_idx: Mevcut kare indeksi
            width: Goruntu genisligi
            height: Goruntu yuksekligi
            active_tracks: Aktif track bbox listesi: [{"bbox": (x1,y1,x2,y2), "id": int}, ...]
            lost_track_rois: Kaybolmus track ROI listesi: [(x1,y1,x2,y2), ...]

        Returns:
            CropInfo listesi (max_crops_per_frame ile sinirli)
        """
        self._frame_count = frame_idx
        self.stats.total_frames += 1

        crops = []
        active_tracks = active_tracks or []
        lost_track_rois = lost_track_rois or []

        # Slot allocation: discovery slot guarantee
        if self.max_crops_per_frame == 1:
            # Alternating mode: even frames discovery, odd frames track ROI
            if frame_idx % 2 == 0:
                max_track_roi_slots = 0
                discovery_planned = 1
            else:
                max_track_roi_slots = 1
                discovery_planned = 0
        else:
            max_track_roi_slots = self.max_crops_per_frame - self.min_discovery_slots
            discovery_planned = self.min_discovery_slots

        # discovery_interval_frames kontrolu
        if discovery_planned > 0 and \
           self._discovery_frame_counter % self.discovery_interval_frames != 0:
            discovery_planned = 0
            max_track_roi_slots = self.max_crops_per_frame

        # 1. Lost track ROI'leri (en yuksek oncelik)
        for roi_bbox in lost_track_rois:
            if len(crops) >= max_track_roi_slots:
                break
            crop = CropInfo(
                crop_type=CropType.TRACK_ROI,
                bbox=roi_bbox,
                track_id=-1,
                frame_idx=frame_idx,
            )
            if not self._is_redundant(roi_bbox, crops):
                crops.append(crop)
                self.stats.track_roi_crops += 1
            else:
                self.stats.skipped_redundant += 1

        # 2. Active track ROI'leri
        for track in active_tracks:
            if len(crops) >= max_track_roi_slots:
                break
            track_bbox = track.get("bbox")
            track_id = track.get("id", -1)
            if track_bbox is None:
                continue
            roi_bbox = self._build_track_roi(track_bbox, width, height, track_id)
            crop = CropInfo(
                crop_type=CropType.TRACK_ROI,
                bbox=roi_bbox,
                track_id=track_id,
                frame_idx=frame_idx,
            )
            if not self._is_redundant(roi_bbox, crops):
                crops.append(crop)
                self.stats.track_roi_crops += 1
            else:
                self.stats.skipped_redundant += 1

        for track in active_tracks[len(crops):]:
            self.stats.skipped_redundant += 1

        # 3. Discovery tile'lari
        remaining = self.max_crops_per_frame - len(crops)
        discovery_actual = min(discovery_planned, remaining)
        if discovery_actual <= 0 and remaining > 0:
            discovery_actual = remaining
        if discovery_actual > 0:
            discovery_tiles = self._get_next_discovery_tiles(discovery_actual)
            for tile_idx in discovery_tiles:
                tile_bbox = self._compute_tile_bounds(width, height, tile_idx)
                crop = CropInfo(
                    crop_type=CropType.DISCOVERY,
                    bbox=tile_bbox,
                    tile_index=tile_idx,
                    frame_idx=frame_idx,
                )
                if not self._is_redundant(tile_bbox, crops):
                    crops.append(crop)
                    self.stats.discovery_crops += 1
                else:
                    self.stats.skipped_redundant += 1

        if discovery_actual > 0:
            self._discovery_frame_counter += 1

        self.stats.total_crops += len(crops)

        logger.debug(
            "Frame %d: %d crops (%d track_roi, %d discovery, %d skipped)",
            frame_idx, len(crops),
            sum(1 for c in crops if c.crop_type == CropType.TRACK_ROI),
            sum(1 for c in crops if c.crop_type == CropType.DISCOVERY),
            self.stats.skipped_redundant,
        )

        return crops

    def get_stats(self) -> dict:
        """Scheduler istatistiklerini dondur."""
        return self.stats.to_dict()

    def reset(self):
        """Scheduler'i sifirla."""
        self._current_tile_idx = 0
        self._frame_count = 0
        self._cycle_start_frame = 0
        self._discovery_frame_counter = 0
        self.stats = SchedulerStats()
