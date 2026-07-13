"""Tile scheduler birim testleri (Faz P3).

Test senaryolari:
- Deterministik tile dongusu
- Grid boyutu ve overlap
- Track ROI onceligi
- Crop limiti
- Redundant crop atlama
- Full cycle tamamlama
- Coordinate remap guvenligi
"""

import sys
import os
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from person.tile_scheduler import TileScheduler, CropType, CropInfo


@pytest.fixture
def config():
    return {
        "person_detection": {
            "grid_rows": 2,
            "grid_cols": 3,
            "crop_overlap_ratio": 0.10,
            "max_crops_per_frame": 2,
            "full_cycle_max_frames": 3,
            "track_roi_expand_ratio": 1.8,
            "discovery_interval_frames": 1,
        }
    }


@pytest.fixture
def scheduler(config):
    return TileScheduler(config)


class TestTileSchedulerBasics:
    """Temel scheduler davranislari."""

    def test_grid_bounds_within_image(self, scheduler):
        """Tum tile'lar goruntu sinirlari icinde olmali."""
        W, H = 1920, 1080
        tiles = scheduler.get_all_tile_bounds(W, H)
        assert len(tiles) == 6
        for x1, y1, x2, y2 in tiles:
            assert 0 <= x1 < x2 <= W
            assert 0 <= y1 < y2 <= H

    def test_deterministic_cycle(self, scheduler):
        """Ayni frame idx ile ayni tile'lar gelmeli."""
        W, H = 1920, 1080
        crops1 = scheduler.schedule(0, W, H)
        scheduler.reset()
        crops2 = scheduler.schedule(0, W, H)
        assert len(crops1) == len(crops2)
        for c1, c2 in zip(crops1, crops2):
            assert c1.bbox == c2.bbox

    def test_full_cycle_covers_all_tiles(self, scheduler):
        """Full cycle'da tum tile'lar en az bir kez taranmali."""
        W, H = 1920, 1080
        scheduler.reset()
        seen_tiles = set()
        for frame in range(6):
            crops = scheduler.schedule(frame, W, H)
            for c in crops:
                if c.crop_type == CropType.DISCOVERY:
                    seen_tiles.add(c.tile_index)
        assert len(seen_tiles) == 6, f"Beklenen 6 tile, got {len(seen_tiles)}"

    def test_max_crops_per_frame_limit(self, scheduler):
        """Crop sayisi max_crops_per_frame'i gecmemeli."""
        W, H = 1920, 1080
        for frame in range(10):
            crops = scheduler.schedule(frame, W, H)
            assert len(crops) <= 2, f"Frame {frame}: {len(crops)} crops > limit 2"


class TestTrackROIPriority:
    """Track ROI oncelik testleri."""

    def test_track_roi_before_discovery(self, scheduler):
        """Track ROI'ler discovery tile'lardan once gelmeli."""
        W, H = 1920, 1080
        tracks = [{"bbox": (100, 100, 200, 200), "id": 0}]
        crops = scheduler.schedule(0, W, H, active_tracks=tracks)
        assert len(crops) > 0
        assert crops[0].crop_type == CropType.TRACK_ROI

    def test_lost_roi_highest_priority(self, scheduler):
        """Lost track ROI'leri en yuksek oncelikli olmali."""
        W, H = 1920, 1080
        tracks = [{"bbox": (500, 500, 600, 600), "id": 0}]
        lost_rois = [(100, 100, 300, 300)]
        crops = scheduler.schedule(0, W, H, active_tracks=tracks, lost_track_rois=lost_rois)
        assert len(crops) > 0
        assert crops[0].crop_type == CropType.TRACK_ROI
        assert crops[0].track_id == -1  # lost track

    def test_track_roi_expand_ratio(self, scheduler):
        """Track ROI expand ratio ile genisletilmeli."""
        W, H = 1920, 1080
        track_bbox = (100, 100, 200, 200)
        tracks = [{"bbox": track_bbox, "id": 0}]
        crops = scheduler.schedule(0, W, H, active_tracks=tracks)
        track_crop = crops[0]
        roi = track_crop.bbox
        cx_orig = (100 + 200) / 2
        cy_orig = (100 + 200) / 2
        cx_roi = (roi[0] + roi[2]) / 2
        cy_roi = (roi[1] + roi[3]) / 2
        assert abs(cx_orig - cx_roi) < 2.0
        assert abs(cy_orig - cy_roi) < 2.0
        roi_w = roi[2] - roi[0]
        roi_h = roi[3] - roi[1]
        assert roi_w > 100  # 100 * 1.8 = 180
        assert roi_h > 100


class TestSchedulerStats:
    """Scheduler istatistikleri."""

    def test_stats_tracking(self, scheduler):
        """Istatistikler dogru sayilmali."""
        W, H = 1920, 1080
        scheduler.reset()
        for frame in range(3):
            scheduler.schedule(frame, W, H)
        stats = scheduler.get_stats()
        assert stats["total_frames"] == 3
        assert stats["total_crops"] > 0
        assert stats["discovery_crops"] > 0

    def test_stats_with_track_roi(self, scheduler):
        """Track ROI istatistikleri dogru sayilmali."""
        W, H = 1920, 1080
        scheduler.reset()
        tracks = [{"bbox": (100, 100, 200, 200), "id": 0}]
        scheduler.schedule(0, W, H, active_tracks=tracks)
        stats = scheduler.get_stats()
        assert stats["track_roi_crops"] > 0


class TestRedundancySkip:
    """Redundant crop atlama."""

    def test_redundant_track_roi_skipped(self, scheduler):
        """Ayni bolgedeki track ROI'ler redundant olarak atlanmali."""
        W, H = 1920, 1080
        tracks = [
            {"bbox": (100, 100, 200, 200), "id": 0},
            {"bbox": (110, 110, 210, 210), "id": 1},  # cakisan
        ]
        crops = scheduler.schedule(0, W, H, active_tracks=tracks)
        track_crops = [c for c in crops if c.crop_type == CropType.TRACK_ROI]
        assert len(track_crops) <= 1  # redundant atlandi
        stats = scheduler.get_stats()
        assert stats["skipped_redundant"] >= 1
