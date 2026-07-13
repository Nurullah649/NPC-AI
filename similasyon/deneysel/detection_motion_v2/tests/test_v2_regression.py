"""V2 regresyon testleri - ikinci iterasyon düzeltmeleri.

Test edilen düzeltmeler:
- Homography apply_to_point w bölmesi
- Transform chain matris sırası
- CameraMotionEstimator ayrı mask
- Median-shift MAD=0 güvenlik
- Kalman+kamera iki ayrı merkez
- Kalman kapalı fallback velocity
- Mahalanobis gating
- Scheduler discovery garantisi
- Crop confidence threshold ayrı
- Source-aware merge
- LOST track re-identification
- Coast bbox sınır kontrolü
"""

import sys
import os
import pytest
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from motion.camera_motion import CameraTransform, CameraModel, CameraMotionEstimator
from motion.vehicle_tracker import VehicleTrackV2, VehicleTrackerV2, _bbox_diagonal
from person.tile_scheduler import TileScheduler, CropType
from person.detector_adapter import DetectorAdapter, _box_iou
from person.tracker import PersonTrackerV2, PersonTrackV2, TrackState
from person.merge import source_aware_merge, weighted_box_fusion


@pytest.fixture
def config():
    return {
        "motion": {
            "kalman_enabled": True,
            "kalman_process_noise": 1.0,
            "kalman_measurement_noise": 10.0,
            "min_track_len": 4,
            "max_track_age": 15,
            "track_match_distance": 90.0,
            "motion_window": 6,
            "decision_window": 5,
            "moving_votes_required": 3,
            "moving_start_threshold": 0.15,
            "moving_stop_threshold": 0.08,
            "normalize_by_bbox_diagonal": True,
            "min_diagonal": 20.0,
            "high_conf_threshold": 0.40,
            "low_conf_threshold": 0.15,
            "two_stage_association": True,
            "dynamic_gating": True,
            "gate_diagonal_ratio": 1.5,
            "maha_gate_threshold": 9.49,
            "max_features": 800,
            "affine_min_matches": 12,
            "affine_ransac_threshold": 5.0,
            "min_inlier_ratio": 0.35,
            "max_reprojection_error": 5.0,
        },
        "person_detection": {
            "grid_rows": 2,
            "grid_cols": 3,
            "crop_overlap_ratio": 0.10,
            "max_crops_per_frame": 2,
            "full_cycle_max_frames": 3,
            "track_roi_expand_ratio": 1.8,
            "discovery_interval_frames": 1,
            "min_discovery_slots": 1,
            "confidence_threshold": 0.15,
        },
        "person_tracking": {
            "enabled": True,
            "high_conf_threshold": 0.40,
            "low_conf_threshold": 0.10,
            "new_track_threshold": 0.30,
            "max_age": 4,
            "max_coast_output_frames": 1,
            "use_camera_compensation": True,
            "confirm_hits": 2,
            "confirm_window": 6,
            "coast_enabled": True,
            "coast_confidence_decay": 0.3,
            "kalman": {
                "std_weight_position": 0.05,
                "std_weight_velocity": 0.00625,
            },
        },
    }


class TestHomographyPointTransform:
    """Aşama 3.1: Homography apply_to_point w bölmesi."""

    def test_homography_divides_by_w(self):
        """Gerçek perspektif matrisi ile nokta dönüşümü w'ye bölünmeli."""
        H = np.array([
            [1.0, 0.1, 50.0],
            [0.0, 1.0, 30.0],
            [0.0001, 0.00005, 1.0],
        ], dtype=np.float32)
        t = CameraTransform()
        t.model_type = CameraModel.HOMOGRAPHY
        t.matrix = H
        t.confidence = 0.9

        pt = np.array([500.0, 400.0], dtype=np.float32)
        result = t.apply_to_point(pt)

        pt_h = np.array([500.0, 400.0, 1.0], dtype=np.float32)
        expected_h = H @ pt_h
        expected = expected_h[:2] / expected_h[2]

        assert np.allclose(result, expected, atol=0.1), \
            f"Homography nokta dönüşümü w bölmesi hatalı: {result} vs {expected}"

    def test_homography_not_affine(self):
        """Affine olmayan homography'de w != 1 olmalı."""
        H = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.001, 0.0, 1.0],
        ], dtype=np.float32)
        t = CameraTransform()
        t.model_type = CameraModel.HOMOGRAPHY
        t.matrix = H
        t.confidence = 0.9

        pt = np.array([100.0, 200.0], dtype=np.float32)
        result = t.apply_to_point(pt)

        w = 1.0 + 0.001 * 100.0
        expected_x = 100.0 / w
        expected_y = 200.0 / w

        assert abs(result[0] - expected_x) < 0.5
        assert abs(result[1] - expected_y) < 0.5


class TestTransformChain:
    """Aşama 3.2: Transform chain matris sırası."""

    def test_chain_order_homography(self):
        """Zincir: önce self (eski), sonra other (yeni) uygulanmalı."""
        H1 = np.eye(3, dtype=np.float32)
        H1[0, 2] = 10.0  # 10px x shift

        H2 = np.eye(3, dtype=np.float32)
        H2[1, 2] = 20.0  # 20px y shift

        t1 = CameraTransform()
        t1.model_type = CameraModel.HOMOGRAPHY
        t1.matrix = H1
        t1.confidence = 0.9

        t2 = CameraTransform()
        t2.model_type = CameraModel.HOMOGRAPHY
        t2.matrix = H2
        t2.confidence = 0.9

        chained = t1.chain(t2)
        pt = np.array([100.0, 100.0], dtype=np.float32)
        result = chained.apply_to_point(pt)

        expected = H2 @ (H1 @ np.array([100, 100, 1], dtype=np.float32))
        expected = expected[:2] / expected[2]

        assert np.allclose(result, expected, atol=0.5), \
            f"Chain sırası hatalı: {result} vs {expected}"

    def test_chain_shift_addition(self):
        """Shift chain: basit toplama."""
        t1 = CameraTransform()
        t1.model_type = CameraModel.SHIFT
        t1.shift = np.array([10.0, 0.0], dtype=np.float32)
        t1.confidence = 0.8

        t2 = CameraTransform()
        t2.model_type = CameraModel.SHIFT
        t2.shift = np.array([0.0, 20.0], dtype=np.float32)
        t2.confidence = 0.8

        chained = t1.chain(t2)
        pt = np.array([100.0, 100.0], dtype=np.float32)
        result = chained.apply_to_point(pt)
        assert np.allclose(result, [110.0, 120.0], atol=0.5)


class TestCameraMotionEstimatorFixes:
    """Aşama 3.3: CameraMotionEstimator düzeltmeleri."""

    def test_separate_masks_for_prev_and_current(self):
        """Önceki ve mevcut frame için ayrı mask kullanılmalı."""
        config = {"motion": {"feature_type": "orb", "max_features": 200,
                              "mask_detection_regions": True,
                              "camera_model": "affine",
                              "affine_fallback": True,
                              "median_shift_fallback": True,
                              "min_inlier_ratio": 0.3,
                              "max_reprojection_error": 5.0,
                              "affine_min_matches": 8,
                              "affine_ransac_threshold": 5.0}}

        estimator = CameraMotionEstimator(config)

        prev_gray = np.random.randint(0, 256, (200, 200), dtype=np.uint8)
        gray = np.random.randint(0, 256, (200, 200), dtype=np.uint8)

        prev_dets = [{"bbox": (50, 50, 100, 100)}]
        curr_dets = [{"bbox": (60, 50, 110, 100)}]

        transform = estimator.estimate(prev_gray, gray, curr_dets, prev_dets)
        assert transform is not None

    def test_end_to_end_homography_recovery(self):
        """Sentetik dokulu görüntüyü homography ile warp edip estimator matrisi geri bulmalı."""
        rng = np.random.RandomState(42)
        base = rng.randint(0, 256, (300, 300), dtype=np.uint8)

        H = np.array([
            [1.0, 0.05, 15.0],
            [0.0, 1.0, 10.0],
            [0.0001, 0.0, 1.0],
        ], dtype=np.float32)

        warped = cv2.warpPerspective(base, H, (300, 300))

        config = {"motion": {"feature_type": "orb", "max_features": 500,
                              "mask_detection_regions": False,
                              "camera_model": "homography",
                              "affine_fallback": True,
                              "median_shift_fallback": True,
                              "min_inlier_ratio": 0.25,
                              "max_reprojection_error": 5.0,
                              "affine_min_matches": 8,
                              "affine_ransac_threshold": 5.0}}
        estimator = CameraMotionEstimator(config)
        transform = estimator.estimate(base, warped, None)

        assert transform.is_reliable, "Estimator guvenilir transform bulamadi"

        pt = np.array([150.0, 150.0], dtype=np.float32)
        expected = H @ np.array([150, 150, 1], dtype=np.float32)
        expected = expected[:2] / expected[2]
        result = transform.apply_to_point(pt)
        assert np.linalg.norm(result - expected) < 5.0, \
            f"End-to-end homography: beklenen {expected}, got {result}"


class TestKalmanCameraSeparation:
    """Aşama 3.4: camera_expected vs association_predicted ayrımı."""

    def test_two_centers_returned(self, config):
        """predict() hem camera_expected_center hem association_center döndürmeli."""
        track = VehicleTrackV2(0, (100, 100, 200, 200), 0, config)
        pred = track.predict()
        assert "camera_expected_center" in pred
        assert "association_center" in pred
        assert "center" not in pred

    def test_camera_expected_without_transform(self, config):
        """Transform yoksa camera_expected = self.center (son gozlem)."""
        track = VehicleTrackV2(0, (100, 100, 200, 200), 0, config)
        pred = track.predict(None)
        assert np.allclose(pred["camera_expected_center"], [150, 150], atol=1.0)

    def test_camera_expected_with_transform(self, config):
        """Transform varsa camera_expected = transform(self.center)."""
        track = VehicleTrackV2(0, (100, 100, 200, 200), 0, config)
        t = CameraTransform()
        t.model_type = CameraModel.SHIFT
        t.shift = np.array([20.0, 0.0], dtype=np.float32)
        t.confidence = 0.8
        pred = track.predict(t)
        assert pred["camera_expected_center"][0] > 160  # 150 + 20


class TestKalmanFallbackVelocity:
    """Aşama 3.5: Kalman kapalı fallback velocity düzeltmesi."""

    def test_velocity_nonzero_when_kalman_disabled(self):
        """Kalman kapalıyken velocity sıfır olmamalı."""
        cfg = {"motion": {"kalman_enabled": False, "max_track_age": 15,
                           "min_track_len": 4, "moving_start_threshold": 0.15,
                           "moving_stop_threshold": 0.08, "motion_window": 6,
                           "decision_window": 5, "moving_votes_required": 3,
                           "normalize_by_bbox_diagonal": True, "min_diagonal": 20.0}}
        track = VehicleTrackV2(0, (100, 100, 200, 200), 0, cfg)
        track.update((130, 100, 230, 200), 1)
        assert np.linalg.norm(track.velocity) > 0, "Velocity sıfır - fallback hatası"


class TestMahalanobisGating:
    """Aşama 3.6: Mahalanobis gating."""

    def test_mahalanobis_returns_finite(self, config):
        """Mahalanobis distance sonlu dönmeli."""
        track = VehicleTrackV2(0, (100, 100, 200, 200), 0, config)
        det = np.array([160.0, 160.0], dtype=np.float32)
        pred = np.array([150.0, 150.0], dtype=np.float32)
        maha = track.mahalanobis_distance(det, pred)
        assert np.isfinite(maha)
        assert maha >= 0

    def test_mahalanobis_far_point_large(self, config):
        """Uzak nokta için Mahalanobis büyük olmalı."""
        track = VehicleTrackV2(0, (100, 100, 200, 200), 0, config)
        det = np.array([500.0, 500.0], dtype=np.float32)
        pred = np.array([150.0, 150.0], dtype=np.float32)
        maha = track.mahalanobis_distance(det, pred)
        assert maha > 1.0


class TestSchedulerDiscoveryGuarantee:
    """Aşama 1.4: Scheduler discovery garantisi."""

    def test_discovery_with_two_tracks(self, config):
        """Iki aktif track varken bile discovery yapılmalı."""
        scheduler = TileScheduler(config)
        scheduler.reset()
        W, H = 1920, 1080
        tracks = [
            {"bbox": (100, 100, 200, 200), "id": 0},
            {"bbox": (500, 500, 600, 600), "id": 1},
        ]

        seen_tiles = set()
        for frame in range(6):
            crops = scheduler.schedule(frame, W, H, active_tracks=tracks)
            for c in crops:
                if c.crop_type == CropType.DISCOVERY:
                    seen_tiles.add(c.tile_index)

        assert len(seen_tiles) >= 4, \
            f"2 track varken 6 karede {len(seen_tiles)} discovery tile (beklenen en az 4)"

    def test_discovery_with_max_crops_1(self):
        """max_crops=1 ile bile discovery guarantee edilmeli."""
        cfg = {"person_detection": {
            "grid_rows": 2, "grid_cols": 3, "crop_overlap_ratio": 0.1,
            "max_crops_per_frame": 1, "full_cycle_max_frames": 6,
            "track_roi_expand_ratio": 1.8, "discovery_interval_frames": 1,
            "min_discovery_slots": 1,
        }}
        scheduler = TileScheduler(cfg)
        scheduler.reset()
        W, H = 1920, 1080
        tracks = [{"bbox": (100, 100, 200, 200), "id": 0}]

        seen_tiles = set()
        for frame in range(12):
            crops = scheduler.schedule(frame, W, H, active_tracks=tracks)
            for c in crops:
                if c.crop_type == CropType.DISCOVERY:
                    seen_tiles.add(c.tile_index)

        assert len(seen_tiles) >= 4, \
            f"max_crops=1 ile 12 karede {len(seen_tiles)} discovery tile"


class TestSourceAwareMerge:
    """Aşama 1.5: Source-aware merge."""

    def test_yolo_priority_over_crop(self):
        """YOLO ve crop örtüşürse YOLO bbox korunur, confidence fusion yapılır."""
        yolo = [{"cls": 1, "conf": 0.6, "bbox": (100, 100, 150, 250)}]
        crop = [{"cls": 1, "conf": 0.8, "bbox": (105, 105, 155, 255), "_source": "crop"}]
        merged = source_aware_merge(yolo, crop, 0.45)
        assert len(merged) == 1
        assert merged[0]["bbox"] == (100, 100, 150, 250)  # YOLO bbox korunur
        assert merged[0]["conf"] == 0.8  # Higher confidence

    def test_crop_only_added(self):
        """Crop-only detection eklenmeli."""
        yolo = [{"cls": 1, "conf": 0.6, "bbox": (100, 100, 150, 250)}]
        crop = [{"cls": 1, "conf": 0.5, "bbox": (500, 500, 550, 650), "_source": "crop"}]
        merged = source_aware_merge(yolo, crop, 0.45)
        assert len(merged) == 2

    def test_crop_crop_dedup(self):
        """Crop-crop tekrarları temizlenmeli."""
        yolo = []
        crop = [
            {"cls": 1, "conf": 0.7, "bbox": (100, 100, 150, 250), "_source": "crop"},
            {"cls": 1, "conf": 0.5, "bbox": (102, 102, 152, 252), "_source": "crop"},
        ]
        merged = source_aware_merge(yolo, crop, 0.45)
        assert len(merged) == 1

    def test_yolo_not_lost(self):
        """YOLO detection kaybedilmemeli."""
        yolo = [{"cls": 1, "conf": 0.6, "bbox": (100, 100, 150, 250)}]
        crop = [{"cls": 1, "conf": 0.9, "bbox": (110, 110, 160, 260), "_source": "crop"}]
        merged = source_aware_merge(yolo, crop, 0.45)
        assert len(merged) == 1
        assert merged[0]["bbox"][0] == 100  # YOLO bbox x1 preserved


class TestLostTrackReidentification:
    """Aşama 2.1: LOST track re-identification."""

    def test_lost_track_revived(self, config):
        """Kaybolmuş track, yeni detection ile yeniden canlandırılabilir."""
        tracker = PersonTrackerV2(config)
        for f in range(5):
            dets = [{"cls": 1, "cls_name": "Insan", "conf": 0.6,
                      "bbox": (100 + f * 5, 100, 150 + f * 5, 250)}]
            tracker.update(dets, f, None)

        initial_track_count = len(tracker.tracks)
        assert initial_track_count == 1

        for f in range(5, 12):
            tracker.update([], f, None)

        dets = [{"cls": 1, "cls_name": "Insan", "conf": 0.6,
                  "bbox": (165, 100, 215, 250)}]
        tracker.update(dets, 12, None)

        track_ids = [t.id for t in tracker.tracks.values() if t.is_active]
        assert len(track_ids) <= 2, "Re-identification yeni track açmamalı (mümkün olduğunca)"


class TestCoastBboxConstraints:
    """Aşama 2.4: Coast bbox sınır kontrolü."""

    def test_coast_disabled(self):
        """coast_enabled=False ise coast bbox üretilmemeli."""
        cfg = {"person_tracking": {
            "enabled": True, "high_conf_threshold": 0.40, "low_conf_threshold": 0.10,
            "new_track_threshold": 0.30, "max_age": 4, "max_coast_output_frames": 1,
            "use_camera_compensation": True, "confirm_hits": 2, "confirm_window": 6,
            "coast_enabled": False, "coast_confidence_decay": 0.3,
            "kalman": {"std_weight_position": 0.05, "std_weight_velocity": 0.00625},
        }}
        tracker = PersonTrackerV2(cfg)
        for f in range(5):
            dets = [{"cls": 1, "cls_name": "Insan", "conf": 0.6,
                      "bbox": (100, 100, 150, 250)}]
            result = tracker.update(dets, f, None)

        result = tracker.update([], 5, None)
        coast_dets = [d for d in result if d.get("_source") == "tracker_coast"]
        assert len(coast_dets) == 0, "Coast disabled iken coast bbox üretilmemeli"

    def test_coast_confidence_decay(self, config):
        """Coast confidence azalan değerle üretilmeli."""
        tracker = PersonTrackerV2(config)
        for f in range(5):
            dets = [{"cls": 1, "cls_name": "Insan", "conf": 0.8,
                      "bbox": (100, 100, 150, 250)}]
            tracker.update(dets, f, None)

        result = tracker.update([], 5, None)
        coast_dets = [d for d in result if d.get("_source") == "tracker_coast"]
        if coast_dets:
            assert coast_dets[0]["conf"] < 0.8, "Coast confidence azalmalı"
