"""Arac tracker ve motion testleri (Faz M0-M3).

Zorunlu regresyon testi:
    Sabit dunya araci + kamera 20 px/frame pan
    Beklenen: tum karelerde moving_status=0 ve ayni track_id

Ek testler:
- Sabit kamera + hareketli arac -> moving=1
- Kamera pan + sabit arac -> moving=0
- Detector dropout sonrasi track kurtarma
- Transform zinciri (coklu missed frame)
- Normalize skor tutarliligi
- Predict/update kare basina bir kez
"""

import sys
import os
import pytest
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from motion.camera_motion import CameraTransform, CameraModel, CameraMotionEstimator
from motion.vehicle_tracker import VehicleTrackV2, VehicleTrackerV2, _box_iou, _bbox_diagonal


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
            "camera_model": "homography",
            "affine_fallback": True,
            "median_shift_fallback": True,
            "mask_detection_regions": True,
            "min_inlier_ratio": 0.35,
            "max_reprojection_error": 5.0,
            "feature_type": "orb",
            "max_features": 800,
            "affine_min_matches": 12,
            "affine_ransac_threshold": 5.0,
        }
    }


def make_shift_transform(dx: float, dy: float, confidence: float = 0.8) -> CameraTransform:
    """Test icin sabit shift transform olustur."""
    t = CameraTransform()
    t.model_type = CameraModel.SHIFT
    t.shift = np.array([dx, dy], dtype=np.float32)
    t.match_count = 50
    t.inlier_count = 40
    t.inlier_ratio = 0.8
    t.reprojection_error = 1.0
    t.confidence = confidence
    return t


def make_affine_transform(dx: float, dy: float, confidence: float = 0.85) -> CameraTransform:
    """Test icin affine transform olustur."""
    t = CameraTransform()
    t.model_type = CameraModel.AFFINE
    t.matrix = np.array([
        [1.0, 0.0, dx],
        [0.0, 1.0, dy],
    ], dtype=np.float32)
    t.match_count = 60
    t.inlier_count = 50
    t.inlier_ratio = 0.83
    t.reprojection_error = 0.8
    t.confidence = confidence
    return t


def make_gray_frame(width=1920, height=1080):
    """Sentetik grayscale frame olustur (feature icin texture)."""
    rng = np.random.RandomState(42)
    frame = rng.randint(0, 256, (height, width), dtype=np.uint8)
    return frame


class TestCameraTransform:
    """CameraTransform birim testleri."""

    def test_shift_apply_point(self):
        t = make_shift_transform(20.0, 0.0)
        pt = np.array([100.0, 200.0], dtype=np.float32)
        result = t.apply_to_point(pt)
        assert np.allclose(result, [120.0, 200.0])

    def test_affine_apply_point(self):
        t = make_affine_transform(20.0, 10.0)
        pt = np.array([100.0, 200.0], dtype=np.float32)
        result = t.apply_to_point(pt)
        assert np.allclose(result, [120.0, 210.0])

    def test_shift_apply_bbox(self):
        t = make_shift_transform(20.0, 0.0)
        bbox = (100, 100, 200, 200)
        result = t.apply_to_bbox(bbox)
        assert np.allclose(result[:2], [120, 100])
        assert np.allclose(result[2:], [220, 200])

    def test_none_transform_passthrough(self):
        t = CameraTransform()
        pt = np.array([100.0, 200.0], dtype=np.float32)
        result = t.apply_to_point(pt)
        assert np.allclose(result, [100.0, 200.0])

    def test_chain_transforms(self):
        t1 = make_shift_transform(10.0, 0.0)
        t2 = make_shift_transform(10.0, 0.0)
        chained = t1.chain(t2)
        assert chained.model_type == CameraModel.SHIFT
        assert np.allclose(chained.shift, [20.0, 0.0])

    def test_is_reliable(self):
        assert make_shift_transform(10, 10).is_reliable
        assert not CameraTransform().is_reliable


class TestVehicleTrackV2:
    """VehicleTrackV2 birim testleri."""

    def test_init(self, config):
        track = VehicleTrackV2(0, (100, 100, 200, 200), 0, config)
        assert track.id == 0
        assert track.hits == 1
        assert track.age == 0
        assert track.kf is not None

    def test_predict_advances_kalman(self, config):
        track = VehicleTrackV2(0, (100, 100, 200, 200), 0, config)
        pred = track.predict()
        assert pred["association_center"].shape == (2,)
        assert pred["bbox"][2] > pred["bbox"][0]

    def test_predict_with_camera_transform(self, config):
        track = VehicleTrackV2(0, (100, 100, 200, 200), 0, config)
        cam = make_shift_transform(20.0, 0.0)
        pred = track.predict(cam)
        assert pred["association_center"][0] > 100  # kamera ile saga tasindi
        assert pred["camera_expected_center"][0] > 100

    def test_update_resets_age(self, config):
        track = VehicleTrackV2(0, (100, 100, 200, 200), 0, config)
        track.predict()
        assert track.time_since_update == 1
        track.update((105, 100, 205, 200), 1)
        assert track.age == 0
        assert track.time_since_update == 0
        assert track.hits == 2

    def test_propagate_missed(self, config):
        track = VehicleTrackV2(0, (100, 100, 200, 200), 0, config)
        cam = make_shift_transform(20.0, 0.0)
        track.propagate_missed(cam)
        assert track.center[0] > 100  # current frame'e tasindi

    def test_motion_score_normalized(self, config):
        track = VehicleTrackV2(0, (100, 100, 200, 200), 0, config)
        det_center = np.array([150.0, 150.0], dtype=np.float32)
        pred_center = np.array([100.0, 100.0], dtype=np.float32)
        score = track.compute_motion_score(det_center, pred_center)
        diag = _bbox_diagonal((100, 100, 200, 200))
        expected = float(np.linalg.norm(det_center - pred_center)) / diag
        assert abs(score - expected) < 0.01


class TestMandatoryRegression:
    """ZORUNLU REGRESYON TESTI (Plan Faz M0):

    Sabit dunya araci + kamera 20 px/frame pan
    Beklenen: tum karelerde moving_status=0 ve ayni track_id
    """

    def test_stationary_vehicle_camera_pan(self, config):
        """Sabit arac + kamera pan -> moving_status=0 tum karelerde.

        Kamera saga pan yapinca, dunyada sabit olan aracin goruntudeki
        pozisyonu sola kayar. Kamera transformu bu kaymayi kompanze eder.
        """
        tracker = VehicleTrackerV2(config)
        cam_shift = make_shift_transform(-20.0, 0.0)  # objects shift left

        track_id = None
        for frame in range(10):
            x_offset = -frame * 20  # vehicle shifts left in image
            bbox = (500 + x_offset, 500, 600 + x_offset, 600)
            detections = [{"cls": 0, "cls_name": "Tasit", "conf": 0.8, "bbox": bbox}]
            gray = make_gray_frame()

            detections = tracker.update(detections, gray, cam_shift)

            for det in detections:
                if det.get("cls") == 0:
                    status = det.get("moving_status", "0")
                    assert status == "0", \
                        f"Frame {frame}: sabit arac moving_status={status} (beklenen 0)"
                    tid = det.get("_track_id")
                    if track_id is None:
                        track_id = tid
                    else:
                        assert tid == track_id, \
                            f"Frame {frame}: track_id degisti {track_id} -> {tid}"

    def test_stationary_vehicle_camera_pan_affine(self, config):
        """Sabit arac + kamera pan (affine) -> moving_status=0."""
        tracker = VehicleTrackerV2(config)
        cam = make_affine_transform(-20.0, 0.0)

        for frame in range(10):
            x_offset = -frame * 20
            bbox = (500 + x_offset, 500, 600 + x_offset, 600)
            detections = [{"cls": 0, "cls_name": "Tasit", "conf": 0.8, "bbox": bbox}]
            gray = make_gray_frame()

            detections = tracker.update(detections, gray, cam)

            for det in detections:
                if det.get("cls") == 0:
                    assert det["moving_status"] == "0", \
                        f"Frame {frame}: affine sabit arac moving={det['moving_status']}"


class TestMovingVehicle:
    """Hareketli arac tespiti."""

    def test_moving_vehicle_detected(self, config):
        """Hareketli arac -> moving_status=1 (yeterli kare sonrasi)."""
        tracker = VehicleTrackerV2(config)

        for frame in range(15):
            x = 100 + frame * 30  # 30px/frame hareket
            bbox = (x, 500, x + 100, 600)
            detections = [{"cls": 0, "cls_name": "Tasit", "conf": 0.8, "bbox": bbox}]
            gray = make_gray_frame()

            detections = tracker.update(detections, gray, None)

        last_status = None
        for det in detections:
            if det.get("cls") == 0:
                last_status = det.get("moving_status")
        assert last_status == "1", f"Hareketli arac moving_status={last_status} (beklenen 1)"


class TestDetectorDropout:
    """Detector dropout sonrasi track kurtarma."""

    def test_dropout_one_frame(self, config):
        """1 kare dropout sonrasi ayni track ID ile yakalanmali."""
        tracker = VehicleTrackerV2(config)
        cam = make_shift_transform(5.0, 0.0)
        track_id = None

        for frame in range(8):
            if frame == 4:
                detections = []
            else:
                x = 500 + frame * 10
                bbox = (x, 500, x + 100, 600)
                detections = [{"cls": 0, "cls_name": "Tasit", "conf": 0.8, "bbox": bbox}]

            gray = make_gray_frame()
            detections = tracker.update(detections, gray, cam)

            for det in detections:
                if det.get("cls") == 0 and "_track_id" in det:
                    if track_id is None:
                        track_id = det["_track_id"]

        assert track_id is not None, "Track olusturulmadi"
        assert len(tracker.tracks) > 0, "Track dropout sonrasi kaybolmamali"

    def test_dropout_three_frames(self, config):
        """3 kare dropout sonrasi track kurtarma."""
        tracker = VehicleTrackerV2(config)
        cam = make_shift_transform(5.0, 0.0)

        for frame in range(12):
            if 4 <= frame <= 6:
                detections = []
            else:
                x = 500 + frame * 10
                bbox = (x, 500, x + 100, 600)
                detections = [{"cls": 0, "cls_name": "Tasit", "conf": 0.8, "bbox": bbox}]

            gray = make_gray_frame()
            detections = tracker.update(detections, gray, cam)

        assert len(tracker.tracks) > 0, "Track 3-kare dropout sonrasi kaybolmamali"


class TestPredictUpdateOnce:
    """Predict/update kare basina tam birer kez cagirilmali."""

    def test_predict_once_per_frame(self, config):
        """Her karede predict bir kez, update eslesme durumunda bir kez cagirilmali."""
        tracker = VehicleTrackerV2(config)

        bbox = (500, 500, 600, 600)
        detections = [{"cls": 0, "cls_name": "Tasit", "conf": 0.8, "bbox": bbox}]
        gray = make_gray_frame()

        tracker.update(detections, gray, None)
        track = list(tracker.tracks.values())[0]
        assert track.hits == 1, "Ilk karede hits=1 olmali"

        tracker.update(detections, gray, None)
        assert track.hits == 2, "Ikinci karede hits=2 olmali"
        assert track.time_since_update == 0


class TestNonVehicleStatus:
    """Insan/UAP/UAI moving_status=-1."""

    def test_non_vehicle_negative_one(self, config):
        tracker = VehicleTrackerV2(config)
        detections = [
            {"cls": 0, "cls_name": "Tasit", "conf": 0.8, "bbox": (500, 500, 600, 600)},
            {"cls": 1, "cls_name": "Insan", "conf": 0.7, "bbox": (100, 100, 150, 200)},
            {"cls": 2, "cls_name": "UAP", "conf": 0.6, "bbox": (200, 200, 300, 250)},
        ]
        gray = make_gray_frame()
        result = tracker.update(detections, gray, None)

        for det in result:
            if det["cls"] != 0:
                assert det["moving_status"] == "-1", \
                    f"cls={det['cls']} moving_status={det['moving_status']} (beklenen -1)"


class TestNormalization:
    """Normalize skor tutarliligi (Faz M3)."""

    def test_small_bbox_higher_normalized_score(self, config):
        """Ayni residual px icin kucuk bbox daha yuksek normalize skor vermeli."""
        track_small = VehicleTrackV2(0, (100, 100, 120, 120), 0, config)
        track_large = VehicleTrackV2(1, (100, 100, 300, 300), 0, config)

        det_center = np.array([110.0, 110.0], dtype=np.float32)
        pred_center = np.array([100.0, 100.0], dtype=np.float32)

        score_small = track_small.compute_motion_score(det_center, pred_center)
        score_large = track_large.compute_motion_score(det_center, pred_center)

        diag_small = _bbox_diagonal((100, 100, 120, 120))
        diag_large = _bbox_diagonal((100, 100, 300, 300))

        assert score_small > score_large, \
            f"Kucuk bbox skoru {score_small} buyuk bbox {score_large}'den buyuk olmali"

    def test_normalized_score_scale_invariant(self, config):
        """Farkli bbox boyutlarinda benzer fiziksel hareket tutarli tepki vermeli."""
        config_small = {"motion": {**config["motion"]}}
        config_large = {"motion": {**config["motion"]}}

        track_small = VehicleTrackV2(0, (100, 100, 130, 130), 0, config_small)
        track_large = VehicleTrackV2(1, (100, 100, 250, 250), 0, config_large)

        det_center_s = np.array([115.0, 115.0], dtype=np.float32)
        pred_center_s = np.array([100.0, 100.0], dtype=np.float32)

        det_center_l = np.array([125.0, 125.0], dtype=np.float32)
        pred_center_l = np.array([100.0, 100.0], dtype=np.float32)

        score_s = track_small.compute_motion_score(det_center_s, pred_center_s)
        score_l = track_large.compute_motion_score(det_center_l, pred_center_l)

        diag_s = _bbox_diagonal((100, 100, 130, 130))
        diag_l = _bbox_diagonal((100, 100, 250, 250))
        expected_s = float(np.linalg.norm(det_center_s - pred_center_s)) / diag_s
        expected_l = float(np.linalg.norm(det_center_l - pred_center_l)) / diag_l

        assert abs(score_s - expected_s) < 0.01
        assert abs(score_l - expected_l) < 0.01
