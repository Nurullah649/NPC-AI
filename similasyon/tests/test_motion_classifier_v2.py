import numpy as np

from src.models.motion_camera_v2 import CameraModel, CameraTransform
from src.models.motion_classifier import MotionClassifier


def _config():
    return {
        "motion": {
            "camera_model": "homography",
            "kalman_enabled": True,
            "kalman_process_noise": 1.0,
            "kalman_measurement_noise": 10.0,
            "camera_relative_velocity_enabled": True,
            "relative_velocity_alpha": 1.0,
            "missed_velocity_decay": 0.92,
            "min_track_len": 4,
            "max_track_age": 15,
            "track_match_distance": 90.0,
            "normalize_by_bbox_diagonal": False,
            "moving_start_threshold": 45.0,
            "moving_stop_threshold": 35.0,
            "motion_window": 6,
            "decision_window": 5,
            "moving_votes_required": 3,
            "high_conf_threshold": 0.40,
            "low_conf_threshold": 0.15,
            "two_stage_association": True,
            "dynamic_gating": True,
            "gate_diagonal_ratio": 1.5,
            "maha_gate_threshold": 9.49,
            "maha_gate_min_distance": 60.0,
            "maha_gate_diagonal_ratio": 0.25,
            "use_local_optical_flow": False,
            "deduplicate_vehicle_detections": True,
            "duplicate_iou_threshold": 0.85,
        }
    }


def _vehicle(cx, cy=150.0):
    return {
        "cls": 0,
        "cls_name": "Tasit",
        "conf": 0.95,
        "bbox": (cx - 50.0, cy - 30.0, cx + 50.0, cy + 30.0),
    }


def _shift(dx=0.0, dy=0.0):
    transform = CameraTransform()
    transform.model_type = CameraModel.SHIFT
    transform.shift = np.array([dx, dy], dtype=np.float32)
    transform.confidence = 0.99
    return transform


def test_static_world_vehicle_stays_zero_during_camera_pan(monkeypatch):
    classifier = MotionClassifier(_config())
    monkeypatch.setattr(
        classifier.camera_estimator,
        "estimate",
        lambda *args, **kwargs: _shift(40.0, 0.0),
    )
    gray = np.zeros((300, 500), dtype=np.uint8)

    track_ids = []
    statuses = []
    for frame in range(6):
        result = classifier.update([_vehicle(150.0 + frame * 40.0)], gray)
        track_ids.append(result[0]["_track_id"])
        statuses.append(result[0]["moving_status"])

    assert len(set(track_ids)) == 1
    assert statuses == ["0"] * 6


def test_moving_vehicle_reaches_moving_status_without_id_reset(monkeypatch):
    classifier = MotionClassifier(_config())
    monkeypatch.setattr(
        classifier.camera_estimator,
        "estimate",
        lambda *args, **kwargs: _shift(),
    )
    gray = np.zeros((300, 800), dtype=np.uint8)

    outputs = [
        classifier.update([_vehicle(150.0 + frame * 100.0)], gray)[0]
        for frame in range(5)
    ]

    assert len({det["_track_id"] for det in outputs}) == 1
    assert outputs[-1]["moving_status"] == "1"
    assert outputs[-1]["_motion_score"] > 45.0


def test_duplicate_vehicle_boxes_are_suppressed(monkeypatch):
    classifier = MotionClassifier(_config())
    monkeypatch.setattr(
        classifier.camera_estimator,
        "estimate",
        lambda *args, **kwargs: _shift(),
    )
    gray = np.zeros((300, 500), dtype=np.uint8)
    first = _vehicle(200.0)
    duplicate = dict(first)
    duplicate["conf"] = 0.60
    duplicate["bbox"] = (151.0, 121.0, 251.0, 181.0)

    result = classifier.update([first, duplicate], gray)

    assert len(result) == 1
    assert len(classifier.tracks) == 1
