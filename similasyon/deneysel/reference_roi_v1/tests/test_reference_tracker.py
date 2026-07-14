import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from deneysel.reference_roi_v1.reference_tracker import (
    CameraCompensatedReferenceTracker,
)
from src.models.motion_camera_v2 import CameraModel, CameraTransform


class StaticEstimator:
    def __init__(self, transform):
        self.transform = transform

    def estimate(self, _previous, _current):
        return self.transform


def reliable_shift(dx=0.0, dy=0.0):
    return CameraTransform(
        model_type=CameraModel.SHIFT,
        shift=np.array([dx, dy], dtype=np.float32),
        match_count=30,
        inlier_count=25,
        inlier_ratio=0.83,
        reprojection_error=0.5,
        confidence=0.9,
    )


def tracker():
    return CameraCompensatedReferenceTracker({
        "reference_roi_experiment": {
            "tracker": {
                "min_color_correlation": -1.0,
                "max_color_distance": 1.0,
                "seed_expand_left_ratio": 0.0,
                "seed_expand_right_ratio": 0.0,
                "seed_expand_top_ratio": 0.0,
                "seed_expand_bottom_ratio": 0.0,
            }
        }
    })


def test_unseeded_tracker_never_emits_box():
    frame = np.full((100, 100, 3), 120, dtype=np.uint8)
    result = tracker().update(frame)
    assert result.active is False
    assert result.bbox is None
    assert result.reason == "not_seeded"


def test_verified_seed_is_translated_by_camera_motion():
    frame = np.full((100, 100, 3), 120, dtype=np.uint8)
    instance = tracker()
    instance.seed(frame, (10, 20, 40, 50))
    instance.estimator = StaticEstimator(reliable_shift(5, 7))

    result = instance.update(frame)

    assert result.active is True
    assert np.allclose(result.bbox, (15, 27, 45, 57))
    assert result.reason == "tracked"


def test_tracker_stops_when_polygon_leaves_frame():
    frame = np.full((100, 100, 3), 120, dtype=np.uint8)
    instance = tracker()
    instance.seed(frame, (10, 70, 40, 95))
    instance.estimator = StaticEstimator(reliable_shift(0, 80))

    result = instance.update(frame)

    assert result.active is False
    assert result.bbox is None
    assert result.reason == "left_frame"
    assert instance.active is False


def test_seed_expansion_can_cover_asymmetric_attachment():
    frame = np.full((200, 300, 3), 120, dtype=np.uint8)
    instance = CameraCompensatedReferenceTracker({
        "reference_roi_experiment": {
            "tracker": {
                "seed_expand_left_ratio": 0.25,
                "seed_expand_right_ratio": 0.55,
                "seed_expand_top_ratio": 0.25,
                "seed_expand_bottom_ratio": 0.25,
            }
        }
    })

    expanded = instance.seed(frame, (100, 50, 200, 100))

    assert np.allclose(expanded, (75, 37.5, 255, 112.5))
