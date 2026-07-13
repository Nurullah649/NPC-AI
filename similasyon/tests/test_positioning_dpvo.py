"""DPVO poz konvansiyonu ve çevrimiçi kalibrasyon regresyon testleri."""

import os
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import cv2
import numpy as np
import pytest

from src.models.positioning_dpvo import PositioningDPVO


def make_positioner(**dpvo_overrides):
    dpvo = {
        "use_direction_guard": False,
        "calibration_width": 1920,
        "calibration_height": 1080,
        "input_width": 640,
        "input_height": 360,
        "min_calib_frames": 3,
    }
    dpvo.update(dpvo_overrides)
    return PositioningDPVO({"dpvo": dpvo})


def test_intrinsics_scale_uses_calibration_native_resolution():
    positioning = make_positioner()
    scaled = positioning._scaled_intrinsics(640, 360)
    np.testing.assert_allclose(
        scaled[0, 0], positioning.intrinsics[0, 0] / 3.0, rtol=1e-7
    )


def test_intrinsics_scale_is_independent_of_source_frame_size():
    positioning = make_positioner()
    frame_4k = np.zeros((2160, 3840, 3), dtype=np.uint8)
    _, intrinsics, height, width = positioning._prepare_dpvo_input(frame_4k)
    expected = np.array(
        [
            positioning.intrinsics[0, 0] / 3.0,
            positioning.intrinsics[1, 1] / 3.0,
            positioning.intrinsics[0, 2] / 3.0,
            positioning.intrinsics[1, 2] / 3.0,
        ],
        dtype=np.float32,
    )
    assert (height, width) == (360, 640)
    np.testing.assert_allclose(intrinsics, expected, rtol=1e-6)


def test_2026_camera_profile_is_selected_and_scaled():
    profile = (
        Path(__file__).resolve().parents[1]
        / "config/camera/profiles/thyz_2026_rgb_1920x1080_v1.yaml"
    )
    positioning = PositioningDPVO(
        {
            "camera": {"profile": str(profile)},
            "dpvo": {
                "use_direction_guard": False,
                "calibration_width": 1920,
                "calibration_height": 1080,
                "input_width": 640,
                "input_height": 360,
            },
        }
    )

    assert positioning.camera_profile_id == "thyz_2026_rgb_1920x1080_v1"
    np.testing.assert_allclose(
        positioning._intrinsics_matrix_to_vec(positioning._scaled_intrinsics(640, 360)),
        [463.233333, 462.366667, 318.002333, 186.298667],
        atol=1e-5,
    )
    np.testing.assert_allclose(
        positioning.camera_distortion,
        [0.1378, -0.2564, 0.0, 0.0],
        atol=1e-12,
    )


def test_camera_profile_rejects_mismatched_native_size():
    profile = (
        Path(__file__).resolve().parents[1]
        / "config/camera/profiles/thyz_2026_rgb_1920x1080_v1.yaml"
    )
    with pytest.raises(ValueError, match="calibration size"):
        PositioningDPVO(
            {
                "camera": {"profile": str(profile)},
                "dpvo": {
                    "calibration_width": 4000,
                    "calibration_height": 3000,
                },
            }
        )


def test_umeyama_recovers_scale_rotation_and_translation():
    positioning = make_positioner()
    rng = np.random.default_rng(7)
    src = rng.normal(size=(40, 3))
    angle = np.deg2rad(70.0)
    rotation = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    scale = 37.5
    translation = np.array([120.0, -44.0, 8.0])
    dst = (scale * (rotation @ src.T)).T + translation

    fitted_rotation, fitted_translation, fitted_scale = positioning._sim3_umeyama(src, dst)

    np.testing.assert_allclose(fitted_scale, scale, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(fitted_rotation, rotation, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(fitted_translation, translation, rtol=1e-10, atol=1e-10)


def test_direction_guard_checks_aligned_candidate_not_raw_axes():
    positioning = make_positioner(
        use_direction_guard=True,
        direction_guard_min_points=3,
        min_direction_similarity=0.8,
        min_allowed_scale=0.5,
        max_allowed_scale=2.0,
    )
    src = np.array([[i, 0.2 * i, 0.1 * i * i] for i in range(10)], dtype=np.float64)
    # Raw trajectory points in nearly opposite XY axes, but Sim3 maps it exactly.
    rotation = np.diag([-1.0, -1.0, 1.0])
    dst = (25.0 * (rotation @ src.T)).T + np.array([10.0, 20.0, -5.0])
    positioning.dpvo_buffer = src.tolist()
    positioning.gt_buffer = dst.tolist()

    positioning._update_calibration()

    assert positioning.is_calibrated
    aligned = np.asarray([positioning._align_dpvo_to_gt(point) for point in src])
    np.testing.assert_allclose(aligned, dst, atol=1e-8)


def test_absolute_linear_alignment_recovers_affine_mapping():
    positioning = make_positioner(fit_method="linear", min_calib_frames=5)
    rng = np.random.default_rng(19)
    src = rng.normal(size=(50, 3))
    coef = np.array(
        [[-40.0, 12.0, -6.0], [15.0, 39.0, -9.0], [2.5, 1.4, 37.0]],
        dtype=np.float64,
    )
    intercept = np.array([1.9, 3.4, -0.5], dtype=np.float64)
    dst = src @ coef.T + intercept
    positioning.dpvo_buffer = src.tolist()
    positioning.gt_buffer = dst.tolist()

    positioning._update_calibration()

    assert positioning.is_calibrated
    np.testing.assert_allclose(positioning.sim3_R, coef, atol=1e-10)
    np.testing.assert_allclose(positioning.sim3_t, intercept, atol=1e-10)
    predicted = np.asarray([positioning._align_dpvo_to_gt(point) for point in src])
    np.testing.assert_allclose(predicted, dst, atol=1e-10)


def test_absolute_alignment_is_anchored_to_last_gt_sample():
    positioning = make_positioner(
        fit_method="linear", min_calib_frames=5, anchor_at_calib_end=True
    )
    rng = np.random.default_rng(23)
    src = rng.normal(size=(30, 3))
    dst = src @ np.array([[20.0, 2.0, 0.0], [-1.0, 18.0, 3.0], [2.0, 0.0, 15.0]]).T
    dst += rng.normal(scale=0.05, size=dst.shape)
    positioning.dpvo_buffer = src.tolist()
    positioning.gt_buffer = dst.tolist()

    positioning._update_calibration()

    np.testing.assert_allclose(positioning._align_dpvo_to_gt(src[-1]), dst[-1], atol=1e-10)


def test_uncalibrated_dpvo_raw_is_not_sent_as_ned(tmp_path):
    positioning = make_positioner(
        input_width=64,
        input_height=36,
        calibration_width=64,
        calibration_height=36,
        fallback_velocity=True,
    )

    class FakeSlam:
        def process_frame(self, *_args, **_kwargs):
            return (999.0, -999.0, 500.0)

    positioning.slam = FakeSlam()
    positioning._dpvo_available = True
    positioning.last_known_position = np.array([1.0, 2.0, 3.0])
    positioning.last_velocity = np.array([0.1, -0.2, 0.3])
    frame_path = tmp_path / "frame.jpg"
    cv2.imwrite(str(frame_path), np.zeros((36, 64, 3), dtype=np.uint8))

    result = positioning.process_frame(0, str(frame_path), "0")

    np.testing.assert_allclose(result, [1.1, 1.8, 3.3])


def test_base_positioner_fails_closed_if_loop_closure_is_accidentally_enabled(monkeypatch):
    import src.models.dpvo_standalone as standalone_module

    class FakeDPVOStandalone:
        def __init__(self, *_args, **_kwargs):
            self.initialized = True
            self.cfg = SimpleNamespace(LOOP_CLOSURE=True)

    monkeypatch.setattr(standalone_module, "DPVOStandalone", FakeDPVOStandalone)
    positioning = make_positioner()

    positioning._init_dpvo(360, 640)

    assert not positioning._dpvo_available
    assert positioning.slam is None


def test_delta_ma_mapping_matches_causal_restart_math():
    positioning = make_positioner(
        fit_method="delta_linear",
        delta_window=3,
        delta_ridge_alpha=0.0,
        delta_fit_intercept=False,
        min_calib_frames=5,
    )
    rng = np.random.default_rng(11)
    raw_delta = rng.normal(scale=0.02, size=(30, 3))
    features = positioning._causal_moving_average(raw_delta, 3)
    expected_coef = np.array(
        [[25.0, 3.0, -2.0], [-4.0, 21.0, 1.5], [2.0, -1.0, 18.0]],
        dtype=np.float64,
    )
    gt_delta = features @ expected_coef.T
    raw_position = np.vstack([np.zeros((1, 3)), np.cumsum(raw_delta, axis=0)])
    gt_position = np.vstack([np.zeros((1, 3)), np.cumsum(gt_delta, axis=0)])
    positioning.dpvo_buffer = raw_position.tolist()
    positioning.gt_buffer = gt_position.tolist()
    positioning.delta_raw_buffer = raw_delta.tolist()
    positioning.delta_gt_buffer = gt_delta.tolist()

    positioning._update_calibration()

    assert positioning.is_calibrated
    np.testing.assert_allclose(positioning.delta_coef, expected_coef, rtol=1e-10, atol=1e-10)

    next_delta = np.array([0.01, -0.03, 0.02])
    positioning.raw_delta_history = [*raw_delta.tolist(), next_delta.tolist()]
    positioning.last_known_position = gt_position[-1].copy()
    expected_step = expected_coef @ np.mean(
        np.vstack([raw_delta[-2:], next_delta]), axis=0
    )
    predicted = positioning._predict_delta_position(next_delta)
    np.testing.assert_allclose(predicted, gt_position[-1] + expected_step, atol=1e-10)
