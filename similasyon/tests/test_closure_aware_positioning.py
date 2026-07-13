"""Regression tests for the experimental loop-closure gauge repair."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np


SIM_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SIM_ROOT))


def load_experimental_positioner():
    path = SIM_ROOT / "deneysel/dpvo_2026/closure_aware_positioning.py"
    spec = importlib.util.spec_from_file_location("closure_aware_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module.ExperimentalPositioningDPVO


def load_gt_kalman_positioner():
    path = SIM_ROOT / "deneysel/dpvo_2026/gt_kalman_positioning.py"
    spec = importlib.util.spec_from_file_location("gt_kalman_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module.ExperimentalPositioningDPVO


def make_exact_ba_event(new_raw, old_raw, timestamps):
    return {
        "event_id": 1,
        "trigger_input_timestamp": int(timestamps[-1]) + 1,
        "n_active": len(timestamps),
        "n_full_edges": 123,
        "active_input_timestamps": timestamps,
        "before_c2w_xyz": old_raw,
        "after_c2w_xyz": new_raw,
    }


def test_gauge_repair_uses_exact_ba_snapshot_and_preserves_ned_continuity():
    Positioner = load_experimental_positioner()
    positioner = Positioner(
        {
            "dpvo": {
                "fit_method": "linear",
                "use_direction_guard": False,
                "calibration_width": 1920,
                "calibration_height": 1080,
                "input_width": 640,
                "input_height": 360,
            }
        }
    )
    rng = np.random.default_rng(42)
    new_raw = rng.normal(size=(12, 3))
    angle = np.deg2rad(24.0)
    rotation = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    old_raw = (0.22 * (rotation @ new_raw.T)).T + np.array([0.7, -1.2, 0.3])
    timestamps = np.arange(300, 312, dtype=np.int64)
    event = make_exact_ba_event(new_raw, old_raw, timestamps)

    positioner.is_calibrated = True
    positioner.sim3_R = np.array(
        [[3.0, 0.2, 0.0], [-0.1, 2.5, 0.4], [0.0, 0.3, 1.7]]
    )
    positioner.sim3_t = np.array([8.0, -4.0, 1.0])
    positioner.alignment_anchor_offset = np.array([0.5, -0.2, 0.1])
    positioner.last_dpvo_raw = new_raw[-1].copy()
    before_ned = np.array([21.0, -13.0, 5.0])

    gauge, telemetry = positioner._fit_event_gauge(event, frame_idx=312)
    assert gauge is not None
    repaired, telemetry = positioner._repair_output_gauge(
        gauge, telemetry, before_ned=before_ned
    )

    assert repaired
    assert telemetry["status"] == "repaired"
    np.testing.assert_allclose(
        positioner._align_dpvo_to_gt(positioner.last_dpvo_raw), before_ned, atol=1e-8
    )


def test_calibration_buffer_is_rebased_only_for_pre_ba_raw_samples():
    Positioner = load_experimental_positioner()
    positioner = Positioner(
        {
            "dpvo": {
                "fit_method": "linear",
                "use_direction_guard": False,
                "calibration_width": 1920,
                "calibration_height": 1080,
            }
        }
    )
    rotation = np.eye(3)
    translation = np.array([3.0, -1.0, 2.0])
    scale = 0.5
    # old = scale * new + translation
    old_raw = np.array([[4.0, 1.0, 2.5], [5.0, -1.0, 3.0]])
    current_new_raw = np.array([6.0, 2.0, -4.0])
    positioner.frame_indices = [10, 11, 12]
    positioner.dpvo_buffer = [*old_raw.tolist(), current_new_raw.tolist()]
    positioner._calibrated_sample_count = 3

    changed = positioner._rebase_calibration_buffer(
        {"rotation": rotation, "translation": translation, "scale": scale},
        {"trigger_input_timestamp": 12},
    )

    assert changed == 2
    np.testing.assert_allclose(
        np.asarray(positioner.dpvo_buffer[:2]), (old_raw - translation) / scale
    )
    np.testing.assert_allclose(positioner.dpvo_buffer[2], current_new_raw)
    assert positioner._calibrated_sample_count == -1


def test_experimental_positioner_explicitly_opted_in_to_loop_closure(monkeypatch):
    import src.models.dpvo_standalone as standalone_module

    class FakeDPVOStandalone:
        def __init__(self, *_args, **_kwargs):
            self.initialized = True
            self.cfg = SimpleNamespace(LOOP_CLOSURE=True)

    monkeypatch.setattr(standalone_module, "DPVOStandalone", FakeDPVOStandalone)
    Positioner = load_experimental_positioner()
    positioner = Positioner(
        {
            "dpvo": {
                "fit_method": "linear",
                "use_direction_guard": False,
                "calibration_width": 1920,
                "calibration_height": 1080,
            }
        }
    )

    positioner._init_dpvo(360, 640)

    assert positioner._dpvo_available
    assert positioner.slam is not None


def test_gt_kalman_only_updates_from_health1_gt(monkeypatch):
    Positioner = load_gt_kalman_positioner()
    positioner = Positioner(
        {
            "dpvo": {
                "fit_method": "linear",
                "use_direction_guard": False,
                "calibration_width": 1920,
                "calibration_height": 1080,
            },
            "experimental_kalman": {"enabled": True},
        }
    )

    # Replace only the base DPVO result. The health=0 call deliberately carries
    # None GT fields, matching the evaluator's anti-leak contract.
    from src.models.positioning_dpvo import PositioningDPVO

    def fake_base_process(self, *_args, **kwargs):
        if str(kwargs["health_status"]) == "1":
            result = np.array([kwargs["gt_x"], kwargs["gt_y"], kwargs["gt_z"]])
        else:
            assert kwargs["gt_x"] is None
            assert kwargs["gt_y"] is None
            assert kwargs["gt_z"] is None
            result = np.array([4.0, 5.0, 6.0])
        self.last_known_position = result.copy()
        return tuple(result)

    monkeypatch.setattr(PositioningDPVO, "process_frame", fake_base_process)
    healthy = positioner.process_frame(
        frame_idx=0, frame_path="", health_status="1", gt_x=1.0, gt_y=2.0, gt_z=3.0
    )
    blind = positioner.process_frame(
        frame_idx=1, frame_path="", health_status="0", gt_x=None, gt_y=None, gt_z=None
    )

    np.testing.assert_allclose(healthy, [1.0, 2.0, 3.0])
    assert np.all(np.isfinite(blind))
    assert positioner.kalman_updates_gt == 1
    assert positioner.kalman_updates_vo == 1
