import numpy as np

from competition_replay import acceptance_gate, apply_causal_relocalization
from offline_fusion import (
    CALIBRATION_FRAMES,
    RELOCALIZATION_CONFIRMATIONS,
    apply_relocalization,
    is_relocalization_candidate,
)


def candidate(frame: int, reference: int) -> dict[str, str]:
    return {
        "frame": str(frame),
        "candidate_frame": str(reference),
        "dbow_score": "0.05",
        "homography_inliers": "100",
        "homography_inlier_ratio": "0.7",
        "fundamental_inliers": "90",
        "fundamental_inlier_ratio": "0.6",
    }


def test_candidate_must_reference_health1_map():
    assert is_relocalization_candidate(candidate(900, CALIBRATION_FRAMES - 1))
    assert not is_relocalization_candidate(candidate(900, CALIBRATION_FRAMES))


def test_relocalization_consumes_calibration_gt_only():
    length = CALIBRATION_FRAMES + RELOCALIZATION_CONFIRMATIONS + 2
    prediction = np.zeros((length, 3), dtype=np.float64)
    calibration_gt = np.zeros((CALIBRATION_FRAMES, 3), dtype=np.float64)
    calibration_gt[10] = [20.0, 0.0, 0.0]
    rows = [candidate(frame, -1) for frame in range(length)]
    for frame in range(
        CALIBRATION_FRAMES,
        CALIBRATION_FRAMES + RELOCALIZATION_CONFIRMATIONS,
    ):
        rows[frame] = candidate(frame, 10)

    output, events = apply_relocalization(prediction, calibration_gt, rows)

    assert len(events) == 1
    assert events[0]["frame"] == CALIBRATION_FRAMES + RELOCALIZATION_CONFIRMATIONS - 1
    assert np.allclose(output[-1], [4.0, 0.0, 0.0])


def test_competition_relocalization_never_rewrites_committed_prefix():
    event_frame = CALIBRATION_FRAMES + RELOCALIZATION_CONFIRMATIONS - 1
    length = event_frame + 3
    prediction = np.zeros((length, 3), dtype=np.float64)
    calibration_gt = np.zeros((CALIBRATION_FRAMES, 3), dtype=np.float64)
    calibration_gt[10] = [20.0, 0.0, 0.0]
    rows = [candidate(frame, -1) for frame in range(length)]
    for frame in range(CALIBRATION_FRAMES, event_frame + 1):
        rows[frame] = candidate(frame, 10)

    output, events = apply_causal_relocalization(
        prediction, calibration_gt, rows, gain=0.5
    )

    assert len(events) == 1
    assert np.allclose(output[:event_frame], prediction[:event_frame])
    assert np.allclose(output[event_frame:], [10.0, 0.0, 0.0])


def test_acceptance_gate_requires_tail_metrics_too():
    offline = {"E_3d": 10.0, "RMSE_3d": 11.0, "p95_3d": 18.0, "max_3d": 20.0}
    candidate_metrics = {
        "E_3d": 10.5,
        "RMSE_3d": 11.5,
        "p95_3d": 25.0,
        "max_3d": 21.0,
    }

    result = acceptance_gate(candidate_metrics, offline)

    assert result["checks"]["E_3d"]
    assert not result["checks"]["p95_3d"]
    assert not result["passed"]
