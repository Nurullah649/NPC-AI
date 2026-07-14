import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.models.positioning_causal_fusion import CausalPositionFusion


INTRINSICS = np.array(
    [[463.2, 0.0, 318.0], [0.0, 462.4, 186.3], [0.0, 0.0, 1.0]],
    dtype=np.float64,
)


def config(vocabulary: Path, **overrides) -> dict:
    fusion = {
        "enabled": True,
        "require_relocalization": True,
        "vocabulary_path": str(vocabulary),
        "confirmations": 5,
        "relocalization_gain": 0.2,
    }
    fusion.update(overrides)
    return {"dpvo": {"causal_fusion": fusion}}


def safe_candidate(frame: int, reference: int = 10) -> dict:
    return {
        "frame": frame,
        "candidate_frame": reference,
        "dbow_score": 0.05,
        "homography_inliers": 100,
        "homography_inlier_ratio": 0.7,
        "fundamental_inliers": 90,
        "fundamental_inlier_ratio": 0.6,
    }


def test_missing_required_relocalization_fails_closed(tmp_path):
    missing = tmp_path / "missing-vocabulary.txt"
    fusion = CausalPositionFusion(config(missing), INTRINSICS)
    image = np.zeros((360, 640, 3), dtype=np.uint8)

    fusion.observe_frame(image, "1", np.zeros(3))

    base = np.array([1.0, 2.0, 3.0])
    assert not fusion.enabled
    assert np.array_equal(fusion.fuse_position(base, "0"), base)


def test_plane_mapping_uses_healthy_calibration_pairs(tmp_path):
    vocabulary = tmp_path / "vocabulary.txt"
    vocabulary.write_text("test", encoding="utf-8")
    fusion = CausalPositionFusion(
        config(vocabulary, require_relocalization=False), INTRINSICS
    )
    design = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 1.0],
        ],
        dtype=np.float64,
    )
    fusion.min_calibration_pairs = len(design)
    fusion._plane_design = [row.copy() for row in design]
    fusion._plane_target = [row.copy() for row in design]

    fusion._fit_plane_mapping()

    assert fusion.telemetry["plane_calibrated"]
    assert np.allclose(fusion._plane_weights, np.eye(3), atol=2e-6)


def test_relocalization_changes_current_and_future_not_committed_prefix(tmp_path):
    vocabulary = tmp_path / "vocabulary.txt"
    vocabulary.write_text("test", encoding="utf-8")
    fusion = CausalPositionFusion(
        config(vocabulary, require_relocalization=False), INTRINSICS
    )
    fusion._healthy_map[10] = np.array([20.0, 0.0, 0.0])
    outputs = []
    correction_deltas = []
    for frame in range(450, 457):
        fusion._visual_frame_index = frame + 1
        fusion._pending_candidate = safe_candidate(frame)
        outputs.append(fusion.fuse_position(np.zeros(3), "0"))
        correction_deltas.append(fusion.last_applied_correction_delta.copy())

    assert np.allclose(outputs[:4], 0.0)
    assert np.allclose(outputs[4:], [4.0, 0.0, 0.0])
    assert len(fusion.events) == 1
    assert fusion.events[0]["frame"] == 454
    assert np.allclose(correction_deltas[4], [4.0, 0.0, 0.0])
    assert np.allclose(correction_deltas[5:], 0.0)


def test_candidate_without_healthy_map_position_is_rejected(tmp_path):
    vocabulary = tmp_path / "vocabulary.txt"
    vocabulary.write_text("test", encoding="utf-8")
    fusion = CausalPositionFusion(
        config(vocabulary, require_relocalization=False), INTRINSICS
    )
    fusion._visual_frame_index = 901
    fusion._pending_candidate = safe_candidate(900, reference=899)

    output = fusion.fuse_position(np.array([1.0, 2.0, 3.0]), "0")

    assert np.allclose(output, [1.0, 2.0, 3.0])
    assert not fusion.events
