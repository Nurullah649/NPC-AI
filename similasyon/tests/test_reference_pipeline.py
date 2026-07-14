from types import SimpleNamespace

import numpy as np

from src.models.reference_pipeline import ReferenceDetectionPipeline


def _bare_pipeline():
    pipeline = ReferenceDetectionPipeline.__new__(ReferenceDetectionPipeline)
    pipeline.full_min_frame_coverage = 0.0005
    pipeline.full_max_frame_coverage = 0.85
    pipeline.full_max_frame_coverage_by_order = {5: 0.45}
    pipeline.scene_track_min_visible_ratio = 0.50
    pipeline._last_processed_frame = {}
    pipeline._last_results = {}
    pipeline.last_diagnostics = {}
    return pipeline


class StubTracker:
    def __init__(self):
        self.reset_called = False
        self.seed_called = False

    def reset(self):
        self.reset_called = True

    def seed(self, _frame, bbox, expand=False):
        self.seed_called = True
        assert expand is False
        return bbox


def _scene(visible_ratio):
    return SimpleNamespace(
        accepted=True,
        bbox=(10.0, 20.0, 50.0, 80.0),
        visible_ratio=visible_ratio,
        to_dict=lambda: {"accepted": True, "visible_ratio": visible_ratio},
    )


def test_full_bbox_gate_rejects_implausibly_large_ref5_projection():
    pipeline = _bare_pipeline()
    frame = np.zeros((100, 200, 3), dtype=np.uint8)

    accepted, coverage = pipeline._full_bbox_is_sane(
        (0, 10, 190, 100), frame, order=5
    )

    assert accepted is False
    assert coverage > 0.80


def test_partial_scene_match_emits_current_box_without_seeding_tracker():
    pipeline = _bare_pipeline()
    tracker = StubTracker()
    frame = np.zeros((100, 200, 3), dtype=np.uint8)

    result = pipeline._accept_scene_result(
        "ref5", 2067, frame, tracker, 5, _scene(0.34), {}
    )

    assert result.bbox == (10.0, 20.0, 50.0, 80.0)
    assert result.source == "scene_match"
    assert tracker.reset_called is True
    assert tracker.seed_called is False


def test_well_visible_scene_match_seeds_tracker():
    pipeline = _bare_pipeline()
    tracker = StubTracker()
    frame = np.zeros((100, 200, 3), dtype=np.uint8)

    result = pipeline._accept_scene_result(
        "ref5", 1980, frame, tracker, 5, _scene(0.90), {}
    )

    assert result.source == "scene_seed"
    assert tracker.seed_called is True
