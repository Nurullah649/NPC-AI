import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from deneysel.reference_roi_v1.roi_matcher import (
    RoiMatchDiagnostics,
    RoiReferenceMatcher,
    _bbox_iou,
    _hull_coverage,
)


def test_bbox_iou_identical():
    assert _bbox_iou((10, 20, 100, 200), (10, 20, 100, 200)) == 1.0


def test_hull_coverage_spread_points_is_positive():
    points = np.array([[10, 10], [90, 10], [90, 90], [10, 90]], dtype=np.float32)
    assert _hull_coverage(points, 100, 100) > 0.60


def test_candidate_deduplication_keeps_high_confidence():
    matcher = RoiReferenceMatcher.__new__(RoiReferenceMatcher)
    matcher.candidate_min_conf = 0.25
    matcher.duplicate_iou = 0.85
    detections = [
        {"cls": 0, "conf": 0.95, "bbox": (100, 100, 200, 200)},
        {"cls": 0, "conf": 0.50, "bbox": (101, 101, 201, 201)},
        {"cls": 1, "conf": 0.99, "bbox": (400, 400, 500, 500)},
    ]

    kept = matcher._deduplicate_candidates(detections)

    assert len(kept) == 1
    assert kept[0][0] == 0


def test_diagnostics_serializes_non_finite_error_as_none():
    diagnostics = RoiMatchDiagnostics(
        candidate_index=0,
        candidate_bbox=(1, 2, 3, 4),
        candidate_conf=0.9,
        crop_bbox=(0, 0, 10, 10),
        scale=2,
    )
    assert diagnostics.to_dict()["reprojection_error"] is None
