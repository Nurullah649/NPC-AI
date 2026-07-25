"""ReferenceMatcher LightGlue cihaz ve çıktı-sözleşmesi regresyon testleri."""

import logging
import os
import sys

import cv2
import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.models.reference_matcher import ReferenceMatcher


class _FakeExtractor:
    def __init__(self, features):
        self.features = features

    def extract(self, _tensor):
        return self.features


class _FakeMatcher:
    def __init__(self, pairs):
        self.pairs = pairs

    def __call__(self, _features):
        # LightGlue 0.2.x: batch başına [K, 2] tensörlerinden oluşan liste.
        return {'matches': [self.pairs]}


def _matcher_with_synthetic_correspondences(frame_points):
    matcher = ReferenceMatcher.__new__(ReferenceMatcher)
    matcher.logger = logging.getLogger('test.reference_matcher')
    matcher._lightglue_available = True
    matcher._device = torch.device('cpu')
    matcher.min_matches = 10
    matcher.min_inliers = 8
    matcher.min_inlier_ratio = 0.25

    ref_points = torch.tensor(
        [[
            [10.0, 10.0], [40.0, 10.0], [70.0, 10.0], [100.0, 10.0],
            [10.0, 40.0], [40.0, 40.0], [70.0, 40.0], [100.0, 40.0],
            [10.0, 70.0], [40.0, 70.0], [70.0, 70.0], [100.0, 70.0],
        ]],
        dtype=torch.float32,
    )
    frame_features = {'keypoints': frame_points}
    pairs = torch.column_stack((torch.arange(12), torch.arange(12)))
    matcher._extractor = _FakeExtractor(frame_features)
    matcher._matcher = _FakeMatcher(pairs)
    ref_features = {
        'lightglue': {'keypoints': ref_points},
        'shape': (80, 120),
    }
    return matcher, ref_features, ref_points


def test_lightglue_list_output_produces_bbox():
    translation = torch.tensor([[[100.0, 50.0]]])
    ref_template = torch.tensor(
        [[
            [10.0, 10.0], [40.0, 10.0], [70.0, 10.0], [100.0, 10.0],
            [10.0, 40.0], [40.0, 40.0], [70.0, 40.0], [100.0, 40.0],
            [10.0, 70.0], [40.0, 70.0], [70.0, 70.0], [100.0, 70.0],
        ]],
        dtype=torch.float32,
    )
    matcher, ref_features, _ = _matcher_with_synthetic_correspondences(
        ref_template + translation
    )

    bbox = matcher._match_lightglue(
        ref_features,
        np.zeros((240, 320, 3), dtype=np.uint8),
    )

    assert bbox is not None
    assert bbox == pytest.approx((100.0, 50.0, 219.0, 129.0), abs=1e-3)


def test_lightglue_rejects_too_few_ransac_inliers(monkeypatch):
    ref_template = torch.zeros((1, 12, 2), dtype=torch.float32)
    matcher, ref_features, _ = _matcher_with_synthetic_correspondences(ref_template)

    def fake_find_homography(_src, _dst, _method, _threshold):
        mask = np.array([[1]] * 5 + [[0]] * 7, dtype=np.uint8)
        return np.eye(3, dtype=np.float64), mask

    monkeypatch.setattr(cv2, 'findHomography', fake_find_homography)

    bbox = matcher._match_lightglue(
        ref_features,
        np.zeros((240, 320, 3), dtype=np.uint8),
    )

    assert bbox is None


def test_geometry_gate_rejects_large_projection_before_clipping():
    matcher = ReferenceMatcher.__new__(ReferenceMatcher)
    matcher.min_matches = 10
    matcher.min_inliers = 8
    matcher.min_inlier_ratio = 0.25

    source = np.array(
        [
            [10, 10], [40, 10], [70, 10], [100, 10],
            [10, 40], [40, 40], [70, 40], [100, 40],
            [10, 70], [40, 70], [70, 70], [100, 70],
        ],
        dtype=np.float32,
    )
    homography = np.array(
        [[5.0, 0.0, 100.0], [0.0, 5.0, 50.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    destination = cv2.perspectiveTransform(
        source.reshape(-1, 1, 2), homography
    ).reshape(-1, 2)

    bbox, diagnostics = matcher._validate_homography_projection(
        source,
        destination,
        homography,
        np.ones((len(source), 1), dtype=np.uint8),
        (80, 120),
        (240, 320),
        "test",
    )

    assert bbox is None
    assert diagnostics["reason"] == "projection_too_large"
    assert diagnostics["projected_area_ratio"] > 1.0


def test_initialized_lightglue_uses_one_device():
    matcher = ReferenceMatcher({'reference': {'device': 'cpu'}})
    if not matcher._lightglue_available:
        pytest.skip('LightGlue bu ortamda kullanılamıyor')

    expected = torch.device('cpu')
    assert matcher._device == expected
    assert next(matcher._extractor.parameters()).device == expected
    assert next(matcher._matcher.parameters()).device == expected
    tensor = matcher._to_lightglue_tensor(
        np.zeros((32, 32, 3), dtype=np.uint8)
    )
    assert tensor.device == expected
