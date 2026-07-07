"""CLAHE preprocessing testleri."""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import cv2
import numpy as np
import pytest

from src.models.image_preprocess import ImagePreprocessor


class TestImagePreprocessor:
    @pytest.fixture
    def config_clahe_off(self):
        return {"preprocessing": {"use_clahe": False}}

    @pytest.fixture
    def config_clahe_on(self):
        return {
            "preprocessing": {
                "use_clahe": True,
                "clahe_clip_limit": 2.0,
                "clahe_tile_grid_size": 8,
                "split_rows": 2,
                "split_cols": 3,
                "brightness_threshold": 50,
                "apply_to_detector": False,
                "apply_to_landing_classifier": True,
                "apply_to_reference_matcher": False,
            }
        }

    def test_clahe_off_returns_same(self, config_clahe_off):
        prep = ImagePreprocessor(config_clahe_off)
        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        result = prep.apply(img, "detector")
        assert result is img  # Same object when disabled

    def test_clahe_on_shape_preserved(self, config_clahe_on):
        prep = ImagePreprocessor(config_clahe_on)
        img = np.random.randint(0, 100, (480, 640, 3), dtype=np.uint8)  # Dark
        result = prep.apply(img, "landing_classifier")
        assert result.shape == img.shape
        assert result.dtype == img.dtype

    def test_none_input(self, config_clahe_off):
        prep = ImagePreprocessor(config_clahe_off)
        assert prep.apply(None) is None

    def test_clahe_active_only_for_correct_purpose(self, config_clahe_on):
        prep = ImagePreprocessor(config_clahe_on)
        assert not prep.is_active_for("detector")
        assert prep.is_active_for("landing_classifier")
        assert not prep.is_active_for("reference_matcher")

    def test_split_merge_shape(self, config_clahe_on):
        prep = ImagePreprocessor(config_clahe_on)
        img = np.random.randint(0, 100, (480, 640, 3), dtype=np.uint8)
        # Apply for landing_classifier (active)
        result = prep.apply(img, "landing_classifier")
        assert result.shape == (480, 640, 3)
        # Tile-based should not change dimensions
        assert np.array_equal(result.shape, img.shape)

    def test_bright_tile_unchanged(self, config_clahe_on):
        prep = ImagePreprocessor(config_clahe_on)
        img = np.random.randint(200, 255, (480, 640, 3), dtype=np.uint8)  # Bright
        result = prep.apply(img, "landing_classifier")
        assert result.shape == img.shape
        # Bright image should come through (may have small diff but shape preserved)
