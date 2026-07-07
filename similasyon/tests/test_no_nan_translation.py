"""Translation NaN testleri."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
from src.detected_translation import DetectedTranslation


class TestNoNanTranslation:
    def test_nan_sanitized(self):
        t = DetectedTranslation(float('nan'), 1.0, 1.0)
        assert t.translation_x == 0.0  # NaN -> fallback

    def test_none_sanitized(self):
        t = DetectedTranslation(None, 1.0, 1.0)
        assert t.translation_x == 0.0

    def test_inf_sanitized(self):
        t = DetectedTranslation(1.0, float('inf'), 1.0)
        assert t.translation_y == 0.0

    def test_normal_values_preserved(self):
        t = DetectedTranslation(1.5, 2.5, 3.5)
        assert t.translation_x == 1.5
        assert t.translation_y == 2.5
        assert t.translation_z == 3.5

    def test_payload_no_nan(self):
        t = DetectedTranslation(float('nan'), 1.0, 1.0)
        payload = t.create_payload()
        assert payload['translation_x'] != 'nan'
        assert payload['translation_x'] == '0.0'
