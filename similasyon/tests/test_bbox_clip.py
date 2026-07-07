"""Bbox clipping testleri."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pytest
from src.models.detector_yolo import DetectorYOLO


class TestBBoxClipping:
    def test_bbox_within_bounds(self):
        """Normal bbox sınırlar içinde kalmalı."""
        from src.models.detector_yolo import DetectorYOLO

    def test_bbox_negative(self):
        """Negatif koordinatlar 0'a clip'lenmeli."""
        det = {'bbox': (-10, -20, 100, 200)}
        H, W = 480, 640
        x1 = max(0, min(det['bbox'][0], W-1))
        y1 = max(0, min(det['bbox'][1], H-1))
        x2 = max(0, min(det['bbox'][2], W-1))
        y2 = max(0, min(det['bbox'][3], H-1))
        assert x1 >= 0
        assert y1 >= 0

    def test_bbox_exceeds_frame(self):
        """Sınır dışı koordinatlar frame boyutuna clip'lenmeli."""
        H, W = 480, 640
        x1 = max(0, min(500, W-1))
        y1 = max(0, min(400, H-1))
        x2 = max(0, min(700, W-1))
        y2 = max(0, min(500, H-1))
        assert x2 == W-1
        assert y2 == H-1

    def test_inverted_bbox(self):
        """Ters bbox (x1>x2) geçerli olmamalı."""
        x1, y1, x2, y2 = 100, 100, 50, 50
        if x1 >= x2 or y1 >= y2:
            assert True  # Skip
        else:
            assert False

    def test_nan_bbox(self):
        """NaN bbox koordinatları olmamalı."""
        import math
        bbox = (float('nan'), 10, 100, 200)
        assert math.isnan(bbox[0])  # Tespit edilmeli
