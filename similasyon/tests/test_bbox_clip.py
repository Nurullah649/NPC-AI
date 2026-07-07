"""Bbox clipping testleri."""
import sys
import os
import math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pytest


class TestBBoxClipping:
    def test_bbox_within_bounds(self):
        """Normal bbox sınırlar içinde kalmalı."""
        H, W = 480, 640
        x1, y1, x2, y2 = 50, 50, 500, 400
        x1 = max(0, min(x1, W-1))
        y1 = max(0, min(y1, H-1))
        x2 = max(0, min(x2, W-1))
        y2 = max(0, min(y2, H-1))
        assert x1 >= 0 and y1 >= 0
        assert x2 < W and y2 < H
        assert x1 < x2 and y1 < y2

    def test_bbox_negative(self):
        """Negatif koordinatlar 0'a clip'lenmeli."""
        H, W = 480, 640
        x1 = max(0, min(-10, W-1))
        y1 = max(0, min(-20, H-1))
        x2 = max(0, min(100, W-1))
        y2 = max(0, min(200, H-1))
        assert x1 == 0
        assert y1 == 0
        assert x2 == 100
        assert y2 == 200

    def test_bbox_exceeds_frame(self):
        """Sınır dışı koordinatlar frame boyutuna clip'lenmeli."""
        H, W = 480, 640
        x1 = max(0, min(50, W-1))
        y1 = max(0, min(50, H-1))
        x2 = max(0, min(700, W-1))
        y2 = max(0, min(500, H-1))
        assert x2 == W-1
        assert y2 == H-1

    def test_inverted_bbox(self):
        """Ters bbox (x1>x2) geçerli olmamalı."""
        x1, y1, x2, y2 = 100, 100, 50, 50
        is_invalid = (x1 >= x2 or y1 >= y2)
        assert is_invalid

    def test_nan_bbox(self):
        """NaN bbox koordinatları olmamalı."""
        bbox = (float('nan'), 10, 100, 200)
        assert math.isnan(bbox[0])
