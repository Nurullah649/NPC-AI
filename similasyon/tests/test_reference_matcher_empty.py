"""Reference matcher boş/None testleri."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pytest


class TestReferenceMatcherEmpty:
    def test_no_active_refs(self):
        """Aktif referans yoksa match çağrılmamalı."""
        config = {}
        from src.models.reference_matcher import ReferenceMatcher
        matcher = ReferenceMatcher(config)
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        result = matcher.match("nonexistent_ref", None, frame)
        assert result is None

    def test_match_no_ref_path(self):
        """Referans yolu None ise match None dönmeli."""
        from src.models.reference_matcher import ReferenceMatcher
        matcher = ReferenceMatcher({})
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        result = matcher.match("ref_url", None, frame)
        assert result is None

    def test_empty_frame(self):
        """Boş frame ile match None dönmeli."""
        from src.models.reference_matcher import ReferenceMatcher
        matcher = ReferenceMatcher({})
        frame = np.zeros((10, 10, 3), dtype=np.uint8)
        result = matcher.match("ref_url", None, frame)
        assert result is None
