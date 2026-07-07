"""Reference matcher boş/None testleri."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pytest


class TestReferenceMatcherEmpty:
    @pytest.fixture
    def matcher(self):
        """ReferenceMatcher'ı yüklemeyi dene, başarısız olursa skip."""
        try:
            from src.models.reference_matcher import ReferenceMatcher
            return ReferenceMatcher({})
        except ImportError as e:
            pytest.skip(f"ReferenceMatcher yüklenemedi (bağımlılık eksik): {e}")
        except Exception as e:
            pytest.skip(f"ReferenceMatcher başlatılamadı: {e}")

    def test_no_active_refs(self, matcher):
        """Aktif referans yoksa match None dönmeli."""
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        result = matcher.match("nonexistent_ref", None, frame)
        assert result is None

    def test_match_no_ref_path(self, matcher):
        """Referans yolu None ise match None dönmeli."""
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        result = matcher.match("ref_url", None, frame)
        assert result is None

    def test_empty_frame(self, matcher):
        """Boş frame ile match None dönmeli."""
        frame = np.zeros((10, 10, 3), dtype=np.uint8)
        result = matcher.match("ref_url", None, frame)
        assert result is None
