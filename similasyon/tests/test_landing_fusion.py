"""Landing fusion testleri."""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pytest

from src.models.landing_status import LandingStatusResolver


class TestLandingFusion:
    @pytest.fixture
    def config(self):
        return {
            "landing": {
                "overlap_threshold": 0.2,
                "near_buffer_px": 20,
                "use_classifier": True,
                "classifier_threshold": 0.5,
                "hard_veto_obstacles": True,
            }
        }

    def test_non_landing_class(self, config):
        """cls 0/1 için landing_status '-1' olmalı."""
        resolver = LandingStatusResolver(config)
        dets = [{"cls": 0, "bbox": (10, 10, 100, 100)}]
        result = resolver.resolve(dets, (480, 640))
        assert result[0]["landing_status"] == "-1"

    def test_uap_blocked_by_person_center(self, config):
        """UAP içine insan merkezi girerse, classifier mock yok, rule veto."""
        resolver = LandingStatusResolver(config)
        dets = [
            {"cls": 1, "bbox": (120, 120, 180, 180)},  # İnsan
            {"cls": 2, "bbox": (100, 100, 300, 300)},  # UAP
        ]
        # İnsan merkezi (150,150) UAP içinde -> blocked
        result = resolver.resolve(dets, (480, 640))
        uap = [d for d in result if d["cls"] == 2][0]
        assert uap["landing_status"] == "0"

    def test_no_obstacle_rule_fallback(self, config):
        """Obstacle yoksa rule fallback '1' olmalı."""
        resolver = LandingStatusResolver(config)
        dets = [{"cls": 2, "bbox": (100, 100, 300, 300)}]
        result = resolver.resolve(dets, (480, 640))
        uap = result[0]
        assert uap["landing_status"] == "1"

    def test_sahi_person_as_obstacle(self, config):
        """SAHI'den gelen insan (source='sahi') da obstacle olmalı."""
        resolver = LandingStatusResolver(config)
        dets = [
            {"cls": 8, "bbox": (140, 140, 160, 160), "source": "sahi"},
            {"cls": 2, "bbox": (100, 100, 300, 300)},
        ]
        result = resolver.resolve(dets, (480, 640))
        uap = [d for d in result if d["cls"] == 2][0]
        assert uap["landing_status"] == "0"

    def test_invalid_bbox_no_crash(self, config):
        """Geçersiz bbox crash etmemeli."""
        resolver = LandingStatusResolver(config)
        dets = [{"cls": 2, "bbox": (200, 200, 100, 100)}]  # ters
        result = resolver.resolve(dets, (480, 640))
        assert len(result) == 1

    def test_payload_no_extra_fields(self, config):
        """Payload'a debug/probability alanı eklenmemeli."""
        resolver = LandingStatusResolver(config)
        dets = [{"cls": 2, "bbox": (100, 100, 300, 300)}]
        result = resolver.resolve(dets, (480, 640))
        keys = set(result[0].keys())
        assert 'landing_source' not in keys
        assert 'landing_probability' not in keys
        assert 'landing_status' in keys
