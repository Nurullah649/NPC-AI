"""DirectionGuard testleri."""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pytest

from src.models.direction_guard import DirectionGuard


class TestDirectionGuard:
    @pytest.fixture
    def config(self):
        return {
            "dpvo": {
                "use_direction_guard": True,
                "direction_guard_min_points": 3,
                "max_allowed_scale": 50.0,
                "min_allowed_scale": 0.02,
                "min_direction_similarity": -0.2,
            }
        }

    def test_same_direction(self, config):
        guard = DirectionGuard(config)
        gt = [(float(i), float(i), 0.0) for i in range(10)]
        dpvo = [(float(i * 2), float(i * 2), 0.0) for i in range(10)]  # Same direction, scale=2
        result = guard.evaluate(gt, dpvo)
        assert result["ok"] is True, f"Expected ok=True, got {result}"

    def test_opposite_direction(self, config):
        guard = DirectionGuard(config)
        gt = [(float(i), float(i), 0.0) for i in range(10)]
        dpvo = [(float(-i), float(-i), 0.0) for i in range(10)]  # Opposite
        result = guard.evaluate(gt, dpvo)
        # Should have low similarity
        assert result["direction_similarity"] < 0

    def test_zero_movement(self, config):
        guard = DirectionGuard(config)
        gt = [(0.0, 0.0, 0.0) for _ in range(10)]
        dpvo = [(0.0, 0.0, 0.0) for _ in range(10)]
        result = guard.evaluate(gt, dpvo)
        assert result["ok"] is False

    def test_nan_input(self, config):
        guard = DirectionGuard(config)
        gt = [(float(i), float(i), float('nan')) for i in range(10)]
        dpvo = [(float(i), float(i), 0.0) for i in range(10)]
        result = guard.evaluate(gt, dpvo)
        assert result["ok"] is False

    def test_extreme_scale(self, config):
        guard = DirectionGuard(config)
        gt = [(float(i), float(i), 0.0) for i in range(10)]
        dpvo = [(float(i * 100), float(i * 100), 0.0) for i in range(10)]  # scale=100
        result = guard.evaluate(gt, dpvo)
        # scale=100 > max_allowed=50
        assert result["ok"] is False

    def test_insufficient_points(self, config):
        guard = DirectionGuard(config)
        gt = [(1.0, 1.0, 0.0)]
        dpvo = [(1.0, 1.0, 0.0)]
        result = guard.evaluate(gt, dpvo)
        assert result["ok"] is False  # min_points=3
