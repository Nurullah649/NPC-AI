from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np


MODULE_PATH = Path(__file__).resolve().parents[1] / "offline_sweep.py"
SPEC = importlib.util.spec_from_file_location("dpvo_drift_offline_sweep", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_causal_average_uses_only_present_and_past_values():
    values = np.arange(15, dtype=np.float64).reshape(5, 3)
    averaged = MODULE.causal_average(values, 3)
    np.testing.assert_allclose(averaged[0], values[0])
    np.testing.assert_allclose(averaged[1], values[:2].mean(axis=0))
    np.testing.assert_allclose(averaged[4], values[2:5].mean(axis=0))


def test_robust_fit_downweights_a_large_delta_outlier():
    x = np.linspace(-2.0, 2.0, 80)[:, None]
    design = np.column_stack([x, np.ones(len(x))])
    target = np.column_stack([2.0 * x[:, 0] + 1.0, -x[:, 0] + 0.5])
    target[40] += np.array([100.0, -80.0])
    least_squares, _ = MODULE.fit_linear(
        design, target, ridge=1e-9, robust=False
    )
    robust, diagnostics = MODULE.fit_linear(
        design, target, ridge=1e-9, robust=True
    )
    truth = np.array([[2.0, -1.0], [1.0, 0.5]])
    assert np.linalg.norm(robust - truth) < np.linalg.norm(least_squares - truth)
    assert diagnostics["downweighted_count"] >= 1


def test_pose_affine_anchor_is_exact_at_training_boundary():
    raw = np.column_stack(
        [np.linspace(0, 2, 20), np.linspace(1, 4, 20), np.linspace(-1, 1, 20)]
    )
    matrix = np.array([[2.0, 0.2, 0.0], [-0.1, 1.5, 0.3], [0.0, 0.0, 4.0]])
    gt = raw @ matrix.T + np.array([3.0, -2.0, 8.0])
    future = raw[-2:]
    prediction, _ = MODULE.pose_affine(raw, gt, future, ridge=0.0)
    np.testing.assert_allclose(prediction[-1], gt[-1], atol=1e-10)
