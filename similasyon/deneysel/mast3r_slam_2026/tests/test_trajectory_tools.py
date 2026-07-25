from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from trajectory_tools import fit_affine, fit_similarity, relative_drift, trajectory_metrics


class TrajectoryToolsTests(unittest.TestCase):
    def test_similarity_recovers_known_transform(self):
        source = np.asarray(
            [[0, 0, 0], [1, 0, 0], [0, 2, 0], [0, 0, 3], [2, 1, 1]],
            dtype=float,
        )
        rotation = np.asarray([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
        target = 2.5 * (source @ rotation.T) + np.asarray([4, -3, 7])
        fitted = fit_similarity(source, target)
        np.testing.assert_allclose(fitted.apply(source), target, atol=1e-10)
        self.assertAlmostEqual(fitted.scale, 2.5)
        self.assertAlmostEqual(fitted.rmse, 0.0, places=10)

    def test_affine_recovers_and_anchors(self):
        source = np.asarray(
            [[0, 0, 0], [1, 0, 0], [0, 2, 0], [0, 0, 3], [2, 1, 1]],
            dtype=float,
        )
        matrix = np.asarray([[2, 0.1, 0], [0, -1, 0.2], [0.3, 0, 1.5]])
        target = source @ matrix.T + np.asarray([10, 20, 30])
        fitted = fit_affine(source, target)
        np.testing.assert_allclose(fitted.apply(source), target, atol=1e-10)
        self.assertFalse(fitted.used_ridge)

    def test_metrics_and_relative_drift_are_zero_for_exact_track(self):
        gt = np.column_stack([np.arange(301, dtype=float), np.zeros(301), np.zeros(301)])
        values = trajectory_metrics(gt, gt)
        self.assertEqual(values["E_3d"], 0.0)
        drift = relative_drift(gt, gt)
        self.assertEqual(drift["10"]["mean_error_m"], 0.0)
        self.assertEqual(drift["250"]["mean_drift_percent"], 0.0)


if __name__ == "__main__":
    unittest.main()

