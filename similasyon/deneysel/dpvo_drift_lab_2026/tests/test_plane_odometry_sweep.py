from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SPEC = importlib.util.spec_from_file_location(
    "dpvo_plane_odometry_sweep", ROOT / "plane_odometry_sweep.py"
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_identity_homographies_produce_zero_plane_motion():
    homographies = np.repeat(np.eye(3)[None], 8, axis=0)
    intrinsics = np.array(
        [[463.2, 0.0, 318.0], [0.0, 462.4, 186.3], [0.0, 0.0, 1.0]]
    )
    delta, diagnostics = MODULE.decompose_plane_motion(
        homographies, intrinsics
    )
    np.testing.assert_allclose(delta, 0.0, atol=1e-12)
    assert diagnostics["plane_distance_end_over_start"] == 1.0
    assert sum(diagnostics["solution_counts"]) == len(homographies) - 1

