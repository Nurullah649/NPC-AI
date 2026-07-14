import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from deneysel.reference_roi_v1.reference_scene_relocalizer import (
    ReferenceSceneRelocalizer,
)


def test_relocalizer_without_verified_scene_memory_never_emits_box():
    frame = np.full((120, 160, 3), 100, dtype=np.uint8)
    relocalizer = ReferenceSceneRelocalizer({})

    result = relocalizer.relocalize(frame)

    assert result.accepted is False
    assert result.bbox is None
    assert result.reason == "no_reliable_scene_homography"


def test_visible_ratio_rejects_polygon_outside_frame():
    points = np.array(
        [[200, 200], [250, 200], [250, 250], [200, 250]], dtype=np.float32
    )

    assert ReferenceSceneRelocalizer._visible_ratio(points, 160, 120) == 0.0
