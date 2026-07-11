"""DPVO wrapper with the online pose convention fixed.

Princeton DPVO stores active graph poses in the world-to-camera convention.
The official ``terminate()`` path applies ``poses.inv()`` before exposing a
trajectory.  The project-local online accessor omitted that inversion.  This
class keeps the existing DPVO core untouched and exposes the current pose in
the same camera-to-world convention as the official completed trajectory.
"""

from __future__ import annotations

import numpy as np
import torch

from Class.DPVO.dpvo.dpvo import DPVO
from Class.DPVO.dpvo.lietorch import SE3


class FixedOnlineDPVO(DPVO):
    """DPVO with a correct camera-to-world online pose accessor."""

    @torch.no_grad()
    def get_current_pose_vectors(self) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(camera_to_world, internal_world_to_camera)`` vectors.

        Both vectors use DPVO's seven-value layout:
        ``[tx, ty, tz, qx, qy, qz, qw]``.
        """
        if self.n == 0:
            raise RuntimeError("Henüz herhangi bir frame işlenmedi; pozisyon mevcut değil.")

        internal = SE3(self.pg.poses_[self.n - 1])
        camera_to_world = internal.inv()
        return (
            camera_to_world.data.detach().float().cpu().numpy().copy(),
            internal.data.detach().float().cpu().numpy().copy(),
        )

    @torch.no_grad()
    def get_current_pose(self) -> np.ndarray:
        """Return the current camera-to-world pose as a 4x4 matrix."""
        if self.n == 0:
            raise RuntimeError("Henüz herhangi bir frame işlenmedi; pozisyon mevcut değil.")
        return (
            SE3(self.pg.poses_[self.n - 1])
            .inv()
            .matrix()
            .detach()
            .float()
            .cpu()
            .numpy()
            .copy()
        )

