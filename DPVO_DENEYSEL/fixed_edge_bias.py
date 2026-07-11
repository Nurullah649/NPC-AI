"""Correct the project-local EDGE_BIAS feature-grid coordinate mismatch.

The local implementation computed a Sobel map at image resolution but sampled
it with coordinates from DPVO's 1/4-resolution feature map.  This patch keeps
the experiment isolated by replacing only the private edge-map helper at
runtime.  The edge map is pooled to the feature-map resolution, matching the
official GRADIENT_BIAS implementation.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _fixed_compute_edge_map(self, images: torch.Tensor) -> torch.Tensor:
    if images.dim() != 5:
        raise ValueError(f"DPVO images tensörü [B,N,C,H,W] olmalı, gelen={tuple(images.shape)}")

    batch, frames, channels, height, width = images.shape
    gray = images.mean(dim=2).reshape(batch * frames, 1, height, width)
    sobel_x = torch.tensor(
        [[1, 0, -1], [2, 0, -2], [1, 0, -1]],
        dtype=gray.dtype,
        device=gray.device,
    ).view(1, 1, 3, 3)
    sobel_y = torch.tensor(
        [[1, 2, 1], [0, 0, 0], [-1, -2, -1]],
        dtype=gray.dtype,
        device=gray.device,
    ).view(1, 1, 3, 3)
    grad_x = F.conv2d(gray, sobel_x, padding=1)
    grad_y = F.conv2d(gray, sobel_y, padding=1)
    magnitude = torch.sqrt(grad_x.square() + grad_y.square() + 1e-12)
    magnitude = F.avg_pool2d(magnitude, kernel_size=4, stride=4)
    return magnitude.view(batch, frames, magnitude.shape[-2], magnitude.shape[-1])


def install_fixed_edge_bias() -> None:
    """Install the corrected helper on the already imported Patchifier class."""
    from Class.DPVO.dpvo.net import Patchifier

    Patchifier._Patchifier__compute_edge_map = _fixed_compute_edge_map

