# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Convex-hull containment primitive (pure torch).

A convex polytope is the intersection of half-spaces ``{x : n·x + d <= 0}``
over its outward face equations. Dropping a face from the input plane set
yields an unbounded polytope along that face's normal — used to model
"open-top" containers, where the rim cap is removed so the polytope extends
infinitely along the container's local opening direction.
"""

from typing import NamedTuple

import numpy as np
import torch


class LocalHull(NamedTuple):
    """Convex hull of a body's mesh, expressed in body-local frame.

    All tensors live on the same device (typically GPU). Built once per body
    at scene-load and cached; never recomputed.

    Fields:
        vertices: (V, 3) — convex hull vertices, used on the *object* side of a
            containment test (object's points transformed into container-local
            frame and tested against the container's planes).
        centroid: (3,) — mean of ``vertices``. Precomputed so the per-step
            ``_obj_centroid_in_container`` doesn't re-reduce V verts each call.
        planes_full: (F, 4) — closed-hull outward face equations, used by
            ``enclosed`` semantics where the object must be fully bounded.
        planes_open_top: (F_kept, 4) — top faces dropped via ``open_top_planes``,
            used by ``inside`` / ``outside_of`` for open-top containers where
            "inside" includes the air column above the rim.
    """
    vertices: torch.Tensor
    centroid: torch.Tensor
    planes_full: torch.Tensor
    planes_open_top: torch.Tensor


def build_local_hull(
    points_np: np.ndarray,
    device: torch.device | str = "cpu",
    open_top_threshold: float = 0.7,
) -> LocalHull:
    """Compute the convex hull of ``points_np`` and package as a ``LocalHull``.

    Pure function — no IsaacSim dependencies. ``scipy.spatial.ConvexHull``
    handles QuickHull; the heavy work happens here, once per body. Resulting
    tensors are pushed to ``device`` for downstream point-in-hull math.

    Args:
        points_np: (P, 3) array of mesh points in body-local frame.
        device: torch device for the returned tensors.
        open_top_threshold: passed to ``open_top_planes`` (axis=2).

    Returns:
        LocalHull with vertices, full planes, and open-top planes.

    Raises:
        scipy.spatial.qhull.QhullError if ``points_np`` is degenerate
        (coplanar, < 4 distinct points, etc.). Caller decides on fallback.
    """
    from scipy.spatial import ConvexHull

    hull = ConvexHull(points_np)
    verts = torch.tensor(points_np[hull.vertices], dtype=torch.float32, device=device)
    centroid = verts.mean(dim=0)
    planes_full = torch.tensor(hull.equations, dtype=torch.float32, device=device)
    planes_ot = open_top_planes(planes_full, axis=2, threshold=open_top_threshold)
    return LocalHull(vertices=verts, centroid=centroid, planes_full=planes_full, planes_open_top=planes_ot)


def point_in_hull(points: torch.Tensor, planes: torch.Tensor) -> torch.Tensor:
    """Return a boolean mask over points lying inside the convex polytope.

    Args:
        points: shape ``(..., 3)``, in the same frame as ``planes``.
        planes: shape ``(F, 4)``, outward face equations ``(nx, ny, nz, d)``.
            A point ``x`` is inside iff ``n·x + d <= 0`` for every face.

    Returns:
        Boolean tensor of shape ``(...,)``. The closed half-space test
        includes the boundary (``<= 0``).
    """
    signed = points @ planes[:, :3].T + planes[:, 3]   # (..., F)
    return signed.amax(dim=-1) <= 0


def open_top_planes(planes: torch.Tensor, axis: int = 2, threshold: float = 0.7) -> torch.Tensor:
    """Drop faces whose outward normal projects ``>= threshold`` onto ``+axis``.

    With default ``axis=2`` and ``threshold=0.7``, this removes faces facing
    within ~45° of straight up — the rim cap of an open-top container in the
    standard +z-up authoring convention. The resulting polytope is unbounded
    along the container's local +z, modelling the "open air above the rim".

    Args:
        planes: shape ``(F, 4)``.
        axis: index of the local axis defining "up". Default 2 (z).
        threshold: minimum normal component to drop. Faces with
            ``planes[:, axis] >= threshold`` are removed.

    Returns:
        ``(F_kept, 4)`` plane subset. Caller is responsible for caching.
    """
    keep = planes[:, axis] < threshold
    return planes[keep]
