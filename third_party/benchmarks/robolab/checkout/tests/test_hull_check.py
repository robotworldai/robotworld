# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pure-torch unit tests for ``robolab.core.task.hull_check``.

These tests don't need IsaacSim. The ``tests/`` conftest still boots Isaac Sim
once (because other tests need it), so for quick iteration run this file
directly with ``python tests/test_hull_check.py``.
"""

import math
import os
import sys

import numpy as np

# Standalone-runner path shim: when invoked as ``python tests/test_hull_check.py``
# from a worktree, the editable-install hook resolves ``robolab`` to the main
# repo (where the install was registered), not the worktree. Pytest works
# without this because cwd is already on sys.path. No-op when worktree root is
# already on sys.path (e.g. pytest, or running from worktree root).
_WORKTREE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _WORKTREE_ROOT not in sys.path:
    sys.path.insert(0, _WORKTREE_ROOT)

import torch  # noqa: E402

from robolab.core.task.hull_check import LocalHull, build_local_hull, open_top_planes, point_in_hull  # noqa: E402


def cube_planes(half: float = 0.5) -> torch.Tensor:
    """``(6, 4)`` outward face equations for the axis-aligned cube ``[-half, half]^3``.

    Order: +x, -x, +y, -y, +z (top), -z (bottom).
    """
    return torch.tensor([
        [+1.0, 0.0, 0.0, -half],
        [-1.0, 0.0, 0.0, -half],
        [0.0, +1.0, 0.0, -half],
        [0.0, -1.0, 0.0, -half],
        [0.0, 0.0, +1.0, -half],
        [0.0, 0.0, -1.0, -half],
    ], dtype=torch.float32)


def test_unit_cube_interior():
    planes = cube_planes(0.5)
    pts = torch.tensor([[0.0, 0.0, 0.0], [0.4, 0.4, 0.4], [-0.4, 0.4, -0.4]], dtype=torch.float32)
    assert point_in_hull(pts, planes).all()


def test_unit_cube_exterior():
    planes = cube_planes(0.5)
    pts = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-1.0, -1.0, -1.0], [0.0, 0.0, 0.6]], dtype=torch.float32)
    assert not point_in_hull(pts, planes).any()


def test_unit_cube_boundary_inside():
    """Closed half-space (``<= 0``) includes face boundaries."""
    planes = cube_planes(0.5)
    pts = torch.tensor([[0.5, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, -0.5]], dtype=torch.float32)
    assert point_in_hull(pts, planes).all()


def test_open_top_drops_correct_faces():
    """Default open_top_planes drops the +z face only on a standard cube."""
    planes = cube_planes(0.5)
    kept = open_top_planes(planes)  # axis=2, threshold=0.7
    assert kept.shape[0] == 5
    assert (kept[:, 2] < 0.7).all()


def test_open_top_high_above_box():
    """Points high above an open-top box read inside (no upper bound)."""
    planes = open_top_planes(cube_planes(0.5))
    pts = torch.tensor([[0.0, 0.0, 100.0], [0.3, -0.2, 5.0]], dtype=torch.float32)
    assert point_in_hull(pts, planes).all()


def test_open_top_lateral_outside():
    """Lateral wall planes still bound the open-top polytope."""
    planes = open_top_planes(cube_planes(0.5))
    pts = torch.tensor([[5.0, 0.0, 0.0], [0.0, 5.0, 100.0]], dtype=torch.float32)
    assert not point_in_hull(pts, planes).any()


def test_open_top_below_floor():
    """Bottom plane is preserved — points below the floor are still outside."""
    planes = open_top_planes(cube_planes(0.5))
    pts = torch.tensor([[0.0, 0.0, -1.0], [0.3, 0.0, -2.0]], dtype=torch.float32)
    assert not point_in_hull(pts, planes).any()


def test_threshold_tunable():
    """Stricter threshold keeps slightly-upward faces; permissive drops them."""
    half = 0.5
    sloped_normal = torch.tensor([0.0, 0.8, 0.6], dtype=torch.float32)
    sloped_normal = sloped_normal / sloped_normal.norm()  # n_z ≈ 0.6
    sloped_top_plane = torch.cat([sloped_normal, torch.tensor([-half])]).unsqueeze(0)
    walls = cube_planes(half)[:4]
    bottom = cube_planes(half)[5:6]
    planes = torch.cat([walls, bottom, sloped_top_plane])  # 6 faces

    # threshold=0.5: sloped top (n_z ≈ 0.6) is dropped → 5 kept
    assert open_top_planes(planes, threshold=0.5).shape[0] == 5
    # threshold=0.9: sloped top kept → all 6
    assert open_top_planes(planes, threshold=0.9).shape[0] == 6


def test_tilted_polytope():
    """Rotation invariance: rotating planes and points by the same R gives same result."""
    planes = cube_planes(0.5)
    pts = torch.tensor([[0.3, 0.3, 0.3], [0.6, 0.0, 0.0], [-0.6, 0.0, 0.0]], dtype=torch.float32)
    theta = math.radians(30)
    c, s = math.cos(theta), math.sin(theta)
    R = torch.tensor([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=torch.float32)
    rotated_normals = planes[:, :3] @ R.T
    rotated_planes = torch.cat([rotated_normals, planes[:, 3:4]], dim=1)
    rotated_pts = pts @ R.T
    expected = point_in_hull(pts, planes)
    actual = point_in_hull(rotated_pts, rotated_planes)
    assert torch.equal(expected, actual)


def test_batched_points_shape_and_frac():
    """``(N, V, 3)`` → ``(N, V)`` bool; per-batch frac_inside is the dim=-1 mean."""
    planes = cube_planes(0.5)
    pts = torch.tensor([
        [[0.0, 0.0, 0.0], [0.3, 0.0, 0.0], [0.4, 0.4, 0.0], [1.0, 0.0, 0.0]],   # 3 in, 1 out
        [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [-2.0, 0.0, 0.0], [0.0, 2.0, 0.0]],  # 1 in, 3 out
    ], dtype=torch.float32)
    inside = point_in_hull(pts, planes)
    assert inside.shape == (2, 4)
    frac = inside.float().mean(dim=-1)
    assert torch.allclose(frac, torch.tensor([0.75, 0.25]))


def test_only_floor_remains():
    """Aggressive threshold (0.0) drops every plane with non-negative n_z, leaving the floor."""
    planes = open_top_planes(cube_planes(0.5), threshold=0.0)
    # Only the bottom face survives: (0, 0, -1, -0.5) → inside iff z >= -0.5.
    assert planes.shape == (1, 4)
    pts = torch.tensor([[0.0, 0.0, 0.0], [100.0, 100.0, 0.0], [0.0, 0.0, -2.0]], dtype=torch.float32)
    expected = torch.tensor([True, True, False])
    assert torch.equal(point_in_hull(pts, planes), expected)


def test_single_point_shape():
    """Shape ``(3,)`` → scalar bool tensor."""
    planes = cube_planes(0.5)
    inside = point_in_hull(torch.tensor([0.0, 0.0, 0.0]), planes)
    outside = point_in_hull(torch.tensor([2.0, 0.0, 0.0]), planes)
    assert bool(inside) is True
    assert bool(outside) is False


def _cube_points(half: float = 0.5) -> np.ndarray:
    """``(8, 3)`` cube vertices for testing build_local_hull."""
    return np.array([
        [-half, -half, -half], [+half, -half, -half],
        [-half, +half, -half], [+half, +half, -half],
        [-half, -half, +half], [+half, -half, +half],
        [-half, +half, +half], [+half, +half, +half],
    ], dtype=np.float64)


def test_build_local_hull_cube():
    """Cube → 8 verts, 12 simplices (scipy triangulates each face), unit normals.

    scipy.spatial.ConvexHull returns triangulated simplices, so a cube's 6 faces
    become 12 triangles with 6 unique outward normals (each duplicated).
    Duplicates are harmless in ``point_in_hull`` (extra matmul cols of equal value).
    """
    hull = build_local_hull(_cube_points(0.5))
    assert isinstance(hull, LocalHull)
    assert hull.vertices.shape == (8, 3)
    assert hull.planes_full.shape == (12, 4)
    norms = hull.planes_full[:, :3].norm(dim=-1)
    assert torch.allclose(norms, torch.ones(12))
    # Top face (n_z = 1) split into 2 triangles → both dropped → 10 remain
    assert hull.planes_open_top.shape == (10, 4)
    assert (hull.planes_open_top[:, 2] < 0.7).all()


def test_build_local_hull_tetrahedron():
    """Minimal 3D hull: 4 verts, 4 faces."""
    points = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ])
    hull = build_local_hull(points)
    assert hull.vertices.shape == (4, 3)
    assert hull.planes_full.shape == (4, 4)


def test_build_local_hull_redundant_interior_points():
    """Interior points are discarded; only the convex envelope vertices remain."""
    cube = _cube_points(0.5)
    rng = np.random.default_rng(0)
    interior = rng.uniform(-0.4, 0.4, size=(50, 3))
    points = np.concatenate([cube, interior], axis=0)
    hull = build_local_hull(points)
    assert hull.vertices.shape == (8, 3)
    assert hull.planes_full.shape == (12, 4)


def test_build_local_hull_open_top_threshold_passthrough():
    """``open_top_threshold`` is plumbed through to ``open_top_planes``.

    Cube top faces have ``n_z`` exactly 1.0. Strict ``keep = n_z < threshold``
    semantics: threshold > 1.0 keeps everything, threshold ≤ 1.0 drops the top.
    """
    points = _cube_points(0.5)
    permissive = build_local_hull(points, open_top_threshold=2.0)   # > 1, keeps top
    strict = build_local_hull(points, open_top_threshold=0.5)       # drops top
    assert permissive.planes_open_top.shape == (12, 4)  # nothing dropped
    assert strict.planes_open_top.shape == (10, 4)      # top 2 triangles dropped


def test_build_local_hull_point_in_hull_roundtrip():
    """Build a hull, then verify ``point_in_hull`` agrees with axis-aligned bounds."""
    half = 0.3
    hull = build_local_hull(_cube_points(half))
    pts = torch.tensor([
        [0.0, 0.0, 0.0], [0.2, 0.2, 0.2],   # interior
        [0.5, 0.0, 0.0], [0.0, 0.5, 0.0],   # exterior
    ], dtype=torch.float32)
    expected = torch.tensor([True, True, False, False])
    assert torch.equal(point_in_hull(pts, hull.planes_full), expected)


if __name__ == "__main__":
    # Standalone runner — fast iteration without paying conftest's IsaacSim startup.
    # Run: python tests/test_hull_check.py
    import sys
    tests = [v for k, v in list(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {t.__name__}: {e}")
    if failed:
        print(f"\n{failed}/{len(tests)} failed")
        sys.exit(1)
    print(f"\nAll {len(tests)} tests passed.")
