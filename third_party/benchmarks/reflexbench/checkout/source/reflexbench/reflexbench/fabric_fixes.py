"""Workarounds for upstream Isaac Lab / Isaac Sim Fabric bugs."""
from __future__ import annotations

import re
from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def _quat_wxyz_to_rot3(qw: float, qx: float, qy: float, qz: float) -> np.ndarray:
    """Build 3x3 rotation matrix from a ``(w, x, y, z)`` quaternion (normalized internally)."""
    n = float(np.sqrt(qw * qw + qx * qx + qy * qy + qz * qz))
    if n == 0.0:
        return np.eye(3, dtype=np.float64)
    qw, qx, qy, qz = qw / n, qx / n, qy / n, qz / n
    return np.array(
        [
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
        ],
        dtype=np.float64,
    )


def mark_wrist_cam_usd_dirty(
    env: "ManagerBasedEnv",
    env_ids: torch.Tensor | None,
    sensor_name: str = "wrist_cam",
) -> None:
    """Workaround for an upstream Isaac Lab / Isaac Sim Fabric-sync bug.

    Directly authors the correct ``omni:fabric:localMatrix`` for a child-mounted
    camera from its configured offset (``cam.cfg.offset``), then propagates the
    change through Fabric with ``update_world_xforms``.

    Why it is needed:
        ``isaaclab.sim.views.xform_prim_view.XformPrimView._sync_fabric_from_usd_once``
        is invoked on the first Fabric read. It writes Fabric ``worldMatrix`` first
        and lets ``update_world_xforms`` back-compute ``local = world * parent_world^-1``.
        When the parent (e.g. ``panda_hand``) Fabric ``worldMatrix`` has tiny float
        drift from the USD-composed parent world, the back-computed local becomes
        slightly off (observed ~3 cm position offset and a small rotation on
        ``wrist_cam``). The corrupted local persists because Fabric ignores
        subsequent USD notifications unless a ``sim.render()`` follows, and
        ``ManagerBasedEnv.reset`` does not guarantee a render after events.

    The fix authors the exact target local matrix in Fabric (matching USD
    ``xformOp:translate`` and ``xformOp:orient``) and then updates the Fabric
    world hierarchy, which is invariant to render scheduling.

    Usage:
        Register as a reset-mode ``EventTerm`` **after** all other reset events in
        a DataCollection env cfg (i.e. the env that actually attaches a wrist
        camera). The function is a no-op on envs without ``sensor_name``, so it is
        safe to register in shared base configs.
    """
    import isaaclab.sim as sim_utils
    from isaaclab.utils.math import convert_camera_frame_orientation_convention
    import usdrt
    from usdrt import Gf as GfRt

    if sensor_name not in env.scene.keys():
        return
    cam = env.scene[sensor_name]
    if not hasattr(cam, "cfg") or not hasattr(cam.cfg, "offset"):
        return

    if env_ids is None:
        ids: list[int] = list(range(env.num_envs))
    elif hasattr(env_ids, "tolist"):
        ids = env_ids.tolist()
    else:
        ids = list(env_ids)

    # Convert configured offset.rot (wxyz, in cfg.offset.convention) to opengl
    # convention -- this matches what ``Camera._initialize_impl`` spawned into USD
    # ``xformOp:orient``, so Fabric local will agree with USD local.
    rot_in = torch.tensor(cam.cfg.offset.rot, dtype=torch.float32).unsqueeze(0)
    rot_gl = convert_camera_frame_orientation_convention(
        rot_in, origin=cam.cfg.offset.convention, target="opengl"
    ).squeeze(0).cpu().numpy()
    qw, qx, qy, qz = (float(rot_gl[0]), float(rot_gl[1]), float(rot_gl[2]), float(rot_gl[3]))
    rot3 = _quat_wxyz_to_rot3(qw, qx, qy, qz)
    tx, ty, tz = (float(cam.cfg.offset.pos[0]), float(cam.cfg.offset.pos[1]), float(cam.cfg.offset.pos[2]))

    # USD ``GfMatrix4d`` / Fabric ``omni:fabric:localMatrix`` is row-major, with
    # rotation rows storing columns of the linear map and translation in row 3.
    m4 = np.eye(4, dtype=np.float64)
    m4[0, :3] = rot3[:, 0]
    m4[1, :3] = rot3[:, 1]
    m4[2, :3] = rot3[:, 2]
    m4[3, :3] = (tx, ty, tz)

    stage_id = sim_utils.get_current_stage_id()
    fabric_stage = usdrt.Usd.Stage.Attach(stage_id)
    prim_tpl: str = cam.cfg.prim_path

    # Isaac Lab expands ``{ENV_REGEX_NS}`` into a regex fragment ``env_.*`` at
    # cfg-parse time, so ``cam.cfg.prim_path`` at runtime contains ``env_.*`` (or
    # similar) rather than the raw placeholder. Resolve back to concrete per-env
    # paths via a regex substitution.
    _ENV_RE = re.compile(r"(/env)_(?:\.\*|\[[^\]]+\]|\{ENV_REGEX_NS\})")

    wrote_any = False
    for eid in ids:
        path = _ENV_RE.sub(lambda m, e=int(eid): f"{m.group(1)}_{e}", prim_tpl, count=1)
        rt_prim = fabric_stage.GetPrimAtPath(path)
        if not rt_prim.IsValid():
            continue
        attr = rt_prim.GetAttribute("omni:fabric:localMatrix")
        if not attr.IsValid():
            continue
        mat = GfRt.Matrix4d(
            m4[0, 0], m4[0, 1], m4[0, 2], m4[0, 3],
            m4[1, 0], m4[1, 1], m4[1, 2], m4[1, 3],
            m4[2, 0], m4[2, 1], m4[2, 2], m4[2, 3],
            m4[3, 0], m4[3, 1], m4[3, 2], m4[3, 3],
        )
        attr.Set(mat)
        wrote_any = True

    # Propagate the authored local into Fabric world matrices so any downstream
    # consumer that caches world (``get_world_poses``, renderer hydra) sees the
    # corrected pose. Harmless no-op if the view has not been initialised yet;
    # in that case the first read will compute world from our authored local.
    if wrote_any and hasattr(cam, "_view") and cam._view is not None:
        hierarchy = getattr(cam._view, "_fabric_hierarchy", None)
        if hierarchy is not None:
            hierarchy.update_world_xforms()
