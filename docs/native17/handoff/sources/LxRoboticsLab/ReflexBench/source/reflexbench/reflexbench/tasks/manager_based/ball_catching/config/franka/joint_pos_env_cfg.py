# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim.schemas.schemas_cfg import CollisionPropertiesCfg, RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.sim.utils.prims import clone
from isaaclab.utils import configclass

from isaaclab.markers.config import FRAME_MARKER_CFG  # noqa: E402
from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG  # noqa: E402

from ... import mdp
from ...ball_catching_env_cfg import BallCatchingEnvCfg


@clone
def spawn_mug(
    prim_path: str,
    cfg: UsdFileCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
):
    """Spawn mug USD with rigid body and collision APIs explicitly applied."""
    from pxr import Usd, UsdPhysics, PhysxSchema
    import omni.usd

    prim = sim_utils.spawn_from_usd(prim_path, cfg, translation, orientation)

    stage = omni.usd.get_context().get_stage()
    root_prim = stage.GetPrimAtPath(prim_path)

    # Apply RigidBodyAPI on the root xform
    if not root_prim.HasAPI(UsdPhysics.RigidBodyAPI):
        UsdPhysics.RigidBodyAPI.Apply(root_prim)
    if cfg.rigid_props is not None:
        rb_api = UsdPhysics.RigidBodyAPI(root_prim)
        rb_api.CreateRigidBodyEnabledAttr(True)
        if cfg.rigid_props.disable_gravity:
            PhysxSchema.PhysxRigidBodyAPI.Apply(root_prim)
            PhysxSchema.PhysxRigidBodyAPI(root_prim).CreateDisableGravityAttr(True)

    # Apply MassAPI on root
    if not root_prim.HasAPI(UsdPhysics.MassAPI):
        UsdPhysics.MassAPI.Apply(root_prim)
    if cfg.mass_props is not None:
        UsdPhysics.MassAPI(root_prim).CreateMassAttr(cfg.mass_props.mass)

    # Apply per-mesh collision as convex decomposition so the mug opening
    # remains traversable instead of being sealed by one large convex hull.
    for child in Usd.PrimRange(root_prim):
        if child.GetTypeName() == "Mesh":
            if not child.HasAPI(UsdPhysics.CollisionAPI):
                UsdPhysics.CollisionAPI.Apply(child)
            UsdPhysics.CollisionAPI(child).CreateCollisionEnabledAttr(True)
            if not child.HasAPI(UsdPhysics.MeshCollisionAPI):
                UsdPhysics.MeshCollisionAPI.Apply(child)
            mesh_collision_api = UsdPhysics.MeshCollisionAPI(child)
            mesh_collision_api.GetApproximationAttr().Set(UsdPhysics.Tokens.convexDecomposition)
            if not child.HasAPI(PhysxSchema.PhysxConvexDecompositionCollisionAPI):
                PhysxSchema.PhysxConvexDecompositionCollisionAPI.Apply(child)
    return prim


@configclass
class FrankaBallCatchingEnvCfg(BallCatchingEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()

        self.scene.robot = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.spawn.activate_contact_sensors = True

        self.actions.arm_action = mdp.RelativeJointPositionActionCfg(
            asset_name="robot",
            joint_names=["panda_joint.*"],
            scale=0.1,
            use_zero_offset=True,
        )
        self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["panda_finger.*"],
            # Slightly closed so the gripper appears to hold the catcher.
            open_command_expr={"panda_finger_.*": 0.015},
            close_command_expr={"panda_finger_.*": 0.0},
        )

        self.scene.ball = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Ball",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[1.5, 0.0, 0.45], rot=[1, 0, 0, 0]),
            spawn=sim_utils.SphereCfg(
                radius=0.025,
                rigid_props=RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=1,
                    max_angular_velocity=1000.0,
                    max_linear_velocity=10.0,
                    max_depenetration_velocity=5.0,
                    disable_gravity=False,
                ),
                mass_props=sim_utils.MassPropertiesCfg(mass=0.04),
                collision_props=CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.3, 0.0)),
            ),
        )

        # -- bucket (visual container; gravity-free placeholder, not yet wrist-attached) --
        # Plan: inner diameter 0.12-0.16 m, depth 0.12 m, wall >= 4 mm
        # Solid cylinder approximation; catch logic uses virtual zone (mdp/events.py)
        self.scene.bucket = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Bucket",
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=[0.55, 0.0, 0.28],
                rot=[1, 0, 0, 0],
            ),
            spawn=UsdFileCfg(
                usd_path="https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/Isaac/Props/Mugs/SM_Mug_A2.usd",
                scale=(0.015, 0.015, 0.015),
                func=spawn_mug,
                rigid_props=RigidBodyPropertiesCfg(
                    disable_gravity=True,
                ),
                mass_props=sim_utils.MassPropertiesCfg(mass=0.5),
                collision_props=CollisionPropertiesCfg(),
            ),
        )

        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/panda_link0",
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/panda_hand",
                    name="end_effector",
                    offset=OffsetCfg(pos=[0.0, 0.0, 0.1034]),
                ),
            ],
        )


@configclass
class FrankaBallCatchingEnvCfg_PLAY(FrankaBallCatchingEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
