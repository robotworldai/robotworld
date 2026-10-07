"""Isaac Lab asset configuration for the drone-rope-beam system."""

from omni.isaac.lab.assets import ArticulationCfg
from omni.isaac.lab.actuators import ImplicitActuatorCfg
import omni.isaac.lab.sim as sim_utils

try:
    from ...utils.paths import resource_path
except ImportError:  # Allows importing this module as top-level ``environments``.
    from utils.paths import resource_path


DRONE_ROPE_PLANK_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(resource_path("asserts", "drone_rope_plank_horizontal_slide_block.usd")),
        variants=None,
        visible=True,
        semantic_tags=None,
        copy_from_source=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            enable_gyroscopic_forces=True,
            kinematic_enabled=None,
            disable_gravity=False,
            rigid_body_enabled=True,
            solver_position_iteration_count=255,
            solver_velocity_iteration_count=255, # 64
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            articulation_enabled=True,
            enabled_self_collisions=True,
            solver_position_iteration_count=255,
            solver_velocity_iteration_count=255, # 64
            fix_root_link=True,
        ),
    ),
    collision_group=0,
    debug_vis=False,
    init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 2.0)),
    actuators={
        "drone_rotors": ImplicitActuatorCfg(
            joint_names_expr=["rotor_.*_joint"],
            stiffness=0.0,
            damping=0.0,
        ),
        "passive_joints": ImplicitActuatorCfg(
            joint_names_expr=["joint_.*"],
            stiffness=0.0,
            damping=0.00001,
        ),
        "fixed_endpoint_slider": ImplicitActuatorCfg(
            joint_names_expr=["slider_holder_joint"],
            stiffness=0.0,
            damping=1000.0,
        ),
        "beam_hinge": ImplicitActuatorCfg(
            joint_names_expr=["holder_plank_joint"],
            stiffness=0.0,
            damping=0.0,
        ),
    },
)
