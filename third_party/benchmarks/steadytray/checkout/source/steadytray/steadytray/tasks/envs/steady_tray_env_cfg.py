import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import ContactSensorCfg, FrameTransformerCfg
from isaaclab.utils import configclass
from steadytray.tasks import mdp

from .locomotion_env_cfg import RobotEnvCfg, RobotSceneCfg, EventCfg, RewardsCfg, TerminationsCfg

# Initial positions for the tray and tray holders relative to the robot's pelvis
TRAY_INITIAL_POS = [0.30225, 0.0, 0.141]
LEFT_TRAY_HOLDER_POS = [0.24127, 0.14865, 0.09523]
RIGHT_TRAY_HOLDER_POS = [0.24127, -0.14865, 0.09523]
TORSO_IN_PELVIS = [0.0, 0.0, 0.044]


@configclass
class TraySceneCfg(RobotSceneCfg):
    """Configuration for the tray scene."""

    tray: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Tray",
        spawn=sim_utils.CuboidCfg(
            # size=(0.254, 0.352, 0.015),
            size=(0.254, 0.352, 0.018),
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.002),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0), metallic=0.2),
            physics_material=sim_utils.RigidBodyMaterialCfg(compliant_contact_stiffness=8e4, compliant_contact_damping=400),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(),
    )
    tray_contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Tray", 
        track_air_time=True,
        history_length=10,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Robot/left_tray_holder_link", "{ENV_REGEX_NS}/Robot/right_tray_holder_link"],
    )


@configclass
class TrayEventCfg(EventCfg):
    """Configuration for the tray events."""

    random_tray_friction = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("tray"),
            "static_friction_range": (1.2, 2.0),
            "dynamic_friction_range": (1.0, 1.8),
            "restitution_range": (0.0, 0.05),
            "num_buckets": 256,
        },
    )

    random_tray_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("tray"),
            "mass_distribution_params": (0.3, 0.7),
            "operation": "abs",
        },
    )

    reset_tray_pos = EventTerm(
        func=mdp.set_rigid_object_relative_to_robot,
        mode="reset",
        params={
            "base_asset_cfg": SceneEntityCfg("robot", body_names="pelvis"),
            "target_asset_cfg": SceneEntityCfg("tray"),
            "relative_pose": {
                "x": TRAY_INITIAL_POS[0],
                "y": TRAY_INITIAL_POS[1],
                "z": TRAY_INITIAL_POS[2],
                "roll": 0.0,
                "pitch": 0.0,
                "yaw": 0.0,
            },
            "relative_velocity": {
                "x": 0.0,
                "y": 0.0,
                "z": 0.0,
                "roll": 0.0,
                "pitch": 0.0,
                "yaw": 0.0,
            },
        },
    )


@configclass
class TrayRewardsCfg(RewardsCfg):
    """Configuration for the tray rewards."""

    # Release locomotion penalty to allow more natural movement
    joint_deviation_arms = RewTerm(
        func=mdp.joint_deviation_exp,
        weight=0.5,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    ".*_shoulder_.*_joint",
                    ".*_elbow_joint",
                    ".*_wrist_.*",
                ],
            ),
            "lambda_exp": 0.3,
        },    
    )     # MAX = 0.2

    # Reward for keeping the tray stable
    tray_flat_orientation = RewTerm(
        func=mdp.object_upright_bonus_exp,
        weight=0.25,
        params={"object_cfg": SceneEntityCfg('tray'), "lambda_exp": 4.0},
    )   # MAX = 0.25 x 2 = 0.5
    tray_lin_vel = RewTerm(
        func=mdp.object_lin_vel_z_exp, 
        weight=0.2,
        params={"object_cfg": SceneEntityCfg("tray"), "lambda_exp": 2.0}
    )     # MAX = 0.2
    tray_ang_vel = RewTerm(
        func=mdp.object_ang_vel_xy_exp, 
        weight=0.2,
        params={"object_cfg": SceneEntityCfg("tray"), "lambda_exp": 1.0}
    )     # MAX = 0.2

    # Reward for maintaining tray contact with holders
    tray_contact = RewTerm(
        func=mdp.desired_contacts_count,
        weight=0.01,
        params={
            "sensor_cfg": SceneEntityCfg("tray_contact_sensor"),
            "threshold": 0.1,
        },
    )   # MAX = 0.01 x 2 x 10 = 0.2
    tray_contact_force = RewTerm(
        func=mdp.contact_force_exp,
        weight=0.2,
        params={
            "sensor_cfg": SceneEntityCfg("tray_contact_sensor"),
            "lambda_exp": 0.005,
        },
    )   # MAX = 0.2

    left_relative_quat_deviation = RewTerm(
        func=mdp.entity_quat_exp,
        weight=0.5,
        params={
            "entity1_cfg": SceneEntityCfg('tray'),
            "entity2_cfg": SceneEntityCfg('robot', body_names='left_tray_holder_link'),
            "lambda_exp": 2.0,
        },
    )
    right_relative_quat_deviation = RewTerm(
        func=mdp.entity_quat_exp,
        weight=0.5,
        params={
            "entity1_cfg": SceneEntityCfg('tray'),
            "entity2_cfg": SceneEntityCfg('robot', body_names='right_tray_holder_link'),
            "lambda_exp": 2.0,
        },
    )

    # penalty for using too much torque
    torque_penalty = RewTerm(
        func=mdp.joint_torques_l2,
        weight=-2e-5,
        params={"asset_cfg": SceneEntityCfg("robot")}
    )


@configclass
class TrayTerminationsCfg(TerminationsCfg):
    """Configuration for the tray termination conditions."""

    tray_fallen = DoneTerm(
        func=mdp.link_height_below_minimum,
        params={
            "minimum_height": 0.7,  # Terminate if tray drops below 0.7m (more strict)
            "asset_cfg": SceneEntityCfg("tray"),
        },
        track_only=True,
        track_only_delay=1.0,
    )


@configclass
class SteadyTrayEnvCfg(RobotEnvCfg):
    """Configuration for the steady tray environment."""

    scene: TraySceneCfg = TraySceneCfg(num_envs=4096, env_spacing=2.5, replicate_physics=True)
    events: TrayEventCfg = TrayEventCfg()
    rewards: TrayRewardsCfg = TrayRewardsCfg()
    terminations: TrayTerminationsCfg = TrayTerminationsCfg()


@configclass
class TrayTerminationsPlayCfg(TerminationsCfg):
    """Configuration for the tray termination conditions."""
    base_height = DoneTerm(
        func=mdp.root_height_below_minimum, 
        params={"minimum_height": 0.4},
        track_only=True,
        track_only_delay=0.0,
    )
    bad_orientation = DoneTerm(
        func=mdp.bad_orientation, 
        params={"limit_angle": 0.7, "asset_cfg": SceneEntityCfg("robot", body_names="torso_link")},
        track_only=True,
        track_only_delay=0.0,
    )
    tray_fallen = DoneTerm(
        func=mdp.link_height_below_minimum,
        params={
            "minimum_height": 0.7,  # Terminate if tray drops below 0.7m (more strict)
            "asset_cfg": SceneEntityCfg("tray"),
        },
        track_only=True,
        track_only_delay=0.0,
    )

@configclass
class SteadyTrayPlayEnvCfg(SteadyTrayEnvCfg):

    terminations: TrayTerminationsPlayCfg = TrayTerminationsPlayCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 5

        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges
