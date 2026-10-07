import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import ContactSensorCfg, FrameTransformerCfg
from isaaclab.utils import configclass
from steadytray.tasks import mdp
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from .steady_tray_env_cfg import TraySceneCfg, TrayEventCfg, TrayRewardsCfg, TrayTerminationsCfg, TrayTerminationsPlayCfg, SteadyTrayEnvCfg
from .locomotion_env_cfg import ObservationsCfg, CurriculumCfg

@configclass
class ObjectSceneCfg(TraySceneCfg):
    """Configuration for the steady tray with object scene."""

    object: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        spawn=sim_utils.CylinderCfg(
            radius=0.03,
            height=0.10,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.002),
            activate_contact_sensors=True,
        ),
        init_state=RigidObjectCfg.InitialStateCfg(),
    )

    object_contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Object", 
        track_air_time=True,
        history_length=10,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Tray"],
    )

    object_tray_transform: FrameTransformerCfg = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Tray",  # Reference frame
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Object",
                name="object",
            ),
        ],
    )

    robot_transform: FrameTransformerCfg = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/torso_link",  # Reference frame
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Tray",
                name="tray",
            ),
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Object",
                name="object",
            ),
        ],
    )


@configclass
class ObjectEventCfg(TrayEventCfg):
    """Configuration for events in the steady tray with object environment."""
    
    random_object_scale = EventTerm(
        func=mdp.randomize_cylinder_scale,
        mode="prestartup",
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "radius_scale_range": (0.7, 1.5),
            "height_scale_range": (0.75, 2.0),
        },
    )

    random_object_friction = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "static_friction_range": (0.5, 1.0),
            "dynamic_friction_range": (0.4, 0.9),
            "restitution_range": (0.0, 0.5),
            "num_buckets": 256,
        },
    )
    random_object_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "mass_distribution_params": (0.05, 1.0),
            "operation": "abs",
        },
    )
    random_object_com = EventTerm(
        func=mdp.randomize_rigid_body_com_fixed,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "com_range": {"x": (-0.01, 0.01), "y": (-0.01, 0.01), "z": (-0.02, 0.02)},
        },
    )
    reset_object_pos = EventTerm(
        func=mdp.set_rigid_object_relative_to_robot,
        mode="reset",
        params={
            "base_asset_cfg": SceneEntityCfg("tray"),
            "target_asset_cfg": SceneEntityCfg("object"),
            "relative_pose": {
                "x": (-0.05, 0.05),
                "y": (-0.08, 0.08),
                "z": 0.0,
                "roll": 0.0,
                "pitch": 0.0,
                "yaw": (-3.14, 3.14),
            },
            "relative_velocity": {
                "x": 0.0,
                "y": 0.0,
                "z": 0.0,
                "roll": 0.0,
                "pitch": 0.0,
                "yaw": 0.0,
            },
            "z_offset": 0.02,  # Optional: gap between object bottom and tray top
        },
    )

    push_object = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(2.0, 4.0),
        params={
            "velocity_range": {
                "x": (-0.3, 0.3), 
                "y": (-0.3, 0.3), 
                "roll": (-0.3, 0.3), 
                "pitch": (-0.3, 0.3)
            },
            'asset_cfg': SceneEntityCfg("object")
        },
    )

@configclass
class ObjectObservationsCfg(ObservationsCfg):
    """Configuration for observations in the steady tray with object environment."""

    @configclass
    class EncoderCfg(ObsGroup):
        """Configuration for encoder observations."""

        # Locomotion robot observations
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2, noise=Unoise(n_min=-0.2, n_max=0.2), clip=(-25.0, 25.0))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05, noise=Unoise(n_min=-1.5, n_max=1.5), clip=(-100.0, 100.0))
        last_action = ObsTerm(func=mdp.last_action)

        # robot base velocity
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1), clip=(-3.0, 3.0))

        # Tray observations
        tray_projected_gravity = ObsTerm(func=mdp.projected_gravity, params={"asset_cfg": SceneEntityCfg("tray")}, noise=Unoise(n_min=-0.05, n_max=0.05))
        tray_pos_rel = ObsTerm(func=mdp.object_rel_pos, params={"sensor_cfg": SceneEntityCfg("robot_transform"), "target_frame_name": "tray"}, noise=Unoise(n_min=-0.03, n_max=0.03), clip=(-1.0, 1.0))

        # Object observations (with stochastic delays for sim-to-real)
        # This ensures all object-related measurements share the same delay buffer, preventing temporal misalignment.
        combined_object_obs = ObsTerm(
            func=mdp.CombinedObjectObservationsDict,
            params={
                "include_pos": True,
                "include_ang_vel": True,
                "include_lin_vel": True,
                "include_gravity": True,
                # Individual noise
                "pos_noise": Unoise(n_min=-0.03, n_max=0.03),
                "ang_vel_noise": Unoise(n_min=-0.2, n_max=0.2),
                "lin_vel_noise": Unoise(n_min=-0.1, n_max=0.1),
                "gravity_noise": Unoise(n_min=-0.05, n_max=0.05),
                # Individual clipping
                "pos_clip": (-1.0, 1.0),
                "ang_vel_clip": (-50.0, 50.0),
                "lin_vel_clip": (-10.0, 10.0),
                "gravity_clip": None,
                # Individual scaling
                "pos_scale": 1.0,
                "ang_vel_scale": 0.2,
                "lin_vel_scale": 0.5,
                "gravity_scale": 1.0,
            },
            delay_min_lag=0,        # Minimum delay in steps (0 = no delay)
            delay_max_lag=3,        # Maximum delay in steps (3 steps x 0.02s = 0.06s)
            delay_per_env=True,     # Each environment has independent delay
            delay_hold_prob=0.2,    # 20% chance to keep previous lag
            delay_update_period=10, # Update lag every 10 steps (0.2s)
            delay_per_env_phase=True, # Stagger lag updates across environments
        )
        
        def __post_init__(self):
            self.history_length = 32
            self.enable_corruption = True
            self.flatten_history_dim = False

    encoder: EncoderCfg = EncoderCfg()

    @configclass
    class AdaptedCriticCfg(ObsGroup):
        """Configuration for adapted critic observation group."""

        # Locomotion robot observations
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2, clip=(-25.0, 25.0))  
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05, clip=(-100.0, 100.0))
        last_action = ObsTerm(func=mdp.last_action)

        # robot base velocity
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, clip=(-3.0, 3.0))

        # Tray observations
        tray_projected_gravity = ObsTerm(func=mdp.projected_gravity, params={"asset_cfg": SceneEntityCfg("tray")})
        tray_pos_rel = ObsTerm(func=mdp.object_rel_pos, params={"sensor_cfg": SceneEntityCfg("robot_transform"), "target_frame_name": "tray"}, clip=(-1.0, 1.0))
        tray_ang_vel_rel = ObsTerm(func=mdp.object_rel_ang_vel, params={"target_asset_cfg": SceneEntityCfg("tray"), "reference_asset_cfg": SceneEntityCfg("robot", body_names="torso_link")}, scale=0.2, clip=(-50.0, 50.0))
        tray_lin_vel_rel = ObsTerm(func=mdp.object_rel_lin_vel, params={"target_asset_cfg": SceneEntityCfg("tray"), "reference_asset_cfg": SceneEntityCfg("robot", body_names="torso_link")}, scale=0.5, clip=(-10.0, 10.0))
        tray_holder_contact_forces = ObsTerm(func=mdp.tray_holder_contact_forces, params={"sensor_cfg": SceneEntityCfg("tray_contact_sensor")}, scale=0.1, clip=(-50.0, 50.0))

        # Object observations
        object_pos_rel = ObsTerm(func=mdp.object_rel_pos, params={"sensor_cfg": SceneEntityCfg("object_tray_transform"), "target_frame_name": "object"}, clip=(-1.0, 1.0))
        object_ang_vel_rel = ObsTerm(func=mdp.object_rel_ang_vel, params={"target_asset_cfg": SceneEntityCfg("object"), "reference_asset_cfg": SceneEntityCfg("robot", body_names="torso_link")}, scale=0.2, clip=(-50.0, 50.0))
        object_lin_vel_rel = ObsTerm(func=mdp.object_rel_lin_vel, params={"target_asset_cfg": SceneEntityCfg("object"), "reference_asset_cfg": SceneEntityCfg("robot", body_names="torso_link")}, clip=(-10.0, 10.0))
        object_projected_gravity = ObsTerm(func=mdp.projected_gravity, params={"asset_cfg": SceneEntityCfg("object")})

        def __post_init__(self):
            self.history_length = 5

    critic: AdaptedCriticCfg = AdaptedCriticCfg()


@configclass
class ObjectRewardsCfg(TrayRewardsCfg):
    """Configuration for rewards in the steady tray with object environment."""

    # Disabled rewards from tray environment
    tray_lin_vel = None
    tray_ang_vel = None

    # Reward to keep the object on the tray and upright
    object_contact = RewTerm(
        func=mdp.desired_contacts_count,
        weight=0.05,
        params={
            "sensor_cfg": SceneEntityCfg("object_contact_sensor"),
            "threshold": 0.1,
        },
    )   # MAX = 0.02 x 10 = 0.2
    upright_object = RewTerm(
        func=mdp.object_upright_bonus_exp,
        weight=1.0,
        params={"object_cfg": SceneEntityCfg("object"),
                "lambda_exp": 4.0,
        },
    )   # MAX = 1.0 x 2 = 2.0


@configclass
class ObjectTerminationsCfg(TrayTerminationsCfg):
    """Configuration for termination conditions in the steady tray with object environment."""

    object_fallen = DoneTerm(
        func=mdp.link_height_below_minimum,
        params={
            "minimum_height": 0.7,  # Terminate if object drops below 0.7m (more strict)
            "asset_cfg": SceneEntityCfg("object"),
        },
        track_only=True,
        track_only_delay=1.0,
    )
    object_bad_orientation = DoneTerm(
        func=mdp.bad_orientation,
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "limit_angle": 0.7,
        },
        track_only=True,
        track_only_delay=1.0,
    )


@configclass
class ObjectCommandCfg:
    """Configuration for the velocity command in the steady tray with object environment."""

    base_velocity = mdp.DelayedUniformVelocityCommandCfg(
        asset_name="robot",
        body_name="torso_link",
        delay_time=1.0,  # Zero velocity for first second
        heading_command=True,
        rel_heading_envs=0.5,
        rel_standing_envs=0.02,
        resampling_time_range=(10.0, 10.0),
        ranges=mdp.DelayedUniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.5, 1.0), lin_vel_y=(-0.5, 0.5), ang_vel_z=(-0.5, 0.5), heading=(-3.14, 3.14)
        ),
        limit_ranges=mdp.DelayedUniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.5, 1.0), lin_vel_y=(-0.5, 0.5), ang_vel_z=(-0.5, 0.5), heading=(-3.14, 3.14)
        ),
        debug_vis=True,
        class_type=mdp.DelayedUniformVelocityCommand,
    )

@configclass
class ObjectCurriculumCfg(CurriculumCfg):
    """Configuration for curriculum in the steady tray with object environment."""

    lin_vel_cmd_levels = None
    ang_vel_cmd_levels = None


@configclass
class SteadyObjectEnvCfg(SteadyTrayEnvCfg):
    """Configuration for the steady tray with object environment."""

    scene: ObjectSceneCfg = ObjectSceneCfg(num_envs=4096, env_spacing=2.5, replicate_physics=False)
    events: ObjectEventCfg = ObjectEventCfg()
    observations: ObjectObservationsCfg = ObjectObservationsCfg()
    rewards: ObjectRewardsCfg = ObjectRewardsCfg()
    terminations: ObjectTerminationsCfg = ObjectTerminationsCfg()
    commands: ObjectCommandCfg = ObjectCommandCfg()
    curriculum: ObjectCurriculumCfg = ObjectCurriculumCfg()


@configclass
class ObjectTerminationsPlayCfg(TrayTerminationsPlayCfg):
    """Configuration for termination conditions in the steady tray with object environment."""

    object_fallen = DoneTerm(
        func=mdp.link_height_below_minimum,
        params={
            "minimum_height": 0.7,  # Terminate if object drops below 0.7m (more strict)
            "asset_cfg": SceneEntityCfg("object"),
        },
        track_only=True,
        track_only_delay=0.0,
    )
    object_bad_orientation = DoneTerm(
        func=mdp.bad_orientation,
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "limit_angle": 0.7,
        },
        track_only=True,
        track_only_delay=0.0,
    )

@configclass
class SteadyObjectPlayEnvCfg(SteadyObjectEnvCfg):

    terminations: ObjectTerminationsPlayCfg = ObjectTerminationsPlayCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 5

        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges