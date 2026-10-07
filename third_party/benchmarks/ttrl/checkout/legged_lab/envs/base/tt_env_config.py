# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# Original code is licensed under BSD-3-Clause.
#
# Copyright (c) 2025-2026, The Legged Lab Project Developers.
# All rights reserved.
# Modifications are licensed under BSD-3-Clause.
#
# This file contains code derived from Isaac Lab Project (BSD-3-Clause license)
# with modifications by Legged Lab Project (BSD-3-Clause license).

import math
from dataclasses import MISSING

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (  # noqa:F401
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
    RslRlRndCfg,
    RslRlSymmetryCfg,
)

import legged_lab.mdp as mdp

from .tt_config import (
    PerceptionDelayCfg,
    ActionDelayCfg,
    BaseSceneCfg,
    CommandRangesCfg,
    CommandsCfg,
    DomainRandCfg,
    EventCfg,
    HeightScannerCfg,
    NoiseCfg,
    NoiseScalesCfg,
    NormalizationCfg,
    ObsScalesCfg,
    PhysxCfg,
    RewardCfg,
    CurriculumCfg,
    RobotCfg,
    BallCfg,
    TableCfg,
    SimCfg,
    ActionsCfg,
    ObservationsCfg,
)


@configclass
class TTEnvCfg:
    device: str = "cuda:0"
    scene: BaseSceneCfg = BaseSceneCfg(
        max_episode_length_s= 10, #9,#15.0,#10.0 ,#20.0 # not effective if < max_servers_per_episode* max_ball_eposide_length_s 
        num_envs=4096,
        env_spacing=5.0,
        robot=MISSING,
        table=MISSING,
        ball=MISSING,
        terrain_type=MISSING,
        terrain_generator=None,
        max_init_terrain_level=5,
        height_scanner=HeightScannerCfg(
            enable_height_scan=False,
            prim_body_name=MISSING,
            resolution=0.1,
            size=(1.6, 1.0),
            debug_vis=False,
            drift_range=(0.0, 0.0),  # (0.3, 0.3)
        ),
    )
    robot: RobotCfg = RobotCfg(
        actor_obs_history_length=5,#10
        critic_obs_history_length=5,
        action_scale=0.25,
        terminate_contacts_body_names=MISSING,
        feet_body_names=MISSING,
        effort_limit_scale=1.0,
    )
    
    ball: BallCfg = BallCfg(
        # ! The following setting aligns with Vicon Mocap data range
        # #fast low ball
        # ball_speed_x_range=(-7.1, -6.2),#(-5.5, -4.8) (-6.9, -6.5)
        # ball_speed_y_range= (-0.8, 0.4),#(-0.8, 0.8)
        # ball_speed_z_range=(0.9, 1.6), #(1.7, 1.9),
        #slow high ball
        ball_speed_x_range=(-6.5, -5.0),#(-5.5, -4.8) (-6.9, -6.5)4.8->6.0
        # ball_speed_x_range=(-6.5, -5.2),#
        # ball_speed_y_range= (-0.6, 0.2),#(-0.8, 0.8)
        ball_speed_y_range= (-0.8, 0.4),#(-0.8, 0.8)
        # ball_speed_z_range=(1.5, 1.9), #(1.7, 1.9),1.9 ->2.2
        ball_speed_z_range=(1.5, 2.0), #(1.7, 1.9),1.9 ->2.2
        # testing bouncing ball
        # ball_speed_x_range=(-0.0, -0.0),
        # ball_speed_y_range= (-0., 0.),
        # ball_speed_z_range=(-0.0, -0.0), 
        ball_pos_y_range=(-0.1, 0.1), #(-0.2, 0.2),
        # ball_speed_x_range=(-6.56, -6.56),
        # ball_speed_y_range= (-0.0, 0.0),
        # ball_speed_z_range=(1.11, 1.11), 
        # ball_pos_y_range=(-0.0, 0.0), 
        contact_threshold=0.05,  # radius of contact region from paddle center, subtracted by ball radius
        ball_max_eposide_length = 1.8,
        ball_reset_repeat = 1,
        num_new_serves = 2,
        max_serve_per_episode= 5
    )

    table: TableCfg = TableCfg(
        table_opponent_contact_x=(0.0, 1.35),  # opponent table(0.0, 1.37)
        table_opponent_contact_y=(-0.7625, 0.7625),
        table_opponent_contact_z=(0.76, 0.83),
        table_own_contact_x=(-1.37, -0.0),  # own table
        table_own_contact_y=(-0.7625, 0.7625),
        table_own_contact_z=(0.76, 0.83),
    )

    reward = RewardCfg()
    curriculum = CurriculumCfg()
    normalization: NormalizationCfg = NormalizationCfg(
        obs_scales=ObsScalesCfg(
            lin_vel=1.0,
            ang_vel=1.0,
            projected_gravity=1.0,
            commands=1.0,
            joint_pos=1.0,
            joint_vel=1.0,
            actions=1.0,
            height_scan=1.0,
            # ball_pos=1.0,
            perception=1.0, #Mocap pos scale
            # ball_linvel=1.0, #Mocap vel scale
            # robot_pos=1.0,
        ),
        clip_observations=100.0,
        clip_actions=100.0,
        height_scan_offset=0.5,
    )
    commands: CommandsCfg = CommandsCfg(
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.2,
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=False,#True,
        ranges=CommandRangesCfg(
            lin_vel_x=(0.0, 0.0), lin_vel_y=(-0.5, 0.5), ang_vel_z=(-1.57, 1.57), heading=(-math.pi, math.pi)
        ),
    )
    noise: NoiseCfg = NoiseCfg(
        add_noise=True,
        # add_noise=False,
        noise_scales=NoiseScalesCfg(
            ang_vel=0.2,
            projected_gravity=0.05,
            joint_pos=0.01,
            joint_vel=1.5,
            height_scan=0.1,
            # ball_pos=0.1, 
            # ball_linvel=0.2, #Mocap velocity scale
            perception=0.007, #Mocap position scale #0.007
        ),
    )
    domain_rand: DomainRandCfg = DomainRandCfg(
        events=EventCfg(
            physics_material=EventTerm(
                func=mdp.randomize_rigid_body_material,
                mode="startup",
                params={
                    "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
                    "static_friction_range": (0.8, 1.0),
                    "dynamic_friction_range": (0.5, 0.8),
                    "restitution_range": (0.0, 0.005), #(0.0, 0.005)
                    "num_buckets": 64,
                },
            ),
            add_base_mass=EventTerm(
                func=mdp.randomize_rigid_body_mass,
                mode="startup",
                params={
                    "asset_cfg": SceneEntityCfg("robot", body_names=MISSING),
                    "mass_distribution_params": (-5.0, 5.0),
                    "operation": "add",
                },
            ),
            reset_base=EventTerm(
                func=mdp.reset_root_state_uniform,
                mode="reset",
                params={
                    # "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-0.8, 0.8)}, #set to match TT playing range
                    "pose_range": {"x": (-0.6, 0.02), "y": (-0.6, 0.8), "yaw": (-0.3, 0.3)}, #training setup # used to be -0.8,0.1, -0.5,0.8
                    # "pose_range": {"x": (-0.41, -0.4), "y": (0.4, 0.4), "yaw": (-0.1, -0.1)}, #testing setup
                    "velocity_range": {
                        "x": (-0.2, 0.2),
                        "y": (-0.2, 0.2),
                        "z": (-0.2, 0.2),
                        "roll": (-0.2, 0.2),
                        "pitch": (-0.2, 0.2),
                        "yaw": (-0.5, 0.5),
                    },
                    # #testing setup
                    # "velocity_range": {
                    #     "x": (-0.02, 0.02),
                    #     "y": (-0.02, 0.02),
                    #     "z": (-0.02, 0.02),
                    #     "roll": (-0.02, 0.02),
                    #     "pitch": (-0.02, 0.02),
                    #     "yaw": (-0.05, 0.05),
                    # },
                },
            ),
            reset_locomotion_joints=EventTerm(
                func=mdp.reset_joints_by_scale_debug,
                mode="reset",
                params={
                    "position_range": (0.5, 1.5),
                    "velocity_range": (0.0, 0.0),
                    "asset_cfg": SceneEntityCfg("robot", joint_names=["Waist",".*_Hip_.*", ".*_Knee_.*",".*_Ankle_.*","Left_Elbow_.*","Left_Shoulder_.*","AAHead_yaw","Head_pitch"]),
                    # "asset_cfg": SceneEntityCfg("robot", joint_ids=[0, 1, 3, 4, 5, 7, 8, 9, 11, 12, 13, 15, 16, 17, 18, 19, 20, 21, 22]),
                },
            ),
            reset_manipulation_joints=EventTerm(
                func=mdp.reset_joints_by_offset_debug,
                mode="reset",
                params={
                    "position_range": (-0.5, 0.5),
                    "velocity_range": (0.0, 0.0),
                    "asset_cfg": SceneEntityCfg("robot", joint_names=["Right_Elbow_.*","Right_Shoulder_.*"]),
                    # "asset_cfg": SceneEntityCfg("robot", joint_ids=[2, 6, 10, 14]),
                },
            ),
            push_robot=EventTerm(
                func=mdp.push_by_setting_velocity,
                mode="interval",
                interval_range_s=(5.0, 15.0),
                # params={"velocity_range": {"x": (-1.0, 1.0), "y": (-1.0, 1.0)}},
                params={"velocity_range": {"x": (-0.2, 0.2), "y": (-0.2, 0.2)}},
            ),
        ),
        action_delay=ActionDelayCfg(enable=False, params={"max_delay": 1, "min_delay": 1}),
        perception_delay=PerceptionDelayCfg(enable=True, params={"max_delay": 5, "min_delay": 2}), #NOTE: Adjust based on expected delay, expected 4-10 ms dt 2 ms
    )
    # sim: SimCfg = SimCfg(dt=0.005, decimation=4, physx=PhysxCfg(gpu_max_rigid_patch_count=10 * 2**15))
    sim: SimCfg = SimCfg(dt=0.002, decimation=10, physx=PhysxCfg(gpu_max_rigid_patch_count=10 * 2**15))

    actions: ActionsCfg = ActionsCfg(
        joint_names=MISSING, 
        preserve_order=True,
    )

    observations: ObservationsCfg = ObservationsCfg(
        joint_names=MISSING,
        preserve_order=True,
    )

    def __post_init__(self):
        pass


@configclass
class TTAgentCfg(RslRlOnPolicyRunnerCfg):
    seed = 42
    device = "cuda:0"
    num_steps_per_env = 24#24 #1second->50 steps
    max_iterations = 50000
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        class_name="ActorCritic",
        init_noise_std=1.0,
        noise_std_type="scalar",
        actor_hidden_dims=[512, 512, 128],
        critic_hidden_dims=[512, 512, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        class_name="PPO",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.006,# 0.005, alt 0.01
        num_learning_epochs= 5,#5,
        num_mini_batches=4, #4,
        learning_rate=1.0e-3,#1.0e-3,
        schedule="adaptive",
        gamma=0.95, #0.99
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        normalize_advantage_per_mini_batch=False,
        symmetry_cfg=None,  # RslRlSymmetryCfg()
        rnd_cfg=None,  # RslRlRndCfg()
    )
    clip_actions = None
    save_interval = 100
    experiment_name = ""
    run_name = ""
    logger = "wandb"
    neptune_project = "leggedlab"
    wandb_project = "leggedlab"
    resume = False
    load_run = ".*"
    load_checkpoint = "model_.*.pt"

    def __post_init__(self):
        pass
