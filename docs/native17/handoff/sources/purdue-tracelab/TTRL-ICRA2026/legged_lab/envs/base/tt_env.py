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

import isaaclab.sim as sim_utils
import isaacsim.core.utils.torch as torch_utils  # type: ignore
import isaaclab.utils.math as math_utils
import numpy as np
import torch
from isaaclab.assets.articulation import Articulation
from isaaclab.assets.rigid_object import RigidObject
from isaaclab.assets.rigid_object import RigidObjectCfg
from isaaclab.envs.mdp.commands import UniformVelocityCommand, UniformVelocityCommandCfg
from isaaclab.managers import EventManager, RewardManager, CurriculumManager
from isaaclab.managers.scene_entity_cfg import SceneEntityCfg
from isaaclab.scene import InteractiveScene
from isaaclab.sensors import ContactSensor, RayCaster
from isaaclab.sim import PhysxCfg, SimulationContext
from isaaclab.utils.buffers import CircularBuffer, DelayBuffer
from isaaclab.utils import configclass
from rsl_rl.env import VecEnv

from legged_lab.envs.base.tt_env_config import TTEnvCfg
from legged_lab.envs.base.tt_config import BaseSceneCfg
from legged_lab.utils.env_utils.scene import SceneCfg

# ! Aerodynamics : BEGIN
from legged_lab.physics.aerodynamics import AeroForceField
# ! Aerodynamics : END

#TODO: move this functio to tt_config.py
@configclass
class TTSceneCfg(SceneCfg):
    def __init__(self, config: BaseSceneCfg, physics_dt, step_dt):
        super().__init__(config, physics_dt, step_dt)
        self.table: RigidObjectCfg = config.table
        self.table.prim_path = "{ENV_REGEX_NS}/Table"

        self.ball: RigidObjectCfg = config.ball
        self.ball.prim_path = "{ENV_REGEX_NS}/Ball"

        # visualization object for predicted ball pose
        self.ball_future: RigidObjectCfg = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/BallFuture",
            spawn=sim_utils.SphereCfg(
                radius=0.02,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    kinematic_enabled=True,
                    disable_gravity=True,
                ),
                collision_props=sim_utils.CollisionPropertiesCfg(
                    collision_enabled=False,
                ),
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.0, 1.0, 0.0),
                    metallic=0.0,
                    roughness=0.5,
                ),
            ),
        )
        # visualization object for learned model ball prediction (different color)
        self.ball_pred: RigidObjectCfg = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/BallPred",
            spawn=sim_utils.SphereCfg(
                radius=0.02,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    kinematic_enabled=True,
                    disable_gravity=True,
                ),
                collision_props=sim_utils.CollisionPropertiesCfg(
                    collision_enabled=False,
                ),
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(1.0, 0.7, 0.0),  # orange/yellow to distinguish
                    metallic=0.0,
                    roughness=0.5,
                ),
            ),
        )
        # self.robot_future: RigidObjectCfg = RigidObjectCfg(
        #     prim_path="{ENV_REGEX_NS}/RobotFuturePos",
        #     spawn=sim_utils.SphereCfg(
        #         radius=0.03,  # slightly larger than BallFuture (0.02)
        #         rigid_props=sim_utils.RigidBodyPropertiesCfg(
        #             kinematic_enabled=True,
        #             disable_gravity=True,
        #         ),
        #         collision_props=sim_utils.CollisionPropertiesCfg(
        #             collision_enabled=False,
        #         ),
        #         visual_material=sim_utils.PreviewSurfaceCfg(
        #             diffuse_color=(0.1, 0.4, 1.0),  # different color, e.g., blue-ish
        #             metallic=0.0,
        #             roughness=0.5,
        #         ),
        #     ),
        # )

        # visualization object for predicted robot velocity
        # ArrowCfg = getattr(sim_utils, "ArrowCfg", None)
        # if ArrowCfg is not None:
        #     arrow_spawn = ArrowCfg(
        #         shaft_length=0.9,
        #         shaft_radius=0.01,
        #         head_length=0.1,
        #         head_radius=0.03,
        #         rigid_props=sim_utils.RigidBodyPropertiesCfg(
        #             kinematic_enabled=True,
        #             disable_gravity=True,
        #         ),
        #         collision_props=sim_utils.CollisionPropertiesCfg(
        #             collision_enabled=False,
        #         ),
        #         visual_material=sim_utils.PreviewSurfaceCfg(
        #             diffuse_color=(1.0, 0.0, 0.0),
        #             metallic=0.0,
        #             roughness=0.5,
        #         ),
        #     )
        # else:
        #     arrow_spawn = sim_utils.CuboidCfg(
        #         size=(1.0, 0.02, 0.02),
        #         rigid_props=sim_utils.RigidBodyPropertiesCfg(
        #             kinematic_enabled=True,
        #             disable_gravity=True,
        #         ),
        #         collision_props=sim_utils.CollisionPropertiesCfg(
        #             collision_enabled=False,
        #         ),
        #         visual_material=sim_utils.PreviewSurfaceCfg(
        #             diffuse_color=(1.0, 0.0, 0.0),
        #             metallic=0.0,
        #             roughness=0.5,
        #         ),
        #     )

        # self.robot_future_vel: RigidObjectCfg = RigidObjectCfg(
        #     prim_path="{ENV_REGEX_NS}/RobotFutureVel",
        #     spawn=arrow_spawn,
        # )

class TTEnv(VecEnv):
    def __init__(self, cfg: TTEnvCfg, headless):
        self.cfg: TTEnvCfg

        self.cfg = cfg
        self._is_closed = False
        self.headless = headless
        self.device = self.cfg.device
        self.physics_dt = self.cfg.sim.dt
        self.step_dt = self.cfg.sim.decimation * self.cfg.sim.dt
        self.num_envs = self.cfg.scene.num_envs
        self.seed(cfg.scene.seed)

        sim_cfg = sim_utils.SimulationCfg(
            device=cfg.device,
            dt=cfg.sim.dt,
            render_interval=cfg.sim.decimation,
            physx=PhysxCfg(gpu_max_rigid_patch_count=cfg.sim.physx.gpu_max_rigid_patch_count),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="min", 
                restitution_combine_mode="min", 
                restitution=0.8,
            ),
        )
        self.sim = SimulationContext(sim_cfg)

        scene_cfg = TTSceneCfg(config=cfg.scene, physics_dt=self.physics_dt, step_dt=self.step_dt)
        self.scene = InteractiveScene(scene_cfg)
        self.sim.reset()

        self.robot: Articulation = self.scene["robot"]
        self.table: RigidObject = self.scene["table"]
        self.ball: RigidObject = self.scene["ball"]
        self.ball_future_visual: RigidObject = self.scene["ball_future"]
        self.ball_pred_visual: RigidObject = self.scene["ball_pred"]
        # self.robot_future_vel_visual: RigidObject = self.scene["robot_future_vel"]
        # self.robot_future_pos_visual: RigidObject = self.scene["robot_future"]

        self.contact_sensor: ContactSensor = self.scene.sensors["contact_sensor"]
        if self.cfg.scene.height_scanner.enable_height_scan:
            self.height_scanner: RayCaster = self.scene.sensors["height_scanner"]

        command_cfg = UniformVelocityCommandCfg(
            asset_name="robot",
            resampling_time_range=self.cfg.commands.resampling_time_range,
            rel_standing_envs=self.cfg.commands.rel_standing_envs,
            rel_heading_envs=self.cfg.commands.rel_heading_envs,
            heading_command=self.cfg.commands.heading_command,
            heading_control_stiffness=self.cfg.commands.heading_control_stiffness,
            debug_vis=self.cfg.commands.debug_vis,
            ranges=self.cfg.commands.ranges,
        )
        self.command_generator = UniformVelocityCommand(cfg=command_cfg, env=self)
        self.reward_manager = RewardManager(self.cfg.reward, self)
        self.curriculum_manager = CurriculumManager(self.cfg.curriculum, self)
        
        # ! Aerodynamics Init : BEGIN
        # ! Initialize before the buffer and the environment reset
        self.aero = AeroForceField(
            device=str(self.device), 
            radius_m=0.020,
            air_density=1.225,
            # drag_coeff = 0.0 # ! Set to zero to disable aerodynamics
            drag_coeff = 0.4378 # ! Match the real world testing
        )
        # ! Aerodynamics Init : am

        self.init_buffers()

        env_ids = torch.arange(self.num_envs, device=self.device)
        self.event_manager = EventManager(self.cfg.domain_rand.events, self)
        if "startup" in self.event_manager.available_modes:
            self.event_manager.apply(mode="startup")
        self.reset(env_ids)
    
    def __del__(self):
        """Cleanup for the environment."""
        self.close()

    def init_buffers(self):
        self.extras = {}

        self.max_ball_episode_length_s = self.cfg.ball.ball_max_eposide_length
        self.max_ball_serve_per_episode = self.cfg.ball.max_serve_per_episode
        self.max_ball_episode_length = np.ceil(self.max_ball_episode_length_s / self.step_dt)
        self.max_episode_length_s = self.cfg.scene.max_episode_length_s
        # self.max_episode_length_s = self.max_ball_episode_length_s * self.cfg.ball.ball_reset_repeat * self.cfg.ball.num_new_serves
        self.max_episode_length = np.ceil(self.max_episode_length_s / self.step_dt)

        # self.num_actions = self.robot.data.default_joint_pos.shape[1]
        self.num_actions = self.cfg.robot.num_actions   # actuated joints only
        self.num_joints = self.cfg.robot.num_joints
        self.clip_actions = self.cfg.normalization.clip_actions
        self.clip_obs = self.cfg.normalization.clip_observations
        self.num_perception = 6 # ball_pos(3) + robot_pos(3) = 9

        self.action_scale = self.cfg.robot.action_scale
        self.action_buffer = DelayBuffer(
            self.cfg.domain_rand.action_delay.params["max_delay"], self.num_envs, device=self.device
        )
        self.action_buffer.compute(
            torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        )
        if self.cfg.domain_rand.action_delay.enable:
            time_lags = torch.randint(
                low=self.cfg.domain_rand.action_delay.params["min_delay"],
                high=self.cfg.domain_rand.action_delay.params["max_delay"] + 1,
                size=(self.num_envs,),
                dtype=torch.int,
                device=self.device,
            )
            self.action_buffer.set_time_lag(time_lags, torch.arange(self.num_envs, device=self.device))

        self.perception_buffer = DelayBuffer(
            self.cfg.domain_rand.perception_delay.params["max_delay"], self.num_envs, device=self.device
        )
        self.perception_buffer.compute(
            torch.zeros(self.num_envs, self.num_perception, dtype=torch.float, device=self.device, requires_grad=False)
        )
        if self.cfg.domain_rand.perception_delay.enable:
            time_lags = torch.randint(
                low=self.cfg.domain_rand.perception_delay.params["min_delay"],
                high=self.cfg.domain_rand.perception_delay.params["max_delay"] + 1,
                size=(self.num_envs,),
                dtype=torch.int,
                device=self.device,
            )
            self.perception_buffer.set_time_lag(time_lags, torch.arange(self.num_envs, device=self.device))

        # resolve the joints over which the action term is applied
        self.action_joint_ids, self.action_joint_names = self.robot.find_joints(
            self.cfg.actions.joint_names, preserve_order=self.cfg.actions.preserve_order
        )
        self.obs_joint_ids, self.obs_joint_names = self.robot.find_joints(
            self.cfg.observations.joint_names, preserve_order=self.cfg.observations.preserve_order
        )

        self.robot_cfg = SceneEntityCfg(name="robot")
        self.robot_cfg.resolve(self.scene)

        self.table_cfg = SceneEntityCfg(name="table")
        self.table_cfg.resolve(self.scene)

        self.ball_cfg = SceneEntityCfg(name="ball")
        self.ball_cfg.resolve(self.scene)

        self.termination_contact_cfg = SceneEntityCfg(
            name="contact_sensor", body_names=self.cfg.robot.terminate_contacts_body_names
        )
        self.termination_contact_cfg.resolve(self.scene)
        self.feet_cfg = SceneEntityCfg(name="contact_sensor", body_names=self.cfg.robot.feet_body_names)
        self.feet_cfg.resolve(self.scene)

        self.obs_scales = self.cfg.normalization.obs_scales
        self.add_noise = self.cfg.noise.add_noise

        self.episode_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.ball_episode_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.ball_reset_counter = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        # will store env ids that had their ball reset in the most recent step
        self.ball_reset_ids = torch.empty(0, dtype=torch.long, device=self.device)
        self.reset_ball_state_buf = torch.zeros(self.num_envs, 13, device=self.device, dtype=torch.float)
        self.sim_step_counter = 0
        self.time_out_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)

        # ball-related buffers
        self.has_touch_paddle = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.has_touch_paddle_rew = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.ball_landing_dis_rew = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.ball_contact_rew = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.has_first_bounce = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.has_first_bounce_prev = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.has_touch_own_table = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.has_touch_own_table_prev = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.has_touch_opo_table_prev = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.has_return_own_table2_prev = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)

        self.penalty_ball_to_floor = torch.zeros(self.num_envs, device=self.device)
        self.reward_table_success = torch.zeros(self.num_envs, device=self.device)
        self.reward_vel_prev = torch.zeros(self.num_envs, device=self.device)
        self.penalty_table_fail = torch.zeros(self.num_envs, device=self.device)

        self.touch_info = []

        # storage for predicted future ball pose
        self.ball_future_pose_vis = torch.zeros(self.num_envs, 3, device=self.device)
        self.ball_future_pose = torch.zeros(self.num_envs, 3, device=self.device)
        # learned prediction (from auxiliary model)
        self.ball_prediction_vis = torch.zeros(self.num_envs, 3, device=self.device)
        self.ball_prediction = torch.zeros(self.num_envs, 3, device=self.device)
        self.robot_future_vel = torch.zeros(self.num_envs, 3, device=self.device)
        self.robot_future_pos = torch.zeros(self.num_envs, 3, device=self.device)
        self.ball_future_t = torch.zeros(self.num_envs, 1, device=self.device)
        self.mask_invalid = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.mask_terminal = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.robot_future_vel = torch.zeros(self.num_envs, 3, device=self.device)

        self.paddle_touch_point = torch.zeros(self.num_envs, 3, device=self.device)
        self.robot.write_joint_effort_limit_to_sim(self.robot.data.joint_effort_limits[:, self.action_joint_ids] * self.cfg.robot.effort_limit_scale, self.action_joint_ids)

        self.init_obs_buffer()
        # --- Quadratic-drag model constant for ball dynamics (scalar k) ---
        # k = 0.5 * rho * Cd * A / m
        try:
            rho = 1.225  # kg/m^3
            cd = 0.47    # sphere drag coefficient
            radius = 0.02  # m
            mass = 0.0027  # kg
            area = float(np.pi) * (radius ** 2)
            self.ball_drag_k = float(0.5 * rho * cd * area / max(1e-6, mass))
        except Exception:
            self.ball_drag_k = 0.13

    def update_ball_future_visual(self):
        if self.headless:
            return
        pose = torch.zeros((self.num_envs, 7), device=self.device)
        pose[:, :3] = self.ball_future_pose_vis
        pose[:, 3] = 1.0
        env_ids = torch.arange(self.num_envs, device=self.device)
        self.ball_future_visual.write_root_pose_to_sim(pose, env_ids)
        self.scene.write_data_to_sim()
    def update_ball_pred_visual(self):
        if self.headless:
            return
        pose = torch.zeros((self.num_envs, 7), device=self.device)
        pose[:, :3] = self.ball_prediction_vis
        pose[:, 3] = 1.0
        env_ids = torch.arange(self.num_envs, device=self.device)
        self.ball_pred_visual.write_root_pose_to_sim(pose, env_ids)
        self.scene.write_data_to_sim()
    def update_robot_future_pos_visual(self):
        if self.headless:
            return
        pose = torch.zeros((self.num_envs, 7), device=self.device)
        # robot_future_pos is in env-local coords; add env origins to place in world
        pose[:, :3] = self.robot_future_pos + self.scene.env_origins
        pose[:, 3] = 1.0  # identity quat (w, x, y, z) where w=1, xyz=0
        env_ids = torch.arange(self.num_envs, device=self.device)
        self.robot_future_pos_visual.write_root_pose_to_sim(pose, env_ids)
        self.scene.write_data_to_sim()
    # def update_robot_future_vel_visual(self):
    #     if self.headless:
    #         return
    #     base_pos = self.robot.data.root_link_pos_w
    #     vel = self.robot_future_vel
    #     pose = torch.zeros((self.num_envs, 7), device=self.device)
    #     ref_dir = torch.zeros_like(vel)
    #     ref_dir[:, 0] = 1.0
    #     dirs = torch.nn.functional.normalize(vel, dim=-1, eps=1e-6)
    #     # place arrow above robot and start at robot origin
    #     pose[:, :3] = base_pos + torch.tensor([0.0, 0.0, 0.5], device=self.device) + dirs * 0.2
    #     cross = torch.cross(ref_dir, dirs, dim=-1)
    #     w = 1.0 + (ref_dir * dirs).sum(dim=-1, keepdim=True)
    #     quat = torch.cat([w, cross], dim=-1)
    #     quat = torch.nn.functional.normalize(quat, dim=-1, eps=1e-6)
    #     pose[:, 3:] = quat
    #     env_ids = torch.arange(self.num_envs, device=self.device)
    #     self.robot_future_vel_visual.write_root_pose_to_sim(pose, env_ids)
    #     self.scene.write_data_to_sim()
        

    def compute_current_observations(self):
        robot = self.robot
        table = self.table
        ball = self.ball
        net_contact_forces = self.contact_sensor.data.net_forces_w_history

        ang_vel = robot.data.root_ang_vel_b
        projected_gravity = robot.data.projected_gravity_b
        heading = robot.data.heading_w
        # command = self.command_generator.command
        joint_pos = robot.data.joint_pos[:, self.obs_joint_ids] - robot.data.default_joint_pos[:, self.obs_joint_ids]
        joint_vel = robot.data.joint_vel[:, self.obs_joint_ids]
        action = self.action_buffer._circular_buffer.buffer[:, -1, :]
        ball_pos = ball.data.root_link_pos_w - table.data.root_link_pos_w
        # ball_pos = self.ball.data.root_pos_w - self.scene.env_origins  
        ball_linvel = self.ball.data.root_lin_vel_w
        robot_pos = robot.data.root_link_pos_w - table.data.root_link_pos_w
        # Relative target offset (x,y): use ball prediction shifted by (-0.1, +0.6) as desired robot target
        ball_target_xy = torch.stack([
            self.ball_prediction[:, 0] - 0.1,
            self.ball_prediction[:, 1] + 0.6,
        ], dim=1)
        rel_target_xy = (ball_target_xy - robot_pos[:, :2]) * self.obs_scales.robot_pos

        current_actor_obs = torch.cat(
            [
                ang_vel * self.obs_scales.ang_vel,
                projected_gravity * self.obs_scales.projected_gravity,
                joint_pos * self.obs_scales.joint_pos,
                joint_vel * self.obs_scales.joint_vel,
                action * self.obs_scales.actions,
                ball_pos * self.obs_scales.ball_pos,
                robot_pos * self.obs_scales.robot_pos,
                # self.paddle_touch_point* self.obs_scales.ball_pos,
                self.ball_prediction * self.obs_scales.ball_pos, # use learned prediction in actor obs
                rel_target_xy,  # 2D relative target base pos
                heading.unsqueeze(-1) * self.obs_scales.projected_gravity,
            ],
            dim=-1,
        )
        alt_critic_obs = torch.cat(
            [
                ang_vel * self.obs_scales.ang_vel,
                projected_gravity * self.obs_scales.projected_gravity,
                joint_pos * self.obs_scales.joint_pos,
                joint_vel * self.obs_scales.joint_vel,
                action * self.obs_scales.actions,
                ball_pos * self.obs_scales.ball_pos,
                ball_linvel * self.obs_scales.ball_linvel,
                # self.ball_prediction * self.obs_scales.ball_pos,
                self.ball_future_pose * self.obs_scales.ball_pos, # use ground-truth future in critic obs
                self.paddle_touch_point* self.obs_scales.ball_pos,
                robot_pos * self.obs_scales.robot_pos,
                heading.unsqueeze(-1) * self.obs_scales.projected_gravity,
                (self.robot_future_pos * self.obs_scales.robot_pos - robot_pos * self.obs_scales.robot_pos),
                self.ball_future_t,  # [N,1]
                (self.ball_episode_length_buf.float() / float(self.max_ball_episode_length)).clamp(0.0, 1.0).unsqueeze(-1),
                # (self.episode_length_buf.float() / float(self.max_episode_length)).clamp(0.0, 1.0).unsqueeze(-1),
                (self.ball_reset_counter/self.max_ball_serve_per_episode).unsqueeze(-1),
                self.has_touch_own_table_prev.unsqueeze(-1) * self.obs_scales.ball_state,
                self.has_touch_paddle.unsqueeze(-1) * self.obs_scales.ball_state,
            ],
            dim=-1,
        )
        root_lin_vel = robot.data.root_lin_vel_b
        feet_contact = torch.max(torch.norm(net_contact_forces[:, :, self.feet_cfg.body_ids], dim=-1), dim=1)[0] > 0.5
        current_critic_obs = torch.cat(
            [alt_critic_obs, root_lin_vel * self.obs_scales.lin_vel, feet_contact],
            dim=-1
        )

        return current_actor_obs, current_critic_obs
    
    def compute_current_observations_perception(self): 
        robot = self.robot
        table = self.table
        ball = self.ball
        net_contact_forces = self.contact_sensor.data.net_forces_w_history

        ang_vel = robot.data.root_ang_vel_b
        projected_gravity = robot.data.projected_gravity_b
        heading = robot.data.heading_w
        # command = self.command_generator.command
        joint_pos = robot.data.joint_pos[:, self.obs_joint_ids] - robot.data.default_joint_pos[:, self.obs_joint_ids]
        joint_vel = robot.data.joint_vel[:, self.obs_joint_ids]
        action = self.action_buffer._circular_buffer.buffer[:, -1, :]
        ball_pos = ball.data.root_link_pos_w - table.data.root_link_pos_w
        # ball_pos = self.ball.data.root_pos_w - self.scene.env_origins  
        ball_linvel = self.ball.data.root_lin_vel_w
        robot_pos = robot.data.root_link_pos_w - table.data.root_link_pos_w
        # Relative target offset (x,y) for actor obs with perception
        ball_target_xy = torch.stack([
            self.ball_prediction[:, 0] - 0.1,
            self.ball_prediction[:, 1] + 0.6,
        ], dim=1)
        rel_target_xy = (ball_target_xy - robot_pos[:, :2]) * self.obs_scales.robot_pos

        current_actor_obs = torch.cat(
            [
                ang_vel * self.obs_scales.ang_vel,
                projected_gravity * self.obs_scales.projected_gravity,
                joint_pos * self.obs_scales.joint_pos,
                joint_vel * self.obs_scales.joint_vel,
                action * self.obs_scales.actions,
                self.delayed_perception * self.obs_scales.perception,
                # self.paddle_touch_point* self.obs_scales.ball_pos,
                self.ball_prediction * self.obs_scales.ball_pos, # use learned prediction in actor obs
                rel_target_xy,  # 2D relative target base pos
                heading.unsqueeze(-1) * self.obs_scales.projected_gravity,
            ],
            dim=-1,
        )
        alt_critic_obs = torch.cat(
            [
                ang_vel * self.obs_scales.ang_vel,
                projected_gravity * self.obs_scales.projected_gravity,
                joint_pos * self.obs_scales.joint_pos,
                joint_vel * self.obs_scales.joint_vel,
                action * self.obs_scales.actions,
                ball_pos * self.obs_scales.ball_pos,
                ball_linvel * self.obs_scales.ball_linvel,
                # self.ball_prediction * self.obs_scales.ball_pos,
                self.ball_future_pose * self.obs_scales.ball_pos, # use ground-truth future in critic obs
                self.paddle_touch_point* self.obs_scales.ball_pos,
                robot_pos * self.obs_scales.robot_pos,
                heading.unsqueeze(-1) * self.obs_scales.projected_gravity,
                (self.robot_future_pos * self.obs_scales.robot_pos - robot_pos * self.obs_scales.robot_pos),
                self.ball_future_t,  # [N,1]
                (self.ball_episode_length_buf.float() / float(self.max_ball_episode_length)).clamp(0.0, 1.0).unsqueeze(-1),
                (self.episode_length_buf.float() / float(self.max_episode_length)).clamp(0.0, 1.0).unsqueeze(-1),
                self.has_touch_own_table_prev.unsqueeze(-1) * self.obs_scales.ball_state,
                self.has_touch_paddle.unsqueeze(-1) * self.obs_scales.ball_state,
            ],
            dim=-1,
        )
        root_lin_vel = robot.data.root_lin_vel_b
        feet_contact = torch.max(torch.norm(net_contact_forces[:, :, self.feet_cfg.body_ids], dim=-1), dim=1)[0] > 0.5
        current_critic_obs = torch.cat(
            [alt_critic_obs, root_lin_vel * self.obs_scales.lin_vel, feet_contact],
            dim=-1
        )

        return current_actor_obs, current_critic_obs

    def update_prediction(self, preds: torch.Tensor):
        """Update learned ball prediction and its visualization.

        Args:
            preds: Tensor of shape [N, 3] in env-local coordinates per env.
        """
        try:
            if preds is None:
                return
            # Ensure correct shape and device
            if not isinstance(preds, torch.Tensor):
                preds = torch.as_tensor(preds, dtype=torch.float32)
            preds = preds.to(self.device)
            if preds.shape != (self.num_envs, 3):
                # Attempt to reshape if possible (e.g., [N, H*3] is invalid here)
                if preds.ndim == 2 and preds.shape[1] >= 3:
                    preds = preds[:, :3]
                else:
                    return
            # Store prediction (env-local frame)
            self.ball_prediction = preds
            # Compute visualization positions in world frame
            self.ball_prediction_vis = self.ball_prediction + self.scene.env_origins
            # debug print once
            if not hasattr(self, "_printed_pred_once"):
                try:
                    p0 = self.ball_prediction[0].detach().cpu().numpy()
                    print(f"[TTEnv] update_prediction received, first env: {p0}")
                except Exception:
                    pass
                self._printed_pred_once = True
            if not self.headless:
                self.update_ball_pred_visual()
        except Exception:
            # Be robust against any runtime issues; avoid breaking the sim loop
            pass
    
    def compute_observations(self):
        # current_actor_obs, current_critic_obs = self.compute_current_observations()
        current_actor_obs, current_critic_obs = self.compute_current_observations_perception()
        self.current_actor_obs=current_actor_obs
        if self.add_noise:
            current_actor_obs += (2 * torch.rand_like(current_actor_obs) - 1) * self.noise_scale_vec

        self.actor_obs_buffer.append(current_actor_obs)
        self.critic_obs_buffer.append(current_critic_obs)

        actor_obs = self.actor_obs_buffer.buffer.reshape(self.num_envs, -1)
        critic_obs = self.critic_obs_buffer.buffer.reshape(self.num_envs, -1)
        if self.cfg.scene.height_scanner.enable_height_scan:
            height_scan = (
                self.height_scanner.data.pos_w[:, 2].unsqueeze(1)
                - self.height_scanner.data.ray_hits_w[..., 2]
                - self.cfg.normalization.height_scan_offset
            ) * self.obs_scales.height_scan
            critic_obs = torch.cat([critic_obs, height_scan], dim=-1)
            if self.add_noise:
                height_scan += (2 * torch.rand_like(height_scan) - 1) * self.height_scan_noise_vec
            actor_obs = torch.cat([actor_obs, height_scan], dim=-1)

        actor_obs = torch.clip(actor_obs, -self.clip_obs, self.clip_obs)
        critic_obs = torch.clip(critic_obs, -self.clip_obs, self.clip_obs)

        return actor_obs, critic_obs

    def reset(self, env_ids):
        if len(env_ids) == 0:
            return

        self.extras["log"] = dict()
        if self.cfg.scene.terrain_generator is not None:
            if self.cfg.scene.terrain_generator.curriculum:
                terrain_levels = self.update_terrain_levels(env_ids)
                self.extras["log"].update(terrain_levels)

        self.scene.reset(env_ids)
        if "reset" in self.event_manager.available_modes:
            self.event_manager.apply(
                mode="reset",
                env_ids=env_ids,
                dt=self.step_dt,
                global_env_step_count=self.sim_step_counter // self.cfg.sim.decimation,
            )

        reward_extras = self.reward_manager.reset(env_ids)
        self.extras["log"].update(reward_extras)
        self.extras["time_outs"] = self.time_out_buf

        self.command_generator.reset(env_ids)
        self.actor_obs_buffer.reset(env_ids)
        self.critic_obs_buffer.reset(env_ids)
        self.action_buffer.reset(env_ids)
        self.perception_buffer.reset(env_ids)
        self.episode_length_buf[env_ids] = 0
        self.ball_reset_counter[env_ids] = 0

        # Reset ball state
        self.reset_ball(env_ids)

        self.scene.write_data_to_sim()
        self.sim.forward()

    def reset_ball(self, env_ids):
        """Reset only the ball state for specified environments."""
        if len(env_ids) == 0:
            return
            
        # Reset ball-related buffers for these environments
        self.has_touch_paddle[env_ids] = False
        self.ball_landing_dis_rew[env_ids] = False
        self.has_touch_paddle_rew[env_ids] = False
        self.ball_contact_rew[env_ids] = 0.0
        self.has_first_bounce[env_ids] = False
        self.has_first_bounce_prev[env_ids] = False
        self.has_touch_own_table[env_ids] = False
        self.has_touch_own_table_prev[env_ids] = False
        self.has_touch_opo_table_prev[env_ids] = False
        self.has_return_own_table2_prev[env_ids] = False
        self.reward_vel_prev[env_ids] = 0.0
        self.ball_episode_length_buf[env_ids] = 0
        generate_new = (self.ball_reset_counter[env_ids] % self.cfg.ball.ball_reset_repeat) == 0
        reuse_old = ~generate_new
        self.ball_reset_counter[env_ids] += 1
        self.touch_info = []

        if generate_new.any():
            new_state_env_ids = env_ids[generate_new]
            # Reset ball position and velocity (copied from reset method)
            ball_state = self.ball.data.default_root_state.clone()[new_state_env_ids]
            ball_state[:, :3] += self.scene.env_origins[new_state_env_ids]
            # Random y-noise and velocity
            pos_noise = torch.empty(len(new_state_env_ids), 1, device=self.device).uniform_(
                *self.cfg.ball.ball_pos_y_range
            )
            ball_state[:, 1:2] += pos_noise
            v_x = torch.empty(len(new_state_env_ids), 1, device=self.device).uniform_(
                *self.cfg.ball.ball_speed_x_range
            )
            v_y = torch.empty(len(new_state_env_ids), 1, device=self.device).uniform_(
                *self.cfg.ball.ball_speed_y_range
            )
            v_z = torch.empty(len(new_state_env_ids), 1, device=self.device).uniform_(
                *self.cfg.ball.ball_speed_z_range
            )
            # With small probability (≈1%), create zero-velocity, random-position serves
            #disabled for testing 
            # try:
            #     special_mask = torch.rand(len(new_state_env_ids), device=self.device) < 0.01
            #     if special_mask.any():
            #         sel = torch.nonzero(special_mask, as_tuple=False).squeeze(-1)
            #         # Sample XYZ uniformly from the given ranges (env-local) and offset by env origins
            #         x_rand = torch.empty(len(sel), 1, device=self.device).uniform_(-2.0, 2.0)
            #         y_rand = torch.empty(len(sel), 1, device=self.device).uniform_(-2.0, 2.0)
            #         z_rand = torch.empty(len(sel), 1, device=self.device).uniform_(0.75, 1.5)
            #         pos_rand = torch.cat((x_rand, y_rand, z_rand), dim=1)
            #         ball_state[sel, :3] = self.scene.env_origins[new_state_env_ids][sel] + pos_rand
            #         # Zero linear velocity for these special cases
            #         v_x[sel] = 0.0
            #         v_y[sel] = 0.0
            #         v_z[sel] = 0.0
            # except Exception:
            #     pass
            lin_vel = torch.cat((v_x, v_y, v_z), dim=1)
            ang_vel = torch.zeros(len(new_state_env_ids), 3, device=self.device)
            ball_state[:, 7:] = torch.cat((lin_vel, ang_vel), dim=1)
            # Store new states in buffer
            self.reset_ball_state_buf[new_state_env_ids] = ball_state
            # Apply the new state to the simulation
            self.ball.write_root_pose_to_sim(ball_state[:, :7], new_state_env_ids)
            self.ball.write_root_velocity_to_sim(ball_state[:, 7:], new_state_env_ids)

        if reuse_old.any():
            old_state_env_ids = env_ids[reuse_old]
            old_states = self.reset_ball_state_buf[old_state_env_ids]
            # Apply old states to simulation
            self.ball.write_root_pose_to_sim(old_states[:, :7], old_state_env_ids)
            self.ball.write_root_velocity_to_sim(old_states[:, 7:], old_state_env_ids)

    def step(self, actions: torch.Tensor):

        delayed_actions = self.action_buffer.compute(actions)

        cliped_actions = torch.clip(delayed_actions, -self.clip_actions, self.clip_actions).to(self.device)
        processed_actions = cliped_actions * self.action_scale + self.robot.data.default_joint_pos[:, self.action_joint_ids]

        for _ in range(self.cfg.sim.decimation):
            self.sim_step_counter += 1
            self.robot.set_joint_position_target(processed_actions, self.action_joint_ids)
            # ! Aerodynamics: Step : BEGIN
            # ! Step before self.scene.write_data_to_sim(), update per decimation step
            self.aero.apply_to_rigid_object(self.ball)
            # ! Aerodynamics: Step : END
            self.scene.write_data_to_sim()
            self.sim.step(render=False)
            self.scene.update(dt=self.physics_dt)
            self.compute_perception()
            self.compute_paddle_touch()

        if not self.headless:
            self.sim.render()

        self.episode_length_buf += 1
        self.ball_episode_length_buf += 1
        self.command_generator.compute(self.step_dt)
        if "interval" in self.event_manager.available_modes:
            self.event_manager.apply(mode="interval", dt=self.step_dt)

        self.compute_intermediate_values()

        # Check for balls on floor and reset them without resetting the entire environment
        ball_on_floor = self.ball.data.root_pos_w[:, 2] < 0.1  # Adjust threshold as needed
        ball_timeout = self.ball_episode_length_buf >= self.max_ball_episode_length
        ball_reset_condition = ball_on_floor | ball_timeout
        # ball_reset_condition = ball_timeout
        ball_reset_ids = ball_reset_condition.nonzero(as_tuple=False).flatten()
        # expose ball reset ids for downstream modules (e.g., predictor)
        # always set as a tensor on device to avoid attribute errors
        self.ball_reset_ids = ball_reset_ids
        # ball_reset_ids = ball_on_floor.nonzero(as_tuple=False).flatten()
        if len(ball_reset_ids) > 0:
            self.reset_ball(ball_reset_ids)

        self.reset_buf, self.time_out_buf = self.check_reset()

        self.curriculum_manager.compute()
        
        reward_buf = self.reward_manager.compute(self.step_dt)

        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset(env_ids)

        actor_obs, critic_obs = self.compute_observations()
        self.extras["observations"] = {"critic": critic_obs}

        return actor_obs, reward_buf, self.reset_buf, self.extras

    def check_reset(self):
        # net_contact_forces = self.contact_sensor.data.net_forces_w_history

        # reset_buf = torch.any(
        #     torch.max(
        #         torch.norm(
        #             net_contact_forces[:, :, self.termination_contact_cfg.body_ids],
        #             dim=-1,
        #         ),
        #         dim=1,
        #     )[0]
        #     > 1.0,
        #     dim=1,
        # )
        
        # reset_buf = (
        #     (self.robot_pos[..., 2] < 0.50) |
        #     (self.robot_pos[..., 0] < -3.6) |
        #     (self.robot_pos[..., 0] > -1.35) |
        #     (self.robot_pos[..., 1] < -2.0) |
        #     (self.robot_pos[..., 1] > 2.0)
        # )
        reset_buf = (
            (self.robot_pos[..., 2] < 0.50) |
            (self.robot_pos[..., 0] < -3.6) |
            (self.robot_pos[..., 0] > -1.35) |
            (self.robot_pos[..., 1] < -1.1) |
            (self.robot_pos[..., 1] > 1.1)
        )       
        time_out_buf = self.episode_length_buf >= self.max_episode_length
        time_out_buf |= self.ball_reset_counter >self.max_ball_serve_per_episode
        # print(self.episode_length_buf,self.max_episode_length)
        # print(self.max_episode_length_s,self.step_dt)
        # print ('time_out_buf',time_out_buf)
        reset_buf |= time_out_buf
        # print ('reset_buf',reset_buf)
        return reset_buf, time_out_buf

    def compute_perception(self):
        self.ball_pos = self.ball.data.root_pos_w - self.scene.env_origins  # Local (offset) position
        self.robot_pos = self.robot.data.root_link_pos_w - self.table.data.root_link_pos_w  # Local (offset) position wrt table
        self.current_perception = torch.cat(
            [
                self.ball_pos,
                # self.ball_linvel,
                self.robot_pos,
            ],
            dim=-1,
        )
        self.delayed_perception = self.perception_buffer.compute(self.current_perception)

    def compute_paddle_touch(self):
        self.ball_global_pos = self.ball.data.root_pos_w 

        # --- Compute Paddle Position and Contact ---
        paddle_index = 15  # paddle belongs to 'right_hand_link'
        paddle_pos = self.robot.data.body_pos_w[:, paddle_index, :]
        # print("paddle_pos: ", paddle_pos[0, :])
        # print("ball_pos: ", self.ball_global_pos[0,:])

        paddle_quat = self.robot.data.body_quat_w[:, paddle_index, :]
        # 1) Normalize the quaternion (just in case):
        paddle_quat = paddle_quat / paddle_quat.norm(dim=1, keepdim=True)
        # 2) Build the local offset (0, -0.345, 0) and expand to (N,3):
        local_offset = (
            torch.tensor(
                [0.0, -0.345, 0.0], # Good to double check.
                device=paddle_pos.device,
                dtype=paddle_pos.dtype,
            )
            .unsqueeze(0)
            .expand_as(paddle_pos)
        )
        rotated_offset: torch.Tensor = math_utils.quat_apply(paddle_quat, local_offset)
        # 4) Compute your touch point:
        self.paddle_touch_point = paddle_pos + rotated_offset # paddle_position in the world frame.
        # 5) Compute touch reward:

        distance = torch.norm(self.ball_global_pos - self.paddle_touch_point, dim=1) - 0.02 # corrected for ball radius
        self.paddel_ball_distance = distance
        contact_score = (
            self.cfg.ball.contact_threshold - distance
        ) / self.cfg.ball.contact_threshold
        self.ball_contact = torch.clamp(contact_score, min=0.0, max=1.0) # determine if in contact region
        self.ball_contact_rew = torch.maximum(self.ball_contact_rew, self.ball_contact) # finds reward for closest ball paddle distance
        # self.ball_contact = torch.where(contact_score > 0.0, torch.ones_like(contact_score), torch.zeros_like(contact_score))
        # self.ball_contact = self.ball_contact * ~self.has_touch_paddle # mask invalid if previous ball_contact True
        # new_hits = contact_score > 0  # Tensor[N] bool
        new_hits = (contact_score > 0) & (self.ball_contact < self.ball_contact_rew)  # Tensor[N] bool
        still_false = ~self.has_touch_paddle  # Tensor[N] bool
        self.has_touch_paddle[still_false] = new_hits[still_false] # set has_touch_paddle True for env with ball_contact True

    def compute_intermediate_values(self):
        # print("has_touch_paddle", self.has_touch_paddle)
        # self.ball_pos = self.ball.data.root_pos_w - self.table.data.root_pos_w  # Local (offset) position wrt table
        self.ball_quat = self.ball.data.root_quat_w
        self.ball_vel = self.ball.data.root_vel_w
        self.ball_linvel = self.ball.data.root_lin_vel_w
        self.ball_angvel = self.ball.data.root_ang_vel_w
        self.robot_linvel= self.robot.data.root_lin_vel_w
        self.ball_contact_rew = self.ball_contact_rew * (self.has_touch_paddle * ~self.has_touch_paddle_rew) # Mask if previously already gained reward
        self.ball_landing_dis_rew = self.has_touch_paddle & ~self.has_touch_paddle_rew # Set True if has_touch_paddle_rew is True, compute landing dis reward once, set False if previously True

        # --- Compute Contact with table ---
        bx, by, bz = (self.ball_pos[:, 0], self.ball_pos[:, 1], self.ball_pos[:, 2])
        # 3) Load your table‐contact bounds from self
        tcx_min, tcx_max = self.cfg.table.table_opponent_contact_x
        tcy_min, tcy_max = self.cfg.table.table_opponent_contact_y
        tcz_min, tcz_max = self.cfg.table.table_opponent_contact_z

        ncx_min, ncx_max = self.cfg.table.table_own_contact_x
        ncy_min, ncy_max = self.cfg.table.table_own_contact_y
        ncz_min, ncz_max = self.cfg.table.table_own_contact_z
        # print(f'bz{bz}')
        # print(f'ncz_max{ncz_max}')
        # 4) Build masks
        self.has_touch_opponent_table_just_now = (
            (bx >= tcx_min)
            & (bx <= tcx_max)
            & (by >= tcy_min)
            & (by <= tcy_max)
            & (bz >= tcz_min)
            & (bz <= tcz_max)
        )
        has_touch_own_table_just_now = (
            (bx >= ncx_min)
            & (bx <= ncx_max)
            & (by >= ncy_min)
            & (by <= ncy_max)
            & (bz >= ncz_min)
            & (bz <= ncz_max)
            # & (~self.has_touch_own_table_prev) # have not touched own table before
            # bz <= ncz_max
        )
        # print(f'env.has_touch_own_table_just_now={has_touch_own_table_just_now}')
        self.has_touch_own_table_prev = (
            self.has_touch_own_table_prev | has_touch_own_table_just_now
        )
        self.has_touch_opo_table_prev = (
            self.has_touch_opo_table_prev | self.has_touch_opponent_table_just_now
        )      
        self.has_return_own_table2_prev = (
            (self.has_touch_own_table_prev & self.has_touch_paddle & has_touch_own_table_just_now) | self.has_return_own_table2_prev
        )

        self.touched_paddel_no_bounce_table=(
            self.has_touch_paddle & ~(self.has_return_own_table2_prev | self.has_touch_opo_table_prev)
        )
        # print(f"{self.has_touch_paddle=},{self.has_return_own_table2_prev=},{self.has_touch_opo_table_prev=}")
        # print(f'{self.touched_paddel_no_bounce_table=}')
        # self.has_touch_own_table = has_touch_own_table_just_now
        self.has_first_bounce_prev = self.has_first_bounce.clone()
        still_false = ~self.has_first_bounce
        self.has_first_bounce[still_false] = self.has_touch_own_table[still_false]


        self.touch_info.append(
            {
                "has_touch_own_table": self.has_touch_own_table[0].cpu().numpy(),
                "has_touch_own_table_prev": self.has_touch_own_table_prev[0]
                .cpu()
                .numpy(),
                "has_touch_opponent_table": self.has_touch_opponent_table_just_now[0]
                .cpu()
                .numpy(),
                "has_first_bounce": self.has_first_bounce[0].cpu().numpy(),
                "has_first_bounce_prev": self.has_first_bounce_prev[0].cpu().numpy(),
                "ball_contact": self.ball_contact[0].cpu().numpy(),
                "has_touch_paddle": self.has_touch_paddle[0].cpu().numpy(),
                "has_first_bounce_prev": self.has_first_bounce_prev[0].cpu().numpy(),
            }
        )

        self.paddle_pos = self.paddle_touch_point - self.scene.env_origins 
        has_bounced = self.has_touch_own_table_prev
        vz = self.ball_linvel[:, 2]
        z = self.ball_pos[:, 2]
        x = self.ball_pos[:, 0]
        y = self.ball_pos[:, 1]
        vx = self.ball_linvel[:, 0]
        vy = self.ball_linvel[:, 1]

        g=9.81
        body_height=0.69
        vel_max=7.0
        paddle_y_offset = -0.60

        self.mask_before = (has_bounced == 0).squeeze(-1)
        self.mask_after = (has_bounced == 1).squeeze(-1)

        h = 0.78
        D = vz.pow(2) + 2.0 * g * (z - h)
        sqrtD = torch.sqrt(torch.clamp(D, min=0.0))
        t_land = (vz + sqrtD) / g
        t_land = torch.clamp(t_land, min=0.0)
        self.valid_before = (D >= 0.0) & self.mask_before
        t1 = torch.where(self.valid_before, t_land, torch.zeros_like(t_land))
        vz_prime = vz - g * t1
        vz_dd = -0.9 * vz_prime
        # Drag-adjusted ascent after first bounce
        k_scalar = torch.tensor(max(1e-8, getattr(self, 'ball_drag_k', 0.13)), device=self.device, dtype=vz.dtype)
        sqrt_gk = torch.sqrt(torch.clamp(torch.tensor(g, device=self.device, dtype=vz.dtype) * k_scalar, min=1e-12))
        sqrt_kg = torch.sqrt(torch.clamp(k_scalar / torch.tensor(g, device=self.device, dtype=vz.dtype), min=0.0))
        vz_up = torch.clamp(vz_dd, min=0.0)
        t2_drag = torch.atan(torch.clamp(vz_up * sqrt_kg, min=0.0)) / torch.clamp(sqrt_gk, min=1e-12)
        delta_z_drag = (0.5 / k_scalar) * torch.log1p(torch.clamp((k_scalar / torch.tensor(g, device=self.device, dtype=vz.dtype)) * vz_up.pow(2), min=0.0))
        self.t_before = t1 + t2_drag
        zpb = delta_z_drag + h

        # Horizontal displacement under quadratic drag: s(T) = (1/k) * ln(1 + k |v0| T) * sign(v0)
        def horiz_disp(v0, T):
            vabs = torch.abs(v0)
            return (1.0 / k_scalar) * torch.sign(v0) * torch.log1p(torch.clamp(k_scalar * vabs * T, min=0.0))

        # Drag-adjusted ascent for current upward motion (case 'a')
        vz_up_a = torch.clamp(vz, min=0.0)
        t_after_drag = torch.atan(torch.clamp(vz_up_a * sqrt_kg, min=0.0)) / torch.clamp(sqrt_gk, min=1e-12)
        delta_z_drag_a = (0.5 / k_scalar) * torch.log1p(torch.clamp((k_scalar / torch.tensor(g, device=self.device, dtype=vz.dtype)) * vz_up_a.pow(2), min=0.0))
        self.t_after = t_after_drag
        zpa = z + delta_z_drag_a
        self.predict_land_t = t_land
        self.predict_x_land = x + horiz_disp(vx, t_land) # acounts for both landing on our side and the opponent's side of table
        self.predict_y_land = y + horiz_disp(vy, t_land)
        predict_ball_land_vis=torch.stack([self.predict_x_land, self.predict_y_land, torch.ones_like(x)*h], dim=-1)


        dx_ascent = horiz_disp(0.7 *vx, t2_drag)
        dy_ascent = horiz_disp(0.7 *vy, t2_drag)
        xpb = self.predict_x_land + dx_ascent
        ypb = self.predict_y_land + dy_ascent
        dx_after = horiz_disp(vx, t_after_drag)
        dy_after = horiz_disp(vy, t_after_drag)
        xpa = x + dx_after
        ypa = y + dy_after

        xpb=torch.clamp(xpb, max=-1.6)
        xpa=torch.clamp(xpa, max=-1.6)

        self.pos_pred_before = torch.stack([xpb, ypb, zpb], dim=-1)
        self.pos_pred_after = torch.stack([xpa, ypa, zpa], dim=-1)

        self.pos_pred_before_ro = torch.stack([xpb - 0.1, ypb - paddle_y_offset, torch.ones_like(xpb) * body_height], dim=-1)
        self.pos_pred_after_ro = torch.stack([xpa - 0.1, ypa - paddle_y_offset, torch.ones_like(xpb) * body_height], dim=-1)

        self.ball_future_pose = torch.where(
            self.mask_before.unsqueeze(-1), self.pos_pred_before, self.pos_pred_after
        )
        # self.mask_invalid = (self.ball_pos[:, 0] < -1.6) | (vx > 0) | (z < 0.7)
        # Invalid mask: use explicit parentheses to avoid bitwise ops on floats
        self.mask_invalid = (
            (self.ball_pos[:, 0] < -1.9)
            | (vx > 0)
            | (z < 0.7)
            | ((self.ball_pos[:, 0] < -1.35) & (vz < 0))
            | self.has_touch_paddle
        )
        self.mask_terminal = (self.ball_pos[:, 0] > -1.5) | (self.ball_pos[:, 0] < -1.9) | self.has_touch_paddle_rew | (vz < 0.0) | (self.ball_pos[:, 2] < 0.6) 
            #mask_terminal: true-> future,mask_terminal: false->distance
        self.has_touch_paddle_rew = self.has_touch_paddle.clone() # finally set mask True for reward computation
        # Expand to match shape (N, 3)
        mask_invalid_expanded = self.mask_invalid.unsqueeze(-1).expand_as(self.ball_future_pose)
        # Zero out those poses
        modified_ball_pos = torch.clone(self.robot_pos)
        modified_ball_pos[:, 1] += paddle_y_offset  # Offset to paddle position
        modified_ball_pos[:, 2] = body_height+ 0.2  # Set z-coordinate to body_height
        self.ball_future_pose = torch.where(
            mask_invalid_expanded,
            modified_ball_pos,
            self.ball_future_pose
        )

        # self.ball_future_pose_vis = self.ball_future_pose + self.scene.env_origins
        mask = self.touched_paddel_no_bounce_table.unsqueeze(-1)  # (N, 1)
        self.ball_future_pose_vis = torch.where(
            mask,
            predict_ball_land_vis,
            self.ball_future_pose,
        ) + self.scene.env_origins
        self.robot_future_pos = torch.where(
            self.mask_before.unsqueeze(-1), self.pos_pred_before_ro, self.pos_pred_after_ro
        )
        self.robot_future_pos = torch.where(
            mask_invalid_expanded,      # [N,3] bool
            self.robot_future_pos.new_tensor([-1.80, 0.3, body_height]).expand_as(self.robot_future_pos),               
            # [-0.9, 0.2, body_height] for all envs
            self.robot_future_pos
        )

        # self.vel_ro_before = torch.clamp((self.pos_pred_before_ro - self.robot_pos ) /torch.clamp(self.t_before.unsqueeze(-1).expand_as(self.ball_future_pose), min=0.2),min=-vel_max, max=vel_max)
        # self.vel_ro_after = torch.clamp((self.pos_pred_after_ro - self.robot_pos ) /torch.clamp(self.t_after.unsqueeze(-1).expand_as(self.ball_future_pose), min=0.2),min=-vel_max, max=vel_max)
        self.vel_ro_before = torch.clamp((self.pos_pred_before_ro - self.robot_pos ) *4,min=-vel_max, max=vel_max)
        self.vel_ro_after = torch.clamp((self.pos_pred_after_ro - self.robot_pos ) *4,min=-vel_max, max=vel_max)

        self.robot_future_vel = torch.where(
            self.mask_before.unsqueeze(-1), self.vel_ro_before, self.vel_ro_after
        )
        self.robot_future_vel = torch.where(
            mask_invalid_expanded,
            torch.zeros_like(self.robot_future_vel),
            self.robot_future_vel
        )

        tb = self.t_before.unsqueeze(-1)  # [N,1]
        ta = self.t_after.unsqueeze(-1)   # [N,1]
        self.ball_future_t = torch.where(self.mask_before.unsqueeze(-1), tb, ta)
        self.ball_future_t = torch.where(
            self.mask_invalid.unsqueeze(-1),
            torch.zeros_like(self.ball_future_t),
            self.ball_future_t
        )

        if not self.headless:
            # self.update_ball_future_visual()
            # self.update_robot_future_pos_visual()
            # self.update_robot_future_vel_visual()
            pass
    def init_obs_buffer(self):
        if self.add_noise:
            actor_obs, _ = self.compute_current_observations()
            noise_vec = torch.zeros_like(actor_obs[0])
            noise_scales = self.cfg.noise.noise_scales
            noise_vec[:3] = noise_scales.ang_vel * self.obs_scales.ang_vel
            noise_vec[3:6] = noise_scales.projected_gravity * self.obs_scales.projected_gravity
            noise_vec[6 : 6 + self.num_actions] = noise_scales.joint_pos * self.obs_scales.joint_pos
            noise_vec[6 + self.num_actions : 6 + self.num_actions * 2] = (
                noise_scales.joint_vel * self.obs_scales.joint_vel
            )
            noise_vec[6 + self.num_actions * 2 : 6 + self.num_actions * 3] = 0.0
            # noise_vec[6 + self.num_actions * 3 : 6 + self.num_actions * 3+3] = noise_scales.ball_pos * self.obs_scales.ball_pos
            # noise_vec[6 + self.num_actions * 3+3 : 6 + self.num_actions * 3+6] = noise_scales.ball_linvel * self.obs_scales.ball_linvel
            # noise_vec[6 + self.num_actions * 3+6 : 6 + self.num_actions * 3+9] = noise_scales.robot_pos * self.obs_scales.robot_pos
            #perception noise
            noise_vec[6 + self.num_actions * 3 : 6 + self.num_actions * 3+6] = noise_scales.perception * self.obs_scales.ball_pos 
            # noise_vec[6 + self.num_actions * 3+9 : 6 + self.num_actions * 3+11] = noise_scales.ball_state * self.obs_scales.ball_state
            #ball prediction noise is set to zero
            noise_vec[6 + self.num_actions * 3 + 6: 6 + self.num_actions * 3 + 9] = 0.0
            noise_vec[6 + self.num_actions * 3 + 9: 6 + self.num_actions * 3 + 11] = noise_scales.perception * self.obs_scales.ball_pos 
            noise_vec[6 + self.num_actions * 3 + 11] = noise_scales.projected_gravity * self.obs_scales.projected_gravity
            self.noise_scale_vec = noise_vec

            if self.cfg.scene.height_scanner.enable_height_scan:
                height_scan = (
                    self.height_scanner.data.pos_w[:, 2].unsqueeze(1)
                    - self.height_scanner.data.ray_hits_w[..., 2]
                    - self.cfg.normalization.height_scan_offset
                )
                height_scan_noise_vec = torch.zeros_like(height_scan[0])
                height_scan_noise_vec[:] = noise_scales.height_scan * self.obs_scales.height_scan
                self.height_scan_noise_vec = height_scan_noise_vec

        self.actor_obs_buffer = CircularBuffer(
            max_len=self.cfg.robot.actor_obs_history_length, batch_size=self.num_envs, device=self.device
        )
        self.critic_obs_buffer = CircularBuffer(
            max_len=self.cfg.robot.critic_obs_history_length, batch_size=self.num_envs, device=self.device
        )
        self.delayed_perception = actor_obs[..., -self.num_perception:]

    def update_terrain_levels(self, env_ids):
        distance = torch.norm(self.robot.data.root_pos_w[env_ids, :2] - self.scene.env_origins[env_ids, :2], dim=1)
        move_up = distance > self.scene.terrain.cfg.terrain_generator.size[0] / 2
        move_down = (
            distance < torch.norm(self.command_generator.command[env_ids, :2], dim=1) * self.max_episode_length_s * 0.5
        )
        move_down *= ~move_up
        self.scene.terrain.update_env_origins(env_ids, move_up, move_down)
        extras = {"Curriculum/terrain_levels": torch.mean(self.scene.terrain.terrain_levels.float())}
        return extras

    def get_observations(self):
        actor_obs, critic_obs = self.compute_observations()
        self.extras["observations"] = {"critic": critic_obs}
        return actor_obs, self.extras

    def close(self):
        """Cleanup for the environment."""
        if not self._is_closed:
            # destructor is order-sensitive
            del self.reward_manager
            del self.event_manager
            del self.scene
            # clear callbacks and instance
            self.sim.clear_all_callbacks()
            self.sim.clear_instance()

            # update closing status
            self._is_closed = True

    @staticmethod
    def seed(seed: int = -1) -> int:
        try:
            import omni.replicator.core as rep  # type: ignore

            rep.set_global_seed(seed)
        except ModuleNotFoundError:
            pass
        return torch_utils.set_seed(seed)
    
