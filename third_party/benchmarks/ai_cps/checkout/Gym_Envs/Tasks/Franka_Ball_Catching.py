from omniisaacgymenvs.tasks.base.rl_task import RLTask
from Models.Franka.Franka import Franka
from Models.Franka.Franka_view import FrankaView
from Models.ball_catching.tool import Tool

from omni.isaac.core.objects import DynamicSphere
from omni.isaac.core.prims import RigidPrim, RigidPrimView
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.utils.stage import get_current_stage, add_reference_to_stage
from omni.isaac.core.utils.torch.transformations import *
from omni.isaac.core.utils.torch.rotations import *

from omni.isaac.cloner import Cloner

import numpy as np
import torch
import math

from pxr import Gf, Usd, UsdGeom


class FrankaBallCatchingTask(RLTask):
    def __init__(
        self,
        name,
        sim_config,
        env,
        offset=None
    ) -> None:
        self._sim_config = sim_config
        self._cfg = sim_config.config
        self._task_cfg = sim_config.task_config

        self._num_envs = self._task_cfg["env"]["numEnvs"]
        self._env_spacing = self._task_cfg["env"]["envSpacing"]

        self._max_episode_length = self._task_cfg["env"]["episodeLength"]

        self.action_scale = self._task_cfg["env"]["actionScale"]
        self.start_position_noise = self._task_cfg["env"]["startPositionNoise"]
        self.start_rotation_noise = self._task_cfg["env"]["startRotationNoise"]

        self.dof_vel_scale = self._task_cfg["env"]["dofVelocityScale"]
        self.dist_reward_scale = self._task_cfg["env"]["distRewardScale"]
        self.rot_reward_scale = self._task_cfg["env"]["rotRewardScale"]
        self.around_handle_reward_scale = self._task_cfg["env"]["aroundHandleRewardScale"]
        self.open_reward_scale = self._task_cfg["env"]["openRewardScale"]
        self.finger_dist_reward_scale = self._task_cfg["env"]["fingerDistRewardScale"]
        self.action_penalty_scale = self._task_cfg["env"]["actionPenaltyScale"]
        self.finger_close_reward_scale = self._task_cfg["env"]["fingerCloseRewardScale"]

        self.distX_offset = 0.04
        self.dt = 1/60.

        self._num_observations = 27
        self._num_actions = 9

        # Flag for testing
        self.is_test = False
        self.initial_test_value = None
        self.is_action_noise = False

        RLTask.__init__(self, name, env)

        # Extra info for TensorBoard
        self.extras = {}
        torch_zeros = lambda: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.episode_sums = {"ball_vel": torch_zeros(), "ball_tool_dist": torch_zeros()}

        return

    def set_up_scene(self, scene) -> None:

        # Franka
        franka_translation = torch.tensor([0.3, 0.0, 0.0])
        self.get_franka(franka_translation)
        self.get_tool()
        self.get_ball()

        # Here the env is cloned (cannot clone particle systems right now)
        super().set_up_scene(scene)

        # Add Franka
        self._frankas = FrankaView(prim_paths_expr="/World/envs/.*/franka", name="franka_view")

        # Add tool
        self._tool = RigidPrimView(prim_paths_expr="/World/envs/.*/tool/tool/tool_mesh", name="tool_view", reset_xform_properties=False)
        self._tool_center = RigidPrimView(prim_paths_expr="/World/envs/.*/tool/tool/center_cube", name="tool_center_view", reset_xform_properties=False)

        # Add ball

        self._ball = RigidPrimView(prim_paths_expr="/World/envs/.*/ball", name="ball_view", reset_xform_properties=False)
        

        scene.add(self._frankas)
        scene.add(self._frankas._hands)
        scene.add(self._frankas._lfingers)
        scene.add(self._frankas._rfingers)
        scene.add(self._tool)
        scene.add(self._tool_center)
        scene.add(self._ball)

        self.init_data()
        return

    def get_franka(self, translation):

        franka = Franka(prim_path=self.default_zero_env_path + "/franka", name="franka", translation = translation)
        self._sim_config.apply_articulation_settings("franka", get_prim_at_path(franka.prim_path), self._sim_config.parse_actor_config("franka"))

    def get_tool(self):
        tool = Tool(prim_path=self.default_zero_env_path + "/tool", name="tool")
        self._sim_config.apply_articulation_settings("tool", get_prim_at_path(tool.prim_path), self._sim_config.parse_actor_config("tool"))

    def get_ball(self):

        ball = DynamicSphere(
                name = 'ball',
                position=[-0.8,0,1.5],
                orientation=[1,0,0,0],
                prim_path=self.default_zero_env_path + "/ball",
                radius=0.01,
                color=np.array([1, 0, 0]),
                density = 100,
                mass = 0.001
            )

    # Set as testing mode
    def set_as_test(self):
        self.is_test = True

    # Set action noise
    def set_action_noise(self):
        self.is_action_noise = True

    # Set initial test values for testing mode
    def set_initial_test_value(self, value):
        # for ball pushing: initial x,y positions of the ball
        self.initial_test_value = value

    def init_data(self) -> None:
        def get_env_local_pose(env_pos, xformable, device):
            """Compute pose in env-local coordinates"""
            world_transform = xformable.ComputeLocalToWorldTransform(0)
            world_pos = world_transform.ExtractTranslation()
            world_quat = world_transform.ExtractRotationQuat()

            px = world_pos[0] - env_pos[0]
            py = world_pos[1] - env_pos[1]
            pz = world_pos[2] - env_pos[2]
            qx = world_quat.imaginary[0]
            qy = world_quat.imaginary[1]
            qz = world_quat.imaginary[2]
            qw = world_quat.real

            return torch.tensor([px, py, pz, qw, qx, qy, qz], device=device, dtype=torch.float)

        stage = get_current_stage()
        hand_pose = get_env_local_pose(self._env_pos[0], UsdGeom.Xformable(stage.GetPrimAtPath("/World/envs/env_0/franka/panda_link7")), self._device)
        lfinger_pose = get_env_local_pose(
            self._env_pos[0], UsdGeom.Xformable(stage.GetPrimAtPath("/World/envs/env_0/franka/panda_leftfinger")), self._device
        )
        rfinger_pose = get_env_local_pose(
            self._env_pos[0], UsdGeom.Xformable(stage.GetPrimAtPath("/World/envs/env_0/franka/panda_rightfinger")), self._device
        )

        # finger pos
        finger_pose = torch.zeros(7, device=self._device)
        finger_pose[0:3] = (lfinger_pose[0:3] + rfinger_pose[0:3]) / 2.0
        finger_pose[3:7] = lfinger_pose[3:7]
        hand_pose_inv_rot, hand_pose_inv_pos = (tf_inverse(hand_pose[3:7], hand_pose[0:3]))

        # franka grasp local pose
        grasp_pose_axis = 1
        franka_local_grasp_pose_rot, franka_local_pose_pos = tf_combine(hand_pose_inv_rot, hand_pose_inv_pos, finger_pose[3:7], finger_pose[0:3])
        franka_local_pose_pos += torch.tensor([0, 0.04, 0], device=self._device)
        self.franka_local_grasp_pos = franka_local_pose_pos.repeat((self._num_envs, 1))
        self.franka_local_grasp_rot = franka_local_grasp_pose_rot.repeat((self._num_envs, 1))

        self.gripper_forward_axis = torch.tensor([0, 0, 1], device=self._device, dtype=torch.float).repeat((self._num_envs, 1))
        self.gripper_up_axis = torch.tensor([0, 1, 0], device=self._device, dtype=torch.float).repeat((self._num_envs, 1))

        self.franka_default_dof_pos = torch.tensor(
            [0.0, -0.872, 0.0, -2.0, 0.0, 2.618, 0.785, 0.04, 0.04], device=self._device
        )

        self.actions = torch.zeros((self._num_envs, self.num_actions), device=self._device)

    def get_observations(self) -> dict:

        # Franka
        hand_pos, hand_rot = self._frankas._hands.get_world_poses(clone=False)
        franka_dof_pos = self._frankas.get_joint_positions(clone=False)
        franka_dof_vel = self._frankas.get_joint_velocities(clone=False)
        self.franka_dof_pos = franka_dof_pos

        self.franka_lfinger_pos, self.franka_lfinger_rot = self._frankas._lfingers.get_world_poses(clone=False)
        self.franka_rfinger_pos, self.franka_rfinger_rot = self._frankas._lfingers.get_world_poses(clone=False)

        dof_pos_scaled = (
            2.0
            * (franka_dof_pos - self.franka_dof_lower_limits)
            / (self.franka_dof_upper_limits - self.franka_dof_lower_limits)
            - 1.0
        )

        # ball
        self.ball_pos, self.ball_rot = self._ball.get_world_poses(clone=False)
        ball_vel = self._ball.get_velocities()      # ball velocity        
        ball_linvels = ball_vel[:, 0:3]             # ball linear velocity

        # tool
        tool_pos, tool_rot = self._tool_center.get_world_poses(clone=False)   # tool position
        to_target = tool_pos - self.ball_pos                           # ball to tool dist

        self.obs_buf = torch.cat(
            (
                dof_pos_scaled,
                franka_dof_vel * self.dof_vel_scale,
                self.ball_pos,
                to_target,
                ball_linvels,
            ),
            dim=-1,
        )

        observations = {
            self._frankas.name: {
                "obs_buf": self.obs_buf
            }
        }
        return observations

    def pre_physics_step(self, actions) -> None:
        if not self._env._world.is_playing():
            return

        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self.reset_idx(reset_env_ids)

        self.actions = actions.clone().to(self._device)
        # if action noise
        if self.is_action_noise is True:
            # Gaussian white noise with 0.01 variance
            self.actions = self.actions + (0.5)*torch.randn_like(self.actions)
            
        targets = self.franka_dof_targets + self.franka_dof_speed_scales * self.dt * self.actions * self.action_scale
        self.franka_dof_targets[:] = tensor_clamp(targets, self.franka_dof_lower_limits, self.franka_dof_upper_limits)
        env_ids_int32 = torch.arange(self._frankas.count, dtype=torch.int32, device=self._device)

        # fix the finger movement so that the tool will always be grasped in hand
        self.franka_dof_targets[:,7] = self.franka_default_dof_pos[7]
        self.franka_dof_targets[:,8] = self.franka_default_dof_pos[8]

        self._frankas.set_joint_position_targets(self.franka_dof_targets, indices=env_ids_int32)


    def reset_idx(self, env_ids):
        indices = env_ids.to(dtype=torch.int32)
        num_indices = len(indices)

        # reset franka
        pos = torch.clamp(
            self.franka_default_dof_pos.unsqueeze(0),
            #+ 0.25 * (torch.rand((len(env_ids), self.num_franka_dofs), device=self._device) - 0.5),
            self.franka_dof_lower_limits,
            self.franka_dof_upper_limits,
        )
        dof_pos = torch.zeros((num_indices, self._frankas.num_dof), device=self._device)
        dof_vel = torch.zeros((num_indices, self._frankas.num_dof), device=self._device)
        dof_pos[:, :] = pos
        self.franka_dof_targets[env_ids, :] = pos
        self.franka_dof_pos[env_ids, :] = pos

        self._frankas.set_joint_position_targets(self.franka_dof_targets[env_ids], indices=indices)
        self._frankas.set_joint_positions(dof_pos, indices=indices)
        self._frankas.set_joint_velocities(dof_vel, indices=indices)

        # reset tool
        self._tool.set_world_poses(self.default_tool_pos[env_ids], self.default_tool_rot[env_ids], indices = indices)
        self._tool.set_velocities(self.default_tool_velocity[env_ids], indices = indices)

        if not self.is_test:

            # reset ball position within an area: x [-0.1, 0.1], y [-0.1,0.1]
            self.new_ball_pos = self.default_ball_pos.clone().detach()
            self.new_ball_pos[:,0] = self.default_ball_pos[:,0] + (0.05 + 0.05) * torch.rand(self._num_envs, device=self._device) -0.05
            self.new_ball_pos[:,1] = self.default_ball_pos[:,1] + (0.05 + 0.05) * torch.rand(self._num_envs, device=self._device) -0.05

            # reset ball velocity: default_ball_vel = [2.2 0.1, 0.0]
            # x-axis vel: [1.0, 1.5]
            # y-axis vel: [0.0, 0.2]
            self.new_ball_vel = self.default_ball_velocity.clone().detach()
            self.new_ball_vel[:,0] = (1.0 - 1.0) * torch.rand(self._num_envs, device=self._device) + 1.0
            self.new_ball_vel[:,1] = (0.0 - 0.0) * torch.rand(self._num_envs, device=self._device) + 0.0

            # reset ball
            self._ball.set_world_poses(self.new_ball_pos[env_ids], self.default_ball_rot[env_ids], indices = indices)
            self._ball.set_velocities(self.new_ball_vel[env_ids], indices = indices)

        else:

            self.new_ball_pos = self.default_ball_pos.clone().detach()
            self.new_ball_pos[:,0] = self.default_ball_pos[:,0] + self.initial_test_value[0]
            self.new_ball_pos[:,1] = self.default_ball_pos[:,1] + self.initial_test_value[1]

            self.new_ball_vel = self.default_ball_velocity.clone().detach()
            self.new_ball_vel[:,0] = self.initial_test_value[2]
            self.new_ball_vel[:,1] = self.initial_test_value[3]

            # reset ball
            self._ball.set_world_poses(self.new_ball_pos[env_ids], self.default_ball_rot[env_ids], indices = indices)
            self._ball.set_velocities(self.new_ball_vel[env_ids], indices = indices)

        # bookkeeping
        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0


    def post_reset(self):

        # Franka
        self.num_franka_dofs = self._frankas.num_dof
        self.franka_dof_pos = torch.zeros((self.num_envs, self.num_franka_dofs), device=self._device)
        dof_limits = self._frankas.get_dof_limits()
        self.franka_dof_lower_limits = dof_limits[0, :, 0].to(device=self._device)
        self.franka_dof_upper_limits = dof_limits[0, :, 1].to(device=self._device)
        self.franka_dof_speed_scales = torch.ones_like(self.franka_dof_lower_limits)
        self.franka_dof_speed_scales[self._frankas.gripper_indices] = 0.1
        self.franka_dof_targets = torch.zeros(
            (self._num_envs, self.num_franka_dofs), dtype=torch.float, device=self._device
        )

        # tool
        self.default_tool_pos, self.default_tool_rot = self._tool.get_world_poses()
        self.default_tool_velocity = self._tool.get_velocities()

        # ball
        self.default_ball_pos, self.default_ball_rot = self._ball.get_world_poses()
        self.default_ball_velocity = self._ball.get_velocities()

        # change default velocities
        self.default_ball_velocity[:,0] = 2.2
        self.default_ball_velocity[:,1] = 0.1
        self.default_ball_velocity[:,2] = 0.0
        self._ball.set_velocities(self.default_ball_velocity)


        # randomize all envs
        indices = torch.arange(self._num_envs, dtype=torch.int64, device=self._device)
        self.reset_idx(indices)


    def calculate_metrics(self) -> None:

       # variables for reward
        ball_pos = self.ball_pos                    # ball pos
        ball_vel = self._ball.get_velocities()      # ball velocity        
        tool_pos, tool_rot = self._tool_center.get_world_poses()       # tool center pos and rot
        ball_linvels = ball_vel[:, 0:3]             # ball linear velocity

        # 1st reward ball to tool center distance
        ball_center_dist = torch.norm(tool_pos - ball_pos, p=2, dim=-1)
        ball_center_XY_dist = torch.norm(tool_pos[:,0:3] - ball_pos[:,0:3], p=2, dim=-1)
        center_dist_reward = 1.0/(1.0+ball_center_dist*100)

        # 2nd reward: ball is unmoved
        norm_ball_linvel = torch.norm(ball_linvels, p=2, dim=-1)  
        ball_vel_reward = 1.0/(1.0+norm_ball_linvel*100)

        # 3rd reward: rotation not too much
        rot_diff = torch.norm(tool_rot - self.default_tool_rot, p=2, dim=-1)
        tool_rot_reward = 1.0/(1.0+rot_diff)

        # action penalty
        action_penalty = torch.sum(self.actions[:,0:7] ** 2, dim=-1)
        action_penalty = 1 - 1.0 / (1.0 + action_penalty)

        # liveness_reward
        liveness_reward = torch.where(ball_center_XY_dist<0.03, torch.ones_like(center_dist_reward), torch.zeros_like(center_dist_reward))

        # final cumulative reward
        final_reward = 1.0*center_dist_reward + 1.0*ball_vel_reward + 0.0*tool_rot_reward + 0.5*liveness_reward - 0.01*action_penalty

        self.rew_buf[:] = final_reward

        # log additional info
        self.episode_sums["ball_vel"] += norm_ball_linvel
        self.episode_sums["ball_tool_dist"] += ball_center_dist 

    def is_done(self) -> None:

        if not self.is_test: 

            ball_pos = self.ball_pos                                       # ball pos
            tool_pos, tool_rot = self._tool_center.get_world_poses()       # tool center pos and rot
            ball_center_dist = torch.norm(tool_pos - ball_pos, p=2, dim=-1)

            # 1st reset: if ball falls from tool
            # self.reset_buf = torch.where(ball_center_dist > 5.0, torch.ones_like(self.reset_buf), self.reset_buf)

            # 2nd reset: if ball falls to the ground
            self.reset_buf = torch.where(self.ball_pos[:,2] < 0.02, torch.ones_like(self.reset_buf), self.reset_buf)

            # 3rd reset if max length reached
            self.reset_buf = torch.where(self.progress_buf >= self._max_episode_length - 1, torch.ones_like(self.reset_buf), self.reset_buf)

        else:

            self.reset_buf = torch.where(self.progress_buf >= self._max_episode_length - 1, torch.ones_like(self.reset_buf), self.reset_buf)
