from omniisaacgymenvs.tasks.base.rl_task import RLTask
from Models.Franka.Franka import Franka
from Models.Franka.Franka_view import FrankaView
from Models.ball_pushing.table import Table

from omni.isaac.core.objects import DynamicSphere
from omni.isaac.core.prims import RigidPrim, RigidPrimView
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.utils.stage import get_current_stage
from omni.isaac.core.utils.torch.transformations import *
from omni.isaac.core.utils.torch.rotations import *

from omni.isaac.cloner import Cloner

import numpy as np
import torch
import math

from pxr import Usd, UsdGeom


class FrankaBallPushingTask(RLTask):
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

        self.ball_radius = self._task_cfg["env"]["ballRadius"]
        self.ball_initial_position = self._task_cfg["env"]["ballInitialPosition"]
        self.ball_initial_orientation = self._task_cfg["env"]["ballInitialOrientation"]
        # self.ball_initial_position[0] = (0.1 + 0.1) * np.random.rand(1) -0.1
        # self.ball_initial_position[1] = (0.2 + 0.2) * np.random.rand(1) -0.2
        # initial_x = (0.1 + 0.1) * torch.rand(self._num_envs) -0.1
        # initial_y = (0.2 + 0.2) * torch.rand(self._num_envs) -0.2
        self.distX_offset = 0.04
        self.dt = 1/60.

        self._num_observations = 30
        self._num_actions = 9

        # Flag for testing
        self.is_test = False
        self.initial_test_value = None
        self.is_action_noise = False

        RLTask.__init__(self, name, env)
        # Extra info for TensorBoard
        self.extras = {}
        torch_zeros = lambda: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.episode_sums = {"success_rate": torch_zeros(), "ball_hole_XY_dist": torch_zeros()}
        return

    def set_up_scene(self, scene) -> None:

        franka_translation = torch.tensor([0.6, 0.0, 0.0])
        self.get_franka(franka_translation)
        self.get_table()
        self.get_ball()
        
        super().set_up_scene(scene)

        self._frankas = FrankaView(prim_paths_expr="/World/envs/.*/franka", name="franka_view")

        # Add ball
        self._ball = RigidPrimView(prim_paths_expr="/World/envs/.*/ball", name="ball_view", reset_xform_properties=False)

        # Add location_ball
        self._location_ball = RigidPrimView(prim_paths_expr="/World/envs/.*/table/table/location_ball", name="location_ball_view", reset_xform_properties=False)
        

        scene.add(self._frankas)
        scene.add(self._frankas._hands)
        scene.add(self._frankas._lfingers)
        scene.add(self._frankas._rfingers)
        scene.add(self._ball)
        scene.add(self._location_ball)
        
        self.init_data()
        return

    def get_franka(self, translation):

        franka = Franka(prim_path=self.default_zero_env_path + "/franka", name="franka", translation = translation)
        self._sim_config.apply_articulation_settings("franka", get_prim_at_path(franka.prim_path), self._sim_config.parse_actor_config("franka"))

    def get_table(self):
        table = Table(prim_path=self.default_zero_env_path + "/table", name="table")
        self._sim_config.apply_articulation_settings("table", get_prim_at_path(table.prim_path), self._sim_config.parse_actor_config("table"))

    def get_ball(self):

        ball = DynamicSphere(
                name = 'ball',
                position=self.ball_initial_position,
                orientation=self.ball_initial_orientation,
                prim_path=self.default_zero_env_path + "/ball",
                radius=self.ball_radius,
                color=np.array([1, 0, 0]),
                density = 100
            )

        self._sim_config.apply_articulation_settings("ball", get_prim_at_path(ball.prim_path), self._sim_config.parse_actor_config("ball"))

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

        finger_pose = torch.zeros(7, device=self._device)
        finger_pose[0:3] = (lfinger_pose[0:3] + rfinger_pose[0:3]) / 2.0
        finger_pose[3:7] = lfinger_pose[3:7]
        hand_pose_inv_rot, hand_pose_inv_pos = (tf_inverse(hand_pose[3:7], hand_pose[0:3]))

        grasp_pose_axis = 1
        franka_local_grasp_pose_rot, franka_local_pose_pos = tf_combine(hand_pose_inv_rot, hand_pose_inv_pos, finger_pose[3:7], finger_pose[0:3])
        franka_local_pose_pos += torch.tensor([0, 0.04, 0], device=self._device)
        self.franka_local_grasp_pos = franka_local_pose_pos.repeat((self._num_envs, 1))
        self.franka_local_grasp_rot = franka_local_grasp_pose_rot.repeat((self._num_envs, 1))

        self.gripper_forward_axis = torch.tensor([0, 0, 1], device=self._device, dtype=torch.float).repeat((self._num_envs, 1))
        self.gripper_up_axis = torch.tensor([0, 1, 0], device=self._device, dtype=torch.float).repeat((self._num_envs, 1))

        self.franka_default_dof_pos = torch.tensor(
            [1.157, -1.066, -0.155, -2.239, -1.841, 1.003, 0.469, 0.035, 0.035], device=self._device
        )

        self.actions = torch.zeros((self._num_envs, self.num_actions), device=self._device)

    def get_observations(self) -> dict:
        hand_pos, hand_rot = self._frankas._hands.get_world_poses(clone=False)
        franka_dof_pos = self._frankas.get_joint_positions(clone=False)
        franka_dof_vel = self._frankas.get_joint_velocities(clone=False)
        self.franka_dof_pos = franka_dof_pos

        # self.franka_grasp_rot, self.franka_grasp_pos, self.drawer_grasp_rot, self.drawer_grasp_pos = self.compute_grasp_transforms(
        #     hand_rot,
        #     hand_pos,
        #     self.franka_local_grasp_rot,
        #     self.franka_local_grasp_pos,
        #     drawer_rot,
        #     drawer_pos,
        #     self.drawer_local_grasp_rot,
        #     self.drawer_local_grasp_pos,
        # )

        self.franka_lfinger_pos, self.franka_lfinger_rot = self._frankas._lfingers.get_world_poses(clone=False)
        self.franka_rfinger_pos, self.franka_rfinger_rot = self._frankas._lfingers.get_world_poses(clone=False)


        # Ball 
        self.ball_pos, self.ball_rot = self._ball.get_world_poses(clone=False)
        self.ball_vel = self._ball.get_velocities()

        # hole-location ball
        self.location_ball_pos, self.location_ball_rot = self._location_ball.get_world_poses(clone=False)

        to_target = self.location_ball_pos - self.ball_pos

        dof_pos_scaled = (
            2.0
            * (franka_dof_pos - self.franka_dof_lower_limits)
            / (self.franka_dof_upper_limits - self.franka_dof_lower_limits)
            - 1.0
        )

        self.obs_buf = torch.cat(
            (
                dof_pos_scaled,
                franka_dof_vel * self.dof_vel_scale,
                self.ball_vel,
                to_target,
                self.ball_pos,
                # self.location_ball_pos
                # self.cabinet_dof_pos[:, 3].unsqueeze(-1),
                # self.cabinet_dof_vel[:, 3].unsqueeze(-1),
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

        self.franka_dof_targets[:,7] = 0.015
        self.franka_dof_targets[:,8] = 0.015

        self._frankas.set_joint_position_targets(self.franka_dof_targets, indices=env_ids_int32)

    def reset_idx(self, env_ids):
        indices = env_ids.to(dtype=torch.int32)
        num_indices = len(indices)

        # reset franka
        pos = tensor_clamp(
            self.franka_default_dof_pos.unsqueeze(0)
            + 0.25 * (torch.rand((len(env_ids), self.num_franka_dofs), device=self._device) - 0.5),
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
        
        # reset ball 
        # if not test, randomize ball initial positions for training
        if not self.is_test:

            # reset ball position: x [-0.1, 0.1], y [-0.1,0.1]
            self.new_ball_pos = self.default_ball_pos.clone().detach()
            self.new_ball_pos[:,0] = self.default_ball_pos[:,0] + (0.1 + 0.1) * torch.rand(self._num_envs, device=self._device) -0.1
            self.new_ball_pos[:,1] = self.default_ball_pos[:,1] + (0.1 + 0.1) * torch.rand(self._num_envs, device=self._device) -0.1

            self._ball.set_world_poses(self.new_ball_pos[env_ids], self.default_ball_rot[env_ids], indices = indices)
            self._ball.set_velocities(self.default_ball_velocity[env_ids], indices = indices)

        # if is test mode, set the ball to given position (1 environment)
        else:
            self.new_ball_pos = self.default_ball_pos.clone().detach()
            self.new_ball_pos[:,0] = self.default_ball_pos[:,0] + self.initial_test_value[0]
            self.new_ball_pos[:,1] = self.default_ball_pos[:,1] +self.initial_test_value[1]
            self._ball.set_world_poses(self.new_ball_pos[env_ids], self.default_ball_rot[env_ids], indices = indices)
            self._ball.set_velocities(self.default_ball_velocity[env_ids], indices = indices)

        # bookkeeping
        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0
        
        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            if key == "success_rate":
                self.extras["episode"][key] = torch.mean(self.episode_sums[key][env_ids])
            else:
                self.extras["episode"][key] = torch.mean(self.episode_sums[key][env_ids])/self._max_episode_length
            self.episode_sums[key][env_ids] = 0.

    def post_reset(self):

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

        # Ball
        self.default_ball_pos, self.default_ball_rot = self._ball.get_world_poses()
        self.default_ball_velocity = self._ball.get_velocities()

        # randomize all envs
        indices = torch.arange(self._num_envs, dtype=torch.int64, device=self._device)
        self.reset_idx(indices)

    def calculate_metrics(self) -> None:
        
        # get objects positions and orientations
        joint_positions = self.franka_dof_pos               # franka dofs pos
        num_envs = self._num_envs                           # num of sim env
        finger_pos = (self.franka_lfinger_pos + self.franka_lfinger_pos)/2    # franka finger pos (lfinger+rfinger)/2
        self.finger_pos = finger_pos
        gripper_forward_axis = self.gripper_forward_axis
        gripper_up_axis = self.gripper_up_axis
        # franka_grasp_pos = self.franka_grasp_pos
        # franka_grasp_rot = self.franka_grasp_rot
        # ball_grasp_pos = self.ball_grasp_pos
        # ball_grasp_rot = self.ball_grasp_rot
        # ball_inward_axis = self.ball_inward_axis
        # ball_up_axis = self.ball_up_axis
        # franka_dof_pos = self.franka_dof_pos
        ball_init_pos = self.default_ball_pos

        ball_pos = self.ball_pos                    # ball pos
        # ball_rot = self.ball_rot                    # ball rot
        # ball_vel = self._ball.get_velocities()      # ball velocity            

        # table_pos = self.table_pos              # table pos
        # table_rot = self.table_rot              # table rot
        hole_pos = self.location_ball_pos                    # locate hole pos
        # hole_pos[:,1] = hole_pos[:,1] - 0.8     # Y-axis
        # hole_pos[:,2] = hole_pos[:,2] + 0.44    # Z-axis

        # 1st reward: distance from ball to hole
        ball_hole_dist = torch.norm(hole_pos - ball_pos, p=2, dim=-1)
        ball_hole_XY_dist = torch.norm(hole_pos[:,0:2] - ball_pos[:,0:2], p=2, dim=-1)
        # dist_reward = 1.0 / (1.0 + ball_hole_dist ** 2)
        # dist_reward *= 2*dist_reward
        # dist_reward = torch.where(ball_hole_dist <= 0.05, dist_reward+10, dist_reward)

        # ball_hole_dist = torch.norm(hole_pos - ball_pos, p=2, dim=-1)
        # dist_reward = 1.0/(1.0+ball_hole_dist**2)
        dist_reward = 1-torch.tanh(3*ball_hole_XY_dist)      # regulize the dist_reward in [0,1]
        # dist_reward = -(ball_hole_XY_dist)**2

        # 2nd reward: distance from finger to ball 
        # finger_ball_dist = torch.norm(finger_pos - ball_pos, p=2, dim=-1)
        ball_to_init_dist = torch.norm(ball_pos[:,0:2] - ball_init_pos[:,0:2], p=2, dim=-1)
        self.ball_to_init_dist = ball_to_init_dist

        finger_ball_dist = torch.norm(finger_pos - ball_pos, p=2, dim=-1)
        finger_ball_reward = 1.0/(1.0+finger_ball_dist**2)


        # 1st penalty: regularization on the actions (summed for each environment)
        action_penalty = torch.sum(self.actions ** 2, dim=-1)
        action_penalty = 1-torch.tanh(action_penalty/2.5)

        # 5th penalty if ball is not moved
        ball_unmove_penalty = torch.zeros_like(dist_reward)
        ball_unmove_penalty = torch.where(ball_to_init_dist<0.3, torch.tanh(15*(0.3-ball_to_init_dist)), ball_unmove_penalty)

        falling_bonus = torch.where(torch.logical_and(ball_hole_XY_dist < 0.1 , ball_pos[:,2]<0.38), torch.ones_like(dist_reward), torch.zeros_like(dist_reward))

        falling_penalty = torch.zeros_like(dist_reward)
        falling_penalty = torch.where(torch.logical_and(ball_hole_XY_dist > 0.001 , ball_pos[:,2]<0.38), falling_penalty+10, falling_penalty)
        # falling_penalty = torch.where(ball_hole_XY_dist<0.2, falling_penalty-100, falling_penalty)

        # dist_reward = torch.where(ball_hole_XY_dist<0.3, 1-torch.tanh(10*ball_hole_XY_dist), dist_reward)        
        # dist_reward = torch.where(ball_to_init_dist>0.01, dist_reward, dist_reward*0)
        dist_reward = torch.where(ball_pos[:,0]<hole_pos[:,0], torch.zeros_like(dist_reward), dist_reward)

        dist_penalty = torch.tanh(3*ball_hole_XY_dist) 

        final_reward = 10.0*dist_reward - 0.0*ball_unmove_penalty + 100.0*falling_bonus - 0.0*action_penalty \
                        - 0.0*falling_penalty + 0.0*finger_ball_reward - 0.0*dist_penalty

        # final_reward = torch.where(finger_pos[:,2] < (ball_pos[:,2]), final_reward-0.5, final_reward)
        # final_reward = torch.where(torch.logical_and(finger_ball_dist > 0, ball_to_init_dist<0.05), final_reward-0.5, final_reward)
        # final_reward = torch.where(ball_hole_XY_dist>0.2, final_reward-1, final_reward)

        self.is_complete = torch.where(torch.logical_and(ball_hole_XY_dist < 0.01 , ball_pos[:,2]<0.38), torch.ones_like(final_reward), torch.zeros_like(final_reward))   # task complete
        # final_reward = torch.where(ball_hole_XY_dist < 0.6, final_reward+3.0*dist_reward, final_reward)

        self.rew_buf[:] = final_reward

        self.episode_sums["success_rate"] += self.is_complete 
        self.episode_sums["ball_hole_XY_dist"] += ball_hole_XY_dist 


    def is_done(self) -> None:

        if not self.is_test: 

            # reset if ball falls from the edge or in hole
            self.reset_buf = torch.where(self.ball_pos[:, 2] < 0.1, torch.ones_like(self.reset_buf), self.reset_buf)

            # self.reset_buf = torch.where(self.is_complete==1, torch.ones_like(self.reset_buf), self.reset_buf)

            # reset if franka grasp is below the ball and ball is not moved
            # self.reset_buf = torch.where(self.finger_pos[:, 2] < 0.2, torch.ones_like(self.reset_buf), self.reset_buf)
            # self.reset_buf = torch.where(torch.logical_and(self.finger_pos[:, 2] < 0.3, self.ball_to_init_dist < 0.1), torch.ones_like(self.reset_buf), self.reset_buf)

            # reset if max length reached
            self.reset_buf = torch.where(self.progress_buf >= self._max_episode_length - 1, torch.ones_like(self.reset_buf), self.reset_buf)

        else:

            self.reset_buf = torch.where(self.progress_buf >= self._max_episode_length - 1, torch.ones_like(self.reset_buf), self.reset_buf)
