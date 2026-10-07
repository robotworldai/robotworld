from omniisaacgymenvs.tasks.base.rl_task import RLTask
from Models.Franka.Franka import Franka
from Models.Franka.Franka_view import FrankaView
from Models.peg_in_hole.table import Table
from Models.peg_in_hole.tool import Tool

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


class FrankaPegInHoleTask(RLTask):
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

        self.location_ball_radius = self._task_cfg["env"]["locationBallRadius"]
        self.location_ball_initial_position = self._task_cfg["env"]["locationBallPosition"]
        self.location_ball_initial_orientation = self._task_cfg["env"]["locationBallInitialOrientation"]

        self.distX_offset = 0.04
        self.dt = 1/60.

        self._num_observations = 28
        self._num_actions = 9

        # Flag for testing
        self.is_test = False
        self.initial_test_value = None
        self.is_action_noise = False

        RLTask.__init__(self, name, env)

        # Extra info for TensorBoard
        self.extras = {}
        torch_zeros = lambda: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.episode_sums = {"tool_hole_XY_dist": torch_zeros(), "tool_hole_Z_dist": torch_zeros(), "tool_hole_dist": torch_zeros(), 
                            "tool_rot_error": torch_zeros(), "peg_rate": torch_zeros(), "norm_finger_vel": torch_zeros(),  "rewards": torch_zeros()}

        return

    def set_up_scene(self, scene) -> None:

        franka_translation = torch.tensor([0.5, 0.0, 0.0])
        self.get_franka(franka_translation)
        self.get_table()
        self.get_tool()
        
        super().set_up_scene(scene)

        # Add Franka
        self._frankas = FrankaView(prim_paths_expr="/World/envs/.*/franka", name="franka_view")

        # Add Table
        self._table = RigidPrimView(prim_paths_expr="/World/envs/.*/table/table/table_mesh",  name="table_view", reset_xform_properties=False)

        # Add Tool
        self._tool = RigidPrimView(prim_paths_expr="/World/envs/.*/tool/tool/tool",  name="tool_view", reset_xform_properties=False)

        # Add location_ball
        self._location_ball = RigidPrimView(prim_paths_expr="/World/envs/.*/table/table/location_ball", name="location_ball_view", reset_xform_properties=False)
        
        scene.add(self._frankas)
        scene.add(self._frankas._hands)
        scene.add(self._frankas._lfingers)
        scene.add(self._frankas._rfingers)
        scene.add(self._table)
        scene.add(self._tool)
        scene.add(self._location_ball)
        
        self.init_data()
        return

    def get_franka(self, translation):

        franka = Franka(prim_path=self.default_zero_env_path + "/franka", name="franka", translation = translation)
        self._sim_config.apply_articulation_settings("franka", get_prim_at_path(franka.prim_path), self._sim_config.parse_actor_config("franka"))

    def get_table(self):
        table = Table(prim_path=self.default_zero_env_path + "/table", name="table")
        self._sim_config.apply_articulation_settings("table", get_prim_at_path(table.prim_path), self._sim_config.parse_actor_config("table"))

    def get_tool(self):

        tool = Tool(prim_path=self.default_zero_env_path + "/tool", name="tool")
        self._sim_config.apply_articulation_settings("tool", get_prim_at_path(tool.prim_path), self._sim_config.parse_actor_config("tool"))

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

        # tool reference rotation
        self.tool_ref_rot = torch.tensor([0.5, 0.5, 0.5, 0.5], device=self._device)

        # self.gripper_forward_axis = torch.tensor([0, 0, 1], device=self._device, dtype=torch.float).repeat((self._num_envs, 1))
        # self.gripper_up_axis = torch.tensor([0, 1, 0], device=self._device, dtype=torch.float).repeat((self._num_envs, 1))

        # default franka pos: for initially grap the tool
        self.franka_default_dof_pos = torch.tensor(
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.015, 0.015], device=self._device
        )

        self.actions = torch.zeros((self._num_envs, self.num_actions), device=self._device)

    def get_observations(self) -> dict:
        hand_pos, hand_rot = self._frankas._hands.get_world_poses(clone=False)
        franka_dof_pos = self._frankas.get_joint_positions(clone=False)
        franka_dof_vel = self._frankas.get_joint_velocities(clone=False)
        self.franka_dof_pos = franka_dof_pos

        self.franka_lfinger_pos, self.franka_lfinger_rot = self._frankas._lfingers.get_world_poses(clone=False)
        self.franka_rfinger_pos, self.franka_rfinger_rot = self._frankas._lfingers.get_world_poses(clone=False)


        # Tool 
        self.tool_pos, self.tool_rot = self._tool.get_world_poses(clone=False)

        hole_pos, hole_rot = self._location_ball.get_world_poses()

        to_target = self.tool_pos - hole_pos

        dof_pos_scaled = (
            2.0
            * (franka_dof_pos - self.franka_dof_lower_limits)
            / (self.franka_dof_upper_limits - self.franka_dof_lower_limits)
            - 1.0
        )
        # print(torch.norm(to_target, p=2, dim=-1))
        self.obs_buf = torch.cat(
            (
                dof_pos_scaled,
                franka_dof_vel * self.dof_vel_scale,
                self.tool_pos,
                self.tool_rot,
                to_target
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

        # NOTE HERE: right now I fix the finger movement so that the object will always be grasped in hand
        # Later: if the reward is good enough, the hand should be released once the object is in the hole,
        # this means the last two dofs are also in the action
        # self.franka_dof_targets[:,7] = self.franka_default_dof_pos[7]
        # self.franka_dof_targets[:,8] = self.franka_default_dof_pos[8]

        # release the finger if tool is right above the hole
        hole_pos, hole_rot = self._location_ball.get_world_poses()
        tool_pos, tool_rot = self._tool.get_world_poses()
        hole_pos[:,2] = 0.39
        tool_hole_dist = torch.norm(tool_pos - hole_pos, p=2, dim=-1)
        tool_hole_XY_dist = torch.norm(tool_pos[:,0:2] - hole_pos[:,0:2], p=2, dim=-1)
        tool_hole_Z_dist = torch.norm(tool_pos[:,2] - hole_pos[:,2], p=2, dim=-1)
        tool_rot_error = torch.norm(tool_rot - self.tool_ref_rot, p=2, dim=-1)

        # self.release_condition = torch.logical_and(tool_hole_XY_dist <= 0.1, tool_rot_error<=1)
        # self.release_condition = torch.logical_and(self.release_condition, tool_hole_Z_dist<=0.1)

        # self.release_condition = torch.logical_and(tool_hole_dist<0.08, self.is_released)

        self.release_condition = tool_hole_dist<=0.024   

        # self.release_condition = torch.logical_and(tool_hole_XY_dist<=0.04, tool_hole_Z_dist<=0.07)
        # self.release_condition = torch.logical_and(self.release_condition, tool_rot_error<=1)

        # self.is_released = self.release_condition.clone().detach()

        self.franka_dof_targets[:,7] = torch.where(self.release_condition, 0.1, 0.015)
        self.franka_dof_targets[:,8] = torch.where(self.release_condition, 0.1, 0.015)

        # set franka target joint position
        self._frankas.set_joint_position_targets(self.franka_dof_targets, indices=env_ids_int32)

    def reset_idx(self, env_ids):
        indices = env_ids.to(dtype=torch.int32)
        num_indices = len(indices)

        # reset franka (due to initial grasping, cannot randomize)
        pos = torch.clamp(
            self.franka_default_dof_pos.unsqueeze(0),
            # + 0.25 * (torch.rand((len(env_ids), self.num_franka_dofs), device=self._device) - 0.5),
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
            # reset table
            # reset positions: x: [-0.2,0.2], y:[-0.2,0.2]
            random_x = (0.2 + 0.2) * torch.rand(self._num_envs, device=self._device) -0.2
            random_y = (0.2 + 0.2) * torch.rand(self._num_envs, device=self._device) -0.2
            self.new_table_pos = self.default_table_pos.clone().detach()
            self.new_table_pos[:,0] = self.default_table_pos[:,0] + random_x
            self.new_table_pos[:,1] = self.default_table_pos[:,1] + random_y
            self._table.set_world_poses(self.new_table_pos[env_ids], self.default_table_rot[env_ids], indices = indices)
            self._table.set_velocities(self.default_table_velocity[env_ids], indices = indices)

        else:

            self.new_table_pos = self.default_table_pos.clone().detach()
            self.new_table_pos[:,0] = self.default_table_pos[:,0] + self.initial_test_value[0]
            self.new_table_pos[:,1] = self.default_table_pos[:,1] + self.initial_test_value[1]
            self._table.set_world_poses(self.new_table_pos[env_ids], self.default_table_rot[env_ids], indices = indices)
            self._table.set_velocities(self.default_table_velocity[env_ids], indices = indices)


        self.is_released = torch.zeros((1,self._num_envs), device=self._device)
    
        # bookkeeping
        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0

        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            if key == "peg_rate":
                self.extras["episode"][key] = torch.mean(self.episode_sums[key][env_ids])
            else:
                self.extras["episode"][key] = torch.mean(self.episode_sums[key][env_ids]) / self._max_episode_length
            self.episode_sums[key][env_ids] = 0
        

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

        # tool
        self.default_tool_pos, self.default_tool_rot = self._tool.get_world_poses()
        self.default_tool_velocity = self._tool.get_velocities()

        # table
        self.default_table_pos, self.default_table_rot = self._table.get_world_poses()
        self.default_table_velocity = self._table.get_velocities()
        

        # randomize all envs
        indices = torch.arange(self._num_envs, dtype=torch.int64, device=self._device)
        self.reset_idx(indices)

    def calculate_metrics(self) -> None:
        # Envoroonment parameters:
        # table height: 0.4
        # hole depth: 0.05
        # hole radius: 0.01
        # tool at surface: Z = 0.43
        # tool pegged in hole: Z = 0.38
        # tool_pos to tool bottom: Z = 0.03
        # tool body length: 0.06
        # tool cap length: 0.01
        # tool vertical orient: [0.5, 0.5, 0.5, 0.5]

        # tool_ref_rot = self.tool_ref_rot  # tool reference vertical rotation
        num_envs = self._num_envs
        tool_pos, tool_rot = self._tool.get_world_poses(clone=False)
        hole_pos, hole_rot = self._location_ball.get_world_poses(clone=False)
        hole_pos[:,2] = 0.38  # fix hole pos
        hole_surf_pos = hole_pos.clone().detach()
        hole_surf_pos[:,2] = hole_surf_pos[:,2]
        hole_target_pos = hole_pos.clone().detach()
        hole_target_pos[:,2] = 0.39
        # tool_ref_rot = torch.zeros_like(tool_rot)
        # tool_ref_rot[:,:] = self.tool_ref_rot       # tool reference vertical rotation
        tool_ref_rot= self.tool_ref_rot  

        lfinger_pos, lfinger_rot = self._frankas._lfingers.get_world_poses(clone=False)
        rfinger_pos, rfinger_rot = self._frankas._rfingers.get_world_poses(clone=False)

        finger_rot = (lfinger_rot + rfinger_rot)/2 
        finger_pos = (lfinger_pos + rfinger_pos)/2 
        finger_rot_ref = torch.tensor([0.0325, -0.3824,  0.9233, -0.0135], device=self._device)

        # finger velocity
        lfinger_vel = self._frankas._lfingers.get_velocities()
        rfinger_vel = self._frankas._rfingers.get_velocities()
        finger_vel = (lfinger_vel[:,0:3]+rfinger_vel[:,0:3])/2
        norm_finger_vel = torch.norm(finger_vel, p=2, dim=-1)  
        

        # direction vector
        ref_vector = torch.zeros([num_envs,3], device=self._device)
        ref_vector[:,0] = 2*(tool_ref_rot[0]*tool_ref_rot[2] - tool_ref_rot[3]*tool_ref_rot[1])
        ref_vector[:,1] = 2*(tool_ref_rot[1]*tool_ref_rot[2] + tool_ref_rot[3]*tool_ref_rot[0])
        ref_vector[:,2] = 1 - 2*(tool_ref_rot[0]*tool_ref_rot[0] + tool_ref_rot[1]*tool_ref_rot[1])

        tool_vector = torch.zeros([num_envs,3], device=self._device)
        tool_vector[:,0] = 2*(tool_rot[:,0]*tool_rot[:,2] - tool_rot[:,3]*tool_rot[:,1])
        tool_vector[:,1] = 2*(tool_rot[:,1]*tool_rot[:,2] + tool_rot[:,3]*tool_rot[:,0])
        tool_vector[:,2] = 1 - 2*(tool_rot[:,0]*tool_rot[:,0] + tool_rot[:,1]*tool_rot[:,1])

        # roll  = atan2(2.0 * (q.q3 * q.q2 + q.q0 * q.q1) , 1.0 - 2.0 * (q.q1 * q.q1 + q.q2 * q.q2));
        # pitch = asin(2.0 * (q.q2 * q.q0 - q.q3 * q.q1));
        # yaw   = atan2(2.0 * (q.q3 * q.q0 + q.q1 * q.q2) , - 1.0 + 2.0 * (q.q0 * q.q0 + q.q1 * q.q1));

        tool_roll = torch.atan2(2.0*(tool_rot[:,0]*tool_rot[:,1] + tool_rot[:,2]*tool_rot[:,3]), 1.0-2.0*(tool_rot[:,2]*tool_rot[:,2]+tool_rot[:,1]*tool_rot[:,1]))
        tool_yaw= torch.atan2(2.0*(tool_rot[:,3]*tool_rot[:,2] + tool_rot[:,0]*tool_rot[:,1]), 1.0-2.0*(tool_rot[:,1]*tool_rot[:,1]+tool_rot[:,2]*tool_rot[:,2]))
        tool_pitch = torch.asin(2.0*(tool_rot[:,0]*tool_rot[:,2] - tool_rot[:,1]*tool_rot[:,3]))

        tool_ref_roll = torch.atan2(2.0*(tool_ref_rot[0]*tool_ref_rot[1] + tool_ref_rot[2]*tool_ref_rot[3]), 1.0-2.0*(tool_ref_rot[2]*tool_ref_rot[2]+tool_ref_rot[1]*tool_ref_rot[1]))
        tool_ref_yaw = torch.atan2(2.0*(tool_ref_rot[3]*tool_ref_rot[2] + tool_ref_rot[0]*tool_ref_rot[1]), 1.0-2.0*(tool_ref_rot[1]*tool_ref_rot[1]+tool_ref_rot[2]*tool_ref_rot[2]))
        tool_ref_pitch = torch.asin(2.0*(tool_ref_rot[0]*tool_ref_rot[2] - tool_ref_rot[1]*tool_ref_rot[3]))

        tool_roll_error = torch.abs(tool_roll - tool_ref_roll)
        tool_pitch_error = torch.abs(tool_pitch - tool_ref_pitch)

        tool_roll_pitch_reward = 1 - torch.tanh(2*tool_roll_error) + 1 - torch.tanh(2*tool_pitch_error)
        # tool_roll_yaw_reward = 1 - torch.tanh(2*tool_roll_error) + 1 - torch.tanh(2*tool_yaw_error)
        
        # Handle Nan exception
        # tool_roll_pitch_reward = torch.where(torch.isnan(tool_roll_error+tool_pitch_error), torch.ones_like(tool_roll_pitch_reward), tool_roll_pitch_reward)

        # 1st reward: tool XY position
        tool_hole_dist = torch.norm(tool_pos - hole_pos, p=2, dim=-1)
        tool_hole_XY_dist = torch.norm(tool_pos[:,0:2] - hole_pos[:,0:2], p=2, dim=-1)
        # tool_XY_pos_reward = 1.0 / (1.0 + (tool_hole_XY_dist) ** 2)
        tool_XY_pos_reward = 1 - torch.tanh(5*tool_hole_XY_dist)

        tool_hole_surf_dist = torch.norm(tool_pos - hole_surf_pos, p=2, dim=-1)
        # tool_surf_pos_reward = 1.0 / (1.0 + (tool_hole_surf_dist) ** 2)
        tool_surf_pos_reward = 1 - torch.tanh(8*tool_hole_surf_dist)

        # 2nd reward: tool rotation
        # tool_rot_error = torch.norm(tool_rot - tool_ref_rot, p=2, dim=-1)
        tool_rot_error = torch.norm(tool_vector - ref_vector, p=2, dim=-1)
        # tool_rot_reward = 1.0 / (1.0 + (tool_rot_error) ** 2)
        tool_rot_reward = 1 - torch.tanh(3*tool_rot_error)
        self.rot_error = tool_roll_error + tool_pitch_error


        # 3rd reward: pegging in when tool is above the hole
        tool_hole_Z_dist = torch.abs(tool_pos[:,2] - hole_pos[:,2])
        # tool_pegging_reward = 1.0 / (1.0 + (tool_hole_Z_dist) ** 2)
        tool_pegging_reward = 1 - torch.tanh(6*tool_hole_Z_dist)

        # 4th reward: tool hole XYZ position
        tool_hole_dist = torch.norm(tool_pos - hole_pos, p=2, dim=-1)
        tool_target_dist = torch.norm(tool_pos - hole_target_pos, p=2, dim=-1)
        # tool_pos_reward = 1.0 / (1.0 + (tool_hole_dist) ** 2)
        tool_pos_reward = 1 - torch.tanh(5*tool_hole_dist)

        finger_rot_error = torch.norm(finger_rot - finger_rot_ref, p=2, dim=-1) 
        finger_rot_reward = 1.0 / (1.0 + (finger_rot_error) ** 2)

        finger_XY_pos_dist = torch.norm(finger_pos[:,0:2] - hole_pos[:,0:2], p=2, dim=-1) 
        finger_pos_reward = 1 - torch.tanh(5*finger_XY_pos_dist)

        # 1st penalty: action
        action_penalty = torch.sum(self.actions[:,0:7] ** 2, dim=-1)
        action_penalty = 1 - 1.0 / (1.0 + action_penalty)

        finger_vel_penalty = torch.tanh(20*torch.abs(norm_finger_vel-0.1))

        # tool_rot_penalty = 1 - 1.0 / (1.0 + (tool_rot_error) ** 2)
        # tool_pos_penalty = 1 - 1.0 / (1.0 + (tool_hole_dist) ** 2)

        # final cumulative reward
        # final_reward = 5*tool_XY_pos_reward + 5*tool_rot_reward + 2*tool_pegging_reward- 0.001*action_penalty
        # final_reward = 10*tool_surf_pos_reward + 5*tool_rot_reward + 0*tool_hole_XY_dist- 0.001*action_penalty - 1.0*tool_rot_penalty - 1.0*tool_pos_penalty
        # final_reward = torch.where(tool_hole_surf_dist<0.05, 10*tool_pos_reward + 5*tool_rot_reward- 0.001*action_penalty, final_reward)

        # final_reward = torch.where(tool_hole_dist<0.1, 1*tool_pos_reward + 3*tool_rot_reward , 3*tool_pos_reward + 1*tool_rot_reward)

        # final_reward = 2*tool_surf_pos_reward + 2*tool_rot_reward + 0*finger_rot_reward - 0.001*action_penalty

        # final_reward = torch.where(tool_surf_pos_reward<0.1, 2*tool_pos_reward + 2*tool_rot_reward + 0*finger_rot_reward + 2*tool_pegging_reward-0.001*action_penalty, final_reward)

        final_reward = 3.5*tool_XY_pos_reward + 1.48*tool_roll_pitch_reward- 0.001*action_penalty + 2.0*tool_pegging_reward

        final_reward = torch.where((self.rot_error)<0.08, final_reward+0.5, final_reward)
        final_reward = torch.where((self.rot_error)>0.2, final_reward-1, final_reward)

        final_reward = torch.where(tool_hole_Z_dist>0.15, final_reward-1, final_reward)
        final_reward = torch.where(tool_hole_Z_dist<0.05, final_reward+0.1, final_reward)


        final_reward = torch.where(tool_hole_XY_dist<0.05, final_reward+0.5, final_reward)
        final_reward = torch.where(tool_hole_XY_dist>0.1, final_reward-10, final_reward)

        final_reward = torch.where(norm_finger_vel>0.15, final_reward-1, final_reward)


        # amplify different sub-rewards w.r.t. conditions
        # final_reward = torch.where(tool_hole_XY_dist>=0.005, final_reward + 2*tool_XY_pos_reward, final_reward)     # tool-hole XY position 
        # final_reward = torch.where(tool_rot_error > 0.05, final_reward + 2*tool_rot_reward, final_reward)           # tool rotation position 
        # final_reward = torch.where(torch.logical_and(tool_hole_XY_dist<0.05, tool_rot_error<0.05), final_reward + 10*tool_pegging_reward+2*tool_rot_reward, final_reward) # tool-hole Z position 
        # final_reward = torch.where(torch.logical_and(tool_hole_surf_dist<0.05, tool_rot_error<0.06), 
        #                                         10*tool_pos_reward + 5*tool_rot_reward + 2*tool_pegging_reward- 0.001*action_penalty, 
        #                                         final_reward) # tool-hole Z position 
            
        # extra bonus/penalty cases:
        # final_reward = torch.where(tool_hole_XY_dist<=0.01, final_reward+0.1, final_reward)  # tool-hole XY position bonus
        # final_reward = torch.where(tool_rot_error <0.1, final_reward+0.01, final_reward)  
        # final_reward = torch.where(tool_hole_XY_dist <0.005, final_reward+0.01, final_reward)  
        # final_reward = torch.where(tool_hole_Z_dist <0.1, final_reward+0.02, final_reward)  

        # final_reward = 10*tool_pos_reward + 4*tool_rot_reward
        # final_reward = torch.where(tool_hole_XY_dist>0.1, 5.0*tool_pos_reward + 1.0*tool_rot_reward, 1.0*tool_pos_reward + 5.0*tool_rot_reward)

        # final_reward = torch.where(tool_rot_error<0.1, final_reward+2*tool_pos_reward, final_reward)  
        # final_reward = torch.where(tool_hole_XY_dist<0.05, final_reward+5*tool_rot_reward, final_reward)

        # final_reward = torch.where(tool_rot_error <0.1, final_reward+0.2, final_reward)
        # final_reward = torch.where(tool_hole_XY_dist <0.1, final_reward+0.5, final_reward) 
        # final_reward = torch.where(torch.logical_and(tool_hole_Z_dist <0.15, tool_hole_XY_dist <0.1), final_reward+1, final_reward) 

        # final_reward = torch.where(torch.logical_and(tool_hole_XY_dist<=0.005, tool_hole_Z_dist<=0.005), final_reward+10000, final_reward)   # task complete
        final_reward = torch.where(tool_target_dist<0.01, final_reward+100, final_reward)   # task complete


        final_reward = torch.where(torch.isnan(final_reward), torch.zeros_like(final_reward), final_reward)   # task complete

        # trigger to determine if job is done
        self.is_pegged = torch.where(tool_target_dist<0.01, torch.ones_like(final_reward), torch.zeros_like(final_reward))   # task complete

        self.rew_buf[:] = final_reward

        # print("hole_Z_pos", hole_pos[:2])
        # print("tool_Z_pos", tool_pos[:2])
        # print("tool_hole_XY_dist", tool_hole_XY_dist)
        # print("tool_hole_Z_dist", tool_hole_Z_dist)
        # print("tool_target_dist", tool_target_dist)
        # print("hole_surf_pos", hole_surf_pos)
        # print("norm_finger_vel", norm_finger_vel)

        # print("tool_rot", tool_rot)
        # print("tool_rot_error", self.rot_error )
        # print("tool_ref_rot", tool_ref_rot)
        # print("hole_rot", hole_rot)
        # print("finger_rot", finger_rot)
        # finger_rot_ref: 0.0325, -0.3824,  0.9233, -0.0135
        # 0.0     0.92388   0.3826  0

        # hole_pos tensor([[ 1.5000,  0.0000,  0.3800], [-1.5000,  0.0000,  0.3800]], device='cuda:0')
        # tool_hole_Z_dist tensor([0.0820, 0.0789], device='cuda:0')
        # tool_rot_error tensor([0.0629, 0.0621], device='cuda:0')
        # tool_hole_XY_dist tensor([0.0012, 0.0037], device='cuda:0')

        # tool_rot_error tensor([0.7979, 0.7810, 0.7889, 0.7811], device='cuda:0')
        # tool_hole_XY_dist tensor([0.0536, 0.0585, 0.0378, 0.0451], device='cuda:0')
        # tool_hole_Z_dist tensor([0.0343, 0.0353, 0.0368, 0.0350], device='cuda:0')
        # tool_hole_dist tensor([0.0636, 0.0683, 0.0528, 0.0571], device='cuda:0')

        self.episode_sums["tool_hole_XY_dist"] += tool_hole_XY_dist
        self.episode_sums["tool_hole_Z_dist"] += tool_hole_Z_dist
        self.episode_sums["tool_hole_dist"] += tool_hole_dist
        self.episode_sums["tool_rot_error"] += tool_roll_error+tool_pitch_error
        # self.episode_sums["tool_X_pos"] += tool_pos[:,0]
        # self.episode_sums["tool_Y_pos"] += tool_pos[:,1]
        # self.episode_sums["tool_Z_pos"] += tool_pos[:,2]
        # self.episode_sums["tool_rot"] += tool_rot
        self.episode_sums["peg_rate"] += self.is_pegged
        self.episode_sums["norm_finger_vel"] += norm_finger_vel
        self.episode_sums["rewards"] += final_reward


    def is_done(self) -> None:

        if not self.is_test: 
            # reset if tool is pegged in hole
            # self.reset_buf = torch.where(self.is_pegged==1, torch.ones_like(self.reset_buf), self.reset_buf)

            # reset if tool is below the table and not pegged in hole
            # self.reset_buf = torch.where(self.tool_pos[:,2] < 0.3, torch.ones_like(self.reset_buf), self.reset_buf)

            # 
            # self.reset_buf = torch.where(torch.logical_and(self.tool_pos[:,2] < 0.43, self.rot_error>1.5), torch.ones_like(self.reset_buf), self.reset_buf)

            # reset if max length reached
            self.reset_buf = torch.where(self.progress_buf >= self._max_episode_length - 1, torch.ones_like(self.reset_buf), self.reset_buf)

        else:

            self.reset_buf = torch.where(self.progress_buf >= self._max_episode_length - 1, torch.ones_like(self.reset_buf), self.reset_buf)
            # self.reset_buf = torch.where(self.is_pegged==1, torch.ones_like(self.reset_buf), self.reset_buf)
