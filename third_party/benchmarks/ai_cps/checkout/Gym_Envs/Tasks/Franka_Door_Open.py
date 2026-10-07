from omniisaacgymenvs.tasks.base.rl_task import RLTask
from Models.Franka.Franka import Franka
from Models.Franka.Franka_view import FrankaView
from Models.door_open.door import Door
from Models.door_open.door_view import DoorView

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


class FrankaDoorOpenTask(RLTask):
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
        self.episode_sums = {"door_yaw_deg": torch_zeros(), "grasp_handle_dist": torch_zeros(), "handle_yaw_deg": torch_zeros(), 
                            "handle_pos_error": torch_zeros(), "open_rate": torch_zeros(), "rewards": torch_zeros(), "handle_yaw_error": torch_zeros()}

        return

    def set_up_scene(self, scene) -> None:

        franka_translation = torch.tensor([0.5, 0.0, 0.0])
        self.get_franka(franka_translation)
        self.get_door()

        super().set_up_scene(scene)

        # Add Franka
        self._frankas = FrankaView(prim_paths_expr="/World/envs/.*/franka", name="franka_view")

        # Add door
        self._door = DoorView(prim_paths_expr="/World/envs/.*/door/door", name="door_view")

        scene.add(self._frankas)
        scene.add(self._frankas._hands)
        scene.add(self._frankas._lfingers)
        scene.add(self._frankas._rfingers)
        scene.add(self._door)
        scene.add(self._door._handle)
        
        self.init_data()
        return

    def get_franka(self, translation):

        franka = Franka(prim_path=self.default_zero_env_path + "/franka", name="franka", translation = translation)
        self._sim_config.apply_articulation_settings("franka", get_prim_at_path(franka.prim_path), self._sim_config.parse_actor_config("franka"))

    def get_door(self):
        door = Door(prim_path=self.default_zero_env_path + "/door", name="door")
        self._sim_config.apply_articulation_settings("door", get_prim_at_path(door.prim_path), self._sim_config.parse_actor_config("door"))

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

        # XXX assume to be the local pos of the handle
        door_local_handle_pose = torch.tensor([-0.1, -0.23, 0.81, 1.0, 0.0, 0.0, 0.0], device=self._device)
        self.door_local_handle_pos = door_local_handle_pose[0:3].repeat((self._num_envs, 1))
        self.door_local_handle_rot = door_local_handle_pose[3:7].repeat((self._num_envs, 1))

        self.gripper_forward_axis = torch.tensor([0, 0, 1], device=self._device, dtype=torch.float).repeat((self._num_envs, 1))
        self.gripper_up_axis = torch.tensor([0, 1, 0], device=self._device, dtype=torch.float).repeat((self._num_envs, 1))
        self.door_inward_axis = torch.tensor([-1, 0, 0], device=self._device, dtype=torch.float).repeat((self._num_envs, 1))
        self.door_up_axis = torch.tensor([0, 0, 1], device=self._device, dtype=torch.float).repeat((self._num_envs, 1))


        self.franka_default_dof_pos = torch.tensor(
            [1.157, -1.066, -0.155, -2.239, -1.841, 1.003, 0.469, 0.035, 0.035], device=self._device
        )

        self.actions = torch.zeros((self._num_envs, self.num_actions), device=self._device)

    def get_observations(self) -> dict:

        # Franka
        hand_pos, hand_rot = self._frankas._hands.get_world_poses(clone=False)
        self.door_pos, self.door_rot = self._door.get_world_poses(clone=False)
        franka_dof_pos = self._frankas.get_joint_positions(clone=False)
        franka_dof_vel = self._frankas.get_joint_velocities(clone=False)
        self.franka_dof_pos = franka_dof_pos
        self.door_dof_pos = self._door.get_joint_positions(clone=False)
        self.door_dor_vel = self._door.get_joint_velocities(clone=False)

        self.franka_grasp_rot, self.franka_grasp_pos, self.door_handle_rot, self.door_handle_pos = self.compute_grasp_transforms(
            hand_rot,
            hand_pos,
            self.franka_local_grasp_rot,
            self.franka_local_grasp_pos,
            self.door_rot,
            self.door_pos,
            self.door_local_handle_rot,
            self.door_local_handle_pos,
        )
        self.franka_lfinger_pos, self.franka_lfinger_rot = self._frankas._lfingers.get_world_poses(clone=False)
        self.franka_rfinger_pos, self.franka_rfinger_rot = self._frankas._lfingers.get_world_poses(clone=False)


        # handle 
        self.handle_pos, self.handle_rot = self._door._handle.get_world_poses(clone=False)
        self.handle_pos[:,1] = self.handle_pos[:,1] - 0.3 # fix hand-point y-axis error

        # position error: from franka grasp to door handle
        grasp_handle_pos_error = self.handle_pos - self.franka_grasp_pos
        # grasp_handle_pos_error = self.handle_pos - (self.franka_lfinger_pos + self.franka_rfinger_pos)/2

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
                self.handle_pos,
                self.handle_rot,
                grasp_handle_pos_error,
                # self.handle_pos,
                # self.handle_rot,
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

    def compute_grasp_transforms(
        self,
        hand_rot,
        hand_pos,
        franka_local_grasp_rot,
        franka_local_grasp_pos,
        door_rot,
        door_pos,
        door_local_handle_rot,
        door_local_handle_pos,
    ):

        global_franka_rot, global_franka_pos = tf_combine(
            hand_rot, hand_pos, franka_local_grasp_rot, franka_local_grasp_pos
        )
        global_door_rot, global_door_pos = tf_combine(
            door_rot, door_pos, door_local_handle_rot, door_local_handle_pos
        )

        return global_franka_rot, global_franka_pos, global_door_rot, global_door_pos

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

        self._frankas.set_joint_position_targets(self.franka_dof_targets, indices=env_ids_int32)

    def reset_idx(self, env_ids):
        indices = env_ids.to(dtype=torch.int32)
        num_indices = len(indices)

        # reset franka
        pos = torch.clamp(
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
        
        if not self.is_test:
            # reset door: only 1 joint
            # reset door positions: x: [-0.1,0.1], y:[-0.4,0.4]
            self.new_door_pos = self.default_door_pos.clone().detach()
            self.new_door_pos[:,0] = self.default_door_pos[:,0] + (0.05 + 0.05) * torch.rand(self._num_envs, device=self._device) -0.05
            self.new_door_pos[:,1] = self.default_door_pos[:,1] + (0.1 + 0.1) * torch.rand(self._num_envs, device=self._device) -0.1
            self._door.set_world_poses(self.new_door_pos[env_ids], self.default_door_rot[env_ids], indices = indices)
        
        else:

            self.new_door_pos = self.default_door_pos.clone().detach()
            self.new_door_pos[:,0] = self.default_door_pos[:,0] + self.initial_test_value[0]
            self.new_door_pos[:,1] = self.default_door_pos[:,1] + self.initial_test_value[1]
            self._door.set_world_poses(self.new_door_pos[env_ids], self.default_door_rot[env_ids], indices = indices)
        
        # reset door joints
        door_pos = torch.zeros((num_indices, 1), device=self._device)
        door_vel = torch.zeros((num_indices, 1), device=self._device)
        self._door.set_joint_positions(door_pos, indices=indices)
        self._door.set_joint_velocities(door_vel, indices=indices)
        self._door.set_joint_position_targets(self.door_dof_targets[env_ids], indices=indices)

        # bookkeeping
        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0
        
        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            if key == "open_rate":
                self.extras["episode"][key] = torch.mean(self.episode_sums[key][env_ids])
            else:
                self.extras["episode"][key] = torch.mean(self.episode_sums[key][env_ids]) / self._max_episode_length
            self.episode_sums[key][env_ids] = 0.0
        
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

        # Door
        self.door_dof_targets = torch.zeros(
            (self._num_envs, 1), dtype=torch.float, device=self._device
        )
        self.default_door_pos, self.default_door_rot = self._door.get_world_poses()


        # randomize all envs
        indices = torch.arange(self._num_envs, dtype=torch.int64, device=self._device)
        self.reset_idx(indices)

    def calculate_metrics(self) -> None:

        # info extraction
        # env
        num_envs = self._num_envs

        # Franka
        joint_positions = self.franka_dof_pos
        gripper_forward_axis = self.gripper_forward_axis
        gripper_up_axis = self.gripper_up_axis
        franka_grasp_pos, franka_grasp_rot = self.franka_grasp_pos, self.franka_grasp_rot
        franka_lfinger_pos, franka_lfinger_rot = self.franka_lfinger_pos, self.franka_lfinger_rot
        franka_rfinger_pos, franka_rfinger_rot = self.franka_rfinger_pos, self.franka_rfinger_rot
        actions = self.actions
        finger_pos = (franka_lfinger_pos + franka_rfinger_pos)/2
        finger_rot = (franka_lfinger_pos + franka_rfinger_pos)/2

        # door
        door_inward_axis = self.door_inward_axis
        door_up_axis = self.door_up_axis
        door_dof_pos = self.door_dof_pos
        door_pos, door_rot = self.door_pos, self.door_rot

        # handle
        handle_pos, handle_rot = self.handle_pos, self.handle_rot
        # handle_pos[:,1] = handle_pos[:,1] - 0.3 # fix hand-point y-axis error
        handle_local_pos, handle_local_rot = self._door._handle.get_local_poses()

        # preprocessing
        # distance from grasp to handle
        grasp_handle_dist = torch.norm(finger_pos - handle_pos, p=2, dim=-1)

        # distance of each finger to the handle along Z-axis
        lfinger_Z_dist = torch.abs(franka_lfinger_pos[:, 2] - handle_pos[:, 2])
        rfinger_Z_dist = torch.abs(franka_rfinger_pos[:, 2] - handle_pos[:, 2])

        # how far the door has been opened out
        # quaternions to euler angles      
        door_yaw = torch.atan2(2.0*(door_rot[:,0]*door_rot[:,3] + door_rot[:,1]*door_rot[:,2]), 1.0-2.0*(door_rot[:,2]*door_rot[:,2]+door_rot[:,3]*door_rot[:,3]))
        handle_yaw = torch.atan2(2.0*(handle_rot[:,0]*handle_rot[:,3] + handle_rot[:,1]*handle_rot[:,2]), 1.0-2.0*(handle_rot[:,2]*handle_rot[:,2]+handle_rot[:,3]*handle_rot[:,3]))
        door_ref_yaw = torch.deg2rad(torch.tensor([60], device=self._device))
        door_yaw_error = torch.abs(door_ref_yaw - handle_yaw)
        self.door_yaw_error = door_yaw_error.clone().detach()

        # handle destination if opened
        handle_ref_pos = handle_pos.clone().detach()
        # target_open_deg = door_ref_yaw*torch.ones((num_envs,1), device=self._device)        # open the door by 60 degrees
        # target_open_rad = math.radians(60)
        handle_ref_pos[:,0] = handle_ref_pos[:,0]*torch.cos(door_ref_yaw) + handle_ref_pos[:,1]*torch.sin(door_ref_yaw)
        handle_ref_pos[:,1] = -handle_ref_pos[:,0]*torch.sin(door_ref_yaw) + handle_ref_pos[:,1]*torch.cos(door_ref_yaw)
        self.handle_pos_error = torch.norm(handle_ref_pos - handle_pos, p=2, dim=-1)


        # gripper direction alignment 
        axis1 = tf_vector(franka_grasp_rot, gripper_forward_axis)
        axis2 = tf_vector(handle_rot, door_inward_axis)
        axis3 = tf_vector(franka_grasp_rot, gripper_up_axis)
        axis4 = tf_vector(handle_rot, door_up_axis)

        dot1 = torch.bmm(axis1.view(num_envs, 1, 3), axis2.view(num_envs, 3, 1)).squeeze(-1).squeeze(-1)  # alignment of forward axis for gripper
        dot2 = torch.bmm(axis3.view(num_envs, 1, 3), axis4.view(num_envs, 3, 1)).squeeze(-1).squeeze(-1)  # alignment of up axis for gripper

        # reward functions
        # 1st rewards: distance from hand to the drawer
        grasp_dist_reward = 1.0 / (1.0 + grasp_handle_dist ** 2)
        grasp_dist_reward *= grasp_dist_reward
        grasp_dist_reward = torch.where(grasp_handle_dist <= 0.02, grasp_dist_reward * 2, grasp_dist_reward)


        # 2nd reward for matching the orientation of the hand to the drawer (fingers wrapped)
        rot_reward = 0.5 * (torch.sign(dot1) * dot1 ** 2 + torch.sign(dot2) * dot2 ** 2)

        # 3rd reward: bonus if left finger is above the drawer handle and right below
        around_handle_reward = torch.zeros_like(rot_reward)
        around_handle_reward = torch.where(self.franka_lfinger_pos[:, 2] > handle_pos[:, 2],
                                           torch.where(self.franka_rfinger_pos[:, 2] < handle_pos[:, 2],
                                                       around_handle_reward + 0.5, around_handle_reward), around_handle_reward)

        # 4th reward: distance of each finger from the handle
        finger_dist_reward = torch.zeros_like(rot_reward)
        finger_dist_reward = torch.where(franka_lfinger_pos[:, 2] > handle_pos[:, 2],
                                         torch.where(franka_rfinger_pos[:, 2] < handle_pos[:, 2],
                                                     (0.04 - lfinger_Z_dist) + (0.04 - rfinger_Z_dist), finger_dist_reward), finger_dist_reward)

        # 5th reward: finger closeness
        finger_close_reward = torch.zeros_like(rot_reward)
        finger_close_reward = torch.where(grasp_handle_dist <=0.03, (0.04 - joint_positions[:, 7]) + (0.04 - joint_positions[:, 8]), finger_close_reward)

        # 6th reward: how far the door has been opened out
        # instead of using rotation, may use pos as reference
        open_reward = (1.0 / (1.0 + door_yaw_error ** 2)) * around_handle_reward + handle_yaw
        # open_reward = (1.0 / (1.0 + self.handle_pos_error)) * around_handle_reward

        # 1st penalty
        action_penalty = torch.sum(actions ** 2, dim=-1)

        final_reward = 2.0 * grasp_dist_reward + 0.5 * rot_reward + 10.0 * around_handle_reward + 70.0 * open_reward + \
                        100.0 * finger_dist_reward+ 10.0 * finger_close_reward - 0.01 * action_penalty 
        
        # bonus for opening door properly
        final_reward = torch.where(door_yaw_error < 0.7, final_reward + 0.5, final_reward)
        final_reward = torch.where(door_yaw_error < 0.5, final_reward + around_handle_reward, final_reward)
        final_reward = torch.where(door_yaw_error < 0.2, final_reward + (2.0 * around_handle_reward), final_reward)

        # in case: Nan value occur
        final_reward = torch.where(torch.isnan(final_reward), torch.zeros_like(final_reward), final_reward)

        self.rew_buf[:] = final_reward
        # self.rew_buf[:] = torch.rand(self._num_envs)

        # if the door is opened to ref position -> task complete
        self.is_opened = torch.where(torch.rad2deg(handle_yaw)>=70, torch.ones_like(final_reward), torch.zeros_like(final_reward))   

        self.episode_sums["door_yaw_deg"] += torch.rad2deg(door_yaw)
        self.episode_sums["handle_yaw_deg"] += torch.rad2deg(handle_yaw)
        self.episode_sums["handle_pos_error"] += self.handle_pos_error
        self.episode_sums["handle_yaw_error"] += door_yaw_error
        self.episode_sums["grasp_handle_dist"] += grasp_handle_dist
        self.episode_sums["open_rate"] += self.is_opened
        self.episode_sums["rewards"] += final_reward

        # print("handle_pos", handle_pos)
        # print("handle_rot", handle_rot)
        # print("door_pos", door_pos)
        # print("door_rot", door_rot)
        # print("handle_local_pos", handle_local_pos)
        # print("handle_local_rot", handle_local_rot)
        # print("grasp_handle_dist", grasp_handle_dist)
        # print("door_yaw", door_yaw)


    def is_done(self) -> None:

        if not self.is_test: 

            # reset if door is fully opened
            # self.reset_buf = torch.where(self.is_opened==1, torch.ones_like(self.reset_buf), self.reset_buf)

            # reset if max length reached
            self.reset_buf = torch.where(self.progress_buf >= self._max_episode_length - 1, torch.ones_like(self.reset_buf), self.reset_buf)

        else:

            self.reset_buf = torch.where(self.progress_buf >= self._max_episode_length - 1, torch.ones_like(self.reset_buf), self.reset_buf)
            # self.reset_buf = torch.where(self.is_opened==1, torch.ones_like(self.reset_buf), self.reset_buf)
