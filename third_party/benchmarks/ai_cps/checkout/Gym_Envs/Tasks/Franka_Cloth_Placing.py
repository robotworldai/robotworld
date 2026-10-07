from omniisaacgymenvs.tasks.base.rl_task import RLTask
from Models.Franka.Franka import Franka
from Models.Franka.Franka_view import FrankaView
from Models.cloth_placing.target_table import TargetTable
from omni.isaac.core.prims import ParticleSystem, ClothPrim, ClothPrimView
from omni.isaac.core.materials import ParticleMaterial
from omni.physx.scripts import physicsUtils, particleUtils, deformableUtils

from omni.isaac.core.objects import DynamicCuboid
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


class FrankaClothPlacingTask(RLTask):
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
        self.episode_sums = {"center_dist": torch_zeros()}

        return

    def set_up_scene(self, scene) -> None:

        # Franka
        franka_translation = torch.tensor([0.3, 0.0, 0.0])
        self.get_franka(franka_translation)
        self.get_table()

        # Here the env is cloned (cannot clone particle systems right now)
        super().set_up_scene(scene)

        # Add Franka
        self._frankas = FrankaView(prim_paths_expr="/World/envs/.*/franka", name="franka_view")

        # Add bin
        self._target_table = RigidPrimView(prim_paths_expr="/World/envs/.*/target_table/target_table/mesh", name="target_table_view", reset_xform_properties=False)

        # Add location_ball
        self._location_cube = RigidPrimView(prim_paths_expr="/World/envs/.*/target_table/target_table/location_cube", name="location_cube_view", reset_xform_properties=False)
        
        scene.add(self._frankas)
        scene.add(self._frankas._hands)
        scene.add(self._frankas._lfingers)
        scene.add(self._frankas._rfingers)
        scene.add(self._location_cube)
        scene.add(self._target_table)


        # generate cloth near franka
        franka_positions = self._frankas.get_world_poses()[0]
        self.initialize_cloth(franka_positions)

        # Create a view to deal with all the cloths
        self._cloths = ClothPrimView(prim_paths_expr="/World/Env*/cloth", name="cloth_view")
        self._scene.add(self._cloths)

        self.init_data()
        return

    def get_franka(self, translation):

        franka = Franka(prim_path=self.default_zero_env_path + "/franka", name="franka", translation = translation, use_modified_collision = True)
        self._sim_config.apply_articulation_settings("franka", get_prim_at_path(franka.prim_path), self._sim_config.parse_actor_config("franka"))

    def get_table(self):
        target_table = TargetTable(prim_path=self.default_zero_env_path + "/target_table", name="target_table")
        self._sim_config.apply_articulation_settings("target_table", get_prim_at_path(target_table.prim_path), self._sim_config.parse_actor_config("target_table"))

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
            [0.0, 0.0, 0.0, -1.0, 0.0, 1.0, 0.5, 0.0001, 0.0001], device=self._device
        )

        self.actions = torch.zeros((self._num_envs, self.num_actions), device=self._device)

    def get_observations(self) -> dict:

        # Franka
        hand_pos, hand_rot = self._frankas._hands.get_world_poses(clone=False)

        franka_dof_pos = self._frankas.get_joint_positions(clone=False)
        franka_dof_pos = torch.nan_to_num(franka_dof_pos)

        franka_dof_vel = self._frankas.get_joint_velocities(clone=False)
        franka_dof_vel = torch.nan_to_num(franka_dof_vel)

        self.franka_lfinger_pos, self.franka_lfinger_rot = self._frankas._lfingers.get_world_poses(clone=False)
        self.franka_rfinger_pos, self.franka_rfinger_rot = self._frankas._lfingers.get_world_poses(clone=False)

        dof_pos_scaled = (
            2.0
            * (franka_dof_pos - self.franka_dof_lower_limits)
            / (self.franka_dof_upper_limits - self.franka_dof_lower_limits)
            - 1.0
        )

        # Cloth
        self.cloths_pos = self._cloths.get_world_positions(clone=False) 
        self.cloths_pos = torch.nan_to_num(self.cloths_pos) # shape (M,121,3)
        # cloths_pos_flat = torch.flatten(self.cloths_pos, start_dim=1) # shape (M,121,3)

        cloth_mean_x = torch.mean(self.cloths_pos[:,:,0], dim=1).reshape(self.num_envs, 1)
        cloth_mean_y = torch.mean(self.cloths_pos[:,:,1], dim=1).reshape(self.num_envs, 1)
        cloth_mean_z = torch.mean(self.cloths_pos[:,:,2], dim=1).reshape(self.num_envs, 1)
        self.cloths_pos_mean = torch.cat((cloth_mean_x, cloth_mean_y, cloth_mean_z),1)

        # location cube
        self.location_cube_pos, self.location_cube_rot  = self._location_cube.get_world_poses(clone=False)
        self.location_cube_pos = torch.nan_to_num(self.location_cube_pos)
        to_target = self.cloths_pos_mean - self.location_cube_pos 

        self.obs_buf = torch.cat(
            (
                dof_pos_scaled,
                franka_dof_vel * self.dof_vel_scale,
                # cloths_pos_flat,
                self.cloths_pos_mean,
                to_target,
                self.location_cube_pos,
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

        # Release condition
        location_cube_pos, location_cube_rot  = self._location_cube.get_world_poses()
        location_cube_pos = torch.nan_to_num(location_cube_pos)
        cloths_pos = self._cloths.get_world_positions() 
        cloths_pos = torch.nan_to_num(cloths_pos)
        cloth_mean_x = torch.mean(cloths_pos[:,:,0], dim=1).reshape(self.num_envs, 1)
        cloth_mean_y = torch.mean(cloths_pos[:,:,1], dim=1).reshape(self.num_envs, 1)
        cloth_mean_z = torch.mean(cloths_pos[:,:,2], dim=1).reshape(self.num_envs, 1)
        cloths_pos_mean = torch.cat((cloth_mean_x, cloth_mean_y, cloth_mean_z),1)
        center_dist = torch.norm(location_cube_pos[:,0:2] - cloths_pos_mean[:,0:2], p=2, dim=-1)

        cloth_vel = self._cloths.get_velocities() 
        cloth_vel = torch.nan_to_num(cloth_vel)
        cloth_vel_x = torch.mean(cloth_vel[:,:,0], dim=1).reshape(self.num_envs, 1)
        cloth_vel_y = torch.mean(cloth_vel[:,:,1], dim=1).reshape(self.num_envs, 1)
        cloth_vel_z = torch.mean(cloth_vel[:,:,2], dim=1).reshape(self.num_envs, 1)
        cloths_vel_mean = torch.cat((cloth_vel_x, cloth_vel_y, cloth_vel_z),1)
        vel = torch.norm(cloths_vel_mean, p=2, dim=-1) 

        release_condition = torch.logical_and(center_dist<0.07, cloths_pos_mean[:,2] > location_cube_pos[:,2])
        release_condition = torch.logical_and(release_condition, vel<0.1)

        self.franka_dof_targets[:,7] = torch.where(release_condition, 0.15, self.franka_dof_targets[:,7])
        self.franka_dof_targets[:,8] = torch.where(release_condition, 0.15, self.franka_dof_targets[:,8])

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

        # Reset cloth
        self._cloths.set_world_positions(self.default_cloth_pos, indices=indices)
        self._cloths.set_velocities(self.default_cloth_vel, indices=indices)

        if not self.is_test:
            # Reset cloth bin
            # reset positions: x: [-0.1,0.2], y:[-0.35,0.35]
            random_x = (0.2 + 0.1) * torch.rand(self._num_envs, device=self._device) - 0.1
            random_y = (0.35 + 0.35) * torch.rand(self._num_envs, device=self._device) - 0.35
            self.new_location_cube_pos = self.default_target_table_pos.clone().detach()
            self.new_location_cube_pos[:,0] = self.default_target_table_pos[:,0] + random_x
            self.new_location_cube_pos[:,1] = self.default_target_table_pos[:,1] + random_y
            self._target_table.set_world_poses(self.new_location_cube_pos[env_ids], self.default_target_table_rot[env_ids], indices = indices)
            self._target_table.set_velocities(self.default_target_table_velocity[env_ids], indices = indices)

        else:

            random_x = self.initial_test_value[0]
            random_y = self.initial_test_value[1]
            self.new_location_cube_pos = self.default_target_table_pos.clone().detach()
            self.new_location_cube_pos[:,0] = self.default_target_table_pos[:,0] + random_x
            self.new_location_cube_pos[:,1] = self.default_target_table_pos[:,1] + random_y
            self._target_table.set_world_poses(self.new_location_cube_pos[env_ids], self.default_target_table_rot[env_ids], indices = indices)
            self._target_table.set_velocities(self.default_target_table_velocity[env_ids], indices = indices)

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

        # Cloth
        self.default_cloth_pos = self._cloths.get_world_positions()
        self.default_cloth_vel = torch.zeros([self._num_envs, self._cloths.max_particles_per_cloth, 3], device=self._device)

        # Target table
        self.default_target_table_pos, self.default_target_table_rot = self._target_table.get_world_poses()
        self.default_target_table_velocity = self._target_table.get_velocities()

        # randomize all envs
        indices = torch.arange(self._num_envs, dtype=torch.int64, device=self._device)
        self.reset_idx(indices)

    def initialize_cloth(self, franka_positions):

        stage = get_current_stage()

        # parameters
        dimx = 10
        dimy = 10
        scale = 0.3

        for i in range(self._num_envs):

            # Note here: cannot put into the same envs (env/env_i) due to unknown bugs
            env_path = "/World/Env" + str(i)
            env = UsdGeom.Xform.Define(stage, env_path)
            # set up the geometry
            cloth_path = env.GetPrim().GetPath().AppendChild("cloth")
            plane_mesh = UsdGeom.Mesh.Define(stage, cloth_path)
            tri_points, tri_indices = deformableUtils.create_triangle_mesh_square(dimx=dimx, dimy=dimy, scale=scale)
            initial_positions = torch.zeros((self.num_envs, len(tri_points), 3))
            plane_mesh.GetPointsAttr().Set(tri_points)
            plane_mesh.GetFaceVertexIndicesAttr().Set(tri_indices)
            plane_mesh.GetFaceVertexCountsAttr().Set([3] * (len(tri_indices) // 3))
            # initial locations of the cloth
            franka_positions_np = franka_positions.detach().to('cpu').numpy()
            init_loc = Gf.Vec3f(float(franka_positions_np[i][0] - 0.5), float(franka_positions_np[i][1] ), float(franka_positions_np[i][2] + 0.65))
            physicsUtils.setup_transform_as_scale_orient_translate(plane_mesh)
            physicsUtils.set_or_add_translate_op(plane_mesh, init_loc)
            physicsUtils.set_or_add_orient_op(plane_mesh, Gf.Rotation(Gf.Vec3d([1, 0, 0]), 90).GetQuat())
            initial_positions[i] = torch.tensor(init_loc) + torch.tensor(plane_mesh.GetPointsAttr().Get())
            particle_system_path = env.GetPrim().GetPath().AppendChild("particleSystem")
            particle_material_path = env.GetPrim().GetPath().AppendChild("particleMaterial")

            particle_material = ParticleMaterial(
                prim_path=particle_material_path, drag=0.1, lift=0.3, friction=10.0
            )
            # parameters for the properties of the cloth
            # radius = 0.005
            radius = 0.5 * (scale / dimx)   # size rest offset according to plane resolution and width so that particles are just touching at rest
            restOffset = radius
            contactOffset = restOffset * 1.5
            particle_system = ParticleSystem(
                prim_path=particle_system_path,
                simulation_owner=self._env._world.get_physics_context().prim_path,
                rest_offset=restOffset,
                contact_offset=contactOffset,
                solid_rest_offset=restOffset,
                fluid_rest_offset=restOffset,
                particle_contact_offset=contactOffset,
            )
            # note that no particle material is applied to the particle system at this point.
            # this can be done manually via self.particle_system.apply_particle_material(self.particle_material)
            # or to pass the material to the clothPrim which binds it internally to the particle system
            stretch_stiffness = 100000.0
            bend_stiffness = 100.0
            shear_stiffness = 100.0
            spring_damping = 0.1
            particle_mass = 0.005
            cloth = ClothPrim(
                name="clothPrim" + str(i),
                prim_path=str(cloth_path),
                particle_system=particle_system,
                particle_material=particle_material,
                stretch_stiffness=stretch_stiffness,
                bend_stiffness=bend_stiffness,
                shear_stiffness=shear_stiffness,
                spring_damping=spring_damping,
                particle_mass = particle_mass,
                self_collision=True,
                self_collision_filter=True,
            )
            self._scene.add(cloth)


    def calculate_metrics(self) -> None:

        # center_dist = torch.norm(self.location_cube_pos - self.cloths_pos_mean, p=2, dim=-1)
        location_cube_pos = self.location_cube_pos
        center_dist = torch.norm(location_cube_pos - self.cloths_pos_mean, p=2, dim=-1)
        center_dist_reward = 1.0/(1.0+center_dist)

        # finger reward
        # franka_lfinger_pos = torch.nan_to_num(self.franka_lfinger_pos)
        # franka_rfinger_pos = torch.nan_to_num(self.franka_rfinger_pos)
        # finger_center = (franka_lfinger_pos + franka_rfinger_pos)/2
        # target = self.location_cube_pos
        # target[:,2] = target[:,2] + 0.3
        # finger_dist = torch.norm(finger_center - target, p=2, dim=-1)
        # finger_dist_reward = 1.0/(1.0+finger_dist)

        lfinger_vel = torch.nan_to_num(self._frankas._lfingers.get_velocities())
        rfinger_vel = torch.nan_to_num(self._frankas._rfingers.get_velocities())
        finger_vel = (lfinger_vel + rfinger_vel)/2
        finger_vel_norm = torch.norm(finger_vel, p=2, dim=-1)
        finger_vel_reward = 1.0/(1.0+finger_vel_norm)

        # finger rotation 
        franka_lfinger_rot = torch.nan_to_num(self.franka_lfinger_rot)
        franka_rfinger_rot = torch.nan_to_num(self.franka_rfinger_rot)
        mean_rot = (franka_lfinger_rot + franka_rfinger_rot)/2
        rot_target = torch.zeros_like(franka_lfinger_rot)
        rot_target[:,2] = 1
        rot_distance = torch.norm(mean_rot - rot_target, p=2, dim=-1)
        rot_distance_reward = 1.0/(1.0+rot_distance)

        # cloth velocities
        cloth_vel = self._cloths.get_velocities() 
        cloth_vel = torch.nan_to_num(cloth_vel)

        cloth_vel_x = torch.mean(cloth_vel[:,:,0], dim=1).reshape(self.num_envs, 1)
        cloth_vel_y = torch.mean(cloth_vel[:,:,1], dim=1).reshape(self.num_envs, 1)
        cloth_vel_z = torch.mean(cloth_vel[:,:,2], dim=1).reshape(self.num_envs, 1)
        cloths_vel_mean = torch.cat((cloth_vel_x, cloth_vel_y, cloth_vel_z),1)
        vel = torch.norm(cloths_vel_mean, p=2, dim=-1)  
        vel_reward = 1.0/(1.0+vel)

        # stay alive
        live_reward = torch.where(self.cloths_pos_mean[:,2] > 0.3, torch.ones_like(self.cloths_pos_mean[:,2]), torch.zeros_like(self.cloths_pos_mean[:,2]))

        # franka velocities
        # franka_dof_vel = self._frankas.get_joint_velocities()
        # franka_dof_vel = torch.nan_to_num(franka_dof_vel)
        # dof_vel_mean = torch.norm(franka_dof_vel, p=2, dim=-1) 
        # dof_vel_reward = 1.0/(1.0+dof_vel_mean)

        # is complete
        is_complete = torch.where(torch.logical_and(center_dist < 0.05, vel<0.1), torch.ones_like(center_dist), torch.zeros_like(center_dist))

        # if torch.any(torch.isnan(self.cloths_pos_mean)):
        #     print("NAN", self.cloths_pos_mean)

        # x_dist = torch.abs(self.location_cube_pos[:,0] - self.cloths_pos_mean[:,0])
        # x_dist_reward = 1.0/(1.0+x_dist)

        # y_dist = torch.abs(self.location_cube_pos[:,1] - self.cloths_pos_mean[:,1])
        # y_dist_reward = 1.0/(1.0+y_dist)

        # z_dist = torch.abs(self.location_cube_pos[:,2] - self.cloths_pos_mean[:,2])
        # z_dist_reward = 1.0/(1.0+z_dist)

        final_reward = 7.0*center_dist_reward + 10.0*is_complete + 1.0*rot_distance_reward + 1.0*live_reward \
                       + 1.0*vel_reward + 1.0*finger_vel_reward

        # TO BE IMPLEMENTED
        self.rew_buf[:] = final_reward

        # log additional info
        self.episode_sums["center_dist"] += center_dist
        # self.episode_sums["y_dist"] += y_dist
        # self.episode_sums["z_dist"] += z_dist

    def is_done(self) -> None:

        if not self.is_test:

            cloths_pos_z = self.cloths_pos_mean[:,2]
            center_dist = torch.norm(self.location_cube_pos- self.cloths_pos_mean, p=2, dim=-1)

            # if cloth falls to the ground
            self.reset_buf = torch.where( (cloths_pos_z < 0.1), torch.ones_like(self.reset_buf), self.reset_buf)

            # if error in franka positions
            franka_dof_pos = self._frankas.get_joint_positions()
            is_pos_nan = torch.isnan(franka_dof_pos)
            is_pos_fault = torch.any(is_pos_nan,1)

            self.reset_buf = torch.where( is_pos_fault == True, torch.ones_like(self.reset_buf), self.reset_buf)

            franka_dof_vel = self._frankas.get_joint_velocities()
            is_vel_nan = torch.isnan(franka_dof_vel)
            is_vel_fault = torch.any(is_vel_nan,1)

            self.reset_buf = torch.where( is_vel_fault == True, torch.ones_like(self.reset_buf), self.reset_buf)

            # or complete the task
            # reset if max length reached
            self.reset_buf = torch.where(self.progress_buf >= self._max_episode_length - 1, torch.ones_like(self.reset_buf), self.reset_buf)

        else:

            self.reset_buf = torch.where(self.progress_buf >= self._max_episode_length - 1, torch.ones_like(self.reset_buf), self.reset_buf)
