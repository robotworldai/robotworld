# -*- encoding: utf-8 -*-
'''
@File    :   geometry_se3_controller.py
@Time    :   2024/11/22 15:35:12
@Author  :   wenminggong, AIMS, PolyU
@Version :   1.0
@Desc    :   Implementations of geometry controller on SE(3). Refer to the paper: https://ieeexplore.ieee.org/document/5717652
'''


import torch
import torch.nn as nn


def _make_env_gain(gain_values, n_envs: int) -> torch.Tensor:
    gain = torch.as_tensor(gain_values).float()
    if gain.ndim == 0:
        gain = gain.repeat(3)
    if gain.ndim == 1:
        if gain.numel() == 1:
            gain = gain.repeat(3)
        if gain.numel() != 3:
            raise ValueError("Controller gain vectors must have 3 entries.")
        return gain.unsqueeze(0).repeat(n_envs, 1)
    if gain.ndim == 2 and gain.shape == (n_envs, 3):
        return gain
    raise ValueError(f"Controller gain tensor must have shape (3,) or ({n_envs}, 3), got {tuple(gain.shape)}.")


def _set_env_gain(controller: nn.Module, gain_name: str, env_ids, values):
    gain = getattr(controller, gain_name)
    if gain.ndim != 2 or gain.shape[-1] != 3:
        raise ValueError(f"{gain_name} must have shape (num_envs, 3) for per-environment updates.")

    env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=gain.device)
    values = torch.as_tensor(values, dtype=gain.dtype, device=gain.device)
    if values.ndim == 0:
        values = values.repeat(env_ids.numel(), 3)
    elif values.ndim == 1:
        if values.numel() == env_ids.numel():
            values = values.unsqueeze(-1).repeat(1, 3)
        elif values.numel() == 3:
            values = values.unsqueeze(0).repeat(env_ids.numel(), 1)
        else:
            raise ValueError("Gain values must be scalar, shape (num_envs,), shape (3,), or shape (num_envs, 3).")
    elif values.ndim == 2:
        if values.shape != (env_ids.numel(), 3):
            raise ValueError(f"Gain values must have shape ({env_ids.numel()}, 3), got {tuple(values.shape)}.")
    else:
        raise ValueError("Gain values must be scalar, 1-D, or 2-D.")
    gain.data[env_ids] = values


def quaternion_to_euler(quaternion: torch.Tensor):
    w, x, y, z = torch.unbind(quaternion, dim=-1)

    euler_angles: torch.Tensor = torch.stack(
        (
            torch.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y)),
            torch.asin(2.0 * (w * y - z * x)),
            torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)),
        ),
        dim=-1,
    )

    # [..., 3]
    return euler_angles

def quaternion_conjugate(quats: torch.Tensor):
    q0 = quats[..., 0]
    q1 = quats[..., 1]
    q2 = quats[..., 2]
    q3 = quats[..., 3]
    return torch.stack([q0, -q1, -q2, -q3], dim=-1)

def quaternion_multiply(q1: torch.Tensor, q2: torch.Tensor):
    # quaternion multiplication
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    
    return torch.stack([w, x, y, z], dim=-1)

def quat_world_to_body(quat: torch.Tensor, ang_vel: torch.Tensor):
    # convert ang_vel to quat_format
    ang_vel_zeros = torch.zeros((ang_vel.shape[:-1]+(1,)), device=ang_vel.device)
    ang_vel_quats = torch.cat([ang_vel_zeros, ang_vel], dim=-1) # [..., 4]

    # compute quaternion_conjugate
    quat_conjs = quaternion_conjugate(quat)

    # compute uat_conj * ang_vel_quat
    temp_results = quaternion_multiply(quat_conjs, ang_vel_quats)
    # compute (quat_conj * ang_vel_quat) * quat
    ang_vel_body_quats = quaternion_multiply(temp_results, quat)

    # return [..., 3]
    return ang_vel_body_quats[..., 1:] 


def quaternion_to_rotation_matrix(quats: torch.Tensor):
    w = quats[..., 0]
    x = quats[..., 1]
    y = quats[..., 2]
    z = quats[..., 3]

    R11 = 1 - 2 * (y * y + z * z)
    R22 = 1 - 2 * (x * x + z * z)
    R33 = 1 - 2 * (x * x + y * y)
    R12 = 2 * (x * y - w * z)
    R13 = 2 * (x * z + w * y)
    R21 = 2 * (x * y + w * z)
    R23 = 2 * (y * z - w * x)
    R31 = 2 * (x * z - w * y)
    R32 = 2 * (y * z + w * x)

    R = torch.stack(
        [
            R11,
            R12,
            R13,
            R21,
            R22,
            R23,
            R31,
            R32,
            R33
        ],
        dim=-1,
    )
    R = R.reshape((R.shape[:-1] + (3, 3)))
    return R


def normalize(x: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
    return x / (torch.norm(x, dim=-1, keepdim=True) + eps)

def rotation_matrix_to_euler_angles(matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert a rotation matrix to Euler angles (ZYX order).
    
    Args:
        matrix (torch.Tensor): A tensor of shape (..., 3, 3) representing rotation matrices.
        
    Returns:
        torch.Tensor: A tensor of shape (..., 3) representing Euler angles in radians.
    """
    assert matrix.shape[-2:] == (3, 3), "Input must be a batch of 3x3 matrices"
    
    sy = torch.sqrt(matrix[..., 0, 0] ** 2 + matrix[..., 1, 0] ** 2)
    
    singular = sy < 1e-6

    x = torch.atan2(matrix[..., 2, 1], matrix[..., 2, 2])
    y = torch.atan2(-matrix[..., 2, 0], sy)
    z = torch.atan2(matrix[..., 1, 0], matrix[..., 0, 0])

    xs = torch.atan2(-matrix[..., 1, 2], matrix[..., 1, 1])
    ys = torch.atan2(-matrix[..., 2, 0], sy)
    zs = torch.zeros_like(z)

    x = torch.where(singular, xs, x)
    y = torch.where(singular, ys, y)
    z = torch.where(singular, zs, z)

    return torch.stack((x, y, z), dim=-1)

def axis_angle_to_quaternion(angle: torch.Tensor, axis: torch.Tensor):
    axis = axis / torch.norm(axis, dim=-1, keepdim=True)
    return torch.cat([torch.cos(angle / 2), torch.sin(angle / 2) * axis], dim=-1)

def axis_angle_to_matrix(angle: torch.Tensor, axis: torch.Tensor):
    quat = axis_angle_to_quaternion(angle, axis)
    return quaternion_to_rotation_matrix(quat)


class AttitudeController(nn.Module):
    '''
    SE(3) attitude controller
    Inputs:
        root_state
        target_roll
        target_pitch
        target_yaw
        target_height
        target_height_vel
        target_height_acc
        target_angle_vel
        target_angle_acc
    Outputs:
        control cmd: rotors' throttles
    '''
    def __init__(
        self,
        g: torch.Tensor,
        uav_params: dict,
        controller_params: dict,
        n_envs: int, 
        target_value_compute: bool=False,
    ):
        super().__init__()
        self._target_value_compute = target_value_compute # target value compute mode, False: zero target, True: delta_value / delta_t
        self.g = nn.Parameter(g) # [g_x, g_y, g_z]
        self.mass = nn.Parameter(torch.tensor(uav_params["mass"]))
        # pos_gain and vel_gain need to multiplicate mass
        self.pos_gain = nn.Parameter(torch.as_tensor(controller_params["position_gain"]).float())
        self.vel_gain = nn.Parameter(torch.as_tensor(controller_params["velocity_gain"]).float())
        self.attitude_gain = nn.Parameter(_make_env_gain(controller_params["attitude_gain"], n_envs))
        self.angle_vel_gain = nn.Parameter(torch.as_tensor(controller_params["angular_rate_gain"]).float())
        inertia = uav_params["inertia"]
        self.J = nn.Parameter(torch.diag_embed(torch.tensor([inertia["xx"], inertia["yy"], inertia["zz"]])))

        rotor_config = uav_params["rotor_configuration"]
        force_constants = torch.as_tensor(rotor_config["force_constants"])
        max_rot_vel = torch.as_tensor(rotor_config["max_rotation_velocities"])
        self.max_thrusts = nn.Parameter(max_rot_vel.square() * force_constants)

        # set initial targets, under the Isaac-sim frame
        self._init_last_target_height = nn.Parameter(torch.zeros(n_envs)) # [n_envs]
        self._init_last_target_height_vel = nn.Parameter(torch.zeros(n_envs)) # [n_envs]
        self._init_last_target_Rd = nn.Parameter(torch.diag_embed(torch.ones(n_envs, 3))) # [n_envs, 3, 3]
        self._init_last_target_angle_vel = nn.Parameter(torch.zeros(n_envs, 3)) # [n_envs, 3]
        self._init_last_t = nn.Parameter(torch.zeros(n_envs, 1)) # [n_envs, 1]

        # control allocation matrix, under isaac-sim frame
        rotor_angles = torch.as_tensor(rotor_config["rotor_angles"])
        arm_lengths = torch.as_tensor(rotor_config["arm_lengths"])
        moment_constants = torch.as_tensor(rotor_config["moment_constants"])
        directions = torch.as_tensor(rotor_config["directions"])
        ca_matrix = torch.stack(
            [
                torch.ones_like(rotor_angles),
                torch.sin(rotor_angles) * arm_lengths,
                -torch.cos(rotor_angles) * arm_lengths,
                directions * moment_constants / force_constants,
            ]
        )
        self.ca_matrix = nn.Parameter(ca_matrix)

        # setting rotation matrix from isaac-sim frame to NED frame
        self._rotation_matrix_from_isaac_to_ned = nn.Parameter(torch.diag_embed(torch.tensor([1.0, -1.0, -1.0]).repeat(n_envs, 1))) # [n_envs, 3, 3]
        self._rotation_matrix_from_isaac_body_to_ned_body = nn.Parameter(torch.diag_embed(torch.tensor([1.0, -1.0, -1.0]).repeat(n_envs, 1))) # [n_envs, 3, 3]

        self.reset()
        self.requires_grad_(False)

    def reset(self, env_ids=None):
        if env_ids is None:
            self.last_target_height = self._init_last_target_height.data.clone()
            self.last_target_height_vel = self._init_last_target_height_vel.data.clone()
            self.last_target_Rd = self._init_last_target_Rd.data.clone()
            self.last_target_angle_vel = self._init_last_target_angle_vel.data.clone()
            self.last_t = self._init_last_t.data.clone()
        else:
            self.last_target_height = self.last_target_height.to(device=self._init_last_target_height.data.device)
            self.last_target_height_vel = self.last_target_height_vel.to(device=self._init_last_target_height_vel.data.device)
            self.last_target_Rd = self.last_target_Rd.to(device=self._init_last_target_Rd.data.device)
            self.last_target_angle_vel = self.last_target_angle_vel.to(device=self._init_last_target_angle_vel.data.device)
            self.last_t = self.last_t.to(device=self._init_last_t.data.device)

            self.last_target_height[env_ids] = self._init_last_target_height.data[env_ids].clone()
            self.last_target_height_vel[env_ids] = self._init_last_target_height_vel.data[env_ids].clone()
            self.last_target_Rd[env_ids] = self._init_last_target_Rd.data[env_ids].clone()
            self.last_target_angle_vel[env_ids] = self._init_last_target_angle_vel.data[env_ids].clone()
            self.last_t[env_ids] = self._init_last_t.data[env_ids].clone()

    def set_gain(self, gain_name: str, env_ids, values):
        _set_env_gain(self, gain_name, env_ids, values)

    def get_gain(self, gain_name: str) -> torch.Tensor:
        return getattr(self, gain_name).detach()

    def forward(
        self,
        current_t: torch.Tensor,
        root_state: torch.Tensor,
        target_roll: torch.Tensor=None,
        target_pitch: torch.Tensor=None,
        target_yaw: torch.Tensor=None,
        target_height: torch.Tensor=None,
        target_height_vel: torch.Tensor=None,
        target_height_acc: torch.Tensor=None,
        target_angle_vel: torch.Tensor=None,
        target_angle_acc: torch.Tensor=None,
        body_frame: bool=False
    ):
        '''
        root_state: [..., 13],
        target_roll: [...],
        target_pitch: [...],
        target_yaw: [...],
        target_height: [...],
        target_height_vel: [...],
        target_height_acc: [...],
        target_angle_vel: [..., 3],
        target_angle_acc: [..., 3],
        '''
        # all inputs are under Issac-sim frame
        batch_shape = root_state.shape[:-1]
        current_t = current_t.unsqueeze(dim=-1)

        if target_roll is None:
            target_roll = quaternion_to_euler(root_state[..., 3:7])[..., 0] # [...]
        if target_pitch is None:
            target_pitch = quaternion_to_euler(root_state[..., 3:7])[..., 1] # [...]
        if target_yaw is None:
            target_yaw = quaternion_to_euler(root_state[..., 3:7])[..., -1] # [...]
        if target_height is None:
            target_height = root_state[..., 2] # [...]

        if target_height_vel is None:
            if self._target_value_compute:
                target_height_vel = (target_height - self.last_target_height) / (current_t - self.last_t) # [...]
            else:
                target_height_vel = torch.zeros_like(target_height) # [n_envs]

        if target_height_acc is None:
            if self._target_value_compute:
                target_height_acc = (target_height_vel - self.last_target_height_vel) / (current_t - self.last_t) # [...]
            else:
                target_height_acc = torch.zeros_like(target_height_vel) # [n_envs]

        thrust_moments, R_des, target_angle_vel = self._compute(
            current_t,
            root_state.reshape(-1, 13),
            target_roll.reshape(-1, 1),
            target_pitch.reshape(-1, 1),
            target_yaw.reshape(-1, 1),
            target_height.reshape(-1, 1),
            target_height_vel.reshape(-1, 1),
            target_height_acc.reshape(-1, 1),
            target_angle_vel.reshape(-1, 3) if not target_angle_vel is None else None,
            target_angle_acc.reshape(-1, 3) if not target_angle_acc is None else None,
            body_frame
        )
        
        thrust_moments = thrust_moments.reshape(*batch_shape, -1)
        R_des = R_des.reshape(*batch_shape, 3, 3)
        target_angle_vel = target_angle_vel.reshape(*batch_shape, -1)
        # convert thrust_moments to rotors' throttles
        rotor_thrusts = (self.ca_matrix.inverse() @ thrust_moments.mT).mT # [n_envs, 4]
        rotor_thrusts = torch.where(rotor_thrusts<0, 0, rotor_thrusts)

        throttle_cmds = (rotor_thrusts / self.max_thrusts).sqrt() # [n_envs, 4]

        self.last_target_height = target_height
        self.last_target_height_vel = target_height_vel
        self.last_t = current_t
        self.last_target_Rd = R_des
        self.last_target_angle_vel = target_angle_vel
        return throttle_cmds

    def _compute(
        self,
        current_t: torch.Tensor,
        root_state: torch.Tensor,
        target_roll: torch.Tensor=None,
        target_pitch: torch.Tensor=None,
        target_yaw: torch.Tensor=None,
        target_height: torch.Tensor=None,
        target_height_vel: torch.Tensor=None,
        target_height_acc: torch.Tensor=None,
        target_angle_vel: torch.Tensor=None,
        target_angle_acc: torch.Tensor=None,
        body_frame: bool=False
    ):
        '''
        root_state: [n, 13],
        target_roll: [n, 1],
        target_pitch: [n, 1],
        target_yaw: [n, 1],
        target_height: [n, 1],
        target_height_vel: [n, 1],
        target_height_acc: [n, 1],
        target_angle_vel: [n, 3],
        target_angle_acc: [n, 3],
        '''

        device = root_state.device
        # inputs are all under isaac-sim frame
        pos, quat, vel, ang_vel = torch.split(root_state, [3, 4, 3, 3], dim=-1)
        if not body_frame:
            # convert angle_vel from world frame to body frame
            ang_vel = quat_world_to_body(quat, ang_vel)
        R = quaternion_to_rotation_matrix(quat)

        # convert from isaac-sim frame to NED frame
        pos_ned = torch.bmm(self._rotation_matrix_from_isaac_to_ned, pos.unsqueeze(dim=-1)).squeeze(dim=-1)
        target_pos = target_height.repeat(1, 3)
        target_pos_ned = torch.bmm(self._rotation_matrix_from_isaac_to_ned, target_pos.unsqueeze(dim=-1)).squeeze(dim=-1)
        vel_ned = torch.bmm(self._rotation_matrix_from_isaac_to_ned, vel.unsqueeze(dim=-1)).squeeze(dim=-1)
        target_vel = target_height_vel.repeat(1, 3)
        target_vel_ned = torch.bmm(self._rotation_matrix_from_isaac_to_ned, target_vel.unsqueeze(dim=-1)).squeeze(dim=-1)
        target_acc = target_height_acc.repeat(1, 3)
        target_acc_ned = torch.bmm(self._rotation_matrix_from_isaac_to_ned, target_acc.unsqueeze(dim=-1)).squeeze(dim=-1)
        ang_vel_ned_body = torch.bmm(self._rotation_matrix_from_isaac_body_to_ned_body, ang_vel.unsqueeze(dim=-1)).squeeze(dim=-1)
        R_ned = torch.bmm(self._rotation_matrix_from_isaac_to_ned, torch.bmm(R, self._rotation_matrix_from_isaac_body_to_ned_body.inverse()))
        
        # force control
        pos_error = pos_ned - target_pos_ned
        vel_error = vel_ned - target_vel_ned
        total_thrust = self.mass * (self.pos_gain[-1] * pos_error[..., -1] + self.vel_gain[-1] * vel_error[..., -1] + self.g[-1] - target_acc_ned[..., -1]) / R_ned[..., -1, -1]

        # moment control
        target_roll_matrix = axis_angle_to_matrix(target_roll, torch.tensor([1.0, 0.0, 0.0], device=root_state.device))
        target_pitch_matrix = axis_angle_to_matrix(target_pitch, torch.tensor([0.0, 1.0, 0.0], device=root_state.device))
        target_yaw_matrix = axis_angle_to_matrix(target_yaw, torch.tensor([0.0, 0.0, 1.0], device=root_state.device))
        R_des = torch.bmm(torch.bmm(target_yaw_matrix,  target_pitch_matrix), target_roll_matrix)
        R_des_ned = torch.bmm(self._rotation_matrix_from_isaac_to_ned, torch.bmm(R_des, self._rotation_matrix_from_isaac_body_to_ned_body.inverse()))

        # print("R_des: {}".format(R_des))
        # print("R_des_ned: {}".format(R_des_ned))
        R_error_matrix = 0.5 * (
            torch.bmm(R_des_ned.mT, R_ned)
            - torch.bmm(R_ned.mT, R_des_ned)
        )
        R_error = torch.stack([
            R_error_matrix[:, 2, 1],
            R_error_matrix[:, 0, 2], 
            R_error_matrix[:, 1, 0],
        ],dim=-1)

        # print("R_error_matrix: {}".format(R_error_matrix))
        # print("R_error: {}".format(R_error))

        if target_angle_vel is None:
            if self._target_value_compute:
                target_angle_vel_matrix = torch.bmm(R_des.mT, (R_des - self.last_target_Rd) / (current_t - self.last_t).unsqueeze(dim=-1))
                target_angle_vel = torch.stack([
                    target_angle_vel_matrix[:, 2, 1],
                    target_angle_vel_matrix[:, 0, 2],
                    target_angle_vel_matrix[:, 1, 0]
                ],dim=-1)
            else:
                target_angle_vel = torch.zeros_like(ang_vel)
        target_angle_vel_ned_body = torch.bmm(self._rotation_matrix_from_isaac_body_to_ned_body, target_angle_vel.unsqueeze(dim=-1)).squeeze(dim=-1)
        
        if target_angle_acc is None:
            if self._target_value_compute:
                target_angle_acc = (target_angle_vel - self.last_target_angle_vel) / (current_t - self.last_t)
            else:
                target_angle_acc = torch.zeros_like(target_angle_vel)
        target_angle_acc_ned_body = torch.bmm(self._rotation_matrix_from_isaac_body_to_ned_body, target_angle_acc.unsqueeze(dim=-1)).squeeze(dim=-1)

        angle_vel_error = ang_vel_ned_body - torch.bmm(torch.bmm(R_ned.mT, R_des_ned), target_angle_vel_ned_body.unsqueeze(dim=-1)).squeeze(dim=-1)

        moments_ned = - self.attitude_gain * R_error \
                    - self.angle_vel_gain * angle_vel_error \
                    + torch.cross(ang_vel_ned_body, (self.J @ ang_vel_ned_body.mT).mT, dim=-1) \
                    - (self.J @ (torch.cross(ang_vel_ned_body, torch.bmm(torch.bmm(R_ned.mT, R_des_ned), target_angle_vel_ned_body.unsqueeze(dim=-1)).squeeze(dim=-1), dim=-1) - torch.bmm(torch.bmm(R_ned.mT, R_des_ned), target_angle_acc_ned_body.unsqueeze(dim=-1)).squeeze(dim=-1)).mT).mT

        moments = torch.bmm(self._rotation_matrix_from_isaac_to_ned.inverse(), moments_ned.unsqueeze(dim=-1)).squeeze(dim=-1)
        thrust_moments = torch.cat([total_thrust.reshape(-1, 1), moments], dim=-1)
        # thrust_moments: [n, 4]
        return thrust_moments, R_des, target_angle_vel



class PositionController(nn.Module):
    '''
    SE(3) position controller.
    Inputs:
        root_state
        target_pos
        target_yaw
        target_vel
        target_acc
        target_angle_vel
        target_angle_acc
    Outputs:
        control cmd: rotors' throttles
    Attentions:
        1.original controller based on the NED world frame and NED body frame
        2.Isaac Sim provides ENU world frame and ENU body frame
        3.Need transfer between these two frames
    '''
    def __init__(
        self,
        g: torch.Tensor,
        uav_params: dict,
        controller_params: dict,
        n_envs: int, 
        target_value_compute: bool=False,
    ):
        super().__init__()
        self._target_value_compute = target_value_compute # target value compute mode, False: zero target, True: delta_value / delta_t
        self.g = nn.Parameter(g)
        self.mass = nn.Parameter(torch.tensor(uav_params["mass"]))
        # pos_gain and vel_gain need to multiplicate mass
        self.pos_gain = nn.Parameter(_make_env_gain(controller_params["position_gain"], n_envs))
        self.vel_gain = nn.Parameter(torch.as_tensor(controller_params["velocity_gain"]).float())
        self.attitude_gain = nn.Parameter(torch.as_tensor(controller_params["attitude_gain"]).float())
        self.angle_vel_gain = nn.Parameter(torch.as_tensor(controller_params["angular_rate_gain"]).float())
        inertia = uav_params["inertia"]
        self.J = nn.Parameter(torch.diag_embed(torch.tensor([inertia["xx"], inertia["yy"], inertia["zz"]])))

        rotor_config = uav_params["rotor_configuration"]
        force_constants = torch.as_tensor(rotor_config["force_constants"])
        max_rot_vel = torch.as_tensor(rotor_config["max_rotation_velocities"])
        self.max_thrusts = nn.Parameter(max_rot_vel.square() * force_constants)

        # set initial targets, under the Isaac-sim frame
        self._init_last_target_pos = nn.Parameter(torch.zeros(n_envs, 3)) # [n_envs, 3]
        self._init_last_target_vel = nn.Parameter(torch.zeros(n_envs, 3)) # [n_envs, 3]
        self._init_last_target_Rd = nn.Parameter(torch.diag_embed(torch.ones(n_envs, 3))) # [n_envs, 3, 3]
        self._init_last_target_angle_vel = nn.Parameter(torch.zeros(n_envs, 3)) # [n_envs, 3]
        self._init_last_t = nn.Parameter(torch.zeros(n_envs, 1)) # [n_envs, 1]

        # control allocation matrix, under isaac-sim frame
        rotor_angles = torch.as_tensor(rotor_config["rotor_angles"])
        arm_lengths = torch.as_tensor(rotor_config["arm_lengths"])
        moment_constants = torch.as_tensor(rotor_config["moment_constants"])
        directions = torch.as_tensor(rotor_config["directions"])
        ca_matrix = torch.stack(
            [
                torch.ones_like(rotor_angles),
                torch.sin(rotor_angles) * arm_lengths,
                -torch.cos(rotor_angles) * arm_lengths,
                directions * moment_constants / force_constants,
            ]
        )
        self.ca_matrix = nn.Parameter(ca_matrix)

        # setting rotation matrix from isaac-sim frame to NED frame
        self._rotation_matrix_from_isaac_to_ned = nn.Parameter(torch.diag_embed(torch.tensor([1.0, -1.0, -1.0]).repeat(n_envs, 1))) # [n_envs, 3, 3]
        self._rotation_matrix_from_isaac_body_to_ned_body = nn.Parameter(torch.diag_embed(torch.tensor([1.0, -1.0, -1.0]).repeat(n_envs, 1))) # [n_envs, 3, 3]

        self.reset()
        self.requires_grad_(False)

    def reset(self, env_ids=None):
        if env_ids is None:
            self.last_target_pos = self._init_last_target_pos.data.clone()
            self.last_target_vel = self._init_last_target_vel.data.clone()
            self.last_target_Rd = self._init_last_target_Rd.data.clone()
            self.last_target_angle_vel = self._init_last_target_angle_vel.data.clone()
            self.last_t = self._init_last_t.data.clone()
        else:
            self.last_target_pos = self.last_target_pos.to(device=self._init_last_target_pos.data.device)
            self.last_target_vel = self.last_target_vel.to(device=self._init_last_target_vel.data.device)
            self.last_target_Rd = self.last_target_Rd.to(device=self._init_last_target_Rd.data.device)
            self.last_target_angle_vel = self.last_target_angle_vel.to(device=self._init_last_target_angle_vel.data.device)
            self.last_t = self.last_t.to(device=self._init_last_t.data.device)
            self.last_target_pos[env_ids] = self._init_last_target_pos.data[env_ids].clone()
            self.last_target_vel[env_ids] = self._init_last_target_vel.data[env_ids].clone()
            self.last_target_Rd[env_ids] = self._init_last_target_Rd.data[env_ids].clone()
            self.last_target_angle_vel[env_ids] = self._init_last_target_angle_vel.data[env_ids].clone()
            self.last_t[env_ids] = self._init_last_t.data[env_ids].clone()

    def set_gain(self, gain_name: str, env_ids, values):
        _set_env_gain(self, gain_name, env_ids, values)

    def get_gain(self, gain_name: str) -> torch.Tensor:
        return getattr(self, gain_name).detach()

    def forward(
        self,
        current_t: torch.Tensor,
        root_state: torch.Tensor,
        target_pos: torch.Tensor=None,
        target_yaw: torch.Tensor=None,
        target_vel: torch.Tensor=None,
        target_acc: torch.Tensor=None,
        target_angle_vel: torch.Tensor=None,
        target_angle_acc: torch.Tensor=None,
        body_frame: bool=False
    ):
        '''
        root_state: [..., 13],
        target_pos: [..., 3],
        target_yaw: [...],
        target_vel: [..., 3],
        target_acc: [..., 3],
        target_angle_vel: [..., 3],
        target_angle_acc: [..., 3],
        '''

        # all inputs are under Issac-sim frame
        batch_shape = root_state.shape[:-1]
        current_t = current_t.unsqueeze(dim=-1)

        if target_pos is None:
            target_pos = root_state[..., :3] # [..., 3]
        if target_yaw is None:
            target_yaw = quaternion_to_euler(root_state[..., 3:7])[..., -1] # [...]

        if target_vel is None:
            if self._target_value_compute:
                target_vel = (target_pos - self.last_target_pos) / (current_t - self.last_t) # [..., 3]
            else:
                target_vel = torch.zeros_like(target_pos)

        if target_acc is None:
            if self._target_value_compute:
                target_acc = (target_vel - self.last_target_vel) / (current_t - self.last_t) # [..., 3]
            else:
                target_acc = torch.zeros_like(target_vel)

        thrust_moments, R_des, target_angle_vel = self._compute(
            current_t,
            root_state.reshape(-1, 13),
            target_pos.reshape(-1, 3),
            target_yaw.reshape(-1, 1),
            target_vel.reshape(-1, 3),
            target_acc.reshape(-1, 3),
            target_angle_vel.reshape(-1, 3) if not target_angle_vel is None else None,
            target_angle_acc.reshape(-1, 3) if not target_angle_acc is None else None,
            body_frame
        )
        thrust_moments = thrust_moments.reshape(*batch_shape, -1)
        R_des = R_des.reshape(*batch_shape, 3, 3)
        target_angle_vel = target_angle_vel.reshape(*batch_shape, -1)
        # convert thrust_moments to rotors' throttles
        rotor_thrusts = (self.ca_matrix.inverse() @ thrust_moments.mT).mT # [n_envs, 4]
        rotor_thrusts = torch.where(rotor_thrusts<0, 0, rotor_thrusts)

        # print("position controller | rotor thrusts: {}".format(rotor_thrusts))
        throttle_cmds = (rotor_thrusts / self.max_thrusts).sqrt() # [n_envs, 4]

        self.last_target_pos = target_pos
        self.last_target_vel = target_vel
        self.last_t = current_t
        self.last_target_Rd = R_des
        self.last_target_angle_vel = target_angle_vel
        return throttle_cmds

    def _compute(
        self,
        current_t: torch.Tensor,
        root_state: torch.Tensor,
        target_pos: torch.Tensor=None,
        target_yaw: torch.Tensor=None,
        target_vel: torch.Tensor=None,
        target_acc: torch.Tensor=None,
        target_angle_vel: torch.Tensor=None,
        target_angle_acc: torch.Tensor=None,
        body_frame: bool=False
    ):
        '''
        root_state: [..., 13],
        target_pos: [..., 3],
        target_yaw: [...],
        target_vel: [..., 3],
        target_acc: [..., 3],
        target_angle_vel: [..., 3],
        target_angle_acc: [..., 3],
        '''

        device = root_state.device

        # inputs are all under isaac-sim frame
        pos, quat, vel, ang_vel = torch.split(root_state, [3, 4, 3, 3], dim=-1)
        if not body_frame:
            # convert angle_vel from world frame to body frame
            ang_vel = quat_world_to_body(quat, ang_vel)
        R = quaternion_to_rotation_matrix(quat)

        # convert from isaac-sim frame to NED frame
        pos_ned = torch.bmm(self._rotation_matrix_from_isaac_to_ned, pos.unsqueeze(dim=-1)).squeeze(dim=-1)
        target_pos_ned = torch.bmm(self._rotation_matrix_from_isaac_to_ned, target_pos.unsqueeze(dim=-1)).squeeze(dim=-1)
        vel_ned = torch.bmm(self._rotation_matrix_from_isaac_to_ned, vel.unsqueeze(dim=-1)).squeeze(dim=-1)
        target_vel_ned = torch.bmm(self._rotation_matrix_from_isaac_to_ned, target_vel.unsqueeze(dim=-1)).squeeze(dim=-1)
        target_acc_ned = torch.bmm(self._rotation_matrix_from_isaac_to_ned, target_acc.unsqueeze(dim=-1)).squeeze(dim=-1)
        ang_vel_ned_body = torch.bmm(self._rotation_matrix_from_isaac_body_to_ned_body, ang_vel.unsqueeze(dim=-1)).squeeze(dim=-1)
        R_ned = torch.bmm(self._rotation_matrix_from_isaac_to_ned, torch.bmm(R, self._rotation_matrix_from_isaac_body_to_ned_body.inverse()))
        
        # force control
        pos_error = pos_ned - target_pos_ned
        vel_error = vel_ned - target_vel_ned
        vector_total_thrust = self.mass * (-self.pos_gain * pos_error - self.vel_gain * vel_error - self.g + target_acc_ned)
        total_thrust = (-vector_total_thrust * R_ned[:, :, 2]).sum(dim=-1, keepdim=True)
        # print("position controller | total thrust: {}".format(total_thrust))

        # moment control
        b1_des = torch.cat([
            torch.cos(target_yaw),
            torch.sin(target_yaw),
            torch.zeros_like(target_yaw)
        ],dim=-1)
        b1_des_ned = torch.bmm(self._rotation_matrix_from_isaac_to_ned, b1_des.unsqueeze(dim=-1)).squeeze(dim=-1)
        b3_des_ned = -normalize(vector_total_thrust)
        b2_des_ned = normalize(torch.cross(b3_des_ned, b1_des_ned, dim=-1))
        R_des_ned = torch.stack([
            b2_des_ned.cross(b3_des_ned, -1),
            b2_des_ned,
            b3_des_ned
        ], dim=-1)

        R_des = torch.bmm(self._rotation_matrix_from_isaac_to_ned.inverse(), torch.bmm(R_des_ned, self._rotation_matrix_from_isaac_body_to_ned_body))

        R_error_matrix = 0.5 * (
            torch.bmm(R_des_ned.mT, R_ned)
            - torch.bmm(R_ned.mT, R_des_ned)
        )
        R_error = torch.stack([
            R_error_matrix[:, 2, 1],
            R_error_matrix[:, 0, 2],
            R_error_matrix[:, 1, 0],
        ],dim=-1)

        if target_angle_vel is None:
            if self._target_value_compute:
                target_angle_vel_matrix = torch.bmm(R_des.mT, (R_des - self.last_target_Rd) / (current_t - self.last_t).unsqueeze(dim=-1))
                target_angle_vel = torch.stack([
                    target_angle_vel_matrix[:, 2, 1],
                    target_angle_vel_matrix[:, 0, 2],
                    target_angle_vel_matrix[:, 1, 0]
                ],dim=-1)
            else:
                target_angle_vel = torch.zeros_like(ang_vel)
        target_angle_vel_ned_body = torch.bmm(self._rotation_matrix_from_isaac_body_to_ned_body, target_angle_vel.unsqueeze(dim=-1)).squeeze(dim=-1)
        
        if target_angle_acc is None:
            if self._target_value_compute:
                target_angle_acc = (target_angle_vel - self.last_target_angle_vel) / (current_t - self.last_t)
            else:
                target_angle_acc = torch.zeros_like(target_angle_vel)
        target_angle_acc_ned_body = torch.bmm(self._rotation_matrix_from_isaac_body_to_ned_body, target_angle_acc.unsqueeze(dim=-1)).squeeze(dim=-1)

        angle_vel_error = ang_vel_ned_body - torch.bmm(torch.bmm(R_ned.mT, R_des_ned), target_angle_vel_ned_body.unsqueeze(dim=-1)).squeeze(dim=-1)

        moments_ned = - self.attitude_gain * R_error \
                    - self.angle_vel_gain * angle_vel_error \
                    + torch.cross(ang_vel_ned_body, (self.J @ ang_vel_ned_body.mT).mT, dim=-1) \
                    - (self.J @ (torch.cross(ang_vel_ned_body, torch.bmm(torch.bmm(R_ned.mT, R_des_ned), target_angle_vel_ned_body.unsqueeze(dim=-1)).squeeze(dim=-1), dim=-1) - torch.bmm(torch.bmm(R_ned.mT, R_des_ned), target_angle_acc_ned_body.unsqueeze(dim=-1)).squeeze(dim=-1)).mT).mT

        moments = torch.bmm(self._rotation_matrix_from_isaac_to_ned.inverse(), moments_ned.unsqueeze(dim=-1)).squeeze(dim=-1)
        # print("position controller | moments: {}".format(moments))
        thrust_moments = torch.cat([total_thrust, moments], dim=-1)
        # thrust_moments: [n, 4]
        return thrust_moments, R_des, target_angle_vel
    


class VelocityController(nn.Module):
    '''
    SE(3) velocity controller.
    Inputs:
        root_state
        target_yaw
        target_vel
        target_acc
        target_angle_vel
        target_angle_acc
    Outputs:
        control cmd: rotors' throttles
    Attentions:
        1.original controller based on the NED world frame and NED body frame
        2.Isaac Sim provides ENU world frame and ENU body frame
        3.Need transfer between these two frames
    '''
    def __init__(
        self,
        g: torch.Tensor,
        uav_params: dict,
        controller_params: dict,
        n_envs: int, 
        target_value_compute: bool=False,
        random_sample_vel_gain: bool=False,
        random_mass: bool=False,
    ):
        super().__init__()
        self._target_value_compute = target_value_compute # target value compute mode, False: zero target, True: delta_value / delta_t
        self.g = nn.Parameter(g)
        # self.mass = nn.Parameter(torch.tensor(uav_params["mass"]))
        self.mass = nn.Parameter(torch.full((n_envs, 1), uav_params["mass"])) # [n_envs, 1]
        # pos_gain and vel_gain need to multiplicate mass
        # self.vel_gain = nn.Parameter(torch.as_tensor(controller_params["velocity_gain"]).float())
        self.vel_gain = nn.Parameter((torch.as_tensor(controller_params["velocity_gain"]).float()).repeat(n_envs, 1)) # [n_envs, 3]
        self.attitude_gain = nn.Parameter(torch.as_tensor(controller_params["attitude_gain"]).float())
        self.angle_vel_gain = nn.Parameter(torch.as_tensor(controller_params["angular_rate_gain"]).float())
        inertia = uav_params["inertia"]
        self.J = nn.Parameter(torch.diag_embed(torch.tensor([inertia["xx"], inertia["yy"], inertia["zz"]])))

        rotor_config = uav_params["rotor_configuration"]
        force_constants = torch.as_tensor(rotor_config["force_constants"])
        max_rot_vel = torch.as_tensor(rotor_config["max_rotation_velocities"])
        self.max_thrusts = nn.Parameter(max_rot_vel.square() * force_constants)

        # set initial targets, under the Isaac-sim frame
        self._init_last_target_vel = nn.Parameter(torch.zeros(n_envs, 3)) # [n_envs, 3]
        self._init_last_target_Rd = nn.Parameter(torch.diag_embed(torch.ones(n_envs, 3))) # [n_envs, 3, 3]
        self._init_last_target_angle_vel = nn.Parameter(torch.zeros(n_envs, 3)) # [n_envs, 3]
        self._init_last_t = nn.Parameter(torch.zeros(n_envs, 1)) # [n_envs, 1]

        # control allocation matrix, under isaac-sim frame
        rotor_angles = torch.as_tensor(rotor_config["rotor_angles"])
        arm_lengths = torch.as_tensor(rotor_config["arm_lengths"])
        moment_constants = torch.as_tensor(rotor_config["moment_constants"])
        directions = torch.as_tensor(rotor_config["directions"])
        ca_matrix = torch.stack(
            [
                torch.ones_like(rotor_angles),
                torch.sin(rotor_angles) * arm_lengths,
                -torch.cos(rotor_angles) * arm_lengths,
                directions * moment_constants / force_constants,
            ]
        )
        self.ca_matrix = nn.Parameter(ca_matrix)

        # setting rotation matrix from isaac-sim frame to NED frame
        self._rotation_matrix_from_isaac_to_ned = nn.Parameter(torch.diag_embed(torch.tensor([1.0, -1.0, -1.0]).repeat(n_envs, 1))) # [n_envs, 3, 3]
        self._rotation_matrix_from_isaac_body_to_ned_body = nn.Parameter(torch.diag_embed(torch.tensor([1.0, -1.0, -1.0]).repeat(n_envs, 1))) # [n_envs, 3, 3]

        self.random_sample_vel_gain = random_sample_vel_gain
        self.random_mass = random_mass
        self.reset()
        self.requires_grad_(False)

    def reset(self, env_ids=None):
        if env_ids is None:
            self.last_target_vel = self._init_last_target_vel.data.clone()
            self.last_target_Rd = self._init_last_target_Rd.data.clone()
            self.last_target_angle_vel = self._init_last_target_angle_vel.data.clone()
            self.last_t = self._init_last_t.data.clone()

            if self.random_mass:
                # for sampling mass
                # mass_values = torch.rand(self.mass.shape[0], 1, device=self.mass.device) * (0.72 - 0.61775) + 0.61775  # Uniformly sample from [0.61775, 0.72)
                mass_values = torch.rand(self.mass.shape[0], 1, device=self.mass.device) * (0.72 - 0.70) + 0.70
                self.mass = nn.Parameter(mass_values)  # [n_envs, 1]
        else:
            self.last_target_vel = self.last_target_vel.to(device=self._init_last_target_vel.data.device)
            self.last_target_Rd = self.last_target_Rd.to(device=self._init_last_target_Rd.data.device)
            self.last_target_angle_vel = self.last_target_angle_vel.to(device=self._init_last_target_angle_vel.data.device)
            self.last_t = self.last_t.to(device=self._init_last_t.data.device)
            self.last_target_vel[env_ids] = self._init_last_target_vel.data[env_ids].clone()
            self.last_target_Rd[env_ids] = self._init_last_target_Rd.data[env_ids].clone()
            self.last_target_angle_vel[env_ids] = self._init_last_target_angle_vel.data[env_ids].clone()
            self.last_t[env_ids] = self._init_last_t.data[env_ids].clone()

            if self.random_mass:
                # for sampling mass
                # mass_values = torch.rand(len(env_ids), 1, device=self.mass.device) * (0.72 - 0.61775) + 0.61775  # Uniformly sample from [0.61775, 0.72)
                mass_values = torch.rand(len(env_ids), 1, device=self.mass.device) * (0.72 - 0.70) + 0.70
                self.mass.data[env_ids] = mass_values.clone()

    def set_gain(self, gain_name: str, env_ids, values):
        _set_env_gain(self, gain_name, env_ids, values)

    def get_gain(self, gain_name: str) -> torch.Tensor:
        return getattr(self, gain_name).detach()


    def forward(
        self,
        current_t: torch.tensor,
        root_state: torch.Tensor,
        target_yaw: torch.Tensor=None,
        target_vel: torch.Tensor=None,
        target_acc: torch.Tensor=None,
        target_angle_vel: torch.Tensor=None,
        target_angle_acc: torch.Tensor=None,
        body_frame: bool=False
    ):
        '''
        root_state: [..., 13],
        target_pos: [..., 3],
        target_yaw: [...],
        target_vel: [..., 3],
        target_acc: [..., 3],
        target_angle_vel: [..., 3],
        target_angle_acc: [..., 3],
        '''

        # all inputs are under Issac-sim frame
        batch_shape = root_state.shape[:-1]
        current_t = current_t.unsqueeze(dim=-1)

        if target_yaw is None:
            target_yaw = quaternion_to_euler(root_state[..., 3:7])[..., -1] # [...]

        if target_vel is None:
            target_vel = root_state[..., 7:10] # [..., 3]

        if target_acc is None:
            if self._target_value_compute:
                target_acc = (target_vel - self.last_target_vel) / (current_t - self.last_t) # [..., 3]
            else:
                target_acc = torch.zeros_like(target_vel)

        thrust_moments, R_des, target_angle_vel = self._compute(
            current_t,
            root_state.reshape(-1, 13),
            target_yaw.reshape(-1, 1),
            target_vel.reshape(-1, 3),
            target_acc.reshape(-1, 3),
            target_angle_vel.reshape(-1, 3) if not target_angle_vel is None else None,
            target_angle_acc.reshape(-1, 3) if not target_angle_acc is None else None,
            body_frame
        )
        thrust_moments = thrust_moments.reshape(*batch_shape, -1)
        R_des = R_des.reshape(*batch_shape, 3, 3)
        target_angle_vel = target_angle_vel.reshape(*batch_shape, -1)
        # convert thrust_moments to rotors' throttles
        rotor_thrusts = (self.ca_matrix.inverse() @ thrust_moments.mT).mT # [n_envs, 4]
        rotor_thrusts = torch.where(rotor_thrusts<0, 0, rotor_thrusts)

        # print("velocity controller | rotor thrusts: {}".format(rotor_thrusts))
        throttle_cmds = (rotor_thrusts / self.max_thrusts).sqrt() # [n_envs, 4]

        self.last_target_vel = target_vel
        self.last_t = current_t
        self.last_target_Rd = R_des
        self.last_target_angle_vel = target_angle_vel
        return throttle_cmds

    def _compute(
        self,
        current_t: torch.Tensor,
        root_state: torch.Tensor,
        target_yaw: torch.Tensor=None,
        target_vel: torch.Tensor=None,
        target_acc: torch.Tensor=None,
        target_angle_vel: torch.Tensor=None,
        target_angle_acc: torch.Tensor=None,
        body_frame: bool=False
    ):
        '''
        root_state: [..., 13],
        target_yaw: [...],
        target_vel: [..., 3],
        target_acc: [..., 3],
        target_angle_vel: [..., 3],
        target_angle_acc: [..., 3],
        '''

        device = root_state.device

        # inputs are all under isaac-sim frame
        pos, quat, vel, ang_vel = torch.split(root_state, [3, 4, 3, 3], dim=-1)
        if not body_frame:
            # convert angle_vel from world frame to body frame
            ang_vel = quat_world_to_body(quat, ang_vel)
        R = quaternion_to_rotation_matrix(quat)

        # convert from isaac-sim frame to NED frame
        vel_ned = torch.bmm(self._rotation_matrix_from_isaac_to_ned, vel.unsqueeze(dim=-1)).squeeze(dim=-1)
        target_vel_ned = torch.bmm(self._rotation_matrix_from_isaac_to_ned, target_vel.unsqueeze(dim=-1)).squeeze(dim=-1)
        target_acc_ned = torch.bmm(self._rotation_matrix_from_isaac_to_ned, target_acc.unsqueeze(dim=-1)).squeeze(dim=-1)
        ang_vel_ned_body = torch.bmm(self._rotation_matrix_from_isaac_body_to_ned_body, ang_vel.unsqueeze(dim=-1)).squeeze(dim=-1)
        R_ned = torch.bmm(self._rotation_matrix_from_isaac_to_ned, torch.bmm(R, self._rotation_matrix_from_isaac_body_to_ned_body.inverse()))
        
        # force control
        vel_error = vel_ned - target_vel_ned
        vector_total_thrust = self.mass * (- self.vel_gain * vel_error - self.g + target_acc_ned)
        total_thrust = (-vector_total_thrust * R_ned[:, :, 2]).sum(dim=-1, keepdim=True)
        # print("velocity controller | total thrust: {}".format(total_thrust))

        # moment control
        b1_des = torch.cat([
            torch.cos(target_yaw),
            torch.sin(target_yaw),
            torch.zeros_like(target_yaw)
        ],dim=-1)
        b1_des_ned = torch.bmm(self._rotation_matrix_from_isaac_to_ned, b1_des.unsqueeze(dim=-1)).squeeze(dim=-1)
        b3_des_ned = -normalize(vector_total_thrust)
        b2_des_ned = normalize(torch.cross(b3_des_ned, b1_des_ned, dim=-1))
        R_des_ned = torch.stack([
            b2_des_ned.cross(b3_des_ned, -1),
            b2_des_ned,
            b3_des_ned
        ], dim=-1)

        R_des = torch.bmm(self._rotation_matrix_from_isaac_to_ned.inverse(), torch.bmm(R_des_ned, self._rotation_matrix_from_isaac_body_to_ned_body))

        R_error_matrix = 0.5 * (
            torch.bmm(R_des_ned.mT, R_ned)
            - torch.bmm(R_ned.mT, R_des_ned)
        )
        R_error = torch.stack([
            R_error_matrix[:, 2, 1],
            R_error_matrix[:, 0, 2],
            R_error_matrix[:, 1, 0],
        ],dim=-1)

        if target_angle_vel is None:
            if self._target_value_compute:
                target_angle_vel_matrix = torch.bmm(R_des.mT, (R_des - self.last_target_Rd) / (current_t - self.last_t).unsqueeze(dim=-1))
                target_angle_vel = torch.stack([
                    target_angle_vel_matrix[:, 2, 1],
                    target_angle_vel_matrix[:, 0, 2],
                    target_angle_vel_matrix[:, 1, 0]
                ],dim=-1)
            else:
                target_angle_vel = torch.zeros_like(ang_vel)
        target_angle_vel_ned_body = torch.bmm(self._rotation_matrix_from_isaac_body_to_ned_body, target_angle_vel.unsqueeze(dim=-1)).squeeze(dim=-1)
        
        if target_angle_acc is None:
            if self._target_value_compute:
                target_angle_acc = (target_angle_vel - self.last_target_angle_vel) / (current_t - self.last_t)
            else:
                target_angle_acc = torch.zeros_like(target_angle_vel)
        target_angle_acc_ned_body = torch.bmm(self._rotation_matrix_from_isaac_body_to_ned_body, target_angle_acc.unsqueeze(dim=-1)).squeeze(dim=-1)

        angle_vel_error = ang_vel_ned_body - torch.bmm(torch.bmm(R_ned.mT, R_des_ned), target_angle_vel_ned_body.unsqueeze(dim=-1)).squeeze(dim=-1)

        moments_ned = - self.attitude_gain * R_error \
                    - self.angle_vel_gain * angle_vel_error \
                    + torch.cross(ang_vel_ned_body, (self.J @ ang_vel_ned_body.mT).mT, dim=-1) \
                    - (self.J @ (torch.cross(ang_vel_ned_body, torch.bmm(torch.bmm(R_ned.mT, R_des_ned), target_angle_vel_ned_body.unsqueeze(dim=-1)).squeeze(dim=-1), dim=-1) - torch.bmm(torch.bmm(R_ned.mT, R_des_ned), target_angle_acc_ned_body.unsqueeze(dim=-1)).squeeze(dim=-1)).mT).mT

        moments = torch.bmm(self._rotation_matrix_from_isaac_to_ned.inverse(), moments_ned.unsqueeze(dim=-1)).squeeze(dim=-1)
        # print("velocity controller | moments: {}".format(moments))
        thrust_moments = torch.cat([total_thrust, moments], dim=-1)
        # thrust_moments: [n, 4]
        return thrust_moments, R_des, target_angle_vel
