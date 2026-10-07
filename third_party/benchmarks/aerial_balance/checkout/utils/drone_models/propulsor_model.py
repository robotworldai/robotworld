# -*- encoding: utf-8 -*-
'''
@File    :   propulsor_model.py
@Time    :   2024/10/31 21:47:59
@Author  :   wenminggong, AIMS, PolyU
@Version :   1.0
@Desc    :   Implementation of the ideal propulsor model of single rotor.
'''

import torch
import torch.nn as nn


class PropulsorModel(nn.Module):
    '''
    practical model:
        rotation_velocity = throttle * max_rotation_velocity
        max_thrust = force_constant * max_rotation_velocity^2
        max_moment = moment_constant * max_rotation_velocity^2
        thrust = throttle^2 * max_thrust^2
        moment = throttle^2 * max_moment^2
        throttle = tau * expected_throttle + (1-tau)*throttle = throttle + tau * (expected_throttle - throttle)
    input: expected_throttle
    output: thrust, moment, rotation_velocity
    '''
    def __init__(self, rotor_config, n_envs, noise_scale=0.0, random_propulsor_tau=False):
        super().__init__()
        force_constants = torch.as_tensor(rotor_config["force_constants"])
        moment_constants = torch.as_tensor(rotor_config["moment_constants"])
        self.max_rot_vels = nn.Parameter(torch.as_tensor(rotor_config["max_rotation_velocities"]).float())
        self.num_rotors = len(force_constants)
        self.n_envs = n_envs
        self.noise_scale = noise_scale
        self.random_propulsor_tau = random_propulsor_tau

        self.max_thrusts = nn.Parameter(self.max_rot_vels.square() * force_constants)
        self.max_moments = nn.Parameter(self.max_rot_vels.square() * moment_constants)
        self.directions = nn.Parameter(torch.as_tensor(rotor_config["directions"]).float())
        # self.tau = nn.Parameter(0.43 * torch.ones(self.num_rotors)) # how to get tau?
        self.tau = nn.Parameter(0.43 * torch.ones(self.n_envs, self.num_rotors)) # [n_envs, 4]
        self.init_throttles = nn.Parameter(torch.zeros(self.n_envs, self.num_rotors)) # initialize throttle as 0
        self.reset()
        self.requires_grad_(requires_grad=False)

    def reset(self, init_throttles=None, env_ids=None):
        # reset initial throttles
        if init_throttles is None:
            self.throttles = self.init_throttles.data.clone()
        else:
            assert (init_throttles <= 1.0).all() and (init_throttles >= 0.0).all(), "Illegal init throttle!"
            self.throttles = self.throttles.to(device=self.init_throttles.data.device)
            self.throttles[env_ids] = init_throttles

        # random sample tau
        if not (env_ids is None) and self.random_propulsor_tau:
            self.tau[env_ids].data = torch.rand_like(self.tau[env_ids]) * (0.8 - 0.2)+ 0.2 # [0.2, 0.8)

    def forward(self, expected_throttles: torch.Tensor):
        # expected_throttles: [n_envs, rotor_nums], [0, 1]
        expected_throttles = torch.clamp(expected_throttles, 0.0, 1.0)
        self.throttles.add_(self.tau * (expected_throttles - self.throttles))

        noise = torch.rand_like(self.throttles) * self.noise_scale * 2 - self.noise_scale # [-noise_scale, noise_scale]
        noise_throttles = torch.clamp(self.throttles + noise, 0.0, 1.0)

        rot_vels = noise_throttles * self.max_rot_vels * (-self.directions)
        t = torch.square(noise_throttles)
        thrusts = t * self.max_thrusts
        moments = (t * self.max_moments) * (self.directions)

        # print("rotor propulsor model | rotor thrusts: {}".format(thrusts))
        # print("rotor propulsor model | rotor moments: {}".format(moments))

        # return: [n_envs, rotor_nums]
        return thrusts, moments, rot_vels