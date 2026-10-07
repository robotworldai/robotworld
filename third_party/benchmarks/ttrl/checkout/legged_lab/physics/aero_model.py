# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Renhong Zhang

import torch
from typing import Dict, Tuple

class AeroModel:
    """
    Base class for aerodynamic models. Provides a common interface for easy integration
    of new force and torque models in the future. All calculations are batched.
    """
    def __init__(self, device: str = 'cpu'):
        """
        Initializes the model on a specified device.
        Args:
            device (str): The device to run computations on ('cpu' or 'cuda:0').
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

    def calculate_forces_and_torques(
        self,
        linear_velocities: torch.Tensor,
        angular_velocities: torch.Tensor,
        ball_properties: Dict[str, torch.Tensor],
        env_properties: Dict[str, float],
        model_params: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Abstract method to be implemented by subclasses.
        Calculates the aerodynamic forces and torques for a batch of balls.

        Args:
            linear_velocities (torch.Tensor): Tensor of shape (N, 3) for the linear velocity of N balls.
            angular_velocities (torch.Tensor): Tensor of shape (N, 3) for the angular velocity of N balls.
            ball_properties (Dict[str, torch.Tensor]): Dictionary of tensors for ball properties (e.g., 'radii', 'areas').
            env_properties (Dict[str, float]): Dictionary of scalar environment properties (e.g., 'air_density').
            model_params (Dict[str, torch.Tensor]): Dictionary of tensors for model-specific parameters.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - forces (torch.Tensor): Tensor of shape (N, 3) for the calculated aerodynamic forces.
                - torques (torch.Tensor): Tensor of shape (N, 3) for the calculated aerodynamic torques.
        """
        raise NotImplementedError("This method must be implemented by a subclass.")

class MeasuredAeroModel(AeroModel):
    """
    Calculates aerodynamic forces using a model from measurement with tunable coefficients.
    Forces are calculated in a batch for high efficiency.
    """
    def calculate_forces_and_torques(
        self,
        linear_velocities: torch.Tensor,
        angular_velocities: torch.Tensor,
        ball_properties: Dict[str, torch.Tensor],
        env_properties: Dict[str, float],
        model_params: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculates drag and Magnus forces.

        Args:
            model_params (Dict[str, torch.Tensor]): Must contain 'drag_coefficients' (N,) and 'magnus_factors' (N,).
        
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Calculated forces (N, 3) and zero torques (N, 3).
        """
        v = linear_velocities
        omega = angular_velocities
        v_mag = torch.linalg.norm(v, dim=1, keepdim=True)

        # Reshape parameters from (N,) to (N, 1) for broadcasting
        drag_coeffs = model_params['drag_coefficients'].unsqueeze(1)
        magnus_factors = model_params['magnus_factors'].unsqueeze(1)
        radii = ball_properties['radii'].unsqueeze(1)
        areas = ball_properties['areas'].unsqueeze(1)

        # 1. Drag Force: F_d = -0.5 * rho * A * C_d * |v| * v
        drag_force = -0.5 * env_properties['air_density'] * areas * drag_coeffs * v_mag * v

        # 2. Magnus Force: F_m = S_coeff * (omega x v) where S_coeff = 0.5 * rho * A * R * magnus_factor
        s_coeff = 0.5 * env_properties['air_density'] * areas * radii * magnus_factors
        magnus_force = s_coeff * torch.cross(omega, v, dim=1)

        total_force = drag_force + magnus_force
        
        torques = torch.zeros_like(total_force)

        return total_force, torques

