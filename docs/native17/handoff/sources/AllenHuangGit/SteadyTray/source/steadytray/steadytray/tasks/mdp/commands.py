from __future__ import annotations

from dataclasses import MISSING
import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.envs.mdp import UniformVelocityCommandCfg, UniformVelocityCommand
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply_inverse, quat_mul, quat_from_euler_xyz
import isaaclab.utils.math as math_utils

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


@configclass
class UniformVelocityBodyCommandCfg(UniformVelocityCommandCfg):
    """Configuration for uniform velocity command using a specific body instead of root link.
    
    This configuration extends UniformVelocityCommandCfg to support tracking a specific body
    (e.g., torso_link) instead of the root link for velocity commands and metrics.
    """
    
    body_name: str = "torso_link"
    """Name of the body to track for velocity commands. Defaults to 'torso_link'."""


class UniformVelocityBodyCommand(UniformVelocityCommand):
    """Command generator that uses a specific body (e.g., torso) instead of root link.
    
    This class inherits from UniformVelocityCommand but overrides the velocity tracking
    to use a specified body link (like torso_link) instead of the root link (pelvis).
    This ensures velocity commands and metrics are computed relative to the torso.
    """
    
    cfg: UniformVelocityBodyCommandCfg
    """The configuration of the command generator."""
    
    def __init__(self, cfg: UniformVelocityBodyCommandCfg, env):
        """Initialize the command generator.
        
        Args:
            cfg: The configuration of the command generator.
            env: The environment.
        """
        # Initialize the base class
        super().__init__(cfg, env)
        
        # Get the body ID for the specified body
        self.body_idx = self.robot.find_bodies(cfg.body_name)[0][0]
    
    def _update_metrics(self):
        """Update metrics using body velocity instead of root velocity."""
        # time for which the command was executed
        max_command_time = self.cfg.resampling_time_range[1]
        max_command_step = max_command_time / self._env.step_dt
        
        # Get body velocities instead of root velocities
        body_quat_w = self.robot.data.body_link_quat_w[:, self.body_idx, :]
        body_lin_vel_w = self.robot.data.body_link_lin_vel_w[:, self.body_idx, :3]
        body_ang_vel_w = self.robot.data.body_link_ang_vel_w[:, self.body_idx, :]

        # Transform body linear velocity to body frame
        body_lin_vel_b = quat_apply_inverse(body_quat_w, body_lin_vel_w)
        
        # Transform body angular velocity to body frame
        body_ang_vel_b = quat_apply_inverse(body_quat_w, body_ang_vel_w)
        
        # Compute metrics
        self.metrics["error_vel_xy"] += (
            torch.norm(self.vel_command_b[:, :2] - body_lin_vel_b[:, :2], dim=-1) / max_command_step
        )
        self.metrics["error_vel_yaw"] += (
            torch.abs(self.vel_command_b[:, 2] - body_ang_vel_b[:, 2]) / max_command_step
        )
    
    def _compute_body_heading_w(self) -> torch.Tensor:
        """Compute yaw heading of the body frame (in radians). Shape is (num_envs,).
        
        This is similar to robot.data.heading_w but uses the body frame instead of root frame.
        """
        
        # Get body quaternion
        body_quat_w = self.robot.data.body_link_quat_w[:, self.body_idx, :]
        
        # Forward vector in body frame (assuming x-direction is forward)
        # Need to repeat for all environments
        forward_vec_b = torch.tensor([[1.0, 0.0, 0.0]], device=self.device).repeat(self.num_envs, 1)
        
        # Transform forward vector to world frame
        forward_w = math_utils.quat_apply(body_quat_w, forward_vec_b)
        
        # Compute heading angle (yaw) from the forward vector
        return torch.atan2(forward_w[:, 1], forward_w[:, 0])
    
    def _update_command(self):
        """Post-processes the velocity command using body frame instead of root frame.
        
        This function sets velocity command to zero for standing environments and computes angular
        velocity from heading direction if the heading_command flag is set. Unlike the parent class,
        this uses the body heading instead of root heading.
        """
      
        # Compute angular velocity from heading direction (using body heading)
        if self.cfg.heading_command:
            # resolve indices of heading envs
            env_ids = self.is_heading_env.nonzero(as_tuple=False).flatten()
            # compute body heading instead of root heading
            body_heading_w = self._compute_body_heading_w()
            # compute angular velocity using body heading
            heading_error = math_utils.wrap_to_pi(self.heading_target[env_ids] - body_heading_w[env_ids])
            self.vel_command_b[env_ids, 2] = torch.clip(
                self.cfg.heading_control_stiffness * heading_error,
                min=self.cfg.ranges.ang_vel_z[0],
                max=self.cfg.ranges.ang_vel_z[1],
            )
        
        # Enforce standing (i.e., zero velocity command) for standing envs
        standing_env_ids = self.is_standing_env.nonzero(as_tuple=False).flatten()
        self.vel_command_b[standing_env_ids, :] = 0.0
    
    def _debug_vis_callback(self, event):
        """Update debug visualization using body position instead of root position."""
        # check if robot is initialized
        if not self.robot.is_initialized:
            return
        
        # get marker location using body position instead of root
        body_pos_w = self.robot.data.body_pos_w[:, self.body_idx, :].clone()
        body_pos_w[:, 2] += 0.5
        
        # Get body quaternion and velocities
        body_quat_w = self.robot.data.body_link_quat_w[:, self.body_idx, :]
        body_lin_vel_w = self.robot.data.body_link_lin_vel_w[:, self.body_idx, :3]

        # Transform body linear velocity to body frame
        body_lin_vel_b = quat_apply_inverse(body_quat_w, body_lin_vel_w)
        
        # resolve the scales and quaternions
        vel_des_arrow_scale, vel_des_arrow_quat = self._resolve_xy_velocity_to_arrow(
            self.command[:, :2], body_quat_w
        )
        vel_arrow_scale, vel_arrow_quat = self._resolve_xy_velocity_to_arrow(
            body_lin_vel_b[:, :2], body_quat_w
        )
        
        # display markers
        self.goal_vel_visualizer.visualize(body_pos_w, vel_des_arrow_quat, vel_des_arrow_scale)
        self.current_vel_visualizer.visualize(body_pos_w, vel_arrow_quat, vel_arrow_scale)
    
    def _resolve_xy_velocity_to_arrow(
        self, xy_velocity: torch.Tensor, body_quat_w: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Converts the XY velocity command to arrow direction rotation using body frame.
        
        Args:
            xy_velocity: The XY velocity in body frame.
            body_quat_w: The body quaternion in world frame.
            
        Returns:
            Tuple of arrow scale and arrow quaternion.
        """
        # obtain default scale of the marker
        default_scale = self.goal_vel_visualizer.cfg.markers["arrow"].scale
        # arrow-scale
        arrow_scale = torch.tensor(default_scale, device=self.device).repeat(xy_velocity.shape[0], 1)
        arrow_scale[:, 0] *= torch.linalg.norm(xy_velocity, dim=1) * 3.0
        # arrow-direction
        heading_angle = torch.atan2(xy_velocity[:, 1], xy_velocity[:, 0])
        zeros = torch.zeros_like(heading_angle)
        arrow_quat = quat_from_euler_xyz(zeros, zeros, heading_angle)
        # convert everything from body to world frame using body quaternion
        arrow_quat = quat_mul(body_quat_w, arrow_quat)
        
        return arrow_scale, arrow_quat


@configclass
class UniformLevelVelocityBodyCommandCfg(UniformVelocityBodyCommandCfg):
    """Configuration for uniform level velocity command using body frame."""
    limit_ranges: UniformVelocityCommandCfg.Ranges = MISSING


@configclass
class DelayedUniformVelocityCommandCfg(UniformLevelVelocityBodyCommandCfg):
    """Configuration for uniform velocity command with an initial delay.
    
    The command will be zero for the first `delay_time` seconds, then sample
    uniformly from the specified ranges.
    """
    
    # Time (in seconds) to keep velocity at zero before sampling
    delay_time: float = 1.0


class DelayedUniformVelocityCommand(UniformVelocityBodyCommand):
    """Command generator that keeps velocity at zero for a specified time, then samples uniformly.
    
    This command generator will output zero velocity for the first `delay_time` seconds
    after reset, then sample a new velocity command from the uniform distribution.
    Inherits all functionality from UniformVelocityCommand including heading commands,
    standing environments, and debug visualization.
    """
    
    cfg: DelayedUniformVelocityCommandCfg
    """The configuration of the command generator."""
    
    def __init__(self, cfg: DelayedUniformVelocityCommandCfg, env: ManagerBasedRLEnv):
        """Initialize the command generator.
        
        Args:
            cfg: The configuration of the command generator.
            env: The environment.
        """
        # Initialize the base class (UniformVelocityCommand)
        super().__init__(cfg, env)
        
        # Track time since reset for each environment
        self.time_since_reset = torch.zeros(self.num_envs, device=self.device)
        
        # Store sampled commands (that will be used after delay)
        self.sampled_commands = torch.zeros(self.num_envs, 3, device=self.device)
    
    def __str__(self) -> str:
        """Return a string representation of the command generator."""
        msg = "DelayedUniformVelocityCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tDelay time: {self.cfg.delay_time}s\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}\n"
        msg += f"\tHeading command: {self.cfg.heading_command}\n"
        if self.cfg.heading_command:
            msg += f"\tHeading probability: {self.cfg.rel_heading_envs}\n"
        msg += f"\tStanding probability: {self.cfg.rel_standing_envs}"
        return msg
    
    def _resample_command(self, env_ids: Sequence[int]):
        """Resample the command for the specified environments."""
        # Reset time since reset
        self.time_since_reset[env_ids] = 0.0
        
        # Call parent's resample to sample new commands
        super()._resample_command(env_ids)
        
        # Store the sampled commands (will be used after delay)
        self.sampled_commands[env_ids] = self.vel_command_b[env_ids].clone()
        
        # Set current command to zero (will be active during delay period)
        self.vel_command_b[env_ids] = 0.0
    
    def _update_command(self):
        """Update the command based on the current state and time."""
        # Update time since reset
        self.time_since_reset += self._env.step_dt
        
        # Check which environments have passed the delay period
        delay_passed_mask = self.time_since_reset >= self.cfg.delay_time
        
        # For environments that passed the delay, use the sampled command
        self.vel_command_b[delay_passed_mask] = self.sampled_commands[delay_passed_mask]
        
        # For environments still in delay period, keep zero velocity
        self.vel_command_b[~delay_passed_mask] = 0.0
        
        # Call parent's update command to handle heading and standing environments
        super()._update_command()