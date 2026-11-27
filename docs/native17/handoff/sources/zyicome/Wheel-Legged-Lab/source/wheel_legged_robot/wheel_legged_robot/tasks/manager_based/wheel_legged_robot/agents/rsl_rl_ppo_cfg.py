# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlMLPModelCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class WheelLeggedFlatPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 48
    max_iterations = 2000
    save_interval = 100
    experiment_name = "wheel_legged_flat"
    obs_groups = {"actor": ["policy"], "critic": ["critic"]}
    clip_actions = 1.0
    actor = RslRlMLPModelCfg(
        hidden_dims=[256, 128, 64],
        activation="elu",
        obs_normalization=True,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(
            init_std=0.3,
            std_type="scalar",
        ),
    )
    critic = RslRlMLPModelCfg(
        hidden_dims=[256, 128, 64],
        activation="elu",
        obs_normalization=True,
    )
    # ===== PPO 算法配置 =====
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=3.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class WheelLeggedJumpFlatPPORunnerCfg(WheelLeggedFlatPPORunnerCfg):
    """PPO configuration for staged locomotion and jump learning."""

    experiment_name = "wheel_legged_jump_flat"
    max_iterations = 3000
    save_interval = 100
    num_steps_per_env = 64
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=5.0e-4,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=3.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class WheelLeggedHighJumpLandingPPORunnerCfg(WheelLeggedJumpFlatPPORunnerCfg):
    """Stage-C1 runner kept separate from the proven Stage-B2 experiment."""

    experiment_name = "wheel_legged_jump_high_landing_flat"
    max_iterations = 1600
    save_interval = 100
    num_steps_per_env = 64
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=7.5e-4,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=2.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.008,
        max_grad_norm=1.0,
    )


@configclass
class WheelLeggedJumpClearancePPORunnerCfg(WheelLeggedHighJumpLandingPPORunnerCfg):
    """Stage-C2 runner for explicit 8--12 cm wheel clearance."""

    experiment_name = "wheel_legged_jump_clearance_flat"
    max_iterations = 1600
    save_interval = 100


@configclass
class WheelLeggedMovingJumpPPORunnerCfg(WheelLeggedJumpClearancePPORunnerCfg):
    """Stage-C3 first moving-jump level."""

    experiment_name = "wheel_legged_jump_moving_flat"
    max_iterations = 1800
    save_interval = 100


@configclass
class WheelLeggedMovingJumpCurriculumPPORunnerCfg(WheelLeggedMovingJumpPPORunnerCfg):
    """Single-run 0.2→1.0 m/s moving-jump curriculum."""

    experiment_name = "wheel_legged_jump_moving_curriculum_flat"
    max_iterations = 4000
    save_interval = 100


@configclass
class WheelLeggedTargetLandingPPORunnerCfg(
    WheelLeggedMovingJumpCurriculumPPORunnerCfg
):
    """Stage-D1 runner for commanded flat-ground landing positions."""

    experiment_name = "wheel_legged_jump_target_landing_flat"
    max_iterations = 2200
    save_interval = 100


@configclass
class WheelLeggedObstacleOraclePPORunnerCfg(WheelLeggedTargetLandingPPORunnerCfg):
    """Stage-D2 runner for exact-geometry low-obstacle crossing."""

    experiment_name = "wheel_legged_jump_obstacle_oracle_flat"
    max_iterations = 2400
    save_interval = 100
    # The geometry curriculum introduces new takeoff/retraction solutions after
    # the inherited policy has already converged. Keep more exploration than
    # the earlier jump stages so the final obstacle levels can still adapt.
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=1.5e-3,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=2.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.008,
        max_grad_norm=1.0,
    )


@configclass
class WheelLeggedObstaclePerceptivePPORunnerCfg(
    WheelLeggedObstacleOraclePPORunnerCfg
):
    """Depth-camera adaptation initialized from the Oracle obstacle policy."""

    experiment_name = "wheel_legged_jump_obstacle_perceptive_flat"
    max_iterations = 2000
    save_interval = 100


@configclass
class WheelLeggedRecoveryPPORunnerCfg(WheelLeggedFlatPPORunnerCfg):
    """PPO runner for near-fall recovery and disturbance rejection."""

    experiment_name = "wheel_legged_recovery_flat"
    max_iterations = 1800
    save_interval = 100
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=1.0e-3,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=2.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class WheelLeggedTerrainReactivePPORunnerCfg(WheelLeggedRecoveryPPORunnerCfg):
    """PPO runner for proprioception-only mixed-terrain adaptation."""

    experiment_name = "wheel_legged_terrain_reactive"
    max_iterations = 2800
    save_interval = 100
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=1.5e-3,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=2.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class WheelLeggedTerrainPerceptivePPORunnerCfg(WheelLeggedTerrainReactivePPORunnerCfg):
    """Optional actor-side terrain-scan branch."""

    experiment_name = "wheel_legged_terrain_perceptive"
