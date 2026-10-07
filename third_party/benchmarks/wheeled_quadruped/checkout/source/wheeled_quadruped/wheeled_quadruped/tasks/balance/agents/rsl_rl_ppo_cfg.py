# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class WheeledQuadrupedBalancePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 1000
    save_interval = 100
    experiment_name = "wheeled_quadruped_balance"
    empirical_normalization = False
    # Asymmetric actor-critic: the actor reads the onboard-only "policy" group and
    # the critic reads the privileged "critic" group. In rsl-rl (>=3.0) this mapping
    # is resolved from the runner cfg, not from the vec-env wrapper.
    obs_groups = {"policy": ["policy"], "critic": ["critic"]}
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[128, 128],
        critic_hidden_dims=[128, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class WheeledQuadrupedBalanceRoughPPORunnerCfg(WheeledQuadrupedBalancePPORunnerCfg):
    """Balance on rough terrain: same PPO setup, more iterations, separate log dir."""

    max_iterations = 1500
    experiment_name = "wheeled_quadruped_balance_rough"
