from dataclasses import dataclass

import cusrl
from cusrl.environment.isaaclab import TrainerCfg


@dataclass
class WheelLeggedFlatTrainerCfg(TrainerCfg):
    max_iterations = 2000
    save_interval = 100
    experiment_name = "wheel_legged_flat"
    agent_factory = cusrl.ActorCritic.Factory(
        num_steps_per_update=48,
        actor_factory=cusrl.Actor.Factory(
            backbone_factory=cusrl.Mlp.Factory(
                hidden_dims=[256, 128, 64], activation_fn="ELU", ends_with_activation=True
            ),
            # CusRL 默认为 init_std=1.0；配合 24 rad/s 轮速尺度会让未训练
            # 策略大量动作直接撞到 ±1。与 RSL-RL 保持一致，从 0.3 开始探索。
            distribution_factory=cusrl.NormalDist.Factory(init_std=0.3),
        ),
        critic_factory=cusrl.Value.Factory(
            backbone_factory=cusrl.Mlp.Factory(
                hidden_dims=[256, 128, 64], activation_fn="ELU", ends_with_activation=True
            ),
        ),
        optimizer_factory=cusrl.OptimizerFactory("AdamW", defaults={"lr": 3.0e-4}),
        sampler=cusrl.AutoMiniBatchSampler(num_epochs=5, num_mini_batches=4),
        hooks=[
            cusrl.hook.ValueComputation(),
            cusrl.hook.GeneralizedAdvantageEstimation(gamma=0.99, lamda=0.95),
            cusrl.hook.AdvantageNormalization(),
            cusrl.hook.ValueLoss(),
            cusrl.hook.OnPolicyPreparation(),
            cusrl.hook.PpoSurrogateLoss(),
            cusrl.hook.EntropyLoss(weight=0.005),
            cusrl.hook.GradientClipping(max_grad_norm=1.0),
            cusrl.hook.OnPolicyStatistics(sampler=cusrl.AutoMiniBatchSampler()),
            cusrl.hook.AdaptiveLRSchedule(desired_kl_divergence=0.01),
        ],
    )
