# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg, RslRlDistillationAlgorithmCfg


@configclass
class RslRlAdaptedActorCriticCfg(RslRlPpoActorCriticCfg):
    """Configuration for adapter-based actor-critic with temporal encoder.
    
    Inherits standard actor-critic parameters from RslRlPpoActorCriticCfg and adds
    adapter-specific parameters for parameter-efficient fine-tuning.
    
    Supports two adapter types:
    - AdaptedActorCritic: FiLM adapters (feature-wise linear modulation)
    - ResidualActorCritic: Residual action adapter (action-space corrections)
    
    Supports multiple encoder types:
    - GRU: Recurrent encoder (default, lightweight)
    - Transformer: Self-attention encoder (better long-range dependencies)
    - CausalTransformer: Masked self-attention (strictly causal)
    """
    
    class_name: str = "AdaptedActorCritic"
    """The policy class name. Options: 'AdaptedActorCritic' (FiLM) or 'ResidualActorCritic' (residual action)."""
    
    # Adapter type selection
    adapter_type: str = "film"
    """Type of adapter: 'film' (FiLM adapters) or 'residual' (residual action adapter).
    - 'film': Feature-wise linear modulation in hidden layers
    - 'residual': Direct action-space corrections
    Note: When using AdaptedStudentTeacher, this parameter is required."""
    
    # Encoder architecture selection
    encoder_type: str = "gru"
    """Type of encoder: 'gru', 'transformer', or 'causal_transformer'."""
    
    # Common adapter parameters (shared by both FiLM and Residual)
    ctx_dim: int = 64
    """Context embedding dimension from history encoder."""
    encoder_layers: int = 1
    """Number of encoder layers (GRU layers or Transformer blocks)."""
    use_gate: bool = False
    """Whether to use learnable gating parameter α in adapters."""
    
    # FiLM adapter-specific parameters (only used if class_name='AdaptedActorCritic')
    adapter_hidden: int = 64
    """Hidden dimension for FiLM adapter modulation networks."""
    clamp_gamma: float = 2.0
    """Clamping range for FiLM gamma values (±clamp_gamma)."""
    
    # Residual adapter-specific parameters (only used if class_name='ResidualActorCritic')
    residual_hidden_dims: list[int] = [128, 64]
    """Hidden dimensions for residual action MLP. Default: [128, 64]."""
    clamp_residual: float | None = None
    """Clamping range for residual actions (±clamp_residual)."""
    
    # Transformer-specific parameters (only used if encoder_type is 'transformer' or 'causal_transformer')
    num_heads: int = 4
    """Number of attention heads for Transformer encoder (must divide ctx_dim evenly)."""
    encoder_dropout: float = 0.0
    """Dropout rate for Transformer encoder. Default 0.0 (no dropout).
    For RL with large parallel environments, dropout is typically not needed.
    Use 0.1-0.2 only if overfitting (rare with 1000+ envs)."""


@configclass
class RslRlAdapterDistillationAlgorithmCfg(RslRlDistillationAlgorithmCfg):
    """Configuration for adapter-based distillation algorithm.
    
    Extends RslRlDistillationAlgorithmCfg with adapter-specific parameters.
    """
    
    class_name: str = "Distillation"
    """Algorithm class name."""
    
    # Distillation loss weights
    embedding_loss_coef: float = 1.0
    """Weight for embedding distillation loss (λ_embed)."""
    action_loss_coef: float = 1.0
    """Weight for action distribution loss (λ_action)."""
    
    # Loss function type
    loss_type: str = "mse"
    """Loss function type: 'mse' or 'huber'."""
    
    # Maximum gradient norm
    max_grad_norm: float = 1.0
    """Maximum gradient norm for clipping."""
    
    # DAgger (Dataset Aggregation) parameters
    use_dagger: bool = False
    """Whether to use DAgger for distillation warm-up."""
    dagger_beta_start: float = 1.0
    """Initial probability of using teacher actions (β). 1.0 = always teacher."""
    dagger_beta_end: float = 0.0
    """Final probability of using teacher actions. 0.0 = always student."""
    dagger_decay_steps: int = 1000
    """Number of learning iterations over which to decay β from start to end."""
    dagger_schedule_type: str = "linear"
    """Beta decay schedule: 'linear' or 'exponential'."""

@configclass
class RslRlPpoAlgorithmCurriculumCfg(RslRlPpoAlgorithmCfg):
    # Upper body curriculum configuration
    upper_body_cur_cfg: dict | None = None
    """Configuration for upper body curriculum training. Default is None, in which case it is not used."""


@configclass
class BasePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 50000
    save_interval = 100
    experiment_name = ""  # same as task name
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
    clip_actions = 100.0  # action clipping value


@configclass
class G1PPORunnerCfg(BasePPORunnerCfg):
    """PPO configuration for Unitree G1 robot with upper body curriculum."""

    algorithm = RslRlPpoAlgorithmCurriculumCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,

        # Upper body curriculum for G1 29DOF robot
        upper_body_cur_cfg={
            # Regularization coefficient (strength of the penalty)
            "coeff": 1,
            
            # Joint indices for G1 29DOF arms
            "joint_indices": [11, 12, 15, 16, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28],  # Arms only
            
            # Enable curriculum learning to gradually reduce regularization
            "use_curriculum": False,
            
            # Initial curriculum threshold (will be adjusted during training)
            "curriculum_threshold": 1.0,
        },
    )

@configclass
class G1AdapterSteadyTrayRunnerCfg(RslRlOnPolicyRunnerCfg):
    """
    Adapter-based training configuration for G1 steady tray task.
    
    Uses parameter-efficient fine-tuning with:
    - Frozen pre-trained base policy (locomotion)
    - Trainable adapters for task-specific modulation
    - Trainable encoder for temporal context
    - Trainable critic for value estimation
    
    Three observation groups required:
    - "policy": 5-step flattened history for frozen base actor (e.g., 480 dims = 5 * 96)
    - "encoder": 32-step sequence for encoder (e.g., [32, 96] per-timestep)
    - "adapted_critic": Privileged observations for critic (e.g., 495 dims)
    """
    
    num_steps_per_env = 24
    max_iterations = 50000
    save_interval = 100
    experiment_name = ""
    empirical_normalization = False
    
    # Adapter-based policy configuration
    policy = RslRlAdaptedActorCriticCfg(
        class_name="AdaptedActorCritic",  # "AdaptedActorCritic" for FiLM, "ResidualActorCritic" for residual
        init_noise_std=0.3,
        
        # Frozen base actor architecture (must match pre-trained model)
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        
        # Encoder architecture
        encoder_type="transformer",
        ctx_dim=64,             # Context embedding dimension from encoder
        encoder_layers=2,        # Number of Transformer layers
        num_heads=4,             # Attention heads (must divide ctx_dim evenly)
        encoder_dropout=0.0,     # No dropout for stable RL training

        # Learned gating parameter α for adapters
        use_gate=False,           # Use learnable gating parameter α.
        
        # Residual adapter parameters
        residual_hidden_dims=[512, 256, 128],  # MLP architecture for residual actions
        clamp_residual=None,          # Clamping range for residual action(±clamp_residual).

        # FiLM adapter parameters
        clamp_gamma=3.0,         # Relaxed from 2.0 to allow more modulation
        adapter_hidden=128,      # Increased from 64 for more expressiveness
    )
    
    # PPO algorithm configuration
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
    
    clip_actions = 100.0


@configclass
class G1AdapterDistillationRunnerCfg(RslRlOnPolicyRunnerCfg):
    """
    Distillation training configuration for G1 steady tray task.
    
    Uses online encoder distillation to train a student encoder with limited observations
    to mimic a teacher encoder with full privileged observations:
    - Frozen teacher encoder (from pre-trained AdaptedActorCritic)
    - Trainable student encoder (learns from limited observations)
    - Frozen actor body, adapters, action head, and critic (all shared)
    
    Four observation groups required:
    - "policy": 5-step flattened history for frozen base actor (e.g., 480 dims = 5 * 96)
    - "student_encoder": Limited observations for student encoder (e.g., [32, 48] proprioceptive only)
    - "teacher_encoder": Full privileged observations for teacher encoder (e.g., [32, 96] with external state)
    - "critic": Privileged observations for frozen critic (not used in distillation loss)
    """
    
    num_steps_per_env = 24
    max_iterations = 50000
    save_interval = 100
    experiment_name = ""
    empirical_normalization = False
    
    # Student-teacher policy configuration
    # Uses same config as AdaptedActorCritic, just changes class_name to AdaptedStudentTeacher
    policy = RslRlAdaptedActorCriticCfg(
        class_name="AdaptedStudentTeacher",
        adapter_type="residual",  # "film" for FiLM adapters, "residual" for residual action adapter
        init_noise_std=1e-3,
        
        # Frozen base actor architecture (must match pre-trained model)
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        
        # Encoder architecture
        encoder_type="transformer",
        ctx_dim=64,              # Context embedding dimension from encoder
        encoder_layers=2,        # Number of Transformer layers
        num_heads=4,             # Attention heads (must divide ctx_dim evenly)
        encoder_dropout=0.0,     # No dropout for stable RL training

        # Learned gating parameter α for adapters
        use_gate=False,           # Use learnable gating parameter α

        # Residual adapter parameters
        residual_hidden_dims=[512, 256, 128],  # MLP architecture for residual actions
        clamp_residual=None,          # Clamping range for residual action(±clamp_residual).

        # FiLM adapter parameters
        clamp_gamma=3.0,         # Relaxed from 2.0 to allow more modulation
        adapter_hidden=128,      # Increased from 64 for more expressiveness
    )
    
    # Distillation algorithm configuration
    algorithm = RslRlAdapterDistillationAlgorithmCfg(
        class_name="Distillation",
        num_learning_epochs=1,
        learning_rate=5.0e-5,
        gradient_length=1,
        max_grad_norm=0.5,
        embedding_loss_coef=1.0,
        action_loss_coef=1.0,
        loss_type="mse",
        use_dagger=False,
        dagger_beta_start=1.0,
        dagger_beta_end=0.0,
        dagger_decay_steps=500,
        dagger_schedule_type="linear",
    )
    
    clip_actions = 100.0

