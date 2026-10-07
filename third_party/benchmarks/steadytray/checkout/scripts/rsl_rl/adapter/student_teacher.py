# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal

from rsl_rl.utils import resolve_nn_activation
from .encoder import GRUEncoder, TransformerEncoder, CausalTransformerEncoder
from .adapter import FiLMAdapter, ResidualActionAdapter
from .actor_critic import AdapterSequential


class AdaptedStudentTeacher(nn.Module):
    """
    Student-Teacher architecture for online encoder distillation.
    
    Supports both FiLM adapters and Residual action adapters.
    
    Architecture Overview:
    ----------------------
    Teacher (frozen): Complete adapted actor-critic with trained encoder
        - Frozen base actor MLP (or frozen actor with adapters)
        - Frozen FiLM adapters (if using AdaptedActorCritic)
        - Frozen residual adapter (if using ResidualActorCritic)
        - Frozen GRU/Transformer encoder (generates teacher embeddings from full observations)
        - Frozen critic
    
    Student (trainable encoder only): Uses teacher's actor components and critic
        - Frozen base actor MLP (or frozen actor - shared with teacher)
        - Frozen adapters (shared with teacher)
        - Trainable student GRU/Transformer encoder (learns from limited observations)
        - Frozen critic (shared with teacher)
    
    Observation Types:
    ------------------
    1. student_encoder_obs: Limited observations for student encoder
       - Shape: [num_envs, seq_len, num_student_encoder_obs]
       - Example: Proprioceptive only
       - Used by: Student encoder (trainable)
       
    2. teacher_encoder_obs: Full privileged observations for teacher encoder  
       - Shape: [num_envs, seq_len, num_teacher_encoder_obs]
       - Example: Proprioceptive + external state
       - Used by: Teacher encoder (frozen, inference only)
       
    3. policy_obs: Flattened history for base actor MLP
       - Shape: [num_envs, num_actor_obs]
       - Example: 5-step flattened history for frozen base policy
       - Used by: Actor body (frozen)
       
    4. critic_obs: Privileged observations for critic
       - Shape: [num_envs, num_critic_obs]
       - Used by: Critic MLP (frozen)
    
    Distillation Losses:
    -------------------
    1. **Embedding Loss**: MSE between student and teacher encoder outputs
       - Loss = ||student_encoder(student_obs) - teacher_encoder(teacher_obs)||²
       - Enforces the student to produce similar latent representations
       
    2. **Action Distribution Loss**: KL divergence between distributions
       - Student action dist: N(μ_s, σ) where μ_s = actor(policy_obs, student_latent)
       - Teacher action dist: N(μ_t, σ) where μ_t = actor(policy_obs, teacher_latent)
       - Loss = KL(N(μ_t, σ) || N(μ_s, σ))
       - Enforces the student to produce similar action distributions
    
    Training Strategy:
    ------------------
    - **Trainable**: Only student_encoder (GRU/Transformer) parameters
    - **Frozen**: All other components (actor components, adapters, critic, noise_std)
    - **Shared**: Actor components, adapters, and critic are shared between student and teacher
    - **Online**: Distillation happens during environment interaction (no offline dataset needed)
    
    """
    is_recurrent = False

    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        num_actions,
        num_student_encoder_obs,  # Student encoder input dim (e.g., proprioceptive only)
        num_teacher_encoder_obs,  # Teacher encoder input dim (e.g., full observations)
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],
        activation="elu",
        init_noise_std=1.0,
        noise_std_type: str = "scalar",
        # Common adapter parameters
        ctx_dim: int = 64,
        encoder_layers: int = 1,
        use_gate: bool = True,
        # Encoder architecture selection
        encoder_type: str = "gru",
        num_heads: int = 4,
        encoder_dropout: float = 0.1,
        # Adapter type selection
        adapter_type: str = "film",  # "film" or "residual"
        # FiLM adapter-specific parameters
        adapter_hidden: int = 64,
        clamp_gamma: float = 2.0,
        # Residual adapter-specific parameters
        residual_hidden_dims: list[int] = [128, 64],
        clamp_residual: float = 1.0,
        **kwargs,
    ):
        if kwargs:
            print(
                "AdaptedStudentTeacher.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()
        
        activation = resolve_nn_activation(activation)
        self.num_actor_obs = num_actor_obs
        self.num_critic_obs = num_critic_obs
        self.num_actions = num_actions
        self.ctx_dim = ctx_dim
        self.num_student_encoder_obs = num_student_encoder_obs
        self.num_teacher_encoder_obs = num_teacher_encoder_obs
        self.loaded_teacher = False  # indicates if teacher has been loaded
        self.adapter_type = adapter_type.lower()
        self.encoder_type = encoder_type.lower()

        # ============================================================
        # Actor: Build architecture based on adapter type
        # ============================================================
        
        if self.adapter_type == "film":
            # FiLM adapter architecture: Frozen base MLP + FiLM adapters
            actor_body_layers = []
            actor_body_layers.append(nn.Linear(num_actor_obs, actor_hidden_dims[0]))
            actor_body_layers.append(activation)
            for layer_index in range(len(actor_hidden_dims) - 1):
                actor_body_layers.append(nn.Linear(actor_hidden_dims[layer_index], actor_hidden_dims[layer_index + 1]))
                actor_body_layers.append(activation)
            
            # Wrap body linear layers with FiLM adapters
            adapted_body_layers = []
            for layer in actor_body_layers:
                if isinstance(layer, nn.Linear):
                    # Freeze the base linear layer
                    for param in layer.parameters():
                        param.requires_grad_(False)
                    # Wrap with FiLM adapter (frozen)
                    adapter = FiLMAdapter(layer, ctx_dim, hidden=adapter_hidden, 
                                         clamp_gamma=clamp_gamma, use_gate=use_gate)
                    # Freeze adapter parameters (shared with teacher)
                    for param in adapter.parameters():
                        param.requires_grad_(False)
                    adapted_body_layers.append(adapter)
                else:
                    adapted_body_layers.append(layer)
            
            self.actor_body = AdapterSequential(*adapted_body_layers)
            
            # Action head: frozen linear layer (shared)
            self.action_head = nn.Linear(actor_hidden_dims[-1], num_actions)
            for param in self.action_head.parameters():
                param.requires_grad_(False)
            
            self.residual_adapter = None
            
        elif self.adapter_type == "residual":
            # Residual adapter architecture: Frozen full actor + Residual adapter
            actor_layers = []
            actor_layers.append(nn.Linear(num_actor_obs, actor_hidden_dims[0]))
            actor_layers.append(activation)
            for layer_index in range(len(actor_hidden_dims) - 1):
                actor_layers.append(nn.Linear(actor_hidden_dims[layer_index], actor_hidden_dims[layer_index + 1]))
                actor_layers.append(activation)
            actor_layers.append(nn.Linear(actor_hidden_dims[-1], num_actions))
            
            self.frozen_actor = nn.Sequential(*actor_layers)
            
            # Freeze all actor parameters
            for param in self.frozen_actor.parameters():
                param.requires_grad_(False)
            
            # Residual action adapter (frozen - shared with teacher)
            # Input: concatenation of encoder latent (ctx_dim) and proprio observations (num_actor_obs)
            self.residual_adapter = ResidualActionAdapter(
                num_actions, ctx_dim, proprio_dim=num_actor_obs,
                hidden_dims=residual_hidden_dims,
                use_gate=use_gate, clamp_residual=clamp_residual
            )
            for param in self.residual_adapter.parameters():
                param.requires_grad_(False)
            
            self.actor_body = None
            self.action_head = None
            
        else:
            raise ValueError(f"Unknown adapter_type: {self.adapter_type}. Must be 'film' or 'residual'")
        
        # ============================================================
        # Encoders: Student (trainable) + Teacher (frozen)
        # ============================================================
        
        # Student encoder: trainable, uses limited observations
        if self.encoder_type == "gru":
            self.student_encoder = GRUEncoder(num_student_encoder_obs, ctx_dim, num_layers=encoder_layers)
        elif self.encoder_type == "transformer":
            self.student_encoder = TransformerEncoder(
                num_student_encoder_obs, ctx_dim,
                num_layers=encoder_layers,
                num_heads=num_heads,
                dropout=encoder_dropout
            )
        elif self.encoder_type == "causal_transformer":
            self.student_encoder = CausalTransformerEncoder(
                num_student_encoder_obs, ctx_dim,
                num_layers=encoder_layers,
                num_heads=num_heads,
                dropout=encoder_dropout
            )
        else:
            raise ValueError(f"Unknown encoder_type: {self.encoder_type}. Must be 'gru', 'transformer', or 'causal_transformer'")
        
        # Teacher encoder: frozen, uses full privileged observations
        if self.encoder_type == "gru":
            self.teacher_encoder = GRUEncoder(num_teacher_encoder_obs, ctx_dim, num_layers=encoder_layers)
        elif self.encoder_type == "transformer":
            self.teacher_encoder = TransformerEncoder(
                num_teacher_encoder_obs, ctx_dim,
                num_layers=encoder_layers,
                num_heads=num_heads,
                dropout=encoder_dropout
            )
        elif self.encoder_type == "causal_transformer":
            self.teacher_encoder = CausalTransformerEncoder(
                num_teacher_encoder_obs, ctx_dim,
                num_layers=encoder_layers,
                num_heads=num_heads,
                dropout=encoder_dropout
            )
        else:
            raise ValueError(f"Unknown encoder_type: {self.encoder_type}")
        
        for param in self.teacher_encoder.parameters():
            param.requires_grad_(False)
        self.teacher_encoder.eval()
        
        # ============================================================
        # Critic: Frozen MLP (shared)
        # ============================================================
        
        # Critic input: concatenate critic observations + encoder latent vector
        critic_input_dim = num_critic_obs + ctx_dim
        
        critic_layers = []
        critic_layers.append(nn.Linear(critic_input_dim, critic_hidden_dims[0]))
        critic_layers.append(activation)
        for layer_index in range(len(critic_hidden_dims)):
            if layer_index == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[layer_index], 1))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dims[layer_index], critic_hidden_dims[layer_index + 1]))
                critic_layers.append(activation)
        self.critic = nn.Sequential(*critic_layers)
        # Freeze critic (shared with teacher)
        for param in self.critic.parameters():
            param.requires_grad_(False)

        # Action noise
        self.noise_std_type = noise_std_type
        if self.noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
            # Freeze noise std (shared with teacher)
            self.std.requires_grad_(False)
        elif self.noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
            self.log_std.requires_grad_(False)
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")

        # Action distribution (populated in update_distribution)
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args(False)

        print(f"\n{'='*70}")
        print(f"AdaptedStudentTeacher Architecture Initialized:")
        print(f"{'-'*70}")
        print(f"Adapter Type: {self.adapter_type.upper()}")
        print(f"Encoder Type: {self.encoder_type.upper()}")
        print(f"{'-'*70}")
        print(f"Shared Components (frozen):")
        if self.adapter_type == "film":
            print(f"  - Actor body: {actor_hidden_dims} with {len([l for l in self.actor_body if isinstance(l, FiLMAdapter)])} FiLM adapters")
            print(f"  - Action head: {num_actions} actions")
        elif self.adapter_type == "residual":
            print(f"  - Frozen actor: {actor_hidden_dims} -> {num_actions} actions")
            print(f"  - Residual adapter: {residual_hidden_dims}")
        print(f"  - Critic: {critic_hidden_dims}")
        print(f"{'-'*70}")
        print(f"Student Encoder (trainable):")
        print(f"  Input: {num_student_encoder_obs} (limited observations)")
        print(f"  Hidden: {ctx_dim}, Layers: {encoder_layers}")
        print(f"{'-'*70}")
        print(f"Teacher Encoder (frozen):")
        print(f"  Input: {num_teacher_encoder_obs} (full observations)")
        print(f"  Hidden: {ctx_dim}, Layers: {encoder_layers}")
        print(f"{'='*70}\n")

    def reset(self, dones=None, hidden_states=None):
        """Reset method for compatibility. Since we don't maintain hidden states, this is a no-op."""
        pass
    
    def get_hidden_states(self):
        """Get hidden states for compatibility with recurrent interface. Returns None since we're non-recurrent."""
        return None

    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self):
        if self.distribution is None:
            # Return a dummy tensor if distribution hasn't been updated
            # This can happen when using deterministic actions (act_inference) during DAgger
            return torch.zeros(self.num_actions, device=next(self.parameters()).device)
        return self.distribution.mean

    @property
    def action_std(self):
        if self.distribution is None:
            # Return the base std parameter if distribution hasn't been updated
            if self.noise_std_type == "scalar":
                return self.std
            elif self.noise_std_type == "log":
                return torch.exp(self.log_std)
        return self.distribution.stddev

    @property
    def entropy(self):
        if self.distribution is None:
            # Return zero entropy if distribution hasn't been updated
            return torch.zeros(1, device=next(self.parameters()).device)
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, student_encoder_obs, policy_obs):
        """
        Update action distribution using student encoder.
        
        Args:
            student_encoder_obs: Student encoder observations (limited, e.g., proprioceptive only)
            policy_obs: Policy observations for base actor MLP
        """
        # Get context embedding from student encoder
        e_t = self.get_student_encoder_latent(student_encoder_obs)
        
        # Compute action mean based on adapter type
        if self.adapter_type == "film":
            # FiLM: adapted actor body + frozen action head
            h = self.actor_body(policy_obs, e_t)
            mean = self.action_head(h)
        elif self.adapter_type == "residual":
            # Residual: frozen actor + residual adapter with proprio context
            base_actions = self.frozen_actor(policy_obs)
            mean = self.residual_adapter(base_actions, e_t, proprio=policy_obs)

        # Compute standard deviation
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        elif self.noise_std_type == "log":
            std = torch.exp(self.log_std).expand_as(mean)
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}")
        
        # create distribution
        self.distribution = Normal(mean, std)

    def act(self, student_encoder_obs, policy_obs):
        """
        Sample actions from the policy using student encoder.
        
        Args:
            student_encoder_obs: Student encoder observations
            policy_obs: Policy observations
        """
        self.update_distribution(student_encoder_obs, policy_obs)
        return self.distribution.sample()

    def act_inference(self, student_encoder_obs, policy_obs, return_latent=False):
        """
        Get deterministic actions (mean) for inference using student encoder.
        
        Args:
            student_encoder_obs: Student encoder observations
            policy_obs: Policy observations
            return_latent: If True, return (actions_mean, latent). If False, return actions_mean only.
        
        Returns:
            actions_mean: Deterministic action mean
            latent (optional): Encoder latent vector (only if return_latent=True)
        """
        # Get context embedding from student encoder
        e_t = self.get_student_encoder_latent(student_encoder_obs)
        
        # Compute action mean based on adapter type
        if self.adapter_type == "film":
            h = self.actor_body(policy_obs, e_t)
            actions_mean = self.action_head(h)
        elif self.adapter_type == "residual":
            base_actions = self.frozen_actor(policy_obs)
            actions_mean = self.residual_adapter(base_actions, e_t, proprio=policy_obs)

        if return_latent:
            return actions_mean, e_t
        else:
            return actions_mean

    def evaluate(self, critic_observations, student_encoder_obs=None):
        """
        Evaluate the critic value with both critic observations and student encoder latent.
        
        Args:
            critic_observations: Critic observation inputs
            student_encoder_obs: Student encoder observations (if None, uses critic_observations)
        """
        # Use student encoder observations, or fall back to critic observations
        encoder_obs = student_encoder_obs if student_encoder_obs is not None else critic_observations
        
        # Get context embedding from student encoder
        e_t = self.get_student_encoder_latent(encoder_obs)
        
        # Concatenate critic observations with encoder latent
        if critic_observations.dim() == 2:
            critic_input = torch.cat([critic_observations, e_t], dim=-1)
        else:
            raise ValueError(f"Expected critic observations with 2 dims, got {critic_observations.dim()}")
        
        # Evaluate critic with combined input
        value = self.critic(critic_input)
        return value

    def get_student_encoder_latent(self, observations):
        """
        Get student encoder latent vector from observations.
        
        Args:
            observations: Student observation sequences
                         Shape: [num_envs, seq_len, obs_dim] - sequence input (preferred)
                                [num_envs, obs_dim] - single observation (expanded to seq_len=1)
        
        Returns:
            torch.Tensor: Encoder latent vector [num_envs, ctx_dim]
        """
        # Handle both sequence and single observation inputs
        if observations.dim() == 2:
            obs_seq = observations.unsqueeze(1)
        elif observations.dim() == 3:
            obs_seq = observations
        else:
            raise ValueError(
                f"Expected observations with 2 or 3 dimensions, got shape {observations.shape}"
            )
        
        # Transpose for GRU: [seq_len, num_envs, obs_dim]
        obs_seq_t = obs_seq.transpose(0, 1)
        
        # Get context embedding from student encoder
        e_t = self.student_encoder(obs_seq_t)
        
        return e_t

    def get_teacher_encoder_latent(self, observations):
        """
        Get teacher encoder latent vector from observations (inference only).
        
        Args:
            observations: Teacher observation sequences (full privileged observations)
                         Shape: [num_envs, seq_len, obs_dim] - sequence input (preferred)
                                [num_envs, obs_dim] - single observation (expanded to seq_len=1)
        
        Returns:
            torch.Tensor: Encoder latent vector [num_envs, ctx_dim]
        """
        # Handle both sequence and single observation inputs
        if observations.dim() == 2:
            obs_seq = observations.unsqueeze(1)
        elif observations.dim() == 3:
            obs_seq = observations
        else:
            raise ValueError(
                f"Expected observations with 2 or 3 dimensions, got shape {observations.shape}"
            )
        
        # Transpose for GRU: [seq_len, num_envs, obs_dim]
        obs_seq_t = obs_seq.transpose(0, 1)
        
        # Get context embedding from teacher encoder (no grad)
        with torch.no_grad():
            e_t = self.teacher_encoder(obs_seq_t)
        
        return e_t

    def get_teacher_action_mean(self, teacher_encoder_obs, policy_obs, return_latent=False):
        """
        Get teacher's deterministic action mean for distillation (inference only).
        More efficient than get_teacher_action_distribution as it skips std computation.
        
        Args:
            teacher_encoder_obs: Teacher encoder observations (full privileged)
            policy_obs: Policy observations for base actor MLP
            return_latent: If True, return (action_mean, latent). If False, return action_mean only.
        
        Returns:
            action_mean: Deterministic action mean from teacher
            latent (optional): Teacher encoder latent vector (only if return_latent=True)
        """
        with torch.no_grad():
            # Get context embedding from teacher encoder
            e_t = self.get_teacher_encoder_latent(teacher_encoder_obs)
            
            # Compute action mean based on adapter type
            if self.adapter_type == "film":
                h = self.actor_body(policy_obs, e_t)
                mean = self.action_head(h)
            elif self.adapter_type == "residual":
                base_actions = self.frozen_actor(policy_obs)
                mean = self.residual_adapter(base_actions, e_t, proprio=policy_obs)

        if return_latent:
            return mean, e_t
        else:
            return mean

    def load_state_dict(self, state_dict, strict=True):
        """Load the parameters of the student and teacher networks.

        Args:
            state_dict (dict): State dictionary of the model.
            strict (bool): Whether to strictly enforce that the keys in state_dict match the keys returned by this
                           module's state_dict() function.

        Returns:
            bool: Whether this training resumes a previous training. This flag is used by the `load()` function of
                  `OnPolicyRunner` to determine how to load further parameters.
        """
        # Check if state_dict contains teacher checkpoint or resume checkpoint
        has_student_encoder = any('student_encoder' in key for key in state_dict.keys())
        has_actor_body = any('actor_body' in key for key in state_dict.keys())
        has_frozen_actor = any('frozen_actor' in key for key in state_dict.keys())
        has_residual_adapter = any('residual_adapter' in key for key in state_dict.keys())
        has_history_encoder = any('history_encoder' in key for key in state_dict.keys())
        
        if has_student_encoder:
            # This is an AdaptedStudentTeacher checkpoint - resume distillation training
            print("[INFO] Loading AdaptedStudentTeacher checkpoint (resuming distillation)")
            super().load_state_dict(state_dict, strict=strict)
            self.loaded_teacher = True
            self.teacher_encoder.eval()
            return True
            
        elif (has_actor_body or has_frozen_actor or has_residual_adapter) and has_history_encoder:
            # This is an adapted actor-critic checkpoint (FiLM or Residual) - load as teacher
            
            # Determine checkpoint type
            if has_actor_body:
                checkpoint_type = "AdaptedActorCritic (FiLM)"
            elif has_frozen_actor and has_residual_adapter:
                checkpoint_type = "ResidualActorCritic"
            else:
                checkpoint_type = "Unknown adapted"
            
            print(f"[INFO] Loading {checkpoint_type} checkpoint as teacher")
            print("[INFO] Mapping: history_encoder -> teacher_encoder")
            print("[INFO] Freezing all components except student_encoder")
            
            # Create a mapping for loading teacher components
            teacher_state_dict = {}
            for key, value in state_dict.items():
                # Map history_encoder to teacher_encoder
                if 'history_encoder' in key:
                    new_key = key.replace('history_encoder', 'teacher_encoder')
                    teacher_state_dict[new_key] = value
                # Load actor components based on type
                elif any(prefix in key for prefix in ['actor_body', 'action_head', 'frozen_actor', 'residual_adapter', 'critic', 'std', 'log_std']):
                    teacher_state_dict[key] = value
            
            # Load the mapped state dict
            missing_keys, unexpected_keys = super().load_state_dict(teacher_state_dict, strict=False)
            
            # Expected missing keys: student_encoder parameters
            expected_missing = [k for k in missing_keys if 'student_encoder' in k]
            unexpected_missing = [k for k in missing_keys if 'student_encoder' not in k]
            
            if unexpected_missing:
                print(f"[WARNING] Unexpected missing keys (not student_encoder): {unexpected_missing[:5]}...")
            
            print(f"[INFO] Student encoder parameters: {len(expected_missing)} (randomly initialized)")
            print(f"[INFO] Teacher loaded successfully!")
            
            # Set flag for successfully loading the teacher
            self.loaded_teacher = True
            self.teacher_encoder.eval()
            
            # Ensure everything except student_encoder is frozen
            for name, param in self.named_parameters():
                if 'student_encoder' not in name:
                    param.requires_grad_(False)
            
            # Print trainable parameters summary
            self.print_trainable_parameters()

            return False  # Not resuming, starting new distillation
            
        else:
            raise ValueError(
                "state_dict does not contain valid AdaptedActorCritic or AdaptedStudentTeacher parameters. "
                "Expected 'actor_body' + 'history_encoder' or 'student_encoder' keys."
            )

    def detach_hidden_states(self, dones=None):
        """Compatibility method for recurrent interface."""
        pass

    def print_trainable_parameters(self):
        """Print summary of trainable vs frozen parameters."""
        total_params = 0
        trainable_params = 0
        frozen_params = 0
        
        param_groups = {
            'Student Encoder': [],
            'Teacher Encoder': [],
            'Actor Body': [],
            'Critic': [],
            'Other': []
        }
        
        for name, param in self.named_parameters():
            num_params = param.numel()
            total_params += num_params
            
            if param.requires_grad:
                trainable_params += num_params
                if 'student_encoder' in name:
                    param_groups['Student Encoder'].append((name, num_params))
                else:
                    param_groups['Other'].append((name, num_params))
            else:
                frozen_params += num_params
                if 'teacher_encoder' in name:
                    param_groups['Teacher Encoder'].append((name, num_params))
                elif 'actor_body' in name or 'action_head' in name:
                    param_groups['Actor Body'].append((name, num_params))
                elif 'critic' in name:
                    param_groups['Critic'].append((name, num_params))
        
        print(f"\n{'='*70}")
        print(f"Model Parameters: {total_params:,} total")
        print(f"  Trainable: {trainable_params:,} ({100*trainable_params/total_params:.1f}%)")
        print(f"  Frozen: {frozen_params:,} ({100*frozen_params/total_params:.1f}%)")
        print(f"{'-'*70}")
        
        for group_name in ['Student Encoder', 'Teacher Encoder', 'Actor Body', 'Critic', 'Other']:
            params = param_groups[group_name]
            if params:
                group_total = sum(p[1] for p in params)
                if group_name == 'Student Encoder':
                    status = "TRAINABLE"
                else:
                    status = "FROZEN"
                print(f"{group_name} ({status}): {group_total:,}")
        print(f"{'='*70}\n")
