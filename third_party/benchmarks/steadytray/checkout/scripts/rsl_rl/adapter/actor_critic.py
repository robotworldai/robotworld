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


class AdapterSequential(nn.Sequential):
    """
    Sequential module that forwards (x, e) through FiLMAdapter layers.
    Non-adapter layers only receive x.
    """
    def forward(self, x, e):
        for module in self:
            if isinstance(module, FiLMAdapter):
                x = module(x, e)
            else:
                x = module(x)
        return x


class AdaptedActorCritic(nn.Module):
    """
    Actor-Critic module with FiLM adapter-based actor and shared GRU encoder.
    
    Architecture Overview:
    ----------------------
    Actor: Frozen base MLP + FiLM adapters + GRU history encoder
           - Base MLP (frozen): Pre-trained locomotion policy layers
           - FiLM adapters (trainable): Feature-wise modulation y = frozen(x) * (1 + α*γ) + α*β
           - GRU encoder (trainable): Processes observation sequences for temporal context
           
    Critic: Standard feedforward MLP with encoder latent input
            - Takes concatenation of [critic_obs, encoder_latent]
            - Shares encoder with actor (enables gradient flow from both objectives)
    
    Observation Inputs:
    -------------------
    The model expects three types of observations:
    
    1. encoder_obs: For GRU temporal encoding
       - Shape: [num_envs, seq_len, num_encoder_obs] (e.g., 32-step sequence)
       - Purpose: Generate temporal context embedding via GRU
       - Dimension: num_encoder_obs (may differ from actor obs)
       
    2. policy_obs: For frozen base actor MLP
       - Shape: [num_envs, num_actor_obs] (e.g., 5-step flattened history)
       - Purpose: Input to frozen base policy (must match training format)
       - Dimension: num_actor_obs
       
    3. critic_obs: For critic value estimation
       - Shape: [num_envs, num_critic_obs] (privileged observations)
       - Purpose: Critic input (concatenated with encoder latent)
       - Dimension: num_critic_obs
    
    Training Strategy:
    ------------------
    - Frozen: Base actor MLP layers and action head (pre-trained on locomotion)
    - Trainable: FiLM adapter networks, GRU encoder, critic MLP
    - Shared: GRU encoder receives gradients from both actor (via adapters) and critic
    
    Note: This is a non-recurrent architecture. The GRU encoder processes observation
          sequences provided by the observation manager from a zero initial state.
    """
    is_recurrent = False  # Non-recurrent: GRU processes sequences from zero state

    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        num_actions,
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],
        activation="elu",
        init_noise_std=1.0,
        noise_std_type: str = "scalar",
        # Adapter-specific parameters
        ctx_dim: int = 64,
        encoder_layers: int = 1,
        adapter_hidden: int = 64,
        clamp_gamma: float = 2.0,
        use_gate: bool = True,
        # Encoder architecture selection
        encoder_type: str = "gru",
        num_heads: int = 4,
        encoder_dropout: float = 0.1,
        # Encoder observation dimensions (if different from actor obs)
        num_encoder_obs: int | None = None,
        **kwargs,
    ):
        if kwargs:
            print(
                "AdaptedActorCritic.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()
        
        activation = resolve_nn_activation(activation)
        self.num_actor_obs = num_actor_obs
        self.num_critic_obs = num_critic_obs
        self.num_actions = num_actions
        self.ctx_dim = ctx_dim
        self.init_noise_std = init_noise_std
        
        # Determine encoder input dimension
        # If not specified, assume encoder obs has same per-timestep dimension as actor obs
        # BUT: typically actor obs is flattened history (e.g., 5*single_dim) while
        # encoder expects per-timestep dimension (single_dim)
        if num_encoder_obs is not None:
            self.num_encoder_obs = num_encoder_obs
        else:
            # Fallback: assume actor obs is flattened history, use it as-is
            # (User should ideally provide num_encoder_obs explicitly)
            self.num_encoder_obs = num_actor_obs
            print(
                f"Warning: num_encoder_obs not specified. Using num_actor_obs={num_actor_obs} as encoder input dim.\n"
                f"  If actor obs is flattened history, encoder should receive per-timestep dimension instead."
            )

        # ============================================================
        # Actor: Adapter-based architecture with history encoder
        # ============================================================
        
        # Build actor body (hidden layers only, no action head)
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
                # Wrap with FiLM adapter
                adapted_body_layers.append(
                    FiLMAdapter(layer, ctx_dim, hidden=adapter_hidden, 
                               clamp_gamma=clamp_gamma, use_gate=use_gate)
                )
            else:
                adapted_body_layers.append(layer)
        
        self.actor_body = AdapterSequential(*adapted_body_layers)
        
        # Action head: separate frozen linear layer (not wrapped with adapter)
        self.action_head = nn.Linear(actor_hidden_dims[-1], num_actions)
        for param in self.action_head.parameters():
            param.requires_grad_(False)
        
        # History encoder: encodes observation sequences into context embedding
        # Observation manager provides sequences, encoder processes from zero state
        # Note: encoder uses num_encoder_obs (per-timestep dim), which may differ from num_actor_obs (flattened)
        encoder_type = encoder_type.lower()
        if encoder_type == "gru":
            self.history_encoder = GRUEncoder(self.num_encoder_obs, ctx_dim, num_layers=encoder_layers)
        elif encoder_type == "transformer":
            self.history_encoder = TransformerEncoder(
                self.num_encoder_obs, ctx_dim, 
                num_layers=encoder_layers, 
                num_heads=num_heads,
                dropout=encoder_dropout
            )
        elif encoder_type == "causal_transformer":
            self.history_encoder = CausalTransformerEncoder(
                self.num_encoder_obs, ctx_dim,
                num_layers=encoder_layers,
                num_heads=num_heads,
                dropout=encoder_dropout
            )
        else:
            raise ValueError(f"Unknown encoder_type: {encoder_type}. Must be 'gru', 'transformer', or 'causal_transformer'")
        
        self.encoder_type = encoder_type

        print(f"\n{'='*70}")
        print(f"AdaptedActorCritic Architecture Initialized:")
        print(f"{'-'*70}")
        print(f"Actor Base MLP (frozen):")
        print(f"  Input: {num_actor_obs} (flattened history, e.g., 5-step * single_obs_dim)")
        print(f"  Hidden layers: {actor_hidden_dims}")
        print(f"  Output: {num_actions} actions")
        print(f"  FiLM Adapters: {len([l for l in self.actor_body if isinstance(l, FiLMAdapter)])} layers")
        print(f"    - Context dim: {ctx_dim}")
        print(f"    - Adapter hidden dim: {adapter_hidden}")
        print(f"{'-'*70}")
        print(f"History Encoder (trainable):")
        print(f"  Type: {encoder_type.upper()}")
        print(f"  Input: {self.num_encoder_obs} (per-timestep obs dim for sequences)")
        print(f"  Hidden dim: {ctx_dim}")
        print(f"  Layers: {encoder_layers}")
        if encoder_type in ["transformer", "causal_transformer"]:
            print(f"  Attention heads: {num_heads}")
            print(f"  Dropout: {encoder_dropout}")
        print(f"  Note: Processes sequences from observation manager (e.g., 32-step history)")
        print(f"{'-'*70}")
        print(f"Critic MLP (trainable):")
        print(f"  Input: {num_critic_obs + ctx_dim} (critic_obs={num_critic_obs} + encoder_latent={ctx_dim})")
        print(f"  Hidden layers: {critic_hidden_dims}")
        print(f"  Output: 1 (value estimate)")
        print(f"  Note: Shares encoder with actor for gradient flow")
        print(f"{'='*70}\n")

        # ============================================================
        # Critic: Standard MLP with encoder latent input
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

        # Action noise
        self.noise_std_type = noise_std_type
        if self.noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        elif self.noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")

        # Action distribution (populated in update_distribution)
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args(False)

    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        [
            torch.nn.init.orthogonal_(module.weight, gain=scales[idx])
            for idx, module in enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))
        ]

    def reset(self, dones=None):
        """Reset method for compatibility. Since we don't maintain hidden states, this is a no-op."""
        pass
    
    def get_hidden_states(self):
        """Get hidden states for compatibility with recurrent interface. Returns None since we're non-recurrent."""
        return None

    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, encoder_obs, policy_obs, encoder_latent=None):
        """
        Update the action distribution given encoder and policy observations.
        Uses history encoder to generate context embedding from encoder_obs.
        
        Args:
            encoder_obs: Encoder observation sequences from observation manager
                        Shape: [num_envs, seq_len, obs_dim] for sequences (e.g., 32-step history)
                               or [num_envs, obs_dim] for single observations
                        Used by GRU encoder to generate temporal context embedding
                        
            policy_obs: Policy observations for base actor MLP
                       Shape: [num_envs, num_actor_obs] (e.g., 5-step flattened history)
                       Fed directly to the frozen base actor MLP along with context
                       Must match the input format the frozen base policy was originally trained on
            
            encoder_latent: Pre-computed encoder latent (optional)
                           If provided, skips encoder forward pass (optimization)
        
        The forward pass:
            1. encoder_obs → GRU encoder → context embedding e_t [num_envs, ctx_dim]
            2. (policy_obs, e_t) → FiLM-adapted actor body → hidden features
            3. hidden features → frozen action head → action mean
            4. action mean + noise std → Normal distribution
        """
        # Get context embedding from history encoder using encoder_obs
        if encoder_latent is None:
            e_t = self.get_encoder_latent(encoder_obs)
        else:
            e_t = encoder_latent
        
        # Use policy_obs directly for actor body (already processed appropriately)
        # Compute action mean using adapted actor body + frozen action head
        h = self.actor_body(policy_obs, e_t)
        mean = self.action_head(h)
        
        # Compute standard deviation
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        elif self.noise_std_type == "log":
            std = torch.exp(self.log_std).expand_as(mean)
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        # create distribution
        self.distribution = Normal(mean, std)

    def act(self, encoder_obs, policy_obs, encoder_latent=None, **kwargs):
        """
        Sample actions from the policy.
        
        Args:
            encoder_obs: Encoder observation sequences (32-step history, non-flattened)
            policy_obs: Policy observations (5-step history, flattened)
            encoder_latent: Pre-computed encoder latent (optional, for optimization)
        """
        self.update_distribution(encoder_obs, policy_obs, encoder_latent=encoder_latent)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, encoder_obs, policy_obs, encoder_latent=None):
        """
        Get deterministic actions (mean) for inference.
        
        Args:
            encoder_obs: Encoder observation sequences (32-step history, non-flattened)
            policy_obs: Policy observations (5-step history, flattened)
            encoder_latent: Pre-computed encoder latent (optional, for optimization)
        """
        # Get context embedding from history encoder
        if encoder_latent is None:
            e_t = self.get_encoder_latent(encoder_obs)
        else:
            e_t = encoder_latent
        
        # Use policy_obs directly for actor body
        h = self.actor_body(policy_obs, e_t)
        actions_mean = self.action_head(h)
        
        return actions_mean

    def evaluate(self, critic_observations, actor_observations=None, encoder_latent=None, **kwargs):
        """
        Evaluate the critic value with both critic observations and encoder latent.
        
        Args:
            critic_observations: Critic observation inputs
            actor_observations: Actor observation sequences for encoder (if None, uses critic_observations)
            encoder_latent: Pre-computed encoder latent (optional, for optimization)
        """
        # Get context embedding from history encoder (shared with actor)
        if encoder_latent is None:
            # Use actor observations for encoder, or fall back to critic observations
            encoder_obs = actor_observations if actor_observations is not None else critic_observations
            e_t = self.get_encoder_latent(encoder_obs)
        else:
            e_t = encoder_latent
        
        # Concatenate critic observations with encoder latent
        if critic_observations.dim() == 2:
            # Standard case: [num_envs, obs_dim]
            critic_input = torch.cat([critic_observations, e_t], dim=-1)
        else:
            raise ValueError(f"Expected critic observations with 2 dims, got {critic_observations.dim()}")
        
        # Evaluate critic with combined input
        value = self.critic(critic_input)
        return value

    def get_encoder_latent(self, observations):
        """
        Get encoder latent vector from observations.
        
        Args:
            observations: Observation sequences for encoder
                         Shape: [num_envs, seq_len, obs_dim] - sequence input (preferred)
                                [num_envs, obs_dim] - single observation (expanded to seq_len=1)
        
        Returns:
            torch.Tensor: Encoder latent vector [num_envs, ctx_dim]
        
        Note: The encoder expects observations with shape matching num_encoder_obs,
              which may differ from num_actor_obs (policy observations).
        """
        # Handle both sequence and single observation inputs
        if observations.dim() == 2:
            # Single observation [num_envs, obs_dim] - treat as sequence of length 1
            obs_seq = observations.unsqueeze(1)  # [num_envs, 1, obs_dim]
        elif observations.dim() == 3:
            # Sequence [num_envs, seq_len, obs_dim] - use directly
            obs_seq = observations
        else:
            raise ValueError(
                f"Expected observations with 2 or 3 dimensions [num_envs, (seq_len,) obs_dim], "
                f"got shape {observations.shape} with {observations.dim()} dimensions"
            )
        
        # Transpose for GRU: [seq_len, num_envs, obs_dim]
        obs_seq_t = obs_seq.transpose(0, 1)
        
        # Get context embedding from history encoder
        # GRU processes sequence from zero state and returns last hidden state
        e_t = self.history_encoder(obs_seq_t)
        
        return e_t

    def print_trainable_parameters(self):
        """Print summary of trainable vs frozen parameters."""
        total_params = 0
        trainable_params = 0
        frozen_params = 0
        
        param_groups = {
            'Frozen Actor': [],
            'Adapters': [],
            'Encoder': [],
            'Critic': []
        }
        
        for name, param in self.named_parameters():
            num_params = param.numel()
            total_params += num_params
            
            if param.requires_grad:
                trainable_params += num_params
                if 'actor_body' in name:
                    param_groups['Adapters'].append((name, num_params))
                elif 'history_encoder' in name:
                    param_groups['Encoder'].append((name, num_params))
                elif 'critic' in name:
                    param_groups['Critic'].append((name, num_params))
            else:
                frozen_params += num_params
                if 'actor_body' in name or 'action_head' in name:
                    param_groups['Frozen Actor'].append((name, num_params))
        
        print(f"\n{'='*70}")
        print(f"Model Parameters: {total_params:,} total")
        print(f"  Trainable: {trainable_params:,} ({100*trainable_params/total_params:.1f}%)")
        print(f"  Frozen: {frozen_params:,} ({100*frozen_params/total_params:.1f}%)")
        print(f"{'-'*70}")
        
        for group_name in ['Frozen Actor', 'Adapters', 'Encoder', 'Critic']:
            params = param_groups[group_name]
            if params:
                group_total = sum(p[1] for p in params)
                status = "FROZEN" if group_name == 'Frozen Actor' else "TRAINABLE"
                print(f"{group_name} ({status}): {group_total:,}")
        print(f"{'='*70}\n")

    def load_state_dict(self, state_dict, strict=True):
        """Load the parameters of the actor-critic model.

        This method automatically handles two scenarios:
        1. **Complete checkpoint** (has adapters/encoder): Loads everything normally (resuming training)
        2. **Base policy checkpoint** (no adapters/encoder): Automatically calls load_pretrained_base_policy()
        
        Args:
            state_dict (dict): State dictionary of the model.
            strict (bool): Whether to strictly enforce that the keys in state_dict match the keys returned by this
                           module's state_dict() function. Only used for complete checkpoints.

        Returns:
            bool: Whether this training resumes a previous training. This flag is used by the `load()` function of
                  `OnPolicyRunner` to determine how to load further parameters (relevant for, e.g., distillation).
                  - True: Complete checkpoint loaded (resume training mode)
                  - False: Base policy loaded (start new adapter training)
        """
        # Check if this is a complete AdaptedActorCritic checkpoint or a base policy checkpoint
        has_adapters = any('actor_body' in key for key in state_dict.keys())
        has_encoder = any('history_encoder' in key for key in state_dict.keys())
        
        if not has_adapters and not has_encoder:
            # This looks like a base policy checkpoint without adapters
            print("\n" + "="*70)
            print("Detected base policy checkpoint (no adapters/encoder found)")
            print("="*70 + "\n")
            
            # Inspect the checkpoint structure before loading
            print("Pre-trained Model Structure:")
            self._inspect_state_dict(state_dict)
            
            print("\n" + "="*70)
            print("Automatically loading as pre-trained base policy...")
            print("="*70 + "\n")
            
            # Load using the specialized method that maps actor.X -> actor_body.X.frozen_layer
            self._load_pretrained_state_dict(state_dict)
            
            # Print parameter summary after loading
            self.print_trainable_parameters()
            
            # Return False to indicate this is NOT resuming training (new adapter training)
            return False
        
        # This is a complete checkpoint, load normally
        missing_keys, unexpected_keys = super().load_state_dict(state_dict, strict=strict)
        
        if not strict and (missing_keys or unexpected_keys):
            print(f"\nPartial checkpoint loaded:")
            if missing_keys:
                print(f"  Missing keys: {len(missing_keys)} (e.g., {missing_keys[:3]}...)")
            if unexpected_keys:
                print(f"  Unexpected keys: {len(unexpected_keys)} (e.g., {unexpected_keys[:3]}...)")
        
        # Return True to indicate this is resuming training
        return has_adapters and has_encoder
    
    def _inspect_state_dict(self, state_dict: dict, max_params: int = 100):
        """
        Internal method to inspect and print state dict structure.
        
        Args:
            state_dict: State dictionary to inspect
            max_params: Maximum number of parameters to display per group
        """
        # Group and count parameters
        param_groups = {}
        total_params = 0
        
        for name, param in state_dict.items():
            num_params = param.numel() if hasattr(param, 'numel') else 0
            total_params += num_params
            shape_str = f" {list(param.shape)}" if hasattr(param, 'shape') else ""
            
            # Categorize parameter
            if name.startswith('actor.') and not name.startswith('actor.std'):
                group = 'Actor'
            elif name.startswith('critic.'):
                group = 'Critic'
            elif 'std' in name:
                group = 'Noise'
            else:
                group = 'Other'
            
            if group not in param_groups:
                param_groups[group] = []
            param_groups[group].append((name, num_params, shape_str))
        
        # Print parameter summary
        print(f"{'-'*70}")
        print(f"Total Parameters: {total_params:,}")
        print(f"{'-'*70}")
        
        for group_name in ['Actor', 'Critic', 'Noise', 'Other']:
            if group_name in param_groups:
                params = param_groups[group_name]
                group_total = sum(p[1] for p in params)
                print(f"\n{group_name}: {group_total:,} ({len(params)} tensors)")
                
                # Show sample parameters
                display_count = len(params) if max_params == 0 else min(len(params), max_params)
                for name, num_params, shape_str in params[:display_count]:
                    print(f"  • {name}{shape_str}")
                
                if len(params) > display_count:
                    print(f"  ... +{len(params) - display_count} more")
        
        print(f"{'-'*70}")
    
    def _load_pretrained_state_dict(self, pretrained_state: dict):
        """
        Internal method to load pre-trained base policy weights from a state dict.
        
        Handles standard ActorCritic checkpoint structure where:
        - actor.0, actor.2, actor.4, ... are body layers (Linear + Activation)
        - Last actor layer (highest index) is the action head
        - std parameter is action noise
        
        Args:
            pretrained_state: State dictionary from a standard ActorCritic checkpoint
        """
        loaded_params = []
        skipped_params = []
        
        # First pass: identify the last actor layer (action head)
        actor_layer_indices = []
        for name in pretrained_state.keys():
            if name.startswith('actor.') and not name.startswith('actor.std'):
                parts = name.split('.')
                if len(parts) >= 3 and parts[1].isdigit():
                    actor_layer_indices.append(int(parts[1]))
        
        # The highest index is the action head
        action_head_idx = max(actor_layer_indices) if actor_layer_indices else None
        
        print(f"Detected actor layers: {sorted(set(actor_layer_indices))}")
        print(f"Action head layer index: {action_head_idx}")
        print()
        
        # Second pass: load parameters
        for name, param in pretrained_state.items():
            # Handle actor layers
            if name.startswith('actor.') and not name.startswith('actor.std'):
                parts = name.split('.')
                if len(parts) >= 3 and parts[1].isdigit():
                    layer_idx = int(parts[1])
                    param_type = '.'.join(parts[2:])  # "weight" or "bias"
                    
                    # Check if this is the action head or a body layer
                    if layer_idx == action_head_idx:
                        # This is the action head - load directly
                        adapted_name = f'action_head.{param_type}'
                        if adapted_name in self.state_dict():
                            self.state_dict()[adapted_name].copy_(param)
                            loaded_params.append(f"{name} -> {adapted_name}")
                        else:
                            skipped_params.append(f"{name} (target: {adapted_name} not found)")
                    else:
                        # This is a body layer - map to FiLMAdapter's base layer
                        # The indices match directly: actor.0 -> actor_body.0.base, etc.
                        # because actor_body contains both FiLMAdapters and activations at the same indices
                        adapted_name = f'actor_body.{layer_idx}.base.{param_type}'
                        
                        # Try to load into adapted model
                        try:
                            # Access the parameter directly
                            target_param = self.state_dict()[adapted_name]
                            target_param.copy_(param)
                            loaded_params.append(f"{name} -> {adapted_name}")
                        except KeyError:
                            skipped_params.append(f"{name} (target: {adapted_name} not found in state_dict)")
            
            # Handle std parameter
            elif name == 'actor.std' or name == 'std':
                if 'std' in self.state_dict():
                    # Only load pretrained std if init_noise_std was not explicitly provided
                    if self.init_noise_std is None:
                        self.state_dict()['std'].copy_(param)
                        loaded_params.append(f"{name} -> std (from pretrained)")
                    else:
                        skipped_params.append(f"{name} (using init_noise_std={self.init_noise_std} instead)")
                else:
                    skipped_params.append(name)
            
            # Skip critic (will be trained from scratch)
            elif name.startswith('critic.'):
                skipped_params.append(f"{name} (critic not loaded)")
            
            # Skip other parameters
            else:
                skipped_params.append(f"{name} (unknown)")
        
        print(f"Loaded {len(loaded_params)} parameters:")
        for p in loaded_params:
            print(f"  ✓ {p}")
        
        if skipped_params:
            print(f"\nSkipped {len(skipped_params)} parameters (trained from scratch):")
            for p in skipped_params:
                print(f"  ✗ {p}")
        
        print(f"\n✓ Pre-trained base policy loaded successfully!")
        print(f"  Adapters, encoder, and critic initialized randomly\n")


class ResidualActorCritic(nn.Module):
    """
    Actor-Critic with Residual Action adapter that adds learned corrections in action space.
    
    Architecture Overview:
    ----------------------
    Actor: Frozen base MLP + Residual Action adapter + GRU history encoder
           - Base MLP (frozen): Pre-trained locomotion policy that produces base actions
           - Residual adapter (trainable): Learns action corrections a_final = a_base + α * a_residual
           - GRU encoder (trainable): Processes observation sequences for temporal context
           
    Critic: Standard feedforward MLP with encoder latent input
            - Takes concatenation of [critic_obs, encoder_latent]
            - Shares encoder with actor (enables gradient flow from both objectives)
    
    Key Difference from FiLM:
    -------------------------
    - FiLM: Modulates intermediate features layer-by-layer (feature-wise linear modulation)
    - Residual: Adds correction directly to final actions (action-space residual)
    - Residual is simpler and more interpretable, operating in the action manifold
    
    Observation Inputs:
    -------------------
    Same as AdaptedActorCritic:
    1. encoder_obs: For GRU temporal encoding [num_envs, seq_len, num_encoder_obs]
    2. policy_obs: For frozen base actor MLP [num_envs, num_actor_obs]
    3. critic_obs: For critic value estimation [num_envs, num_critic_obs]
    
    Training Strategy:
    ------------------
    - Frozen: Base actor MLP (all layers including action head)
    - Trainable: Residual action adapter, GRU encoder, critic MLP
    - Shared: GRU encoder receives gradients from both actor and critic
    """
    is_recurrent = False  # Non-recurrent: GRU processes sequences from zero state

    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        num_actions,
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],
        activation="elu",
        init_noise_std=1.0,
        noise_std_type: str = "scalar",
        # Residual adapter parameters
        ctx_dim: int = 64,
        encoder_layers: int = 1,
        residual_hidden_dims: list[int] = [128, 64],
        clamp_residual: float = 1.0,
        use_gate: bool = True,
        # Encoder architecture selection
        encoder_type: str = "gru",
        num_heads: int = 4,
        encoder_dropout: float = 0.1,
        # Encoder observation dimensions
        num_encoder_obs: int | None = None,
        **kwargs,
    ):
        if kwargs:
            print(
                "ResidualActorCritic.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()
        
        activation = resolve_nn_activation(activation)
        self.num_actor_obs = num_actor_obs
        self.num_critic_obs = num_critic_obs
        self.num_actions = num_actions
        self.ctx_dim = ctx_dim
        self.init_noise_std = init_noise_std
        
        # Determine encoder input dimension
        if num_encoder_obs is not None:
            self.num_encoder_obs = num_encoder_obs
        else:
            self.num_encoder_obs = num_actor_obs
            print(
                f"Warning: num_encoder_obs not specified. Using num_actor_obs={num_actor_obs} as encoder input dim.\n"
                f"  If actor obs is flattened history, encoder should receive per-timestep dimension instead."
            )

        # ============================================================
        # Actor: Frozen base MLP + Residual Action adapter
        # ============================================================
        
        # Build frozen actor (entire MLP including action head)
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
        
        # Residual action adapter: learns to add corrections to base actions
        # Input: concatenation of encoder latent (ctx_dim) and proprio observations (num_actor_obs)
        self.residual_adapter = ResidualActionAdapter(
            num_actions, ctx_dim, proprio_dim=num_actor_obs,
            hidden_dims=residual_hidden_dims,
            use_gate=use_gate, clamp_residual=clamp_residual
        )

        # History encoder: same as AdaptedActorCritic
        encoder_type = encoder_type.lower()
        if encoder_type == "gru":
            self.history_encoder = GRUEncoder(self.num_encoder_obs, ctx_dim, num_layers=encoder_layers)
        elif encoder_type == "transformer":
            self.history_encoder = TransformerEncoder(
                self.num_encoder_obs, ctx_dim, 
                num_layers=encoder_layers, 
                num_heads=num_heads,
                dropout=encoder_dropout
            )
        elif encoder_type == "causal_transformer":
            self.history_encoder = CausalTransformerEncoder(
                self.num_encoder_obs, ctx_dim,
                num_layers=encoder_layers,
                num_heads=num_heads,
                dropout=encoder_dropout
            )
        else:
            raise ValueError(f"Unknown encoder_type: {encoder_type}. Must be 'gru', 'transformer', or 'causal_transformer'")
        
        self.encoder_type = encoder_type

        print(f"\n{'='*70}")
        print(f"ResidualActorCritic Architecture Initialized:")
        print(f"{'-'*70}")
        print(f"Frozen Base Actor MLP:")
        print(f"  Input: {num_actor_obs} (flattened history)")
        print(f"  Hidden layers: {actor_hidden_dims}")
        print(f"  Output: {num_actions} actions (base)")
        print(f"{'-'*70}")
        print(f"Residual Action Adapter (trainable):")
        print(f"  Input: {ctx_dim + num_actor_obs} (encoder_latent={ctx_dim} + proprio={num_actor_obs})")
        print(f"  Hidden dims: {residual_hidden_dims}")
        print(f"  Output: {num_actions} (residual action)")
        print(f"  Gate parameter α: {use_gate}")
        print(f"  Clamp residual: ±{clamp_residual}")
        print(f"  Note: Final action = base_action + α * residual_mlp(z, proprio)")
        print(f"{'-'*70}")
        print(f"History Encoder (trainable):")
        print(f"  Type: {encoder_type.upper()}")
        print(f"  Input: {self.num_encoder_obs} (per-timestep obs dim)")
        print(f"  Hidden dim: {ctx_dim}")
        print(f"  Layers: {encoder_layers}")
        if encoder_type in ["transformer", "causal_transformer"]:
            print(f"  Attention heads: {num_heads}")
            print(f"  Dropout: {encoder_dropout}")
        print(f"{'-'*70}")
        print(f"Critic MLP (trainable):")
        print(f"  Input: {num_critic_obs + ctx_dim} (critic_obs + encoder_latent)")
        print(f"  Hidden layers: {critic_hidden_dims}")
        print(f"  Output: 1 (value estimate)")
        print(f"{'='*70}\n")

        # ============================================================
        # Critic: Standard MLP with encoder latent input
        # ============================================================
        
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

        # Action noise
        self.noise_std_type = noise_std_type
        if self.noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        elif self.noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")

        self.distribution = None
        Normal.set_default_validate_args(False)

    def reset(self, dones=None):
        """Reset method for compatibility."""
        pass
    
    def get_hidden_states(self):
        """Get hidden states for compatibility."""
        return None

    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, encoder_obs, policy_obs, encoder_latent=None):
        """
        Update action distribution with residual correction.
        
        Args:
            encoder_obs: Encoder observation sequences [num_envs, seq_len, obs_dim]
            policy_obs: Policy observations [num_envs, num_actor_obs]
            encoder_latent: Pre-computed encoder latent (optional, for optimization)
        
        Forward pass:
            1. encoder_obs → encoder → context embedding e_t
            2. policy_obs → frozen actor → base actions
            3. (base_actions, e_t, policy_obs) → residual adapter → final actions
            4. final actions + noise std → Normal distribution
        """
        # Get context embedding from encoder
        if encoder_latent is None:
            e_t = self.get_encoder_latent(encoder_obs)
        else:
            e_t = encoder_latent

        # Get base actions from frozen actor
        base_actions = self.frozen_actor(policy_obs)

        # Add residual correction with proprio context
        mean = self.residual_adapter(base_actions, e_t, proprio=policy_obs)
        
        # Compute standard deviation
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        elif self.noise_std_type == "log":
            std = torch.exp(self.log_std).expand_as(mean)
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}")
        
        self.distribution = Normal(mean, std)

    def act(self, encoder_obs, policy_obs, encoder_latent=None, **kwargs):
        """Sample actions from the policy."""
        self.update_distribution(encoder_obs, policy_obs, encoder_latent=encoder_latent)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, encoder_obs, policy_obs, encoder_latent=None):
        """Get deterministic actions for inference."""
        if encoder_latent is None:
            e_t = self.get_encoder_latent(encoder_obs)
        else:
            e_t = encoder_latent
        base_actions = self.frozen_actor(policy_obs)
        final_actions = self.residual_adapter(base_actions, e_t, proprio=policy_obs)
        return final_actions

    def evaluate(self, critic_observations, actor_observations=None, encoder_latent=None, **kwargs):
        """Evaluate critic value."""
        if encoder_latent is None:
            encoder_obs = actor_observations if actor_observations is not None else critic_observations
            e_t = self.get_encoder_latent(encoder_obs)
        else:
            e_t = encoder_latent
        
        if critic_observations.dim() == 2:
            critic_input = torch.cat([critic_observations, e_t], dim=-1)
        else:
            raise ValueError(f"Expected critic observations with 2 dims, got {critic_observations.dim()}")
        
        value = self.critic(critic_input)
        return value

    def get_encoder_latent(self, observations):
        """Get encoder latent vector from observations."""
        if observations.dim() == 2:
            obs_seq = observations.unsqueeze(1)
        elif observations.dim() == 3:
            obs_seq = observations
        else:
            raise ValueError(f"Expected 2 or 3 dims, got {observations.dim()}")
        
        obs_seq_t = obs_seq.transpose(0, 1)
        e_t = self.history_encoder(obs_seq_t)
        return e_t

    def print_trainable_parameters(self):
        """Print summary of trainable vs frozen parameters."""
        total_params = 0
        trainable_params = 0
        frozen_params = 0
        
        param_groups = {
            'Frozen Actor': [],
            'Residual Adapter': [],
            'Encoder': [],
            'Critic': []
        }
        
        for name, param in self.named_parameters():
            num_params = param.numel()
            total_params += num_params
            
            if param.requires_grad:
                trainable_params += num_params
                if 'residual_adapter' in name:
                    param_groups['Residual Adapter'].append((name, num_params))
                elif 'history_encoder' in name:
                    param_groups['Encoder'].append((name, num_params))
                elif 'critic' in name:
                    param_groups['Critic'].append((name, num_params))
            else:
                frozen_params += num_params
                if 'frozen_actor' in name:
                    param_groups['Frozen Actor'].append((name, num_params))
        
        print(f"\n{'='*70}")
        print(f"Model Parameters: {total_params:,} total")
        print(f"  Trainable: {trainable_params:,} ({100*trainable_params/total_params:.1f}%)")
        print(f"  Frozen: {frozen_params:,} ({100*frozen_params/total_params:.1f}%)")
        print(f"{'-'*70}")
        
        for group_name in ['Frozen Actor', 'Residual Adapter', 'Encoder', 'Critic']:
            params = param_groups[group_name]
            if params:
                group_total = sum(p[1] for p in params)
                status = "FROZEN" if group_name == 'Frozen Actor' else "TRAINABLE"
                print(f"{group_name} ({status}): {group_total:,}")
        print(f"{'='*70}\n")

    def load_state_dict(self, state_dict, strict=True):
        """
        Load parameters with automatic base policy detection.
        
        Returns:
            bool: True if resuming training, False if loading base policy
        """
        has_residual = any('residual_adapter' in key for key in state_dict.keys())
        has_encoder = any('history_encoder' in key for key in state_dict.keys())
        
        if not has_residual and not has_encoder:
            print("\n" + "="*70)
            print("Detected base policy checkpoint (no residual adapter/encoder found)")
            print("="*70 + "\n")
            
            print("Pre-trained Model Structure:")
            self._inspect_state_dict(state_dict)
            
            print("\n" + "="*70)
            print("Loading as pre-trained base policy...")
            print("="*70 + "\n")
            
            self._load_pretrained_state_dict(state_dict)
            self.print_trainable_parameters()
            return False
        
        missing_keys, unexpected_keys = super().load_state_dict(state_dict, strict=strict)
        
        if not strict and (missing_keys or unexpected_keys):
            print(f"\nPartial checkpoint loaded:")
            if missing_keys:
                print(f"  Missing keys: {len(missing_keys)}")
            if unexpected_keys:
                print(f"  Unexpected keys: {len(unexpected_keys)}")
        
        return has_residual and has_encoder
    
    def _inspect_state_dict(self, state_dict: dict, max_params: int = 100):
        """Inspect state dict structure."""
        param_groups = {}
        total_params = 0
        
        for name, param in state_dict.items():
            num_params = param.numel() if hasattr(param, 'numel') else 0
            total_params += num_params
            shape_str = f" {list(param.shape)}" if hasattr(param, 'shape') else ""
            
            if name.startswith('actor.') and not name.startswith('actor.std'):
                group = 'Actor'
            elif name.startswith('critic.'):
                group = 'Critic'
            elif 'std' in name:
                group = 'Noise'
            else:
                group = 'Other'
            
            if group not in param_groups:
                param_groups[group] = []
            param_groups[group].append((name, num_params, shape_str))
        
        print(f"{'-'*70}")
        print(f"Total Parameters: {total_params:,}")
        print(f"{'-'*70}")
        
        for group_name in ['Actor', 'Critic', 'Noise', 'Other']:
            if group_name in param_groups:
                params = param_groups[group_name]
                group_total = sum(p[1] for p in params)
                print(f"\n{group_name}: {group_total:,} ({len(params)} tensors)")
                
                display_count = len(params) if max_params == 0 else min(len(params), max_params)
                for name, num_params, shape_str in params[:display_count]:
                    print(f"  • {name}{shape_str}")
                
                if len(params) > display_count:
                    print(f"  ... +{len(params) - display_count} more")
        
        print(f"{'-'*70}")
    
    def _load_pretrained_state_dict(self, pretrained_state: dict):
        """Load pre-trained base policy into frozen_actor."""
        loaded_params = []
        skipped_params = []
        
        for name, param in pretrained_state.items():
            if name.startswith('actor.') and not name.startswith('actor.std'):
                # Map actor.X to frozen_actor.X
                adapted_name = name.replace('actor.', 'frozen_actor.')
                
                if adapted_name in self.state_dict():
                    self.state_dict()[adapted_name].copy_(param)
                    loaded_params.append(f"{name} -> {adapted_name}")
                else:
                    skipped_params.append(f"{name} (target: {adapted_name} not found)")
            
            # Handle std parameter
            elif name == 'actor.std' or name == 'std':
                if 'std' in self.state_dict():
                    # Only load pretrained std if init_noise_std was not explicitly provided
                    if self.init_noise_std is None:
                        self.state_dict()['std'].copy_(param)
                        loaded_params.append(f"{name} -> std (from pretrained)")
                    else:
                        skipped_params.append(f"{name} (using init_noise_std={self.init_noise_std} instead)")
                else:
                    skipped_params.append(name)
            
            elif name.startswith('critic.'):
                skipped_params.append(f"{name} (critic not loaded)")
            
            else:
                skipped_params.append(f"{name} (unknown)")
        
        print(f"Loaded {len(loaded_params)} parameters:")
        for p in loaded_params:
            print(f"  ✓ {p}")
        
        if skipped_params:
            print(f"\nSkipped {len(skipped_params)} parameters (trained from scratch):")
            for p in skipped_params:
                print(f"  ✗ {p}")
        
        print(f"\n✓ Pre-trained base policy loaded successfully!")
        print(f"  Residual adapter, encoder, and critic initialized randomly\n")
