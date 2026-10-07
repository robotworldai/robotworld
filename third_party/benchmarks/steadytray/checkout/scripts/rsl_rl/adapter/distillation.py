# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal

from .student_teacher import AdaptedStudentTeacher
from .rollout_storage import RolloutStorage


class Distillation:
    """
    Online distillation algorithm for training student encoder to mimic teacher encoder.
    
    This algorithm implements encoder distillation for the AdaptedStudentTeacher architecture:
    - Only student_encoder parameters are trainable
    - All other components (actor_body, adapters, action_head, critic) are frozen and shared
    - Two distillation losses:
      1. Embedding Loss: MSE between student and teacher encoder latents
      2. Action Loss: MSE between student and teacher deterministic action means
    
    Training Loop:
    --------------
    1. Collect rollout data from environment using student encoder
    2. Compute teacher encoder latents (inference only, no grad)
    3. Compute teacher action means (inference only, no grad)
    4. Optimize: loss = λ_embed * embedding_loss + λ_action * action_loss
    5. Only student_encoder weights are updated
    
    Note: This is pure distillation - no RL (PPO), no critic loss, no value functions.
    """

    policy: AdaptedStudentTeacher
    """The student-teacher model."""

    def __init__(
        self,
        policy,
        num_learning_epochs=1,
        learning_rate=1e-3,
        gradient_length=1,
        max_grad_norm=1.0,
        # Distillation loss weights
        embedding_loss_coef=1.0,
        action_loss_coef=1.0,
        loss_type="mse",
        device="cpu",
        # DAgger parameters
        use_dagger=False,
        dagger_beta_start=1.0,
        dagger_beta_end=0.0,
        dagger_decay_steps=1000,
        dagger_schedule_type="linear",  # "linear" or "exponential"
        # Distributed training parameters
        multi_gpu_cfg: dict | None = None,
    ):
        """
        Initialize the adapter distillation algorithm.
        
        Args:
            policy: AdaptedStudentTeacher model
            num_learning_epochs: Number of epochs for each update
            num_mini_batches: Number of mini-batches per epoch
            learning_rate: Learning rate for student encoder
            max_grad_norm: Maximum gradient norm for clipping
            embedding_loss_coef: Weight for embedding distillation loss (λ_embed)
            action_loss_coef: Weight for action distillation loss (λ_action)
            loss_type: Loss function type ("mse" or "huber")
            device: Device to run on
            use_dagger: Whether to use DAgger (Dataset Aggregation)
            dagger_beta_start: Initial probability of using teacher actions (β)
            dagger_beta_end: Final probability of using teacher actions
            dagger_decay_steps: Number of steps/iterations over which to decay β
            dagger_schedule_type: How to decay β ("linear" or "exponential")
            multi_gpu_cfg: Multi-GPU configuration
        """
        # device-related parameters
        self.device = device
        self.is_multi_gpu = multi_gpu_cfg is not None
        # Multi-GPU parameters
        if multi_gpu_cfg is not None:
            self.gpu_global_rank = multi_gpu_cfg["global_rank"]
            self.gpu_world_size = multi_gpu_cfg["world_size"]
        else:
            self.gpu_global_rank = 0
            self.gpu_world_size = 1

        self.rnd = None  # Compatibility with runner

        # distillation components
        self.policy = policy
        self.policy.to(self.device)
        self.storage: RolloutStorage = None  # type: ignore # initialized later
        
        # Optimizer for student encoder only
        student_encoder_params = [p for n, p in self.policy.named_parameters() if 'student_encoder' in n and p.requires_grad]
        self.optimizer = optim.Adam(student_encoder_params, lr=learning_rate)
        
        # Cache student encoder parameters for gradient clipping (avoid repeated filtering)
        self.student_encoder_params = student_encoder_params
        
        self.transition = RolloutStorage.Transition()

        # distillation parameters
        self.num_learning_epochs = num_learning_epochs
        self.learning_rate = learning_rate
        self.max_grad_norm = max_grad_norm
        self.embedding_loss_coef = embedding_loss_coef
        self.action_loss_coef = action_loss_coef

        # DAgger parameters
        self.use_dagger = use_dagger
        self.dagger_beta_start = dagger_beta_start
        self.dagger_beta_end = dagger_beta_end
        self.dagger_decay_steps = dagger_decay_steps
        self.dagger_schedule_type = dagger_schedule_type.lower()
        self.dagger_beta = dagger_beta_start  # Current beta value
        self.dagger_step = 0  # Current step/iteration/episode counter for decay
        
        # DAgger runtime state (per-environment policy selection)
        # Will be initialized in init_storage with num_envs
        self.using_teacher = None  # bool tensor [num_envs]: True = teacher, False = student

        # initialize the loss function
        if loss_type == "mse":
            self.loss_fn = nn.functional.mse_loss
        elif loss_type == "huber":
            self.loss_fn = nn.functional.huber_loss
        else:
            raise ValueError(f"Unknown loss type: {loss_type}. Supported types are: mse, huber")

        self.num_updates = 0
        
        print(f"\n{'='*70}")
        print(f"AdapterDistillation Initialized:")
        print(f"  Embedding loss weight (λ_embed): {embedding_loss_coef}")
        print(f"  Action loss weight (λ_action): {action_loss_coef}")
        print(f"  Learning rate: {learning_rate}")
        print(f"  Trainable parameters: {sum(p.numel() for p in student_encoder_params):,}")
        if self.use_dagger:
            print(f"  DAgger enabled:")
            print(f"    - Beta schedule: {self.dagger_schedule_type}")
            print(f"    - Beta start: {dagger_beta_start}")
            print(f"    - Beta end: {dagger_beta_end}")
            print(f"    - Decay steps: {dagger_decay_steps}")
        print(f"{'='*70}\n")

    def init_storage(
        self,
        training_type,
        num_envs,
        num_transitions_per_env,
        student_encoder_obs_shape,
        teacher_encoder_obs_shape,
        actions_shape,
        policy_obs_shape=None,
    ):
        """
        Initialize rollout storage for distillation.
        
        Args:
            training_type: Should be "distillation"
            num_envs: Number of parallel environments
            num_transitions_per_env: Number of transitions per rollout
            student_encoder_obs_shape: Shape of student encoder observations (limited)
            teacher_encoder_obs_shape: Shape of teacher encoder observations (full)
            actions_shape: Shape of action vectors
            policy_obs_shape: Shape of policy observations (for base actor MLP)
        """
        # create rollout storage
        # encoder_obs_shape will store student encoder observations
        # privileged_obs_shape will store teacher encoder observations
        self.storage = RolloutStorage(
            training_type,
            num_envs,
            num_transitions_per_env,
            student_encoder_obs_shape,  # encoder_observations (student)
            teacher_encoder_obs_shape,  # privileged_observations (teacher)
            actions_shape,
            None,  # rnd_state_shape
            policy_obs_shape,  # policy_observations
            self.device,
        )
        
        # Initialize DAgger per-environment policy selection
        if self.use_dagger:
            self.using_teacher = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
            self._update_dagger_policy_selection()

    def act(self, student_encoder_obs, policy_obs, teacher_encoder_obs):
        """
        Sample actions from the policy using student or teacher encoder based on DAgger.
        
        Args:
            student_encoder_obs: Student encoder observations (limited, for student encoder)
            policy_obs: Policy observations (flattened history, for base actor MLP)
            teacher_encoder_obs: Teacher encoder observations (full privileged, for teacher encoder)
        
        Returns:
            Actions sampled from student or teacher policy (based on DAgger beta)
        """
        # ============================================================
        # Compute teacher outputs (needed for update, and optionally for DAgger)
        # ============================================================
        with torch.no_grad():
            teacher_actions, teacher_latent = self.policy.get_teacher_action_mean(
                teacher_encoder_obs, policy_obs, return_latent=True
            )
        
        if self.use_dagger and self.dagger_beta > 0 and self.using_teacher is not None:
            # DAgger mode: Mix teacher and student actions
            # Use deterministic actions (means) for both teacher and student
            student_actions = self.policy.act_inference(
                student_encoder_obs, policy_obs, return_latent=False
            ).detach()
            
            # Select actions based on per-environment policy assignment
            # using_teacher is True where we should use teacher, False for student
            actions = torch.where(
                self.using_teacher.unsqueeze(-1),  # Shape: [num_envs, 1]
                teacher_actions,  # Use teacher actions
                student_actions   # Use student actions
            )
        else:
            # Pure student mode (no DAgger or beta=0)
            # Use deterministic actions (mean) to match the distillation loss
            actions = self.policy.act_inference(
                student_encoder_obs, policy_obs, return_latent=False
            ).detach()
        
        # Record observations and actions for distillation
        # encoder_observations stores student encoder obs
        self.transition.actions = actions  # type: ignore
        self.transition.encoder_observations = student_encoder_obs
        # policy_observations stores policy obs (for actor body)
        self.transition.policy_observations = policy_obs
        # privileged_observations stores teacher encoder obs
        self.transition.privileged_observations = teacher_encoder_obs
        # Store teacher outputs to avoid recomputing during update
        # Create new tensors to avoid "inference tensor" issues
        self.transition.teacher_latent = teacher_latent.clone().detach()
        self.transition.teacher_action_mean = teacher_actions.clone().detach()
        
        return actions

    def process_env_step(self, rewards, dones, infos):
        """Process environment step and store transition."""
        # record the rewards and dones
        self.transition.rewards = rewards
        self.transition.dones = dones
        # record the transition
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.policy.reset(dones)
        
        # Update DAgger policy selection for environments that just terminated
        if self.use_dagger and dones.any():
            self._update_dagger_policy_selection(dones)

    def update(self):
        """
        Update student encoder using distillation losses.
        
        Computes two losses:
        1. Embedding Loss: MSE between student and teacher encoder latents
        2. Action Loss: MSE between student and teacher deterministic action means
        
        Only student_encoder parameters are updated.
        """
        self.num_updates += 1
        mean_embedding_loss = 0
        mean_action_loss = 0
        mean_total_loss = 0
        num_batches = 0

        # Iterate through all transitions for multiple epochs
        for epoch in range(self.num_learning_epochs):
            # Use the distillation generator
            # Returns: (encoder_obs, privileged_obs, policy_obs, actions, dones, teacher_latent, teacher_action_mean)
            # - encoder_obs: student encoder observations [num_envs, obs_dim]
            # - privileged_obs: teacher encoder observations [num_envs, obs_dim] (not used for loss)
            # - policy_obs: policy observations for actor body [num_envs, policy_obs_dim]
            # - actions: student actions (not used for loss)
            # - dones: episode termination flags (not used)
            # - teacher_latent: cached teacher encoder latent (or None if not cached)
            # - teacher_action_mean: cached teacher action mean (or None if not cached)
            for (
                student_encoder_obs_batch,
                _,  # teacher_encoder_obs (not needed - teacher outputs are cached)
                policy_obs_batch,
                _,  # actions
                _,  # dones
                cached_teacher_latent,
                cached_teacher_action_mean
            ) in self.storage.generator():
                
                num_batches += 1
                
                # ============================================================
                # 1. Compute student outputs and get teacher outputs
                # ============================================================
                # Student: Get both action mean and latent in one call (with grad)
                student_action_mean, student_latent = self.policy.act_inference(
                    student_encoder_obs_batch, policy_obs_batch, return_latent=True
                )
                
                # Clone cached teacher outputs to convert from inference tensors
                teacher_latent = cached_teacher_latent.clone()
                teacher_action_mean = cached_teacher_action_mean.clone()
                
                # ============================================================
                # 2. Embedding Distillation Loss
                # ============================================================
                embedding_loss = self.loss_fn(student_latent, teacher_latent)

                # ============================================================
                # 3. Action Distillation Loss (MSE between deterministic actions)
                # ============================================================
                # Direct MSE loss between student and teacher action means
                # This encourages student to produce the same deterministic actions as teacher
                action_loss = self.loss_fn(student_action_mean, teacher_action_mean)

                # ============================================================
                # 4. Total Loss (weighted combination)
                # ============================================================
                total_loss = (
                    self.embedding_loss_coef * embedding_loss +
                    self.action_loss_coef * action_loss
                )
                
                # ============================================================
                # 5. Gradient Step (only student_encoder is updated)
                # ============================================================
                self.optimizer.zero_grad()
                total_loss.backward()
                
                # Multi-GPU gradient synchronization
                if self.is_multi_gpu:
                    self.reduce_parameters()
                
                # Gradient clipping (use cached parameters)
                if self.max_grad_norm:
                    nn.utils.clip_grad_norm_(
                        self.student_encoder_params,
                        self.max_grad_norm
                    )
                
                self.optimizer.step()
                
                # Accumulate losses for logging
                mean_embedding_loss += embedding_loss.item()
                mean_action_loss += action_loss.item()
                mean_total_loss += total_loss.item()

        # Average losses over all batches
        mean_embedding_loss /= num_batches
        mean_action_loss /= num_batches
        mean_total_loss /= num_batches
        
        # Clear storage
        self.storage.clear()

        # Construct the loss dictionary for logging
        loss_dict = {
            "embedding_loss": mean_embedding_loss,
            "action_loss": mean_action_loss,
            "total_loss": mean_total_loss,
        }

        return loss_dict

    """
    DAgger helper functions
    """

    def _update_dagger_beta(self):
        """Update DAgger beta based on the decay schedule."""
        if not self.use_dagger:
            return
        
        if self.dagger_schedule_type == "linear":
            # Linear decay: β = β_start - (β_start - β_end) * (step / decay_steps)
            progress = min(self.dagger_step / self.dagger_decay_steps, 1.0)
            self.dagger_beta = self.dagger_beta_start - (self.dagger_beta_start - self.dagger_beta_end) * progress
        
        elif self.dagger_schedule_type == "exponential":
            # Exponential decay: β = β_end + (β_start - β_end) * exp(-k * step)
            # where k is chosen such that at decay_steps, we're close to β_end
            k = 5.0 / self.dagger_decay_steps  # decay constant
            self.dagger_beta = self.dagger_beta_end + (self.dagger_beta_start - self.dagger_beta_end) * \
                torch.exp(torch.tensor(-k * self.dagger_step)).item()
        
        else:
            raise ValueError(f"Unknown DAgger schedule type: {self.dagger_schedule_type}")
        
        # Clamp beta to [0, 1]
        self.dagger_beta = max(0.0, min(1.0, self.dagger_beta))

    def _update_dagger_policy_selection(self, dones=None):
        """
        Update which environments use teacher vs student policy.
        
        This is called:
        1. At initialization (dones=None) - assign all envs
        2. After episode termination (dones provided) - reassign only terminated envs
        
        Args:
            dones: Boolean tensor [num_envs] indicating which environments just terminated.
                   If None, updates all environments.
        """
        if not self.use_dagger or self.using_teacher is None:
            return
        
        if dones is None:
            # Initialize all environments
            num_envs = self.using_teacher.shape[0]
            random_vals = torch.rand(num_envs, device=self.device)
            self.using_teacher[:] = random_vals < self.dagger_beta
        else:
            # Update only terminated environments
            envs_to_update = dones.squeeze(-1).bool() if dones.dim() > 1 else dones.bool()
            num_to_update = int(envs_to_update.sum().item())
            
            if num_to_update > 0:
                # Generate random values for environments that need update
                random_vals = torch.rand(num_to_update, device=self.device)
                # Assign teacher where random value < beta
                use_teacher = random_vals < self.dagger_beta
                # Update only the terminated environments
                self.using_teacher[envs_to_update] = use_teacher

    def increment_dagger_step(self):
        """
        Increment the DAgger step counter and update beta.
        
        Call this once per learning iteration to drive the beta decay schedule.
        """
        if not self.use_dagger:
            return
        
        self.dagger_step += 1
        self._update_dagger_beta()

    def get_dagger_stats(self):
        """
        Get current DAgger statistics for logging.
        
        Returns:
            dict: DAgger statistics including beta and teacher usage ratio
        """
        if not self.use_dagger or self.using_teacher is None:
            return {}
        
        teacher_ratio = self.using_teacher.float().mean().item()
        return {
            "dagger_beta": self.dagger_beta,
            "dagger_teacher_ratio": teacher_ratio,
            "dagger_step": self.dagger_step,
        }

    """
    Helper functions
    """

    def broadcast_parameters(self):
        """Broadcast model parameters to all GPUs."""
        # obtain the model parameters on current GPU
        model_params = [self.policy.state_dict()]
        # broadcast the model parameters
        torch.distributed.broadcast_object_list(model_params, src=0)
        # load the model parameters on all GPUs from source GPU
        self.policy.load_state_dict(model_params[0])

    def reduce_parameters(self):
        """Collect gradients from all GPUs and average them.

        This function is called after the backward pass to synchronize the gradients across all GPUs.
        """
        # Create a tensor to store the gradients
        grads = [param.grad.view(-1) for param in self.policy.parameters() if param.grad is not None]
        all_grads = torch.cat(grads)
        # Average the gradients across all GPUs
        torch.distributed.all_reduce(all_grads, op=torch.distributed.ReduceOp.SUM)
        all_grads /= self.gpu_world_size
        # Update the gradients for all parameters with the reduced gradients
        offset = 0
        for param in self.policy.parameters():
            if param.grad is not None:
                numel = param.numel()
                # copy data back from shared buffer
                param.grad.data.copy_(all_grads[offset : offset + numel].view_as(param.grad.data))
                # update the offset for the next parameter
                offset += numel
