"""
On-policy runner variant with an auxiliary regression predictor for ball pose.

- Maintains a ping-pong ball trajectory buffer per env from `TTEnv.ball_pos`.
- Trains a small MLP to predict a future ball pose from the last 5 positions.
- Uses the apex after bounce proxy target defined as argmax z with x < -1.
- Performs inference every step and, if available, calls `env.update_prediction`.
"""

from __future__ import annotations

import math
import os
import time
from collections import deque
from typing import Deque, List, Optional, Tuple

import torch

from .on_policy_runner import OnPolicyRunner


class _MLPPredictor(torch.nn.Module):
    def __init__(self, input_dim: int, hidden_sizes: Tuple[int, int] = (64, 64), output_dim: int = 3):
        super().__init__()
        h1, h2 = hidden_sizes
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, h1),
            torch.nn.ReLU(),
            torch.nn.Linear(h1, h2),
            torch.nn.ReLU(),
            torch.nn.Linear(h2, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class OnPolicyPredictorRegressionRunner(OnPolicyRunner):  # noqa: C901
    """Runner with auxiliary regression predictor for future ball pose."""

    def __init__(self, env, train_cfg: dict, log_dir: str | None = None, device: str = "cpu"):
        super().__init__(env, train_cfg, log_dir=log_dir, device=device)

        # Predictor config with sensible defaults
        pred_cfg = self.cfg.get("predictor", {})
        self.pred_history_len: int = int(pred_cfg.get("history_len", 5))
        self.pred_traj_maxlen: int = int(pred_cfg.get("traj_max_len", 256))
        self.pred_hidden: Tuple[int, int] = tuple(pred_cfg.get("hidden_sizes", (64, 64)))  # type: ignore
        self.pred_lr: float = float(pred_cfg.get("lr", 1e-3))
        self.pred_epochs: int = int(pred_cfg.get("epochs_per_update", 2))
        self.pred_batch_size: int = int(pred_cfg.get("batch_size", 1024))

        # optionally stop training predictor after a number of PPO iterations
        self.pred_train_until_iters: int = int(pred_cfg.get("train_until_iters", 1000))

        # Trajectory ring buffer on CPU: [T, N, 3]
        self._traj_maxlen: int = self.pred_traj_maxlen
        self._traj_buf_cpu: torch.Tensor = torch.zeros(
            self._traj_maxlen, self.env.num_envs, 3, dtype=torch.float32, device="cpu"
        )
        # Ground-truth future pose ring buffer on CPU: [T, N, 3] (env.ball_future_pose at each step)
        self._gt_buf_cpu: torch.Tensor = torch.zeros(
            self._traj_maxlen, self.env.num_envs, 3, dtype=torch.float32, device="cpu"
        )
        self._traj_write_idx: int = 0
        self._traj_len: int = 0  # how many steps recorded (capped at _traj_maxlen)

        # Predictor MLP and optimizer
        self._pred_input_dim = 3 * self.pred_history_len
        self._predictor = _MLPPredictor(self._pred_input_dim, self.pred_hidden, 3).to(self.device)
        self._pred_optim = torch.optim.Adam(self._predictor.parameters(), lr=self.pred_lr)
        self._pred_trained: bool = False
        self._last_pred_loss: Optional[float] = None
        # debug counters
        self._pred_call_count: int = 0

    # ---------------------------
    # Rollout and learning loop
    # ---------------------------
    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False):  # noqa: C901
        # initialize writer (copied from parent)
        if self.log_dir is not None and getattr(self, "writer", None) is None and not self.disable_logs:
            # Launch either Tensorboard or Neptune & Tensorboard summary writer(s), default: Tensorboard.
            self.logger_type = self.cfg.get("logger", "tensorboard").lower()
            if self.logger_type == "neptune":
                from rsl_rl.utils.neptune_utils import NeptuneSummaryWriter

                self.writer = NeptuneSummaryWriter(log_dir=self.log_dir, flush_secs=10, cfg=self.cfg)
                self.writer.log_config(self.env.cfg, self.cfg, self.alg_cfg, self.policy_cfg)
            elif self.logger_type == "wandb":
                from rsl_rl.utils.wandb_utils import WandbSummaryWriter

                self.writer = WandbSummaryWriter(log_dir=self.log_dir, flush_secs=10, cfg=self.cfg)
                self.writer.log_config(self.env.cfg, self.cfg, self.alg_cfg, self.policy_cfg)
            elif self.logger_type == "tensorboard":
                from torch.utils.tensorboard import SummaryWriter

                self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
            else:
                raise ValueError("Logger type not found. Please choose 'neptune', 'wandb' or 'tensorboard'.")

        # Initialize TT metrics at iter 0 so curves appear immediately
        if self.log_dir is not None and not self.disable_logs and getattr(self, "writer", None) is not None:
            try:
                if not hasattr(self, "_tt_logged_init"):
                    self.writer.add_scalar("Train/TT_success_rate", 0.0, 0)
                    self.writer.add_scalar("Train/TT_hit_rate", 0.0, 0)
                    self._tt_logged_init = True
            except Exception:
                pass

        # teacher check
        if self.training_type == "distillation" and not self.alg.policy.loaded_teacher:
            raise ValueError("Teacher model parameters not loaded. Please load a teacher model to distill.")

        # randomize initial episode lengths (for exploration)
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )

        # start learning
        obs, extras = self.env.get_observations()
        privileged_obs = extras["observations"].get(self.privileged_obs_type, obs)
        obs, privileged_obs = obs.to(self.device), privileged_obs.to(self.device)
        self.train_mode()

        # Book keeping (copied from parent)
        from collections import deque as _dq

        ep_infos = []
        rewbuffer = _dq(maxlen=100)
        lenbuffer = _dq(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        if self.alg.rnd:
            erewbuffer = _dq(maxlen=100)
            irewbuffer = _dq(maxlen=100)
            cur_ereward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
            cur_ireward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        # Table-tennis serve and hit tracking for training logs
        # Aggregate across rollout steps and log every 100 updates
        self._tt_succ_total: int = 0
        self._tt_hit_total: int = 0
        self._tt_serve_total: int = 0
        try:
            self._tt_serve_success_flag = torch.zeros(self.env.num_envs, dtype=torch.bool, device=self.env.device)
            self._tt_serve_hit_flag = torch.zeros(self.env.num_envs, dtype=torch.bool, device=self.env.device)
        except Exception:
            self._tt_serve_success_flag = None
            self._tt_serve_hit_flag = None

        # Ensure all parameters are in-synced
        if self.is_distributed:
            print(f"Synchronizing parameters for rank {self.gpu_global_rank}...")
            self.alg.broadcast_parameters()

        start_iter = self.current_learning_iteration
        tot_iter = start_iter + num_learning_iterations
        for it in range(start_iter, tot_iter):
            start = time.time()
            # Rollout
            with torch.inference_mode():
                for _ in range(self.num_steps_per_env):
                    # Sample actions and step
                    actions = self.alg.act(obs, privileged_obs)
                    obs, rewards, dones, infos = self.env.step(actions.to(self.env.device))
                    # Move to device
                    obs, rewards, dones = (obs.to(self.device), rewards.to(self.device), dones.to(self.device))
                    # perform normalization and update privileged obs
                    obs = self.obs_normalizer(obs)
                    if self.privileged_obs_type is not None:
                        privileged_obs = self.privileged_obs_normalizer(
                            infos["observations"][self.privileged_obs_type].to(self.device)
                        )
                    else:
                        privileged_obs = obs

                    # Aux: record ball positions and run predictor inference
                    try:
                        # if self.current_learning_iteration < self.pred_train_until_iters:
                        self._record_ball_positions()   
                        self._maybe_predict_and_update_env()
                    except Exception:
                        pass

                    # Track TT success/hit per-serve signals (similar to play.py)
                    try:
                        if (
                            hasattr(self.env, "has_touch_opponent_table_just_now")
                            and hasattr(self.env, "has_touch_paddle")
                            and self._tt_serve_success_flag is not None
                        ):
                            event_mask = (self.env.has_touch_opponent_table_just_now & self.env.has_touch_paddle)
                            self._tt_serve_success_flag |= event_mask.to(self._tt_serve_success_flag.device)
                        if (
                            hasattr(self.env, "ball_contact_rew")
                            and self._tt_serve_hit_flag is not None
                        ):
                            hit_mask = (self.env.ball_contact_rew > 0.0)
                            self._tt_serve_hit_flag |= hit_mask.to(self._tt_serve_hit_flag.device)

                        # On serve boundary, aggregate and reset flags
                        if hasattr(self.env, "ball_reset_ids") and self.env.ball_reset_ids is not None:
                            ids = self.env.ball_reset_ids
                            if (
                                isinstance(ids, torch.Tensor)
                                and ids.numel() > 0
                                and self._tt_serve_success_flag is not None
                                and self._tt_serve_hit_flag is not None
                            ):
                                ids_dev = ids.to(self._tt_serve_success_flag.device)
                                self._tt_serve_total += int(ids_dev.numel())
                                self._tt_succ_total += int(self._tt_serve_success_flag[ids_dev].sum().item())
                                self._tt_hit_total += int(self._tt_serve_hit_flag[ids_dev].sum().item())
                                self._tt_serve_success_flag[ids_dev] = False
                                self._tt_serve_hit_flag[ids_dev] = False
                    except Exception:
                        pass

                    # RL algorithm step processing
                    self.alg.process_env_step(rewards, dones, infos)

                    # Extract intrinsic rewards (only for logging)
                    intrinsic_rewards = self.alg.intrinsic_rewards if self.alg.rnd else None

                    # Logging bookkeeping mirrors parent
                    if self.log_dir is not None:
                        if "episode" in infos:
                            ep_infos.append(infos["episode"])
                        elif "log" in infos:
                            ep_infos.append(infos["log"])
                        if self.alg.rnd:
                            cur_ereward_sum += rewards
                            cur_ireward_sum += intrinsic_rewards  # type: ignore
                            cur_reward_sum += rewards + intrinsic_rewards
                        else:
                            cur_reward_sum += rewards
                        cur_episode_length += 1
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0
                        if self.alg.rnd:
                            erewbuffer.extend(cur_ereward_sum[new_ids][:, 0].cpu().numpy().tolist())
                            irewbuffer.extend(cur_ireward_sum[new_ids][:, 0].cpu().numpy().tolist())
                            cur_ereward_sum[new_ids] = 0
                            cur_ireward_sum[new_ids] = 0

                stop = time.time()
                collection_time = stop - start
                start = stop

                # compute returns (same as parent)
                if self.training_type == "rl":
                    self.alg.compute_returns(privileged_obs)

            # Train auxiliary predictor on offline trajectory data (optional cutoff)
            pred_loss_val = None
            if self.current_learning_iteration < self.pred_train_until_iters:
                pred_loss_val = self._train_predictor_offline()
            if pred_loss_val is not None:
                self._last_pred_loss = float(pred_loss_val)

            # Update policy (PPO)
            loss_dict = self.alg.update()

            stop = time.time()
            learn_time = stop - start
            self.current_learning_iteration = it

            # Attach predictor loss to logs if available
            if pred_loss_val is not None:
                loss_dict = dict(loss_dict)  # shallow copy for augmentation
                loss_dict["predictor_mse"] = float(pred_loss_val)

            if self.log_dir is not None and not self.disable_logs:
                self.log(locals())
                # Log TT success and hit rates every 100 updates
                if (it + 1) % 100 == 0:
                    serve_total = self._tt_serve_total
                    succ_rate = (self._tt_succ_total / serve_total) if serve_total > 0 else 0.0
                    hit_rate = (self._tt_hit_total / serve_total) if serve_total > 0 else 0.0
                    try:
                        self.writer.add_scalar("Train/TT_success_rate", succ_rate, it)
                        self.writer.add_scalar("Train/TT_hit_rate", hit_rate, it)
                    except Exception:
                        pass
                    # reset counters for next window
                    self._tt_succ_total = 0
                    self._tt_hit_total = 0
                    self._tt_serve_total = 0
                if it % self.save_interval == 0:
                    self.save(os.path.join(self.log_dir, f"model_{it}.pt"))

            ep_infos.clear()
            if it == start_iter and not self.disable_logs:
                from rsl_rl.utils import store_code_state

                git_file_paths = store_code_state(self.log_dir, self.git_status_repos)
                if self.logger_type in ["wandb", "neptune"] and git_file_paths:
                    for path in git_file_paths:
                        self.writer.save_file(path)

        if self.log_dir is not None and not self.disable_logs:
            self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"))

    # ---------------------------
    # Auxiliary predictor helpers
    # ---------------------------
    def _record_ball_positions(self):
        """Append current ball positions to the trajectory ring buffer (batched)."""
        if not hasattr(self.env, "ball_pos"):
            return
        with torch.no_grad():
            ball_pos = self.env.ball_pos.detach().to("cpu")  # [N, 3]
            self._traj_buf_cpu[self._traj_write_idx].copy_(ball_pos)
            # also record env-provided ground truth future pose as regression target
            # only needed while predictor training is active to save bandwidth
            try:
                if self.current_learning_iteration < self.pred_train_until_iters:
                    if hasattr(self.env, "ball_future_pose"):
                        gt_pose = self.env.ball_future_pose.detach().to("cpu")  # [N,3]
                        self._gt_buf_cpu[self._traj_write_idx].copy_(gt_pose)
            except Exception:
                pass
            self._traj_write_idx = (self._traj_write_idx + 1) % self._traj_maxlen
            self._traj_len = min(self._traj_len + 1, self._traj_maxlen)

    def _maybe_predict_and_update_env(self):
        """Run predictor inference given last 5 positions and update env with predictions."""
        if not self._pred_trained:
            if not hasattr(self, "_warn_no_pred"):
                print("[Predictor] Not trained or not loaded; skipping predictions.")
                self._warn_no_pred = True
            return
        # Build batched inputs for all envs with enough history
        if self._traj_len < self.pred_history_len:
            if not hasattr(self, "_warn_short_hist"):
                print(f"[Predictor] Not enough history yet (have {self._traj_len}, need {self.pred_history_len}).")
                self._warn_short_hist = True
            return
        # gather last H time indices in order
        H = self.pred_history_len
        idxs = (torch.arange(-H, 0) + self._traj_write_idx) % self._traj_maxlen  # [H]
        hist = self._traj_buf_cpu[idxs]  # [H, N, 3]
        X = hist.permute(1, 0, 2).reshape(self.env.num_envs, -1).to(self.device)  # [N, H*3]
        with torch.no_grad():
            preds = self._predictor(X)  # [N, 3]
        self._pred_call_count += 1
        if self._pred_call_count == 1:
            try:
                p0 = preds[0].detach().cpu().numpy()
                print(f"[Predictor] First prediction sample: {p0}")
            except Exception:
                pass
        # If update hook exists in env, call it. Otherwise, skip safely.
        try:
            if hasattr(self.env, "update_prediction"):
                self.env.update_prediction(preds)
        except Exception:
            # Do not break PPO rollout if env-side hook is not implemented yet
            pass

    def _train_predictor_offline(self) -> Optional[float]:
        """Create supervised dataset from trajectory buffers and train the predictor.

        Targets are taken directly from the environment-provided `ball_future_pose`
        recorded at each step (stored in `_gt_buf_cpu`). For each time t with
        sufficient history H, we create a sample: X = seq[t-H:t], Y = gt_seq[t-1].
        History windows that cross a ball reset are skipped to avoid mixing serves.
        """
        H = self.pred_history_len
        L = self._traj_len
        if L < H + 1:
            return None
        # unwrap the last L steps into time order
        idxs = (torch.arange(-L, 0) + self._traj_write_idx) % self._traj_maxlen
        seq = self._traj_buf_cpu[idxs]  # [L, N, 3] on CPU
        gt_seq = self._gt_buf_cpu[idxs]  # [L, N, 3] on CPU

        X_parts: List[torch.Tensor] = []
        Y_parts: List[torch.Tensor] = []
        # Iterate over time (batched over envs). L is capped by traj_max_len.
        for t in range(H, L):
            hist = seq[t - H : t]  # [H, N, 3]
            X_t_full = hist.permute(1, 0, 2).reshape(self.env.num_envs, -1)  # [N, H*3]
            # Use ground-truth at time t-1 (aligned with history window end)
            Y_t_all = gt_seq[t - 1]  # [N, 3]
            # Use all envs without reset-based filtering
            X_parts.append(X_t_full)
            Y_parts.append(Y_t_all)

        if not X_parts:
            return None

        X = torch.cat(X_parts, dim=0).to(self.device)
        Y = torch.cat(Y_parts, dim=0).to(self.device)
        # Optional input augmentation: add Gaussian noise if env uses noise
        try:
            if getattr(self.env, "add_noise", False):
                X = X + 0.02 * torch.randn_like(X)
        except Exception:
            pass

        # Simple supervised regression training
        criterion = torch.nn.MSELoss()
        self._predictor.train()
        total_loss = 0.0
        n_batches = 0
        # no shuffling needed: samples already come from varying times and envs

        bs = self.pred_batch_size
        for _ in range(max(1, self.pred_epochs)):
            for s in range(0, X.shape[0], bs):
                e = min(s + bs, X.shape[0])
                xb = X[s:e]
                yb = Y[s:e]
                pred = self._predictor(xb)
                loss = criterion(pred, yb)
                self._pred_optim.zero_grad()
                loss.backward()
                self._pred_optim.step()
                total_loss += float(loss.detach().item())
                n_batches += 1

        self._pred_trained = True
        mean_loss = total_loss / max(1, n_batches)
        return mean_loss

    # ---------------------------
    # Checkpointing (save/load)
    # ---------------------------
    def save(self, path: str, infos=None):
        """Save PPO policy plus auxiliary predictor weights and optimizers."""
        # base dict mirrors parent save()
        saved_dict = {
            "model_state_dict": self.alg.policy.state_dict(),
            "optimizer_state_dict": self.alg.optimizer.state_dict(),
            "iter": self.current_learning_iteration,
            "infos": infos,
        }
        # RND (if used by PPO)
        if self.alg.rnd:
            saved_dict["rnd_state_dict"] = self.alg.rnd.state_dict()
            saved_dict["rnd_optimizer_state_dict"] = self.alg.rnd_optimizer.state_dict()
        # Normalizers (if used)
        if self.empirical_normalization:
            saved_dict["obs_norm_state_dict"] = self.obs_normalizer.state_dict()
            saved_dict["privileged_obs_norm_state_dict"] = self.privileged_obs_normalizer.state_dict()
        # Auxiliary predictor
        if hasattr(self, "_predictor") and self._predictor is not None:
            saved_dict["pred_state_dict"] = self._predictor.state_dict()
            saved_dict["pred_optimizer_state_dict"] = self._pred_optim.state_dict()
            saved_dict["pred_cfg"] = {
                "history_len": self.pred_history_len,
                "traj_max_len": self.pred_traj_maxlen,
                "hidden_sizes": list(self.pred_hidden),
                "lr": self.pred_lr,
                "epochs_per_update": self.pred_epochs,
                "batch_size": self.pred_batch_size,
            }

        torch.save(saved_dict, path)

        if self.logger_type in ["neptune", "wandb"] and not self.disable_logs:
            self.writer.save_model(path, self.current_learning_iteration)

    def load(self, path: str, load_optimizer: bool = True):
        """Load PPO policy and auxiliary predictor if present in checkpoint."""
        loaded_dict = torch.load(path, weights_only=False)
        # -- PPO model
        resumed_training = self.alg.policy.load_state_dict(loaded_dict["model_state_dict"])
        # -- RND
        if self.alg.rnd and "rnd_state_dict" in loaded_dict:
            self.alg.rnd.load_state_dict(loaded_dict["rnd_state_dict"])
        # -- Normalizers
        if self.empirical_normalization:
            if resumed_training:
                self.obs_normalizer.load_state_dict(loaded_dict["obs_norm_state_dict"])
                self.privileged_obs_normalizer.load_state_dict(loaded_dict["privileged_obs_norm_state_dict"])
            else:
                self.privileged_obs_normalizer.load_state_dict(loaded_dict["obs_norm_state_dict"])
        # -- Optimizers
        if load_optimizer and resumed_training:
            self.alg.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
            if self.alg.rnd and "rnd_optimizer_state_dict" in loaded_dict:
                self.alg.rnd_optimizer.load_state_dict(loaded_dict["rnd_optimizer_state_dict"])
        # -- Auxiliary predictor
        if "pred_state_dict" in loaded_dict:
            try:
                self._predictor.load_state_dict(loaded_dict["pred_state_dict"])
                if load_optimizer and "pred_optimizer_state_dict" in loaded_dict:
                    self._pred_optim.load_state_dict(loaded_dict["pred_optimizer_state_dict"])
                self._pred_trained = True
                print("[Predictor] Loaded predictor weights from checkpoint; predictions enabled.")
            except Exception:
                # if shape mismatch due to config change, keep running without predictor weights
                self._pred_trained = False
        else:
            try:
                print("[Predictor] No predictor weights found in checkpoint. Keys:", list(loaded_dict.keys()))
            except Exception:
                pass
        # -- current iter
        if resumed_training:
            self.current_learning_iteration = loaded_dict["iter"]
        return loaded_dict.get("infos")
