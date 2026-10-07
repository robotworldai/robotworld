#!/usr/bin/env python3
# Copyright (c) 2022-2025
# SPDX-License-Identifier: BSD-3-Clause

import argparse
import os
import glob
import shutil
from pathlib import Path

import numpy as np
import torch
import gymnasium as gym
from rsl_rl.runners import OnPolicyRunner
from copy import deepcopy
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from scripts.rsl_rl.adapter.student_teacher import AdaptedStudentTeacher

# ---------- Helper Functions ----------
def _policy_actions(policy, obs: torch.Tensor):
    """
    Unified way to extract actions from a policy:
    1) Prefer policy.act(obs, deterministic=True)
    2) Otherwise try policy.forward(obs)
    3) Handle dict or tuple outputs by extracting the action tensor
    """
    if hasattr(policy, "act"):
        out = policy.act(obs, deterministic=True)
    elif hasattr(policy, "forward"):
        try:
            out = policy.forward(obs)
        except TypeError:
            out = policy.forward()
    else:
        raise RuntimeError("Policy has neither .act(...) nor .forward(...).")

    if isinstance(out, dict):
        if "actions" in out:
            out = out["actions"]
        else:
            for v in out.values():
                if torch.is_tensor(v):
                    out = v
                    break
    elif isinstance(out, tuple):
        for v in out:
            if torch.is_tensor(v):
                out = v
                break

    if not torch.is_tensor(out):
        raise RuntimeError(f"Policy output is not a Tensor: type={type(out)}")
    return out


def _collect_checkpoints(inputs):
    """Collect checkpoints from files / directories / wildcard patterns."""
    results = []
    patterns = ("*.pt", "*.pth")
    for item in inputs:
        p = Path(item).expanduser()
        if any(ch in item for ch in "*?[]"):  # glob pattern
            results.extend([Path(x) for x in glob.glob(item)])
        elif p.is_dir():
            for pat in patterns:
                results.extend(p.rglob(pat))
        elif p.is_file():
            results.append(p)
        else:
            print(f"[WARN] Path does not exist or does not match: {item}")
    return sorted({x.resolve() for x in results if x.is_file()})


def _apply_normalizer(normalizer, x):
    """Apply observation normalizer if available."""
    if normalizer is None:
        return x
    if hasattr(normalizer, "normalize"):
        return normalizer.normalize(x)
    if callable(normalizer):
        return normalizer(x)
    return x


def _is_distillation_checkpoint(state_dict):
    """Check if checkpoint is from distillation training (has student_encoder and teacher_encoder)."""
    has_student = any('student_encoder.' in k for k in state_dict)
    has_teacher = any('teacher_encoder.' in k for k in state_dict)
    return has_student and has_teacher

def _is_standard_checkpoint(state_dict):
    """Check if checkpoint is standard RSL-RL (no encoder, no adapters)."""
    has_encoder = any("encoder." in k for k in state_dict.keys())
    has_adapter = any(".mod." in k or "frozen_actor." in k or "residual_adapter." in k
                      for k in state_dict.keys())
    return not has_encoder and not has_adapter


def _detect_distillation_adapter_type(state_dict):
    """Detect whether a distillation checkpoint uses FiLM or Residual adapter architecture."""
    has_actor_body = any(k.startswith('actor_body.') for k in state_dict)
    has_action_head = any(k.startswith('action_head.') for k in state_dict)
    has_frozen_actor = any(k.startswith('frozen_actor.') for k in state_dict)
    has_residual_adapter = any(k.startswith('residual_adapter.') for k in state_dict)

    if has_actor_body and has_action_head:
        return "film"
    elif has_frozen_actor and has_residual_adapter:
        return "residual"
    else:
        raise ValueError(
            "Cannot detect adapter type from checkpoint keys. "
            "Expected 'actor_body.*' + 'action_head.*' (FiLM) "
            "or 'frozen_actor.*' + 'residual_adapter.*' (Residual)."
        )


def _auto_detect_distillation_params(state_dict, adapter_type):
    """
    Auto-detect architecture parameters from a distillation checkpoint's state_dict.

    Returns:
        dict: Parameters suitable for constructing AdaptedStudentTeacher.
    """
    params = {}

    # --- Encoder dimensions ---
    params['num_student_encoder_obs'] = state_dict["student_encoder.embed.weight"].shape[1]
    params['num_teacher_encoder_obs'] = state_dict["teacher_encoder.embed.weight"].shape[1]
    params['ctx_dim'] = state_dict["student_encoder.embed.weight"].shape[0]

    # --- Encoder type detection ---
    has_gru = any('student_encoder.gru.' in k for k in state_dict)
    has_causal = any('student_encoder.causal_mask' in k for k in state_dict)
    has_transformer_module = any('student_encoder.transformer.' in k for k in state_dict)
    has_direct_layers = any(k.startswith('student_encoder.layers.') for k in state_dict)

    if has_causal or (has_direct_layers and not has_transformer_module and not has_gru):
        params['encoder_type'] = 'causal_transformer'
    elif has_transformer_module:
        params['encoder_type'] = 'transformer'
    elif has_gru:
        params['encoder_type'] = 'gru'
    else:
        params['encoder_type'] = 'gru'
        print("[WARN] Could not detect encoder type, defaulting to 'gru'")

    # --- Encoder layers ---
    if params['encoder_type'] == 'gru':
        gru_layer_keys = [k for k in state_dict if 'student_encoder.gru.weight_ih_l' in k]
        params['encoder_layers'] = len(gru_layer_keys) if gru_layer_keys else 1
    else:
        layer_keys = [k for k in state_dict
                      if 'student_encoder.transformer.layers.' in k
                      or 'student_encoder.layers.' in k]
        indices = set()
        for k in layer_keys:
            parts = k.split('.')
            for i, part in enumerate(parts):
                if part == 'layers' and i + 1 < len(parts) and parts[i + 1].isdigit():
                    indices.add(int(parts[i + 1]))
        params['encoder_layers'] = max(indices) + 1 if indices else 1

    # --- Actor architecture (adapter-type dependent) ---
    if adapter_type == "film":
        params['num_actor_obs'] = state_dict["actor_body.0.base.weight"].shape[1]
        actor_hidden_dims = []
        i = 0
        while f"actor_body.{i}.base.weight" in state_dict:
            actor_hidden_dims.append(state_dict[f"actor_body.{i}.base.weight"].shape[0])
            i += 2  # skip activation layers
        params['actor_hidden_dims'] = actor_hidden_dims
        params['num_actions'] = state_dict["action_head.weight"].shape[0]
        # FiLM-specific parameters
        if "actor_body.0.mod.0.weight" in state_dict:
            params['adapter_hidden'] = state_dict["actor_body.0.mod.0.weight"].shape[0]
        params['use_gate'] = "actor_body.0.alpha" in state_dict

    elif adapter_type == "residual":
        params['num_actor_obs'] = state_dict["frozen_actor.0.weight"].shape[1]
        linear_indices = sorted(
            int(k.split('.')[1]) for k in state_dict
            if k.startswith('frozen_actor.') and k.endswith('.weight')
        )
        actor_hidden_dims = [state_dict[f"frozen_actor.{idx}.weight"].shape[0]
                             for idx in linear_indices[:-1]]
        params['actor_hidden_dims'] = actor_hidden_dims
        params['num_actions'] = state_dict[f"frozen_actor.{linear_indices[-1]}.weight"].shape[0]
        # Residual adapter parameters
        # Only consider 2D weights (nn.Linear), skip 1D weights (nn.LayerNorm)
        res_linear_indices = sorted(
            int(k.split('.')[2]) for k in state_dict
            if k.startswith('residual_adapter.residual_mlp.') and k.endswith('.weight')
            and state_dict[k].dim() == 2  # Linear weights are 2D, LayerNorm weights are 1D
        )
        params['residual_hidden_dims'] = [
            state_dict[f"residual_adapter.residual_mlp.{idx}.weight"].shape[0]
            for idx in res_linear_indices[:-1]  # exclude final output layer
        ]
        params['use_gate'] = "residual_adapter.alpha" in state_dict

    # --- Critic architecture ---
    critic_indices = sorted(
        int(k.split('.')[1]) for k in state_dict
        if k.startswith('critic.') and k.endswith('.weight')
    )
    params['critic_hidden_dims'] = [state_dict[f"critic.{idx}.weight"].shape[0]
                                    for idx in critic_indices[:-1]]
    params['num_critic_obs'] = state_dict["critic.0.weight"].shape[1] - params['ctx_dim']

    # --- Noise std type ---
    if 'std' in state_dict:
        params['noise_std_type'] = 'scalar'
    elif 'log_std' in state_dict:
        params['noise_std_type'] = 'log'
    else:
        params['noise_std_type'] = 'scalar'

    return params


def load_ckpt_smart(path: str | Path, device: str | torch.device):
    """
    Smart loader for either:
      - state_dict-like checkpoints (training dict or pure state_dict)
      - TorchScript archives (ScriptModule)
    Returns:
      {"kind":"state_dict", "state_dict":..., "extra":...}
      or
      {"kind":"script", "module":ScriptModule}
    """
    p = str(path)

    # 1) Try state_dict with weights_only=True (PyTorch 2.6 default)
    try:
        obj = torch.load(p, map_location=device, weights_only=True)
        if isinstance(obj, dict):
            if "model_state_dict" in obj and isinstance(obj["model_state_dict"], dict):
                return {"kind": "state_dict", "state_dict": obj["model_state_dict"], "extra": obj}
            if "state_dict" in obj and isinstance(obj["state_dict"], dict):
                return {"kind": "state_dict", "state_dict": obj["state_dict"], "extra": obj}
            # Some checkpoints are pure state_dict
            if all(isinstance(k, str) for k in obj.keys()):
                return {"kind": "state_dict", "state_dict": obj, "extra": obj}
    except RuntimeError as e:
        # TorchScript archive + weights_only=True will land here
        if "TorchScript archives" not in str(e):
            raise

    # 2) Try as TorchScript
    try:
        script_mod = torch.jit.load(p, map_location=device)
        return {"kind": "script", "module": script_mod}
    except Exception:
        pass

    # 3) Fallback state_dict with weights_only=False (only for trusted sources)
    obj = torch.load(p, map_location=device, weights_only=False)
    if isinstance(obj, dict):
        if "model_state_dict" in obj and isinstance(obj["model_state_dict"], dict):
            return {"kind": "state_dict", "state_dict": obj["model_state_dict"], "extra": obj}
        if "state_dict" in obj and isinstance(obj["state_dict"], dict):
            return {"kind": "state_dict", "state_dict": obj["state_dict"], "extra": obj}
        if all(isinstance(k, str) for k in obj.keys()):
            return {"kind": "state_dict", "state_dict": obj, "extra": obj}

    raise ValueError(f"Unrecognized checkpoint format: {path}")


# ---------- Export Implementation ----------
def export_policy_as_jit(policy, normalizer, path, filename="policy.pt", obs_dim: int = None, device="cpu"):
    assert obs_dim is not None, "export_policy_as_jit requires obs_dim"

    class ExportPolicy(torch.nn.Module):
        def __init__(self, policy, normalizer):
            super().__init__()
            self.policy = policy
            self.normalizer = normalizer

        def forward(self, observations):
            observations = _apply_normalizer(self.normalizer, observations)
            actions = _policy_actions(self.policy, observations)
            return actions

    export_policy = ExportPolicy(policy, normalizer).eval()
    example_obs = torch.randn(1, obs_dim, dtype=torch.float32, device=device)
    os.makedirs(path, exist_ok=True)
    with torch.no_grad():
        traced = torch.jit.trace(export_policy, example_obs)
    traced.save(os.path.join(path, filename))


def export_distillation_policy_as_jit(
    policy, 
    normalizer, 
    path, 
    filename="policy.pt", 
    student_encoder_obs_dim: int = None,
    policy_obs_dim: int = None,
    encoder_seq_len: int = 32,
    adapter_type: str = "film",
    device="cpu"
):
    """
    Export distillation policy (AdaptedStudentTeacher) for inference.
    Supports both FiLM and Residual adapter architectures.
    
    Args:
        policy: AdaptedStudentTeacher policy
        normalizer: Observation normalizer (for policy_obs)
        path: Export directory
        filename: Output filename
        student_encoder_obs_dim: Dimension of student encoder observations (per timestep)
        policy_obs_dim: Dimension of policy (actor) observations
        encoder_seq_len: Sequence length for encoder input (default: 32 for multi-timestep)
        adapter_type: Adapter architecture type ("film" or "residual")
        device: Device to run on
    """
    assert student_encoder_obs_dim is not None, "student_encoder_obs_dim required"
    assert policy_obs_dim is not None, "policy_obs_dim required"

    if adapter_type == "film":
        class ExportDistillationPolicy(torch.nn.Module):
            """Export wrapper for FiLM adapter: actor_body(obs, ctx) -> action_head -> actions."""
            def __init__(self, policy, normalizer):
                super().__init__()
                self.student_encoder = policy.student_encoder
                self.actor_body = policy.actor_body
                self.action_head = policy.action_head
                self.normalizer = normalizer

            def forward(self, student_encoder_obs, policy_obs):
                policy_obs = _apply_normalizer(self.normalizer, policy_obs)
                e_t = self.student_encoder(student_encoder_obs)
                h = self.actor_body(policy_obs, e_t)
                actions = self.action_head(h)
                return actions

    elif adapter_type == "residual":
        class ExportDistillationPolicy(torch.nn.Module):
            """Export wrapper for Residual adapter: frozen_actor(obs) + residual(ctx, proprio) -> actions."""
            def __init__(self, policy, normalizer):
                super().__init__()
                self.student_encoder = policy.student_encoder
                self.frozen_actor = policy.frozen_actor
                self.residual_adapter = policy.residual_adapter
                self.normalizer = normalizer

            def forward(self, student_encoder_obs, policy_obs):
                policy_obs = _apply_normalizer(self.normalizer, policy_obs)
                e_t = self.student_encoder(student_encoder_obs)
                base_actions = self.frozen_actor(policy_obs)
                actions = self.residual_adapter(base_actions, e_t, proprio=policy_obs)
                return actions
    else:
        raise ValueError(f"Unknown adapter_type: {adapter_type}. Must be 'film' or 'residual'")

    export_policy = ExportDistillationPolicy(policy, normalizer).eval()
    
    # Create example inputs
    # student_encoder_obs shape: [seq_len, batch, obs_dim]
    example_student_encoder_obs = torch.randn(encoder_seq_len, 1, student_encoder_obs_dim, dtype=torch.float32, device=device)
    # policy_obs shape: [batch, policy_obs_dim]
    example_policy_obs = torch.randn(1, policy_obs_dim, dtype=torch.float32, device=device)
    
    os.makedirs(path, exist_ok=True)
    with torch.no_grad():
        traced = torch.jit.trace(export_policy, (example_student_encoder_obs, example_policy_obs))
    traced.save(os.path.join(path, filename))


def export_policy_as_onnx(policy, normalizer, path, filename="policy.onnx",
                          input_names=("observations",), output_names=("actions",),
                          obs_dim: int = None, device="cpu"):
    assert obs_dim is not None, "export_policy_as_onnx requires obs_dim"

    class ExportPolicy(torch.nn.Module):
        def __init__(self, policy, normalizer):
            super().__init__()
            self.policy = policy
            self.normalizer = normalizer

        def forward(self, observations):
            observations = _apply_normalizer(self.normalizer, observations)
            actions = _policy_actions(self.policy, observations)
            return actions

    export_policy = ExportPolicy(policy, normalizer).eval()
    example_obs = torch.randn(1, obs_dim, dtype=torch.float32, device=device)
    os.makedirs(path, exist_ok=True)
    with torch.no_grad():
        torch.onnx.export(
            export_policy,
            example_obs,
            os.path.join(path, filename),
            input_names=list(input_names),
            output_names=list(output_names),
            dynamic_axes={input_names[0]: {0: "batch_size"}, output_names[0]: {0: "batch_size"}},
            opset_version=12,
        )


# ---------- Minimal Dummy Env ----------
class DummyEnv:
    """Minimal Dummy Environment: only provides interfaces for OnPolicyRunner init."""
    def __init__(self, obs_dim, action_dim, device):
        self.obs_dim = int(obs_dim)
        self.privileged_obs_dim = 15
        self.action_dim = int(action_dim)
        self.device = torch.device(device)
        self.num_envs = 1

        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32
        )
        self.privileged_observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.privileged_obs_dim,), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(self.action_dim,), dtype=np.float32
        )

        class _U: pass
        self.unwrapped = _U()
        self.unwrapped.device = self.device

        self.num_obs = self.obs_dim
        self.num_privileged_obs = self.privileged_obs_dim
        self.num_actions = self.action_dim

    def get_observations(self):
        obs = torch.zeros(self.num_envs, self.obs_dim, device=self.device, dtype=torch.float32)
        privileged_obs = torch.zeros(self.num_envs, self.privileged_obs_dim, device=self.device, dtype=torch.float32)
        extras = {
            "observations": {},                  # no critic-specific obs
            "privileged_observations": privileged_obs,
            "rew_terms": {},                     # reward terms placeholder
        }
        return obs, extras

    def reset(self): raise RuntimeError("DummyEnv.reset() should not be called")
    def step(self, a): raise RuntimeError("DummyEnv.step() should not be called")


# ---------- CLI ----------
def build_argparser():
    p = argparse.ArgumentParser("Batch export RSL-RL checkpoints to JIT/ONNX (no IsaacSim needed)")

    p.add_argument("--input_path", nargs="*", default=["logs/rsl_rl/"],
                   help="Checkpoint file(s) / directory / wildcard, multiple allowed")
    p.add_argument("--output_path", type=str, default=".",
                   help="Optional: unified export root. Defaults to ckpt folder/exported_<stem>")

    p.add_argument("--obs_dim", type=int, default=480, help="Observation dimension (e.g. G1: 480)")
    p.add_argument("--action_dim", type=int, default=29, help="Action dimension (e.g. G1: 29)")
    
    # For distillation/adapter policies
    p.add_argument("--encoder_seq_len", type=int, default=32,
                   help="Encoder sequence length (default: 32 for multi-timestep deployment)")
    p.add_argument("--num_heads", type=int, default=4,
                   help="Number of attention heads for transformer encoder (default: 4)")

    p.add_argument("--device", type=str, default="cpu",
                   help="Device to run on, e.g. cpu / cuda:0 / cuda:1")
    return p


def main():
    parser = build_argparser()
    args = parser.parse_args()

    ckpts = _collect_checkpoints(args.input_path)
    if not ckpts:
        print("[ERROR] No .pt/.pth checkpoints found")
        return
    print(f"[INFO] Found {len(ckpts)} checkpoints; device={args.device}")

    # Minimal runner config (avoid IsaacLab deps)
    agent_cfg = {
        "device": args.device,
        "clip_actions": True,
        "num_steps_per_env": 16,
        "max_iterations": 1,
        "save_interval": 1000000,
        "experiment_name": "export_only",
        "runner_log_interval": 1000000,
        "empirical_normalization": False,
        "policy": {
            "class_name": "ActorCritic",
            "init_noise_std": 1.0,
            "actor_hidden_dims": [512, 256, 128],
            "critic_hidden_dims": [512, 256, 128],
            "activation": "elu",
        },
        "algorithm": {
            "class_name": "PPO",
            "value_loss_coef": 1.0,
            "use_clipped_value_loss": True,
            "clip_param": 0.2,
            "entropy_coef": 0.01,
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "learning_rate": 1.0e-3,
            "schedule": "adaptive",
            "gamma": 0.99,
            "lam": 0.95,
            "desired_kl": 0.01,
            "max_grad_norm": 1.0,
        },
    }

    for ckpt in ckpts:
        print(f"\n[INFO] Loading: {ckpt}")
        
        info = load_ckpt_smart(ckpt, args.device)

        ckpt_name = ckpt.stem
        if ckpt_name.startswith("model_"):
            model_number = ckpt_name.replace("model_", "")
            jit_filename = f"policy_{model_number}.pt"
            onnx_filename = f"policy_{model_number}.onnx"
        else:
            jit_filename = "policy.pt"
            onnx_filename = "policy.onnx"

        # Export to fixed "exported" folder
        export_dir = Path(args.output_path).expanduser().resolve() / "exported"
        export_dir.mkdir(parents=True, exist_ok=True)

        if info["kind"] == "state_dict":
            model_state_dict = info["state_dict"]
            
            # Detect checkpoint type
            is_distillation = _is_distillation_checkpoint(model_state_dict)
            is_standard = _is_standard_checkpoint(model_state_dict)
            
            if is_distillation:
                print("[INFO] Detected: Distillation Policy (Student Encoder)")
                
                # Auto-detect adapter type and architecture parameters
                adapter_type = _detect_distillation_adapter_type(model_state_dict)
                arch_params = _auto_detect_distillation_params(model_state_dict, adapter_type)
                
                print(f"  - Adapter type: {adapter_type.upper()}")
                print(f"  - Encoder type: {arch_params['encoder_type'].upper()}")
                print(f"  - Student encoder input dim: {arch_params['num_student_encoder_obs']}")
                print(f"  - Teacher encoder input dim: {arch_params['num_teacher_encoder_obs']}")
                print(f"  - Actor input dim: {arch_params['num_actor_obs']}")
                print(f"  - Actor hidden dims: {arch_params['actor_hidden_dims']}")
                print(f"  - Num actions: {arch_params['num_actions']}")
                print(f"  - Context dim: {arch_params['ctx_dim']}")
                if adapter_type == "film":
                    print(f"  - FiLM adapter hidden: {arch_params.get('adapter_hidden', 'N/A')}")
                elif adapter_type == "residual":
                    print(f"  - Residual hidden dims: {arch_params.get('residual_hidden_dims', [])}")
                
                # Create policy with auto-detected parameters
                policy = AdaptedStudentTeacher(
                    num_actor_obs=arch_params['num_actor_obs'],
                    num_critic_obs=arch_params['num_critic_obs'],
                    num_actions=arch_params['num_actions'],
                    num_student_encoder_obs=arch_params['num_student_encoder_obs'],
                    num_teacher_encoder_obs=arch_params['num_teacher_encoder_obs'],
                    actor_hidden_dims=arch_params['actor_hidden_dims'],
                    critic_hidden_dims=arch_params['critic_hidden_dims'],
                    activation='elu',
                    ctx_dim=arch_params['ctx_dim'],
                    encoder_layers=arch_params['encoder_layers'],
                    use_gate=arch_params['use_gate'],
                    encoder_type=arch_params['encoder_type'],
                    num_heads=args.num_heads,
                    encoder_dropout=0.0,
                    adapter_type=adapter_type,
                    adapter_hidden=arch_params.get('adapter_hidden', 128),
                    residual_hidden_dims=arch_params.get('residual_hidden_dims', [128, 64]),
                    noise_std_type=arch_params.get('noise_std_type', 'scalar'),
                ).to(args.device)
                
                # Load weights
                policy.load_state_dict(model_state_dict, strict=False)
                policy.eval()
                
                # Export using distillation-specific function
                export_distillation_policy_as_jit(
                    policy,
                    normalizer=None,  # no normalizer in checkpoint
                    path=str(export_dir),
                    filename=jit_filename,
                    student_encoder_obs_dim=arch_params['num_student_encoder_obs'],
                    policy_obs_dim=arch_params['num_actor_obs'],
                    encoder_seq_len=args.encoder_seq_len,
                    adapter_type=adapter_type,
                    device=args.device
                )
                print(f"[OK] JIT (distillation/{adapter_type}): {export_dir/jit_filename}")
                
            elif is_standard:
                print("[INFO] Detected: Standard RSL-RL Policy")
                # Use standard OnPolicyRunner
                env = DummyEnv(args.obs_dim, args.action_dim, device=args.device)
                cfg = deepcopy(agent_cfg)
                runner = OnPolicyRunner(env, cfg, log_dir=None, device=args.device)
                
                current_state_dict = runner.alg.policy.state_dict()
                filtered_state_dict = {
                    k: v for k, v in model_state_dict.items()
                    if k in current_state_dict and getattr(v, "shape", None) == getattr(current_state_dict[k], "shape", None)
                }
                missing = [k for k in current_state_dict.keys() if k not in filtered_state_dict]
                if missing:
                    print(f"[WARN] {len(missing)} keys missing or shape-mismatch; loading filtered subset.")

                runner.alg.policy.load_state_dict(filtered_state_dict, strict=False)

                # Extract policy module (compatibility across versions)
                try:
                    policy_nn = runner.alg.policy
                except AttributeError:
                    policy_nn = runner.alg.actor_critic

                normalizer = getattr(runner, "obs_normalizer", None) or getattr(runner.alg, "obs_normalizer", None)

                # Export JIT
                try:
                    export_policy_as_jit(policy_nn.actor, normalizer, path=str(export_dir),
                                         filename=jit_filename, obs_dim=args.obs_dim, device=args.device)
                    print(f"[OK] JIT:   {export_dir/jit_filename}")
                except Exception as e:
                    print(f"[FAIL] JIT: {e}")
            else:
                print("[WARN] Unknown checkpoint type")
                print("[SKIP] Skipping checkpoint")
                continue

        elif info["kind"] == "script":
            # Already TorchScript module
            print("[INFO] Detected: TorchScript Module")
            script_mod = info["module"].eval()
            # Save JIT as-is
            try:
                target = export_dir / jit_filename
                # Prefer re-save to ensure device-neutral
                script_mod.save(str(target))
                print(f"[OK] JIT (from TorchScript): {target}")
            except Exception as e:
                # Fallback: copy original file
                print(f"[WARN] Re-saving ScriptModule failed ({e}); copying original file.")
                shutil.copy2(str(ckpt), str(export_dir / jit_filename))
                print(f"[OK] JIT (copied): {export_dir / jit_filename}")

            # ONNX: usually not supported from ScriptModule
            print("[INFO] Skipping ONNX export for TorchScript archive. "
                  "Use a training state_dict checkpoint to export ONNX.")

    print("\n[DONE] All checkpoints processed.")
    print(f"[INFO] Exported policies saved to: {export_dir}")


if __name__ == "__main__":
    main()
