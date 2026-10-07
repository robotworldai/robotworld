#!/usr/bin/env python3

"""Expand a trained flat RSL-RL checkpoint for the jump task observations.

The jump observations are appended after the existing flat observations.  This
tool therefore preserves the learned deterministic policy and adds zero-weight
columns for the new features. New normalizer means are zero and variances/stds
are one. Excessive exploration standard deviations can optionally be capped.
Use the resulting checkpoint with ``train.py --load_weights_only``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def _expand_observation_input(state: dict[str, torch.Tensor], extra_dims: int, label: str) -> None:
    if extra_dims <= 0:
        return

    weight_key = "mlp.0.weight"
    if weight_key not in state:
        raise KeyError(f"{label} state has no {weight_key!r}; unsupported model layout.")
    weight = state[weight_key]
    state[weight_key] = torch.cat(
        (weight, torch.zeros(weight.shape[0], extra_dims, dtype=weight.dtype, device=weight.device)),
        dim=1,
    )

    for key in ("obs_normalizer._mean", "obs_normalizer._var", "obs_normalizer._std"):
        if key not in state:
            raise KeyError(f"{label} state has no {key!r}; observation normalization is required.")
        value = state[key]
        fill = 0.0 if key.endswith("_mean") else 1.0
        extension = torch.full(
            (*value.shape[:-1], extra_dims),
            fill,
            dtype=value.dtype,
            device=value.device,
        )
        state[key] = torch.cat((value, extension), dim=-1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Flat-task model_*.pt checkpoint.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path (default: INPUT with '_jump_init' suffix).",
    )
    parser.add_argument("--actor-extra-dims", type=int, default=12)
    parser.add_argument("--critic-extra-dims", type=int, default=12)
    parser.add_argument(
        "--max-action-std",
        type=float,
        default=0.30,
        help="Cap inherited Gaussian exploration std (default: 0.30; <=0 disables).",
    )
    args = parser.parse_args()

    input_path = args.input.expanduser().resolve()
    output_path = (
        args.output.expanduser().resolve()
        if args.output is not None
        else input_path.with_name(f"{input_path.stem}_jump_init{input_path.suffix}")
    )
    if output_path == input_path:
        raise ValueError("Output must differ from input; the original checkpoint is never overwritten.")

    checkpoint = torch.load(input_path, map_location="cpu", weights_only=False)
    previous_conversion = (checkpoint.get("infos") or {}).get("jump_checkpoint_conversion")
    if previous_conversion is not None:
        raise ValueError(
            "Input checkpoint was already expanded for jump observations: "
            f"{previous_conversion}. Use it directly with --load_weights_only."
        )
    for key in ("actor_state_dict", "critic_state_dict"):
        if key not in checkpoint:
            raise KeyError(f"Checkpoint has no {key!r}; expected an RSL-RL 3 checkpoint.")

    actor_before = checkpoint["actor_state_dict"]["mlp.0.weight"].shape[1]
    critic_before = checkpoint["critic_state_dict"]["mlp.0.weight"].shape[1]
    _expand_observation_input(
        checkpoint["actor_state_dict"], args.actor_extra_dims, "actor"
    )
    _expand_observation_input(
        checkpoint["critic_state_dict"], args.critic_extra_dims, "critic"
    )
    std_before = None
    std_after = None
    actor_state = checkpoint["actor_state_dict"]
    if args.max_action_std > 0.0:
        if "distribution.std_param" in actor_state:
            std_before = actor_state["distribution.std_param"].clone()
            actor_state["distribution.std_param"].clamp_(min=1.0e-3, max=args.max_action_std)
            std_after = actor_state["distribution.std_param"]
        elif "distribution.log_std_param" in actor_state:
            std_before = actor_state["distribution.log_std_param"].exp()
            actor_state["distribution.log_std_param"].clamp_(
                max=torch.log(torch.tensor(args.max_action_std)).item()
            )
            std_after = actor_state["distribution.log_std_param"].exp()
    checkpoint["infos"] = {
        **(checkpoint.get("infos") or {}),
        "jump_checkpoint_conversion": {
            "source": str(input_path),
            "actor_observations": [actor_before, actor_before + args.actor_extra_dims],
            "critic_observations": [critic_before, critic_before + args.critic_extra_dims],
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output_path)
    print(f"Saved: {output_path}")
    print(f"Actor observations:  {actor_before} -> {actor_before + args.actor_extra_dims}")
    print(f"Critic observations: {critic_before} -> {critic_before + args.critic_extra_dims}")
    if std_before is not None:
        print(f"Action std before: {std_before.tolist()}")
        print(f"Action std after:  {std_after.tolist()}")
    print("Load this checkpoint with train.py --resume --load_weights_only.")


if __name__ == "__main__":
    main()
