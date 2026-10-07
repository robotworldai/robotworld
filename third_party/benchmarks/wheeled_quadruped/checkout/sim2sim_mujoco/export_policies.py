# Copyright (c) 2024, Mickyas Tamiru Asfaw. MIT License.
"""Export the trained actor MLPs to plain numpy .npz files.

Run this once with the Isaac Lab venv (which has torch). The resulting .npz files
carry only the actor's Linear layers, so the MuJoCo sim-to-sim runner can do the
forward pass in pure numpy, with no torch / onnxruntime dependency. The actor uses
no observation normalization (empirical_normalization and actor_obs_normalization
are both off in the runner cfg), so the raw observation goes straight into the MLP.

    <isaac-venv>/python export_policies.py
"""

import os

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)

# (name, checkpoint) for all four trained policies
POLICIES = {
    "balance": "pretrained/balance/model_1399.pt",
    "velocity": "pretrained/velocity/model_2999.pt",
    "balance_rough": "pretrained/balance_rough/model_1499.pt",
    "velocity_rough": "pretrained/velocity_rough/model_2999.pt",
}

out_dir = os.path.join(HERE, "policies")
os.makedirs(out_dir, exist_ok=True)

for name, rel in POLICIES.items():
    ckpt = torch.load(os.path.join(REPO, rel), map_location="cpu", weights_only=False)
    sd = ckpt["model_state_dict"]
    # actor is nn.Sequential(Linear, ELU, Linear, ELU, ...): Linear layers sit at
    # even module indices. Collect them in order.
    idx, arrays = 0, {}
    n = 0
    while f"actor.{idx}.weight" in sd:
        arrays[f"W{n}"] = sd[f"actor.{idx}.weight"].numpy().astype(np.float32)
        arrays[f"b{n}"] = sd[f"actor.{idx}.bias"].numpy().astype(np.float32)
        n += 1
        idx += 2
    obs_dim = arrays["W0"].shape[1]
    act_dim = arrays[f"W{n-1}"].shape[0]
    np.savez(os.path.join(out_dir, f"{name}.npz"), n_layers=np.int64(n), **arrays)
    print(f"{name:16s}: {n} linear layers, obs_dim={obs_dim}, act_dim={act_dim}  -> policies/{name}.npz")

print("done.")
