from __future__ import annotations

from typing import Any, Callable

import numpy as np
import torch


def clone_obs_buf(obs_buf: Any) -> Any:
    if isinstance(obs_buf, dict):
        return {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in obs_buf.items()}
    if isinstance(obs_buf, torch.Tensor):
        return obs_buf.clone()
    return obs_buf


def seed_everything(seed: int | None) -> None:
    if seed is None:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def refresh_manager_observations(manager_env) -> Any:
    if manager_env.sim.has_rtx_sensors():
        manager_env.sim.render()
    manager_env.scene.update(dt=manager_env.physics_dt)
    manager_env.obs_buf = manager_env.observation_manager.compute(update_history=True)
    return clone_obs_buf(manager_env.obs_buf)
