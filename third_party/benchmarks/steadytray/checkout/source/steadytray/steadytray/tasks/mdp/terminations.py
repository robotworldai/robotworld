from __future__ import annotations

import torch
from typing import TYPE_CHECKING
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def link_height_below_minimum(
    env: ManagerBasedRLEnv, 
    minimum_height: float, 
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate when the specified link's height is below the minimum height.

    This function checks the z-position of a specific body/link and terminates the episode
    if it goes below the specified minimum height threshold. No conditional checks applied.

    Args:
        env: The environment instance.
        minimum_height: The minimum allowed height for the link.
        asset_cfg: The asset configuration specifying the asset and body/link to check.
                  Must specify body_names or body_ids to identify the specific link.

    Returns:
        A boolean tensor indicating which environments should be terminated.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    
    # Get the z-position of the specified bodies
    if isinstance(asset, RigidObject):
        # For RigidObject, use the first body
        body_pos_w = asset.data.body_pos_w[:, 0, 2]
    elif isinstance(asset, Articulation):
        # For Articulation, use the specified body_ids
        body_pos_w = asset.data.body_link_pos_w[:, asset_cfg.body_ids[0], 2]

    # Check if any body is below the minimum height
    height_violation = body_pos_w < minimum_height
    
    return height_violation
    
