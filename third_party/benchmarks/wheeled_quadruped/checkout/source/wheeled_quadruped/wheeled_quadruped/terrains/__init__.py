# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Wheel-traversable rough terrain for the wheeled quadruped.

Isaac Lab's stock ``ROUGH_TERRAINS_CFG`` mixes in pyramid stairs and box grids
with 5-23 cm steps. A two-wheel segway-style balancer cannot climb stairs or
step onto boxes, so those sub-terrains are excluded here. What remains is what
"uneven ground" actually means for a wheeled robot: low-amplitude random
roughness and gentle slopes it can roll over while staying balanced.

The terrain generator lays out a ``num_rows x num_cols`` grid of tiles whose
difficulty increases with the row index. With ``curriculum=True`` (enabled by
the velocity-rough task) robots are promoted to harder rows as they learn; with
``curriculum=False`` (the balance-rough task) robots are spread across all
difficulties at random.
"""

import isaaclab.terrains as terrain_gen
from isaaclab.terrains import TerrainGeneratorCfg

WHEELED_ROUGH_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        # Dominant case: small random height noise. At the easiest difficulty the
        # amplitude is ~1 cm (near-flat); at the hardest ~6 cm of rolling bumps.
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.6, noise_range=(0.01, 0.06), noise_step=0.01, border_width=0.25
        ),
        # Gentle up/down slopes (max ~0.2 rad ~ 11 deg) the wheels can climb.
        "gentle_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.2, slope_range=(0.0, 0.2), platform_width=2.0, border_width=0.25
        ),
        "gentle_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.2, slope_range=(0.0, 0.2), platform_width=2.0, border_width=0.25
        ),
    },
)
"""Rough terrain tuned for a wheeled balancer: random roughness + gentle slopes, no stairs."""
