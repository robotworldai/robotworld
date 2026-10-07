# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Articulation configuration for the wheeled quadruped robot.

The robot has four actuated joints only: the two rear wheels (continuous) and
the two front thighs (revolute, +/-0.785 rad). The front wheels are fixed. The
robot stands pitched up, balancing on its two rear wheels, with the base at a
height of 0.828 m.
"""

import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

# Resolve the USD relative to this file so the package is location-independent.
_USD_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "quadruped_robot.usd")

WHEELED_QUADRUPED_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=_USD_PATH,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=100.0,
            enable_gyroscopic_forces=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            # Raised 4 -> 8 relative to the legacy cfg for rolling-contact stability.
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.828),
        joint_pos={
            "robot1_rl_wheel_joint": 0.0,
            "robot1_rr_wheel_joint": 0.0,
            "robot1_front_left_thigh_joint": 0.0,
            "robot1_front_right_thigh_joint": 0.0,
        },
    ),
    actuators={
        # effort/velocity_limit_sim are the non-deprecated v2.3.2 fields; the
        # plain effort_limit/velocity_limit are ignored by ImplicitActuator.
        # Wheel effort matches the URDF limit (100), not the legacy cfg's 400.
        "rl_wheel_actuator": ImplicitActuatorCfg(
            joint_names_expr=["robot1_rl_wheel_joint"],
            effort_limit_sim=100.0,
            velocity_limit_sim=100.0,
            stiffness=0.0,
            damping=10.0,
        ),
        "rr_wheel_actuator": ImplicitActuatorCfg(
            joint_names_expr=["robot1_rr_wheel_joint"],
            effort_limit_sim=100.0,
            velocity_limit_sim=100.0,
            stiffness=0.0,
            damping=10.0,
        ),
        # Thigh damping raised from the legacy 0.005 (undamped ringing on a
        # ~2.5 kg assembly at stiffness 1000) into the critically-damped range.
        "fl_thigh_actuator": ImplicitActuatorCfg(
            joint_names_expr=["robot1_front_left_thigh_joint"],
            effort_limit_sim=400.0,
            velocity_limit_sim=100.0,
            stiffness=1000.0,
            damping=20.0,
        ),
        "fr_thigh_actuator": ImplicitActuatorCfg(
            joint_names_expr=["robot1_front_right_thigh_joint"],
            effort_limit_sim=400.0,
            velocity_limit_sim=100.0,
            stiffness=1000.0,
            damping=20.0,
        ),
    },
)
"""Configuration for the wheeled quadruped robot articulation."""
