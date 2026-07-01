# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the Wheel-Legged ("wl") robot.

Migrated from the Isaac Gym project ``clearlab-sustech/Wheel-Legged-Gym``.
The robot is a two-wheel-legged (balance-infantry style) platform with a
2-link serial leg + wheel on each side. It is driven by a custom Virtual
Model Control (VMC) action term, so every joint is pure effort-controlled:
the joint drives carry zero stiffness/damping and the VMC term supplies all
torques via ``set_joint_effort_target``.

Joint order as imported by Isaac Sim (depth-first pairs):
    [lf0_Joint, rf0_Joint, lf1_Joint, rf1_Joint, l_wheel_Joint, r_wheel_Joint]
Always resolve joints by name (never by hard-coded index) because Isaac Sim
reorders them relative to the URDF declaration order assumed by the source.
"""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from lab.flamingo.assets.flamingo import FLAMINGO_ASSETS_DATA_DIR

# Link-length parameters of the planar 2-link leg (used by the VMC kinematics).
# These mirror ``asset.offset / l1 / l2`` in the source WheelLeggedCfg.
WL_LEG_OFFSET = 0.054
WL_LEG_L1 = 0.15
WL_LEG_L2 = 0.25

WL_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{FLAMINGO_ASSETS_DATA_DIR}/Robots/WheelLegged/wl_rev_01_0_0/wl.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.25),
        joint_pos={
            "lf0_Joint": 0.5,
            "lf1_Joint": 0.35,
            "l_wheel_Joint": 0.0,
            "rf0_Joint": -0.5,
            "rf1_Joint": -0.35,
            "r_wheel_Joint": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.95,
    actuators={
        # Leg joints (f0 = hip, f1 = knee): zero gains, the VMC term drives them.
        "legs": ImplicitActuatorCfg(
            joint_names_expr=["lf0_Joint", "lf1_Joint", "rf0_Joint", "rf1_Joint"],
            effort_limit=30.0,  # URDF effort limit for f0/f1 joints
            velocity_limit=30.0,
            stiffness=0.0,
            damping=0.0,
            friction=0.0,
            armature=0.0,
        ),
        # Wheel joints: zero gains, the VMC term applies a velocity-tracking torque.
        "wheels": ImplicitActuatorCfg(
            joint_names_expr=["l_wheel_Joint", "r_wheel_Joint"],
            effort_limit=5.0,  # URDF effort limit for wheels
            velocity_limit=1000.0,
            stiffness=0.0,
            damping=0.0,
            friction=0.0,
            armature=0.0,
        ),
    },
)
"""Configuration for the wheel-legged ``wl`` robot (VMC effort control)."""
