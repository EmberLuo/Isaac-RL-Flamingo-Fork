# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Flat-terrain stand-and-drive task for the wheel-legged ``wl`` robot.

Reward weights are ported from the source ``WheelLeggedCfg.rewards.scales``.
Where the source used a bespoke ``_reward_*`` method, we map to the project's
existing mdp primitives; the only custom term is leg symmetry (``nominal_state``
in the source), implemented as ``mdp.vmc_leg_symmetry_l2``.
"""

import math

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import lab.flamingo.tasks.manager_based.locomotion.velocity.mdp as mdp
from lab.flamingo.tasks.manager_based.locomotion.velocity.wl_env.velocity_env_cfg import (
    LocomotionVelocityFlatEnvCfg,
)
from lab.flamingo.assets.flamingo.wl_rev01_0_0 import WL_CFG  # isort: skip

_LEG_JOINTS = ["lf0_Joint", "lf1_Joint", "rf0_Joint", "rf1_Joint"]
_LEG_LINKS = ["lf0_Link", "lf1_Link", "rf0_Link", "rf1_Link"]


@configclass
class WLRewardsCfg:
    """Reward terms ported from the source WheelLeggedCfg scales."""

    # -- task tracking
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_link_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_link_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )
    base_height = RewTerm(  # source: _reward_base_height, exp(-err/0.001)
        func=mdp.track_pos_z_exp,
        weight=1.0,
        params={"temperature": 1000.0, "asset_cfg": SceneEntityCfg("robot", body_names="base_link")},
    )

    # -- stability penalties
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_link_l2, weight=-2.0)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_link_l2, weight=-0.05)
    flat_orientation = RewTerm(func=mdp.flat_euler_angle_l2, weight=-10.0)
    leg_symmetry = RewTerm(  # source: _reward_nominal_state (-0.1)
        func=mdp.vmc_leg_symmetry_l2, weight=-0.1, params={"action_name": "vmc"}
    )

    # -- effort / smoothness penalties
    dof_vel_l2 = RewTerm(
        func=mdp.joint_vel_l2, weight=-5.0e-5,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=_LEG_JOINTS)},
    )
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-1.0e-4)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)

    # -- contacts / limits
    dof_pos_limits = RewTerm(
        func=mdp.joint_pos_limits, weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=_LEG_JOINTS)},
    )
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts, weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=_LEG_LINKS),
            "threshold": 1.0,
        },
    )

    # -- termination (not in the source scales; modest penalty aids co_rl training)
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)


@configclass
class WLFlatEnvCfg(LocomotionVelocityFlatEnvCfg):
    rewards: WLRewardsCfg = WLRewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = WL_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


@configclass
class WLFlatEnvCfg_PLAY(WLFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 64
        self.scene.env_spacing = 2.5
        self.observations.stack_policy.enable_corruption = False
        self.events.push_robot = None
        # fixed mid-range height command for inspection
        self.commands.base_velocity.ranges.pos_z = (0.25, 0.25)
