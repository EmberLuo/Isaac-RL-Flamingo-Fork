# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Flat stand-and-drive task for the closed-chain standard WL."""

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import lab.flamingo.tasks.manager_based.locomotion.velocity.mdp as mdp
from lab.flamingo.assets.flamingo.standard_wl_rev01_0_0 import STANDARD_WL_CFG
from lab.flamingo.tasks.manager_based.locomotion.velocity.standard_wl_env.velocity_env_cfg import (
    LocomotionVelocityFlatEnvCfg,
)


_ACTIVE_LEG_JOINTS = ["jAB", "jAG", "jIJ", "jIO"]
_LEG_LINKS = ["AG", "GH", "AB", "BE", "EC", "CF", "IJ", "JM", "MK", "KN", "IO", "OP"]
_SOURCE_TOTAL_MASS_KG = 14.546
_PLAY_TOTAL_MASS_KG = 18.0
_TRAIN_TOTAL_MASS_RANGE_KG = (14.0, 22.0)


@configclass
class StandardWLRewardsCfg:
    """WL stand-and-drive rewards using canonical five-bar virtual states."""

    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_link_exp,
        weight=2.0,
        params={"command_name": "base_velocity", "std": 1.5},
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_link_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": 1.5},
    )
    base_height = RewTerm(
        func=mdp.track_pos_z_exp,
        weight=1.0,
        params={
            "temperature": 1000.0,
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
        },
    )

    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_link_l2, weight=-2.0)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_link_l2, weight=-0.05)
    flat_orientation = RewTerm(func=mdp.flat_euler_angle_l2, weight=-5.0)
    leg_symmetry = RewTerm(
        func=mdp.vmc_leg_symmetry_l2,
        weight=-0.1,
        params={"action_name": "vmc"},
    )

    dof_vel_l2 = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-5.0e-5,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=_ACTIVE_LEG_JOINTS)},
    )
    dof_acc_l2 = RewTerm(
        func=mdp.joint_acc_l2,
        weight=-2.5e-7,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=_ACTIVE_LEG_JOINTS)},
    )
    dof_torques_l2 = RewTerm(
        func=mdp.joint_torques_l2,
        weight=-1.0e-4,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=_ACTIVE_LEG_JOINTS)},
    )
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)

    dof_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=_ACTIVE_LEG_JOINTS)},
    )
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=_LEG_LINKS),
            "threshold": 1.0,
        },
    )
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)


@configclass
class StandardWLFlatEnvCfg(LocomotionVelocityFlatEnvCfg):
    rewards: StandardWLRewardsCfg = StandardWLRewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = STANDARD_WL_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.events.add_base_mass.params["mass_distribution_params"] = tuple(
            mass - _SOURCE_TOTAL_MASS_KG for mass in _TRAIN_TOTAL_MASS_RANGE_KG
        )
        self.events.physics_material.params["static_friction_range"] = (0.6, 1.4)
        self.events.physics_material.params["dynamic_friction_range"] = (0.5, 1.2)
        self.events.physics_material.params["restitution_range"] = (0.0, 0.1)
        self.events.push_robot.interval_range_s = (10.0, 15.0)
        self.events.push_robot.params["velocity_range"] = {"x": (-1.0, 1.0), "y": (-1.0, 1.0)}


@configclass
class StandardWLFlatEnvCfg_PLAY(StandardWLFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.scene.env_spacing = 2.5
        self.observations.stack_policy.enable_corruption = False
        self.observations.none_stack_policy.enable_corruption = False
        self.events.push_robot = None
        nominal_mass_addition = _PLAY_TOTAL_MASS_KG - _SOURCE_TOTAL_MASS_KG
        self.events.add_base_mass.params["mass_distribution_params"] = (
            nominal_mass_addition,
            nominal_mass_addition,
        )
        self.events.randomize_com_positions = None
        self.events.physics_material = None
        self.commands.base_velocity.ranges.pos_z = (0.31, 0.31)
