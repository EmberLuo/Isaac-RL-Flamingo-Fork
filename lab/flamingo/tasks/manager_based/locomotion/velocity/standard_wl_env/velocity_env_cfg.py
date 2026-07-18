# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Velocity environment shared by standard WL flat and rough tasks."""

from __future__ import annotations

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import lab.flamingo.tasks.manager_based.locomotion.velocity.mdp as mdp
from lab.flamingo.tasks.manager_based.locomotion.velocity.wl_env.velocity_env_cfg import (
    CommandsCfg as WLCommandsCfg,
)
from lab.flamingo.tasks.manager_based.locomotion.velocity.wl_env.velocity_env_cfg import (
    EventCfg as WLEventCfg,
)
from lab.flamingo.tasks.manager_based.locomotion.velocity.wl_env.velocity_env_cfg import (
    LocomotionVelocityRoughEnvCfg as WLLocomotionVelocityRoughEnvCfg,
)


@configclass
class CommandsCfg(WLCommandsCfg):
    """Standard WL velocity and base-height commands."""

    def __post_init__(self):
        self.base_velocity.rel_standing_envs = 0.05
        self.base_velocity.ranges.lin_vel_x = (-3.0, 3.0)
        self.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.base_velocity.ranges.ang_vel_z = (-3.0, 3.0)
        self.base_velocity.ranges.pos_z = (0.29, 0.37)


@configclass
class ActionsCfg:
    """Six-dimensional five-bar VMC policy action."""

    vmc = mdp.StandardWLVMCActionCfg(asset_name="robot")


@configclass
class ObservationsCfg:
    """WL-compatible observations backed by the five-bar VMC state."""

    @configclass
    class StackPolicyCfg(ObsGroup):
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel_link,
            scale=0.25,
            noise=Unoise(n_min=-0.2, n_max=0.2),
        )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        vmc_theta0 = ObsTerm(
            func=mdp.vmc_theta0,
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        vmc_theta0_dot = ObsTerm(
            func=mdp.vmc_theta0_dot,
            scale=0.05,
            noise=Unoise(n_min=-1.5, n_max=1.5),
        )
        vmc_l0 = ObsTerm(
            func=mdp.vmc_l0,
            scale=5.0,
            noise=Unoise(n_min=-0.02, n_max=0.02),
        )
        vmc_l0_dot = ObsTerm(
            func=mdp.vmc_l0_dot,
            scale=0.25,
            noise=Unoise(n_min=-0.1, n_max=0.1),
        )
        wheel_vel = ObsTerm(
            func=mdp.vmc_wheel_vel,
            scale=0.05,
            noise=Unoise(n_min=-1.5, n_max=1.5),
        )
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class NoneStackPolicyCfg(ObsGroup):
        velocity_commands = ObsTerm(
            func=mdp.generated_scaled_commands,
            params={"command_name": "base_velocity", "scale": (1.0 / 3.0, 1.0, 1.0 / 3.0)},
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class StackCriticCfg(ObsGroup):
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel_link, scale=0.25)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        vmc_theta0 = ObsTerm(func=mdp.vmc_theta0)
        vmc_theta0_dot = ObsTerm(func=mdp.vmc_theta0_dot, scale=0.05)
        vmc_l0 = ObsTerm(func=mdp.vmc_l0, scale=5.0)
        vmc_l0_dot = ObsTerm(func=mdp.vmc_l0_dot, scale=0.25)
        wheel_vel = ObsTerm(
            func=mdp.vmc_wheel_vel,
            scale=0.05,
        )
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class NoneStackCriticCfg(ObsGroup):
        velocity_commands = ObsTerm(
            func=mdp.generated_scaled_commands,
            params={"command_name": "base_velocity", "scale": (1.0 / 3.0, 1.0, 1.0 / 3.0)},
        )
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel_link, scale=0.5)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    stack_policy: StackPolicyCfg = StackPolicyCfg()
    none_stack_policy: NoneStackPolicyCfg = NoneStackPolicyCfg()
    stack_critic: StackCriticCfg = StackCriticCfg()
    none_stack_critic: NoneStackCriticCfg = NoneStackCriticCfg()


@configclass
class EventCfg(WLEventCfg):
    """Domain randomization with a closure-consistent joint reset."""

    # All fourteen tree joints must be reset to one mutually consistent state.
    # Independent offsets on passive joints inject large loop-constraint impulses.
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={"position_range": (0.0, 0.0), "velocity_range": (0.0, 0.0)},
    )
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-0.2, 0.2),
                "y": (-0.2, 0.2),
                "z": (-0.2, 0.2),
                "roll": (-0.2, 0.2),
                "pitch": (-0.2, 0.2),
                "yaw": (-0.2, 0.2),
            },
        },
    )


@configclass
class LocomotionVelocityRoughEnvCfg(WLLocomotionVelocityRoughEnvCfg):
    """Rough terrain base configuration for the standard WL."""

    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    events: EventCfg = EventCfg()


@configclass
class LocomotionVelocityFlatEnvCfg(LocomotionVelocityRoughEnvCfg):
    """Flat terrain variant."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.scene.height_scanner = None
        self.curriculum.terrain_levels = None
