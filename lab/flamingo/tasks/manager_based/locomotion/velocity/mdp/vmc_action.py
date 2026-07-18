# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Virtual Model Control (VMC) action term for the wheel-legged ``wl`` robot.

Migrated from ``clearlab-sustech/Wheel-Legged-Gym`` (Isaac Gym). The policy
outputs, per side, a virtual-leg target ``[theta0_ref, l0_ref, wheel_vel_ref]``
(6 dims total). Every physics step this term:

1. reads fresh joint positions/velocities,
2. computes the virtual-leg state (length ``L0``, angle ``theta0``) of the
   2-link planar leg via forward kinematics,
3. runs virtual PD controllers on ``(theta0, L0)`` and a velocity controller on
   the wheel,
4. maps the virtual force/torque to physical hip/knee torques through the leg
   Jacobian transpose (the "VMC" step), and
5. writes the resulting joint efforts to the simulation.

All six joints are effort-controlled, so the robot's actuators must use zero
stiffness/damping for this term to have full torque authority.

Intermediate quantities ``L0``, ``theta0``, ``L0_dot``, ``theta0_dot`` are
stored as attributes so observation terms can read them via
``env.action_manager.get_term(<name>)``. Joints are always resolved by name
(never by index) because Isaac Sim reorders DOFs relative to the URDF order
assumed by the original Isaac Gym implementation.
"""

from __future__ import annotations

import math
import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

import omni.log

from isaaclab.assets.articulation import Articulation
from isaaclab.managers.action_manager import ActionTerm
from isaaclab.managers.manager_term_cfg import ActionTermCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

class VMCAction(ActionTerm):
    """Virtual Model Control action term (see module docstring)."""

    cfg: VMCActionCfg
    _asset: Articulation

    def __init__(self, cfg: VMCActionCfg, env: ManagerBasedEnv) -> None:
        super().__init__(cfg, env)

        # resolve joint indices BY NAME (Isaac Sim reorders DOFs vs the URDF order)
        def _jid(name: str) -> int:
            ids, _ = self._asset.find_joints(name)
            if len(ids) != 1:
                raise ValueError(f"VMCAction: joint '{name}' resolved to {ids}, expected exactly one.")
            return ids[0]

        self._lf0 = _jid(cfg.left_hip_joint)
        self._lf1 = _jid(cfg.left_knee_joint)
        self._rf0 = _jid(cfg.right_hip_joint)
        self._rf1 = _jid(cfg.right_knee_joint)
        self._l_wheel = _jid(cfg.left_wheel_joint)
        self._r_wheel = _jid(cfg.right_wheel_joint)
        # column order of the effort tensor written to sim
        self._effort_joint_ids = [self._lf0, self._lf1, self._l_wheel, self._rf0, self._rf1, self._r_wheel]
        omni.log.info(f"VMCAction effort joint ids (lf0,lf1,lw,rf0,rf1,rw) = {self._effort_joint_ids}")

        # geometry / gains
        self._offset, self._l1, self._l2 = cfg.offset, cfg.l1, cfg.l2
        self._pi = math.pi

        # raw + processed action buffers (layout: [th0_L, l0_L, wv_L, th0_R, l0_R, wv_R])
        self._raw_actions = torch.zeros(self.num_envs, 6, device=self.device)
        self._processed_actions = torch.zeros_like(self._raw_actions)
        # per-step reference targets (shape [N, 2]; column 0 = left, 1 = right)
        self._theta0_ref = torch.zeros(self.num_envs, 2, device=self.device)
        self._l0_ref = torch.zeros(self.num_envs, 2, device=self.device)
        self._wheel_vel_ref = torch.zeros(self.num_envs, 2, device=self.device)
        # virtual-leg state, exposed to observation terms
        self.L0 = torch.zeros(self.num_envs, 2, device=self.device)
        self.theta0 = torch.zeros(self.num_envs, 2, device=self.device)
        self.L0_dot = torch.zeros(self.num_envs, 2, device=self.device)
        self.theta0_dot = torch.zeros(self.num_envs, 2, device=self.device)

    @property
    def action_dim(self) -> int:
        return 6

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    def _forward_kinematics(self, theta1: torch.Tensor, theta2: torch.Tensor):
        """Planar 2-link FK → virtual-leg length L0 and angle theta0. Shapes [N, 2]."""
        end_x = self._offset + self._l1 * torch.cos(theta1) + self._l2 * torch.cos(theta1 + theta2)
        end_y = self._l1 * torch.sin(theta1) + self._l2 * torch.sin(theta1 + theta2)
        L0 = torch.sqrt(end_x**2 + end_y**2)
        theta0 = torch.arctan2(end_y, end_x) - self._pi / 2
        return L0, theta0

    def _vmc(self, theta1, theta2, L0, theta0, F, T):
        """Map virtual (force F along leg, torque T about hip) to joint torques (T1 hip, T2 knee)."""
        theta0v = theta0 + self._pi / 2
        t11 = self._l1 * torch.sin(theta0v - theta1) - self._l2 * torch.sin(theta1 + theta2 - theta0v)
        # t12 = (self._l1 * torch.cos(theta0v - theta1) - self._l2 * torch.cos(theta1 + theta2 - theta0v)) / L0
        t12 = (-self._l1 * torch.cos(theta0v - theta1) - self._l2 * torch.cos(theta1 + theta2 - theta0v)) / L0
        # t12 = -d(theta0)/d(theta1): the l1 term is NEGATIVE (cos is even, so the l2 term keeps its sign)
        t21 = -self._l2 * torch.sin(theta1 + theta2 - theta0v)
        t22 = (-self._l2 * torch.cos(theta1 + theta2 - theta0v)) / L0
        T1 = t11 * F - t12 * T
        T2 = t21 * F - t22 * T
        return T1, T2

    def process_actions(self, actions: torch.Tensor):
        # called once per env step: cache the (constant-within-step) virtual-leg references
        #
        # Clamp the raw policy output to [-1, 1] BEFORE using it. This mirrors
        # play.py (which clamps the policy action), so train and play behave
        # identically. Critically, it also bounds what the stock reward/obs
        # terms see: ``action_rate_l2`` and ``last_action`` read
        # ``env.action_manager.action`` (the RAW, unclamped buffer), not the
        # VMC-processed efforts. Without this clamp an occasional large raw
        # action makes action_rate = sum((a - a_prev)^2) blow up to ~1e23,
        # driving the value loss to inf, the gradients to NaN, and crashing PPO
        # with ``Normal(loc=nan)``. The VMC torque clamp does NOT protect these
        # terms because they never see the processed efforts.
        #
        # WL's VMC term is the ONLY action term (total action dim == 6), so the
        # manager's action buffer IS these 6 dims; clamping it in place is safe
        # and does not affect any other task.
        actions = torch.clamp(actions, -1.0, 1.0)
        self._env.action_manager._action[:] = torch.clamp(self._env.action_manager._action, -1.0, 1.0)
        self._raw_actions[:] = actions
        a = actions
        self._theta0_ref = torch.stack((a[:, 0], a[:, 3]), dim=1) * self.cfg.action_scale_theta
        self._l0_ref = torch.stack((a[:, 1], a[:, 4]), dim=1) * self.cfg.action_scale_l0 + self.cfg.l0_offset
        self._wheel_vel_ref = torch.stack((a[:, 2], a[:, 5]), dim=1) * self.cfg.action_scale_vel
        self._processed_actions[:] = torch.cat(
            (self._theta0_ref[:, :1], self._l0_ref[:, :1], self._wheel_vel_ref[:, :1],
             self._theta0_ref[:, 1:], self._l0_ref[:, 1:], self._wheel_vel_ref[:, 1:]), dim=1
        )

    def apply_actions(self):
        # called every physics substep with fresh joint state
        jp = self._asset.data.joint_pos
        jv = self._asset.data.joint_vel
        # virtual-leg angles (left as-is; right mirrored), matching the source convention
        theta1 = torch.stack((jp[:, self._lf0], -jp[:, self._rf0]), dim=1)
        theta2 = torch.stack((jp[:, self._lf1] + self._pi / 2, -jp[:, self._rf1] + self._pi / 2), dim=1)
        theta1_dot = torch.stack((jv[:, self._lf0], -jv[:, self._rf0]), dim=1)
        theta2_dot = torch.stack((jv[:, self._lf1], -jv[:, self._rf1]), dim=1)

        self.L0, self.theta0 = self._forward_kinematics(theta1, theta2)
        dt = 1.0e-3
        L0_t, theta0_t = self._forward_kinematics(theta1 + theta1_dot * dt, theta2 + theta2_dot * dt)
        self.L0_dot = (L0_t - self.L0) / dt
        self.theta0_dot = (theta0_t - self.theta0) / dt

        # virtual PD controllers
        torque_leg = self.cfg.kp_theta * (self._theta0_ref - self.theta0) - self.cfg.kd_theta * self.theta0_dot
        force_leg = self.cfg.kp_l0 * (self._l0_ref - self.L0) - self.cfg.kd_l0 * self.L0_dot
        wheel_vel = torch.stack((jv[:, self._l_wheel], jv[:, self._r_wheel]), dim=1)
        torque_wheel = self.cfg.wheel_damping * (self._wheel_vel_ref - wheel_vel)

        T1, T2 = self._vmc(theta1, theta2, self.L0, self.theta0, force_leg + self.cfg.feedforward_force, torque_leg)

        # assemble efforts in the column order of self._effort_joint_ids
        efforts = torch.stack(
            (T1[:, 0], T2[:, 0], torque_wheel[:, 0], -T1[:, 1], -T2[:, 1], torque_wheel[:, 1]), dim=1
        )
        leg_lim, wheel_lim = self.cfg.leg_effort_limit, self.cfg.wheel_effort_limit
        lim = torch.tensor([leg_lim, leg_lim, wheel_lim, leg_lim, leg_lim, wheel_lim], device=self.device)
        efforts = torch.clamp(efforts, -lim, lim)
        self._asset.set_joint_effort_target(efforts, joint_ids=self._effort_joint_ids)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._raw_actions[env_ids] = 0.0
        self._processed_actions[env_ids] = 0.0

@configclass
class VMCActionCfg(ActionTermCfg):
    """Configuration for the :class:`VMCAction` term.

    Defaults mirror the source ``WheelLeggedVMCCfg`` control block.
    """

    class_type: type[ActionTerm] = VMCAction
    asset_name: str = "robot"

    # joint names (resolved by name at init)
    left_hip_joint: str = "lf0_Joint"
    left_knee_joint: str = "lf1_Joint"
    left_wheel_joint: str = "l_wheel_Joint"
    right_hip_joint: str = "rf0_Joint"
    right_knee_joint: str = "rf1_Joint"
    right_wheel_joint: str = "r_wheel_Joint"

    # leg geometry (planar 2-link)
    offset: float = 0.054
    l1: float = 0.15
    l2: float = 0.25

    # action scaling
    action_scale_theta: float = 0.15
    action_scale_l0: float = 0.05
    action_scale_vel: float = 12.0
    l0_offset: float = 0.24

    # virtual PD gains
    kp_theta: float = 50.0
    kd_theta: float = 3.0
    kp_l0: float = 900.0
    kd_l0: float = 80.0
    wheel_damping: float = 0.5
    feedforward_force: float = 40.0

    # torque clamps (URDF effort limits)
    leg_effort_limit: float = 30.0
    wheel_effort_limit: float = 5.0
