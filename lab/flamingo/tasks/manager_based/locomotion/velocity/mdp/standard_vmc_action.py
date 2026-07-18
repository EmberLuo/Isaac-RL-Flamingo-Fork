# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Five-bar virtual model control for the closed-chain ``standard_wl`` robot."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

import omni.log
import torch

from isaaclab.assets.articulation import Articulation
from isaaclab.managers.action_manager import ActionTerm
from isaaclab.managers.manager_term_cfg import ActionTermCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class StandardWLVMCAction(ActionTerm):
    """Map virtual leg targets to the four motors of the five-bar mechanism.

    The policy interface intentionally matches :class:`VMCAction`:
    ``[theta_left, length_left, wheel_left, theta_right, length_right,
    wheel_right]``.  Only the physical kinematics and torque mapping differ.

    The equations are the vectorized form of ``wheel_leg_mujoco/VMC.py``.  The
    mirrored joint frames are converted to one canonical convention, so equal
    left/right poses produce equal virtual states and policy actions.
    """

    cfg: StandardWLVMCActionCfg
    _asset: Articulation

    def __init__(self, cfg: StandardWLVMCActionCfg, env: ManagerBasedEnv) -> None:
        super().__init__(cfg, env)

        def _joint_id(name: str) -> int:
            ids, _ = self._asset.find_joints(name)
            if len(ids) != 1:
                raise ValueError(
                    f"StandardWLVMCAction: joint '{name}' resolved to {ids}, expected exactly one."
                )
            return ids[0]

        self._left_front = _joint_id(cfg.left_front_joint)
        self._left_rear = _joint_id(cfg.left_rear_joint)
        self._left_wheel = _joint_id(cfg.left_wheel_joint)
        self._right_front = _joint_id(cfg.right_front_joint)
        self._right_rear = _joint_id(cfg.right_rear_joint)
        self._right_wheel = _joint_id(cfg.right_wheel_joint)

        # Effort columns are kept in policy-side order.  Left: phi4, phi1,
        # wheel. Right: phi1, phi4, wheel (matching the source actuator map).
        self._effort_joint_ids = [
            self._left_front,
            self._left_rear,
            self._left_wheel,
            self._right_front,
            self._right_rear,
            self._right_wheel,
        ]
        omni.log.info(
            "StandardWLVMCAction effort joint ids "
            f"(jIJ,jIO,lw,jAB,jAG,rw) = {self._effort_joint_ids}"
        )

        self._raw_actions = torch.zeros(self.num_envs, 6, device=self.device)
        self._processed_actions = torch.zeros_like(self._raw_actions)
        self._theta0_ref = torch.zeros(self.num_envs, 2, device=self.device)
        self._l0_ref = torch.zeros(self.num_envs, 2, device=self.device)
        self._wheel_vel_ref = torch.zeros(self.num_envs, 2, device=self.device)

        self.L0 = torch.zeros(self.num_envs, 2, device=self.device)
        self.theta0 = torch.zeros(self.num_envs, 2, device=self.device)
        self.L0_dot = torch.zeros(self.num_envs, 2, device=self.device)
        self.theta0_dot = torch.zeros(self.num_envs, 2, device=self.device)
        self.wheel_vel = torch.zeros(self.num_envs, 2, device=self.device)

        # theta = sign * (phi0 - pi/2) - nominal_theta.  This mirrors the
        # right leg so equal physical postures produce equal observations.
        self._theta_sign = torch.tensor([[-1.0, 1.0]], device=self.device)

    @property
    def action_dim(self) -> int:
        return 6

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    @staticmethod
    def _signed_clamp(value: torch.Tensor, minimum_abs: float) -> torch.Tensor:
        return torch.where(
            value >= 0.0,
            torch.clamp(value, min=minimum_abs),
            torch.clamp(value, max=-minimum_abs),
        )

    def _five_bar_kinematics(self, phi1: torch.Tensor, phi4: torch.Tensor):
        """Return virtual state and Jacobian coefficients for both legs."""
        l1, l2, l3, l4, l5 = self.cfg.l1, self.cfg.l2, self.cfg.l3, self.cfg.l4, self.cfg.l5

        xb = l1 * torch.cos(phi1)
        yb = l1 * torch.sin(phi1)
        xd = l5 + l4 * torch.cos(phi4)
        yd = l4 * torch.sin(phi4)
        dx_bd = xd - xb
        dy_bd = yd - yb
        l_bd_sq = dx_bd.square() + dy_bd.square()

        a0 = 2.0 * l2 * dx_bd
        b0 = 2.0 * l2 * dy_bd
        c0 = l2 * l2 + l_bd_sq - l3 * l3
        discriminant = torch.clamp(a0.square() + b0.square() - c0.square(), min=0.0)
        phi2 = 2.0 * torch.atan2(b0 + torch.sqrt(discriminant), a0 + c0)
        phi3 = torch.atan2(
            yb - yd + l2 * torch.sin(phi2),
            xb - xd + l2 * torch.cos(phi2),
        )

        xc = xb + l2 * torch.cos(phi2)
        yc = yb + l2 * torch.sin(phi2)
        xc_centered = xc - 0.5 * l5
        l0 = torch.sqrt(torch.clamp(xc_centered.square() + yc.square(), min=self.cfg.kinematic_eps**2))
        phi0 = torch.atan2(yc, xc_centered)

        sin_32 = self._signed_clamp(torch.sin(phi3 - phi2), self.cfg.kinematic_eps)
        j11 = l1 * torch.sin(phi0 - phi3) * torch.sin(phi1 - phi2) / sin_32
        j12 = l1 * torch.cos(phi0 - phi3) * torch.sin(phi1 - phi2) / (l0 * sin_32)
        j21 = l4 * torch.sin(phi0 - phi2) * torch.sin(phi3 - phi4) / sin_32
        j22 = l4 * torch.cos(phi0 - phi2) * torch.sin(phi3 - phi4) / (l0 * sin_32)
        return l0, phi0, j11, j12, j21, j22

    def process_actions(self, actions: torch.Tensor) -> None:
        # Keep train/play and all action-based reward terms on the same bounded
        # policy action, matching the existing WL implementation.
        actions = torch.clamp(actions, -1.0, 1.0)
        self._env.action_manager._action[:] = torch.clamp(self._env.action_manager._action, -1.0, 1.0)
        self._raw_actions[:] = actions

        self._theta0_ref = torch.stack((actions[:, 0], actions[:, 3]), dim=1) * self.cfg.action_scale_theta
        self._l0_ref = (
            torch.stack((actions[:, 1], actions[:, 4]), dim=1) * self.cfg.action_scale_l0
            + self.cfg.l0_offset
        )
        self._wheel_vel_ref = (
            torch.stack((actions[:, 2], actions[:, 5]), dim=1) * self.cfg.action_scale_vel
        )
        self._processed_actions[:] = torch.cat(
            (
                self._theta0_ref[:, :1],
                self._l0_ref[:, :1],
                self._wheel_vel_ref[:, :1],
                self._theta0_ref[:, 1:],
                self._l0_ref[:, 1:],
                self._wheel_vel_ref[:, 1:],
            ),
            dim=1,
        )

    def _update_virtual_state(self):
        """Update canonical leg and wheel states from the articulation state."""
        joint_pos = self._asset.data.joint_pos
        joint_vel = self._asset.data.joint_vel

        # Convert the mirrored USD joint frames to the five-bar coordinates.
        # Hardware encoder calibration offsets from the source controller are
        # intentionally omitted for this ideal, symmetric simulation model.
        # right: phi1 = jAB + pi,       phi4 = jAG + 1.3
        # left:  phi1 = jIO + pi - 1.3, phi4 = jIJ
        phi1 = torch.stack(
            (
                joint_pos[:, self._left_rear] + self.cfg.left_phi1_offset,
                joint_pos[:, self._right_front] + self.cfg.right_phi1_offset,
            ),
            dim=1,
        )
        phi4 = torch.stack(
            (
                joint_pos[:, self._left_front] + self.cfg.left_phi4_offset,
                joint_pos[:, self._right_rear] + self.cfg.right_phi4_offset,
            ),
            dim=1,
        )
        phi1_dot = torch.stack(
            (joint_vel[:, self._left_rear], joint_vel[:, self._right_front]), dim=1
        )
        phi4_dot = torch.stack(
            (joint_vel[:, self._left_front], joint_vel[:, self._right_rear]), dim=1
        )

        self.L0, phi0, j11, j12, j21, j22 = self._five_bar_kinematics(phi1, phi4)
        self.theta0 = self._theta_sign * (phi0 - 0.5 * math.pi) - self.cfg.nominal_theta
        self.L0_dot = j11 * phi1_dot + j21 * phi4_dot
        phi0_dot = j12 * phi1_dot + j22 * phi4_dot
        self.theta0_dot = self._theta_sign * phi0_dot
        # The wheel joint axes are mirrored in the source MJCF/USD.  Expose a
        # canonical velocity where positive means forward on both sides.
        self.wheel_vel = torch.stack(
            (-joint_vel[:, self._left_wheel], joint_vel[:, self._right_wheel]), dim=1
        )
        return j11, j12, j21, j22

    def apply_actions(self) -> None:
        j11, j12, j21, j22 = self._update_virtual_state()

        virtual_torque = (
            self.cfg.kp_theta * (self._theta0_ref - self.theta0)
            - self.cfg.kd_theta * self.theta0_dot
        )
        virtual_force = (
            self.cfg.kp_l0 * (self._l0_ref - self.L0)
            - self.cfg.kd_l0 * self.L0_dot
            + self.cfg.feedforward_force
        )

        # VMC.py's angular Jacobian is d(phi0)/dq. Convert our canonical
        # mirrored angle torque back to its phi0 convention before mapping.
        source_tp = self._theta_sign * virtual_torque
        tau_phi1 = j11 * virtual_force + j12 * source_tp
        tau_phi4 = j21 * virtual_force + j22 * source_tp

        # Positive forward wheel velocity is -jwheel_left and +jwheel_right.
        wheel_torque = self.cfg.wheel_damping * (self._wheel_vel_ref - self.wheel_vel)

        efforts = torch.stack(
            (
                tau_phi4[:, 0],
                tau_phi1[:, 0],
                -wheel_torque[:, 0],
                tau_phi1[:, 1],
                tau_phi4[:, 1],
                wheel_torque[:, 1],
            ),
            dim=1,
        )
        limits = torch.tensor(
            [
                self.cfg.leg_effort_limit,
                self.cfg.leg_effort_limit,
                self.cfg.wheel_effort_limit,
                self.cfg.leg_effort_limit,
                self.cfg.leg_effort_limit,
                self.cfg.wheel_effort_limit,
            ],
            device=self.device,
        )
        efforts = torch.nan_to_num(efforts).clamp(-limits, limits)
        self._asset.set_joint_effort_target(efforts, joint_ids=self._effort_joint_ids)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._raw_actions[env_ids] = 0.0
        self._processed_actions[env_ids] = 0.0
        self._theta0_ref[env_ids] = 0.0
        self._l0_ref[env_ids] = self.cfg.l0_offset
        self._wheel_vel_ref[env_ids] = 0.0
        # Reset events have already written the new joint state at this point.
        # Recompute now so the first observation of the next episode does not
        # contain the terminal episode's virtual-leg state.
        self._update_virtual_state()


@configclass
class StandardWLVMCActionCfg(ActionTermCfg):
    """Configuration for the closed-chain five-bar VMC action term."""

    class_type: type[ActionTerm] = StandardWLVMCAction
    asset_name: str = "robot"

    left_front_joint: str = "jIJ"
    left_rear_joint: str = "jIO"
    left_wheel_joint: str = "jwheel_left"
    right_front_joint: str = "jAB"
    right_rear_joint: str = "jAG"
    right_wheel_joint: str = "jwheel_right"

    # Five-bar dimensions from wheel_leg_mujoco/VMC.py.
    l1: float = 0.215
    l2: float = 0.258
    l3: float = 0.258
    l4: float = 0.215
    l5: float = 0.0
    kinematic_eps: float = 1.0e-6

    # Structural joint-frame transforms; these are not hardware calibration.
    left_phi1_offset: float = math.pi - 1.3
    left_phi4_offset: float = 0.0
    right_phi1_offset: float = math.pi
    right_phi4_offset: float = 1.3
    # Zero policy angle when the virtual leg is vertical (phi0 = pi/2).
    nominal_theta: float = 0.0

    action_scale_theta: float = 0.2
    action_scale_l0: float = 0.05
    # With the 0.077 m wheel radius this gives 3.85 m/s of wheel-speed
    # authority, leaving tracking margin for the 3.0 m/s command range.
    action_scale_vel: float = 50.0
    l0_offset: float = 0.36

    kp_theta: float = 50.0
    kd_theta: float = 3.0
    kp_l0: float = 900.0
    kd_l0: float = 20.0
    wheel_damping: float = 0.5
    # Half of the nominal 18 kg vehicle weight.
    feedforward_force: float = 88.3

    leg_effort_limit: float = 20.0
    wheel_effort_limit: float = 4.0
