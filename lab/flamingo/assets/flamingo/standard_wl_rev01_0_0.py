# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the closed-chain standard wheel-legged robot."""

from __future__ import annotations

import math
from collections.abc import Callable

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from lab.flamingo.assets.flamingo import FLAMINGO_ASSETS_DATA_DIR


Vec2 = tuple[float, float]
Residual = Callable[[list[float], float, float], list[float]]

_MOTOR_LINK_LENGTH = 0.215
_PASSIVE_LINK_LENGTH = 0.258
_MOTOR_LIMITS = {
    "jAG": (-1.57, 0.30),
    "jIO": (-0.30, 1.57),
    "jAB": (-1.57, 0.50),
    "jIJ": (-0.50, 1.57),
}

# Joint anchors and closure-pin anchors in the source CAD's zero pose, projected
# into its sagittal X-Z plane. Each closure pin is the midpoint used by the USD
# converter, so q=0 is exactly stress-free in PhysX.
_P: dict[str, Vec2] = {
    "AG": (-0.013750000000, 0.334000000000),
    "GH": (-0.073902000000, 0.127589853631),
    "AB": (-0.013750000000, 0.334000000000),
    "BE": (0.084116000000, 0.339123855154),
    "EC": (0.075039900000, 0.221873815275),
    "CF": (-0.101540100000, 0.266571867883),
    "IJ": (-0.013750000000, 0.334000000000),
    "JM": (0.084116000000, 0.339123744846),
    "MK": (0.075039900000, 0.221873784727),
    "KN": (-0.101540100000, 0.266571732118),
    "IO": (-0.013750000000, 0.334000000000),
    "OP": (-0.073902000000, 0.127590146373),
    "right_D": (-0.040842050000, 0.239627928851),
    "right_F": (-0.134521050000, 0.154530924397),
    "left_L": (-0.040842050000, 0.239627871151),
    "left_F": (-0.134521050000, 0.154530875607),
}


def _add(a: Vec2, b: Vec2) -> Vec2:
    return a[0] + b[0], a[1] + b[1]


def _sub(a: Vec2, b: Vec2) -> Vec2:
    return a[0] - b[0], a[1] - b[1]


def _rotate(vector: Vec2, angle: float) -> Vec2:
    cos_angle = math.cos(angle)
    sin_angle = math.sin(angle)
    return (
        cos_angle * vector[0] - sin_angle * vector[1],
        sin_angle * vector[0] + cos_angle * vector[1],
    )


def _point_from(origin: Vec2, zero_origin: Vec2, zero_point: Vec2, angle: float) -> Vec2:
    return _add(origin, _rotate(_sub(zero_point, zero_origin), angle))


def _right_loop_residual(passive: list[float], j_ag: float, j_ab: float) -> list[float]:
    """Closure error for [jGH, jBE, jEC, jCF]."""
    j_gh, j_be, j_ec, j_cf = passive

    right_d_from_ag = _point_from(_P["AG"], _P["AG"], _P["right_D"], j_ag)
    gh_origin = _point_from(_P["AG"], _P["AG"], _P["GH"], j_ag)
    right_f_from_gh = _point_from(gh_origin, _P["GH"], _P["right_F"], j_ag + j_gh)

    be_origin = _point_from(_P["AB"], _P["AB"], _P["BE"], j_ab)
    ec_origin = _point_from(be_origin, _P["BE"], _P["EC"], j_ab + j_be)
    ec_angle = j_ab + j_be - j_ec
    right_d_from_ec = _point_from(ec_origin, _P["EC"], _P["right_D"], ec_angle)
    cf_origin = _point_from(ec_origin, _P["EC"], _P["CF"], ec_angle)
    right_f_from_cf = _point_from(cf_origin, _P["CF"], _P["right_F"], ec_angle - j_cf)

    error_d = _sub(right_d_from_ec, right_d_from_ag)
    error_f = _sub(right_f_from_cf, right_f_from_gh)
    return [error_d[0], error_d[1], error_f[0], error_f[1]]


def _left_loop_residual(passive: list[float], j_ij: float, j_io: float) -> list[float]:
    """Closure error for [jJM, jMK, jKN, jOP]."""
    j_jm, j_mk, j_kn, j_op = passive

    left_l_from_io = _point_from(_P["IO"], _P["IO"], _P["left_L"], -j_io)
    op_origin = _point_from(_P["IO"], _P["IO"], _P["OP"], -j_io)
    left_f_from_op = _point_from(op_origin, _P["OP"], _P["left_F"], -j_io - j_op)

    jm_origin = _point_from(_P["IJ"], _P["IJ"], _P["JM"], -j_ij)
    mk_origin = _point_from(jm_origin, _P["JM"], _P["MK"], -j_ij - j_jm)
    mk_angle = -j_ij - j_jm - j_mk
    left_l_from_mk = _point_from(mk_origin, _P["MK"], _P["left_L"], mk_angle)
    kn_origin = _point_from(mk_origin, _P["MK"], _P["KN"], mk_angle)
    left_f_from_kn = _point_from(kn_origin, _P["KN"], _P["left_F"], mk_angle + j_kn)

    error_l = _sub(left_l_from_mk, left_l_from_io)
    error_f = _sub(left_f_from_kn, left_f_from_op)
    return [error_l[0], error_l[1], error_f[0], error_f[1]]


def _solve_linear_4x4(matrix: list[list[float]], rhs: list[float]) -> list[float]:
    augmented = [row[:] + [value] for row, value in zip(matrix, rhs)]
    for column in range(4):
        pivot = max(range(column, 4), key=lambda row: abs(augmented[row][column]))
        if abs(augmented[pivot][column]) < 1.0e-12:
            raise ValueError("Standard WL closure Jacobian is singular at the requested pose")
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        pivot_value = augmented[column][column]
        for index in range(column, 5):
            augmented[column][index] /= pivot_value
        for row in range(4):
            if row == column:
                continue
            factor = augmented[row][column]
            for index in range(column, 5):
                augmented[row][index] -= factor * augmented[column][index]
    return [augmented[row][4] for row in range(4)]


def _error_norm(error: list[float]) -> float:
    return math.sqrt(sum(value * value for value in error))


def _newton_solve(residual: Residual, active_0: float, active_1: float, initial: list[float]) -> list[float]:
    solution = initial[:]
    epsilon = 1.0e-6
    for _ in range(30):
        error = residual(solution, active_0, active_1)
        error_norm = _error_norm(error)
        if error_norm < 1.0e-11:
            return solution

        jacobian = [[0.0] * 4 for _ in range(4)]
        for column in range(4):
            plus = solution[:]
            minus = solution[:]
            plus[column] += epsilon
            minus[column] -= epsilon
            error_plus = residual(plus, active_0, active_1)
            error_minus = residual(minus, active_0, active_1)
            for row in range(4):
                jacobian[row][column] = (error_plus[row] - error_minus[row]) / (2.0 * epsilon)

        delta = _solve_linear_4x4(jacobian, [-value for value in error])
        step_scale = 1.0
        for _ in range(12):
            candidate = [value + step_scale * change for value, change in zip(solution, delta)]
            if _error_norm(residual(candidate, active_0, active_1)) < error_norm:
                solution = candidate
                break
            step_scale *= 0.5
        else:
            raise ValueError("Standard WL closure solve did not converge at the requested pose")

    raise ValueError("Standard WL closure solve exceeded its iteration limit")


def _solve_along_path(residual: Residual, active_0: float, active_1: float) -> list[float]:
    # Following the assembly continuously from q=0 avoids jumping to the other
    # valid five-bar branch when the requested motor angles are far from zero.
    path_steps = max(1, math.ceil(max(abs(active_0), abs(active_1)) / 0.05))
    passive = [0.0, 0.0, 0.0, 0.0]
    for step in range(1, path_steps + 1):
        fraction = step / path_steps
        passive = _newton_solve(
            residual,
            active_0 * fraction,
            active_1 * fraction,
            passive,
        )
    return passive


def _closed_chain_joint_pos(
    *,
    j_ag: float,
    j_io: float,
    j_ab: float,
    j_ij: float,
) -> dict[str, float]:
    """Return a complete, stress-free reset pose from the four leg motor angles."""
    active = {"jAG": j_ag, "jIO": j_io, "jAB": j_ab, "jIJ": j_ij}
    for name, value in active.items():
        lower, upper = _MOTOR_LIMITS[name]
        if not lower <= value <= upper:
            raise ValueError(
                f"Standard WL motor {name}={value:.6f} rad is outside [{lower:.2f}, {upper:.2f}] rad"
            )

    j_gh, j_be, j_ec, j_cf = _solve_along_path(_right_loop_residual, j_ag, j_ab)
    j_jm, j_mk, j_kn, j_op = _solve_along_path(_left_loop_residual, j_ij, j_io)
    if not -0.85 <= j_gh <= 1.0:
        raise ValueError(f"Solved passive joint jGH={j_gh:.6f} rad exceeds its USD limit")
    if not -1.0 <= j_op <= 0.85:
        raise ValueError(f"Solved passive joint jOP={j_op:.6f} rad exceeds its USD limit")
    return {
        "jAG": j_ag,
        "jIO": j_io,
        "jAB": j_ab,
        "jIJ": j_ij,
        "jGH": j_gh,
        "jOP": j_op,
        "jBE": j_be,
        "jJM": j_jm,
        "jEC": j_ec,
        "jMK": j_mk,
        "jCF": j_cf,
        "jKN": j_kn,
        "jwheel_right": 0.0,
        "jwheel_left": 0.0,
    }


def _joint_pos_from_virtual_leg(*, l0: float, phi0_deg: float) -> dict[str, float]:
    """Return the full mirrored reset pose from virtual-leg length and angle."""
    minimum_l0 = abs(_PASSIVE_LINK_LENGTH - _MOTOR_LINK_LENGTH)
    maximum_l0 = _PASSIVE_LINK_LENGTH + _MOTOR_LINK_LENGTH
    if not minimum_l0 < l0 < maximum_l0:
        raise ValueError(
            f"Standard WL l0={l0:.6f} m is outside the open geometric workspace "
            f"({minimum_l0:.3f}, {maximum_l0:.3f}) m"
        )
    if not 0.0 < phi0_deg < 180.0:
        raise ValueError("Standard WL phi0_deg must be between 0 and 180 degrees")

    cosine_alpha = (
        l0 * l0 + _MOTOR_LINK_LENGTH**2 - _PASSIVE_LINK_LENGTH**2
    ) / (2.0 * l0 * _MOTOR_LINK_LENGTH)
    alpha = math.acos(max(-1.0, min(1.0, cosine_alpha)))
    left_phi0 = math.radians(phi0_deg)
    right_phi0 = math.pi - left_phi0

    left_phi1 = left_phi0 + alpha
    left_phi4 = left_phi0 - alpha
    right_phi1 = right_phi0 + alpha
    right_phi4 = right_phi0 - alpha

    # The pi/1.3 terms are structural joint-frame rotations, not encoder calibration.
    return _closed_chain_joint_pos(
        j_ag=right_phi4 - 1.3,
        j_io=left_phi1 - (math.pi - 1.3),
        j_ab=right_phi1 - math.pi,
        j_ij=left_phi4,
    )

STANDARD_WL_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=(
            f"{FLAMINGO_ASSETS_DATA_DIR}/Robots/WheelLegged/"
            "standard_wl_rev_01_0_0/standard_wl.usd"
        ),
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
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=4,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        # Symmetric five-bar initial pose. l0 is the virtual leg length and
        # phi0_deg is its physical angle from the forward horizontal axis:
        #   alpha = acos((l0^2 + l1^2 - l2^2) / (2*l0*l1))
        #   phi1 = phi0 + alpha, phi4 = phi0 - alpha
        # The right leg uses the mirrored angle 180 - phi0_deg. These equations
        # determine all four motor angles; the four loop-pin constraints then
        # determine the eight passive joint angles.
        pos=(0.0, 0.0, 0.31),
        joint_pos=_joint_pos_from_virtual_leg(
            l0=0.36,
            phi0_deg=90.0,
        ),
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.95,
    actuators={
        "leg_motors": ImplicitActuatorCfg(
            joint_names_expr=["jAB", "jAG", "jIJ", "jIO"],
            effort_limit_sim=20.0,
            velocity_limit_sim=30.0,
            stiffness=0.0,
            damping=0.0,
            friction=0.0,
            armature=0.0,
        ),
        "wheel_motors": ImplicitActuatorCfg(
            joint_names_expr=["jwheel_left", "jwheel_right"],
            effort_limit_sim=4.0,
            velocity_limit_sim=100.0,
            stiffness=0.0,
            damping=0.0,
            friction=0.0,
            armature=0.0,
        ),
        "passive_links": ImplicitActuatorCfg(
            joint_names_expr=["jGH", "jBE", "jEC", "jCF", "jJM", "jMK", "jKN", "jOP"],
            effort_limit_sim=20.0,
            velocity_limit_sim=30.0,
            stiffness=0.0,
            damping=0.02,
            friction=0.0,
            armature=0.0,
        ),
    },
)
"""Closed-chain standard WL configuration with five-bar VMC effort control."""
