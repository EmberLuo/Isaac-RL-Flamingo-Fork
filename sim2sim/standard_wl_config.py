from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
STANDARD_WL_MJCF = REPO_ROOT / "wheel_leg_mujoco/MJCF/env.xml"

STANDARD_WL_JOINTS = (
    "jAG",
    "jGH",
    "jwheel_right",
    "jAB",
    "jBE",
    "jEC",
    "jCF",
    "jIJ",
    "jJM",
    "jMK",
    "jKN",
    "jIO",
    "jOP",
    "jwheel_left",
)

STANDARD_WL_INIT_JOINT_POS = {
    "jAG": -0.5164193845814288,
    "jIO": 0.5164193845814287,
    "jAB": -0.7835806154185714,
    "jIJ": 0.7835806154185713,
    "jGH": -0.22655560991529222,
    "jOP": 0.22655576840772548,
    "jBE": 0.22599489195250008,
    "jJM": -0.225995167777391,
    "jEC": 0.18510463079716583,
    "jMK": 0.18510490014585093,
    "jCF": -0.22634852051601614,
    "jKN": 0.22634804441465434,
    "jwheel_right": 0.0,
    "jwheel_left": 0.0,
}


@dataclass(frozen=True)
class StandardSimConfig:
    # MuJoCo needs a smaller internal step for the stiff closed-chain constraints.
    physics_dt: float = 0.001
    # Isaac Lab applies the VMC once per simulation step and the policy every
    # ``policy_decimation`` simulation steps.
    vmc_dt: float = 0.005
    policy_decimation: int = 4
    duration: float = 20.0
    root_height: float = 0.31
    total_mass: float = 18.0
    passive_joint_damping: float = 0.02
    ground_friction: tuple[float, float, float] = (1.0, 0.005, 0.0001)
    base_contact_force_threshold: float = 1.0
    terminate_on_base_contact: bool = True
    analytic_wheel_collisions: bool = True

    @property
    def vmc_substeps(self) -> int:
        return round(self.vmc_dt / self.physics_dt)

    @property
    def policy_dt(self) -> float:
        return self.vmc_dt * self.policy_decimation


@dataclass(frozen=True)
class StandardCommand:
    lin_vel_x: float = 0.0
    lin_vel_y: float = 0.0
    ang_vel_z: float = 0.0
    pos_z: float = 0.31


@dataclass(frozen=True)
class StandardVmcConfig:
    l1: float = 0.215
    l2: float = 0.258
    l3: float = 0.258
    l4: float = 0.215
    l5: float = 0.0
    kinematic_eps: float = 1.0e-6

    left_phi1_offset: float = 1.841592653589793
    left_phi4_offset: float = 0.0
    right_phi1_offset: float = 3.141592653589793
    right_phi4_offset: float = 1.3
    nominal_theta: float = 0.0

    action_scale_theta: float = 0.2
    action_scale_l0: float = 0.05
    action_scale_vel: float = 50.0
    l0_offset: float = 0.36

    kp_theta: float = 50.0
    kd_theta: float = 3.0
    kp_l0: float = 900.0
    kd_l0: float = 20.0
    wheel_damping: float = 0.5
    feedforward_force: float = 88.3

    leg_effort_limit: float = 20.0
    wheel_effort_limit: float = 4.0


@dataclass(frozen=True)
class StandardObsConfig:
    num_policy_stacks: int = 2
    stack_dim: int = 22
    nonstack_dim: int = 4
    command_scale: tuple[float, float, float] = (1.0 / 3.0, 1.0, 1.0 / 3.0)

    @property
    def total_frames(self) -> int:
        return self.num_policy_stacks + 1

    @property
    def policy_dim(self) -> int:
        return self.stack_dim * self.total_frames + self.nonstack_dim
