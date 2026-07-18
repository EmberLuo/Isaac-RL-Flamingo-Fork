from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
WL_URDF = REPO_ROOT / "lab/flamingo/assets/data/Robots/WheelLegged/wl_rev_01_0_0/urdf/wl.urdf"


@dataclass(frozen=True)
class SimConfig:
    sim_dt: float = 0.005
    decimation: int = 4
    duration: float = 20.0
    root_height: float = 0.25
    ground_size: tuple[float, float, float] = (20.0, 20.0, 0.05)
    ground_friction: tuple[float, float, float] = (1.0, 0.005, 0.0001)

    @property
    def control_dt(self) -> float:
        return self.sim_dt * self.decimation


@dataclass(frozen=True)
class Command:
    lin_vel_x: float = 0.0
    lin_vel_y: float = 0.0
    ang_vel_z: float = 0.0
    pos_z: float = 0.25

    def scaled(self) -> tuple[float, float, float, float]:
        return (self.lin_vel_x * 2.0, self.lin_vel_y, self.ang_vel_z * 0.25, self.pos_z)


@dataclass(frozen=True)
class VmcConfig:
    offset: float = 0.054
    l1: float = 0.15
    l2: float = 0.25
    action_scale_theta: float = 0.15
    action_scale_l0: float = 0.05
    action_scale_vel: float = 12.0
    l0_offset: float = 0.24
    kp_theta: float = 50.0
    kd_theta: float = 3.0
    kp_l0: float = 900.0
    kd_l0: float = 80.0
    wheel_damping: float = 0.5
    feedforward_force: float = 40.0
    leg_effort_limit: float = 30.0
    wheel_effort_limit: float = 5.0


@dataclass(frozen=True)
class ObsConfig:
    num_policy_stacks: int = 2
    stack_dim: int = 22
    nonstack_dim: int = 4

    @property
    def total_frames(self) -> int:
        return self.num_policy_stacks + 1

    @property
    def policy_dim(self) -> int:
        return self.stack_dim * self.total_frames + self.nonstack_dim
