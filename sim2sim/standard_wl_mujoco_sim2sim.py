#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import re
import time
from collections import deque
from dataclasses import fields, replace
from pathlib import Path

import mujoco
import numpy as np
import torch
import torch.nn as nn
import yaml

from sim2sim.standard_wl_config import (
    STANDARD_WL_INIT_JOINT_POS,
    STANDARD_WL_JOINTS,
    STANDARD_WL_MJCF,
    StandardCommand,
    StandardObsConfig,
    StandardSimConfig,
    StandardVmcConfig,
)


ACTIVE_JOINTS = ("jAB", "jAG", "jIJ", "jIO", "jwheel_right", "jwheel_left")
PASSIVE_JOINTS = ("jGH", "jBE", "jEC", "jCF", "jJM", "jMK", "jKN", "jOP")
LOOP_SITE_PAIRS = (
    ("EC-D", "AG-D"),
    ("CF-F", "GH-F"),
    ("IO-L", "MK-L"),
    ("OP-N", "KN-N"),
)
WHEEL_BODIES = frozenset(("wheel_left", "wheel_right"))


def _id(model: mujoco.MjModel, obj_type: mujoco.mjtObj, name: str) -> int:
    obj_id = mujoco.mj_name2id(model, obj_type, name)
    if obj_id < 0:
        raise RuntimeError(f"MuJoCo object not found: {name}")
    return obj_id


class _IsaacYamlLoader(yaml.SafeLoader):
    pass


def _python_tuple(loader: _IsaacYamlLoader, node: yaml.Node) -> tuple:
    return tuple(loader.construct_sequence(node))


def _python_object_apply(loader: _IsaacYamlLoader, tag_suffix: str, node: yaml.Node) -> None:
    return None


_IsaacYamlLoader.add_constructor("tag:yaml.org,2002:python/tuple", _python_tuple)
_IsaacYamlLoader.add_multi_constructor("tag:yaml.org,2002:python/object/apply:", _python_object_apply)


def _load_isaac_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as stream:
        return yaml.load(stream, Loader=_IsaacYamlLoader)


def _checkpoint_index(path: Path) -> int:
    match = re.fullmatch(r"model_(\d+)\.pt", path.name)
    return int(match.group(1)) if match else -1


def resolve_policy_source(path: Path, checkpoint: Path | None) -> tuple[Path, Path | None]:
    """Resolve a TorchScript export or raw PPO checkpoint and its run directory."""
    path = path.expanduser().resolve()
    if path.is_file():
        if checkpoint is not None:
            raise ValueError("--checkpoint is only valid when --policy points to a run directory")
        if path.parent.name == "exported" and path.name == "policy.pt":
            return path, path.parent.parent
        run_dir = path.parent if _checkpoint_index(path) >= 0 else None
        return path, run_dir

    if not path.is_dir():
        raise FileNotFoundError(path)

    run_dir = path.parent if path.name == "exported" else path
    if checkpoint is not None:
        candidate = checkpoint.expanduser()
        if not candidate.is_absolute():
            candidate = run_dir / candidate
        candidate = candidate.resolve()
        if not candidate.exists():
            raise FileNotFoundError(candidate)
        return candidate, run_dir

    exported = run_dir / "exported/policy.pt"
    if exported.exists():
        return exported, run_dir

    checkpoints = sorted(run_dir.glob("model_*.pt"), key=_checkpoint_index)
    if checkpoints:
        return checkpoints[-1], run_dir
    raise FileNotFoundError(f"No exported/policy.pt or model_*.pt found in {run_dir}")


def _activation(name: str) -> nn.Module:
    activations: dict[str, type[nn.Module]] = {
        "elu": nn.ELU,
        "relu": nn.ReLU,
        "selu": nn.SELU,
        "lrelu": nn.LeakyReLU,
        "leaky_relu": nn.LeakyReLU,
        "tanh": nn.Tanh,
        "sigmoid": nn.Sigmoid,
    }
    key = name.lower()
    if key not in activations:
        raise ValueError(f"Unsupported actor activation in checkpoint metadata: {name}")
    return activations[key]()


class InferencePolicy:
    """CPU inference wrapper for either exported TorchScript or a raw PPO checkpoint."""

    def __init__(self, path: Path | None, run_dir: Path | None):
        self.path = path
        self.model: torch.jit.ScriptModule | nn.Module | None = None
        self.input_dim: int | None = None
        self.backend = "zero"
        if path is None:
            return

        if path.name == "policy.pt":
            try:
                self.model = torch.jit.load(str(path), map_location="cpu")
                self.model.eval()
                if hasattr(self.model, "reset"):
                    self.model.reset()
                self.backend = "torchscript"
                return
            except RuntimeError:
                pass

        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        state = checkpoint.get("model_state_dict", checkpoint)
        actor_weights = []
        for key, value in state.items():
            match = re.fullmatch(r"actor\.(\d+)\.weight", key)
            if match:
                actor_weights.append((int(match.group(1)), key, value))
        actor_weights.sort()
        if not actor_weights:
            raise ValueError(f"Checkpoint has no actor.*.weight tensors: {path}")

        activation_name = "elu"
        if run_dir is not None and (run_dir / "params/agent.yaml").exists():
            agent_cfg = _load_isaac_yaml(run_dir / "params/agent.yaml")
            activation_name = str(agent_cfg.get("policy", {}).get("activation", activation_name))

        modules: list[nn.Module] = []
        for index, (_, weight_key, weight) in enumerate(actor_weights):
            bias_key = weight_key.removesuffix("weight") + "bias"
            if bias_key not in state:
                raise ValueError(f"Missing actor bias tensor: {bias_key}")
            linear = nn.Linear(weight.shape[1], weight.shape[0])
            with torch.no_grad():
                linear.weight.copy_(weight)
                linear.bias.copy_(state[bias_key])
            modules.append(linear)
            if index + 1 < len(actor_weights):
                modules.append(_activation(activation_name))
        self.model = nn.Sequential(*modules).eval()
        self.input_dim = int(actor_weights[0][2].shape[1])
        if actor_weights[-1][2].shape[0] != 6:
            raise ValueError(f"Expected six policy actions, got {actor_weights[-1][2].shape[0]}")
        self.backend = "checkpoint"

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        if self.model is None:
            return np.zeros(6, dtype=np.float32)
        with torch.inference_mode():
            obs_t = torch.from_numpy(obs.astype(np.float32, copy=False)).unsqueeze(0)
            action = self.model(obs_t)
            if isinstance(action, tuple):
                action = action[0]
            action_np = action.squeeze(0).cpu().numpy()
        return np.clip(action_np, -1.0, 1.0).astype(np.float32)


def load_training_metadata(
    run_dir: Path | None,
    sim_cfg: StandardSimConfig,
    vmc_cfg: StandardVmcConfig,
    obs_cfg: StandardObsConfig,
) -> tuple[StandardSimConfig, StandardVmcConfig, StandardObsConfig, dict[str, float]]:
    init_joint_pos = dict(STANDARD_WL_INIT_JOINT_POS)
    if run_dir is None:
        return sim_cfg, vmc_cfg, obs_cfg, init_joint_pos

    env_yaml = run_dir / "params/env.yaml"
    if env_yaml.exists():
        env_cfg = _load_isaac_yaml(env_yaml)
        sim_cfg = replace(
            sim_cfg,
            vmc_dt=float(env_cfg.get("sim", {}).get("dt", sim_cfg.vmc_dt)),
            policy_decimation=int(env_cfg.get("decimation", sim_cfg.policy_decimation)),
        )
        robot_init = env_cfg.get("scene", {}).get("robot", {}).get("init_state", {})
        root_pos = robot_init.get("pos")
        if root_pos is not None and len(root_pos) == 3:
            sim_cfg = replace(sim_cfg, root_height=float(root_pos[2]))
        yaml_joint_pos = robot_init.get("joint_pos", {})
        for name in STANDARD_WL_JOINTS:
            if name in yaml_joint_pos:
                init_joint_pos[name] = float(yaml_joint_pos[name])

        vmc_data = env_cfg.get("actions", {}).get("vmc", {})
        updates = {
            field.name: float(vmc_data[field.name])
            for field in fields(StandardVmcConfig)
            if field.name in vmc_data
        }
        if updates:
            vmc_cfg = replace(vmc_cfg, **updates)

        command_scale = (
            env_cfg.get("observations", {})
            .get("none_stack_policy", {})
            .get("velocity_commands", {})
            .get("params", {})
            .get("scale")
        )
        if command_scale is not None and len(command_scale) == 3:
            obs_cfg = replace(obs_cfg, command_scale=tuple(float(value) for value in command_scale))

    agent_yaml = run_dir / "params/agent.yaml"
    if agent_yaml.exists():
        agent_cfg = _load_isaac_yaml(agent_yaml)
        if agent_cfg.get("empirical_normalization", False):
            raise ValueError("Raw checkpoint sim2sim does not yet support empirical observation normalization")
        if "num_policy_stacks" in agent_cfg:
            obs_cfg = replace(obs_cfg, num_policy_stacks=int(agent_cfg["num_policy_stacks"]))

    return sim_cfg, vmc_cfg, obs_cfg, init_joint_pos


class StandardWlVmc:
    """NumPy equivalent of mdp.StandardWLVMCAction."""

    def __init__(self, cfg: StandardVmcConfig):
        self.cfg = cfg
        self.theta_sign = np.array([-1.0, 1.0], dtype=np.float64)
        self.theta0_ref = np.zeros(2, dtype=np.float64)
        self.l0_ref = np.full(2, cfg.l0_offset, dtype=np.float64)
        self.wheel_vel_ref = np.zeros(2, dtype=np.float64)
        self.theta0 = np.zeros(2, dtype=np.float64)
        self.l0 = np.zeros(2, dtype=np.float64)
        self.theta0_dot = np.zeros(2, dtype=np.float64)
        self.l0_dot = np.zeros(2, dtype=np.float64)
        self.wheel_vel = np.zeros(2, dtype=np.float64)
        self.virtual_force = np.zeros(2, dtype=np.float64)
        self.virtual_torque = np.zeros(2, dtype=np.float64)

    def process_action(self, action: np.ndarray) -> np.ndarray:
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        self.theta0_ref = action[[0, 3]] * self.cfg.action_scale_theta
        self.l0_ref = action[[1, 4]] * self.cfg.action_scale_l0 + self.cfg.l0_offset
        self.wheel_vel_ref = action[[2, 5]] * self.cfg.action_scale_vel
        return action.astype(np.float32)

    @staticmethod
    def _signed_clamp(value: np.ndarray, minimum_abs: float) -> np.ndarray:
        return np.where(value >= 0.0, np.maximum(value, minimum_abs), np.minimum(value, -minimum_abs))

    def _five_bar_kinematics(self, phi1: np.ndarray, phi4: np.ndarray) -> tuple[np.ndarray, ...]:
        cfg = self.cfg
        xb = cfg.l1 * np.cos(phi1)
        yb = cfg.l1 * np.sin(phi1)
        xd = cfg.l5 + cfg.l4 * np.cos(phi4)
        yd = cfg.l4 * np.sin(phi4)
        dx_bd = xd - xb
        dy_bd = yd - yb
        l_bd_sq = dx_bd**2 + dy_bd**2

        a0 = 2.0 * cfg.l2 * dx_bd
        b0 = 2.0 * cfg.l2 * dy_bd
        c0 = cfg.l2**2 + l_bd_sq - cfg.l3**2
        discriminant = np.maximum(a0**2 + b0**2 - c0**2, 0.0)
        phi2 = 2.0 * np.arctan2(b0 + np.sqrt(discriminant), a0 + c0)
        phi3 = np.arctan2(
            yb - yd + cfg.l2 * np.sin(phi2),
            xb - xd + cfg.l2 * np.cos(phi2),
        )

        xc = xb + cfg.l2 * np.cos(phi2)
        yc = yb + cfg.l2 * np.sin(phi2)
        xc_centered = xc - 0.5 * cfg.l5
        l0 = np.sqrt(np.maximum(xc_centered**2 + yc**2, cfg.kinematic_eps**2))
        phi0 = np.arctan2(yc, xc_centered)

        sin_32 = self._signed_clamp(np.sin(phi3 - phi2), cfg.kinematic_eps)
        j11 = cfg.l1 * np.sin(phi0 - phi3) * np.sin(phi1 - phi2) / sin_32
        j12 = cfg.l1 * np.cos(phi0 - phi3) * np.sin(phi1 - phi2) / (l0 * sin_32)
        j21 = cfg.l4 * np.sin(phi0 - phi2) * np.sin(phi3 - phi4) / sin_32
        j22 = cfg.l4 * np.cos(phi0 - phi2) * np.sin(phi3 - phi4) / (l0 * sin_32)
        return l0, phi0, j11, j12, j21, j22

    def compute_controls(self, q: dict[str, float], qd: dict[str, float]) -> np.ndarray:
        # Arrays are consistently [left, right].
        phi1 = np.array(
            [q["jIO"] + self.cfg.left_phi1_offset, q["jAB"] + self.cfg.right_phi1_offset]
        )
        phi4 = np.array(
            [q["jIJ"] + self.cfg.left_phi4_offset, q["jAG"] + self.cfg.right_phi4_offset]
        )
        phi1_dot = np.array([qd["jIO"], qd["jAB"]])
        phi4_dot = np.array([qd["jIJ"], qd["jAG"]])

        self.l0, phi0, j11, j12, j21, j22 = self._five_bar_kinematics(phi1, phi4)
        self.theta0 = self.theta_sign * (phi0 - 0.5 * math.pi) - self.cfg.nominal_theta
        self.l0_dot = j11 * phi1_dot + j21 * phi4_dot
        self.theta0_dot = self.theta_sign * (j12 * phi1_dot + j22 * phi4_dot)
        self.wheel_vel = np.array([-qd["jwheel_left"], qd["jwheel_right"]])

        self.virtual_torque = (
            self.cfg.kp_theta * (self.theta0_ref - self.theta0)
            - self.cfg.kd_theta * self.theta0_dot
        )
        self.virtual_force = (
            self.cfg.kp_l0 * (self.l0_ref - self.l0)
            - self.cfg.kd_l0 * self.l0_dot
            + self.cfg.feedforward_force
        )
        source_tp = self.theta_sign * self.virtual_torque
        tau_phi1 = j11 * self.virtual_force + j12 * source_tp
        tau_phi4 = j21 * self.virtual_force + j22 * source_tp
        wheel_torque = self.cfg.wheel_damping * (self.wheel_vel_ref - self.wheel_vel)

        # MJCF actuator order is right front/rear, left front/rear, right/left wheel.
        # Left_Wheel_act has gainprm=-1, so its ctrl is the canonical wheel torque.
        controls = np.array(
            [tau_phi1[1], tau_phi4[1], tau_phi4[0], tau_phi1[0], wheel_torque[1], wheel_torque[0]],
            dtype=np.float64,
        )
        limits = np.array(
            [
                self.cfg.leg_effort_limit,
                self.cfg.leg_effort_limit,
                self.cfg.leg_effort_limit,
                self.cfg.leg_effort_limit,
                self.cfg.wheel_effort_limit,
                self.cfg.wheel_effort_limit,
            ]
        )
        return np.clip(np.nan_to_num(controls), -limits, limits)


class ObservationBuilder:
    def __init__(self, cfg: StandardObsConfig, command: StandardCommand):
        self.cfg = cfg
        self.command = command
        self.frames: deque[np.ndarray] = deque(maxlen=cfg.total_frames)

    @staticmethod
    def projected_gravity(data: mujoco.MjData, body_id: int) -> np.ndarray:
        rotation = data.xmat[body_id].reshape(3, 3)
        return rotation.T @ np.array([0.0, 0.0, -1.0])

    @staticmethod
    def body_velocity(
        model: mujoco.MjModel, data: mujoco.MjData, body_id: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Angular and linear velocity in the body LINK frame (matching Isaac's ``root_link_*_vel_b``).

        ``mj_objectVelocity`` with ``flg_local=1`` expresses the velocity in the
        COM-based inertial frame (``ximat``), which the MJCF compiler rotates
        onto the principal inertia axes.  For this model's ``base`` body that
        frame is ~90 degrees away from the link frame (``xmat``), so the local
        output cannot be used directly.  Query the world-frame velocity and
        rotate it into the link frame instead.
        """
        velocity = np.zeros(6, dtype=np.float64)
        mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, body_id, velocity, 0)
        rotation = data.xmat[body_id].reshape(3, 3)
        return rotation.T @ velocity[:3], rotation.T @ velocity[3:]

    def stack_frame(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        base_body_id: int,
        vmc: StandardWlVmc,
        last_action: np.ndarray,
    ) -> np.ndarray:
        angular_velocity_b, _ = self.body_velocity(model, data, base_body_id)
        frame = np.concatenate(
            [
                angular_velocity_b * 0.25,
                self.projected_gravity(data, base_body_id),
                vmc.theta0,
                vmc.theta0_dot * 0.05,
                vmc.l0 * 5.0,
                vmc.l0_dot * 0.25,
                vmc.wheel_vel * 0.05,
                last_action,
            ]
        ).astype(np.float32)
        if frame.shape != (self.cfg.stack_dim,):
            raise RuntimeError(f"Unexpected stack frame shape: {frame.shape}")
        return frame

    def reset(self, frame: np.ndarray) -> np.ndarray:
        self.frames.clear()
        for _ in range(self.cfg.total_frames):
            self.frames.append(frame.copy())
        return self.policy_obs()

    def update(self, frame: np.ndarray) -> np.ndarray:
        self.frames.appendleft(frame.copy())
        return self.policy_obs()

    def policy_obs(self) -> np.ndarray:
        if len(self.frames) != self.cfg.total_frames:
            raise RuntimeError("Observation stack is not initialized")
        command = np.asarray(
            (
                self.command.lin_vel_x * self.cfg.command_scale[0],
                self.command.lin_vel_y * self.cfg.command_scale[1],
                self.command.ang_vel_z * self.cfg.command_scale[2],
                self.command.pos_z,
            ),
            dtype=np.float32,
        )
        return np.concatenate([*self.frames, command])


class StandardWlMujocoRunner:
    def __init__(
        self,
        model_path: Path,
        policy: InferencePolicy,
        command: StandardCommand,
        sim_cfg: StandardSimConfig,
        vmc_cfg: StandardVmcConfig,
        obs_cfg: StandardObsConfig,
        init_joint_pos: dict[str, float],
    ):
        self.sim_cfg = sim_cfg
        self.command = command
        self.policy = policy
        self.vmc = StandardWlVmc(vmc_cfg)
        self.obs_builder = ObservationBuilder(obs_cfg, command)
        self.init_joint_pos = init_joint_pos
        self.model = mujoco.MjModel.from_xml_path(str(model_path.resolve()))
        self.model.opt.timestep = sim_cfg.physics_dt
        self.data = mujoco.MjData(self.model)

        self.base_body_id = _id(self.model, mujoco.mjtObj.mjOBJ_BODY, "base")
        self.floor_geom_id = _id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        self.free_joint_id = _id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "base_free")
        self.free_qpos_adr = int(self.model.jnt_qposadr[self.free_joint_id])
        self.free_dof_adr = int(self.model.jnt_dofadr[self.free_joint_id])
        self.joint_ids = {
            name: _id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in STANDARD_WL_JOINTS
        }
        self.qpos_adrs = {name: int(self.model.jnt_qposadr[jid]) for name, jid in self.joint_ids.items()}
        self.dof_adrs = {name: int(self.model.jnt_dofadr[jid]) for name, jid in self.joint_ids.items()}
        self.loop_site_ids = [
            (
                _id(self.model, mujoco.mjtObj.mjOBJ_SITE, first),
                _id(self.model, mujoco.mjtObj.mjOBJ_SITE, second),
            )
            for first, second in LOOP_SITE_PAIRS
        ]

        self._align_loop_sites_to_midplanes()
        self._configure_dynamics()
        mujoco.mj_setConst(self.model, self.data)
        self.model_total_mass = float(np.sum(self.model.body_mass))
        self.last_action = np.zeros(6, dtype=np.float32)
        self.last_controls = np.zeros(6, dtype=np.float64)
        self.termination_reason: str | None = None
        self.max_loop_error = 0.0
        self.max_base_contact_force = 0.0
        self.max_leg_contact_force = 0.0
        self.last_ncon = 0

    def _align_loop_sites_to_midplanes(self) -> None:
        """Use the same stress-free loop-pin midpoints as the training USD converter."""
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        for first_id, second_id in self.loop_site_ids:
            anchor_world = 0.5 * (self.data.site_xpos[first_id] + self.data.site_xpos[second_id])
            for site_id in (first_id, second_id):
                body_id = int(self.model.site_bodyid[site_id])
                rotation = self.data.xmat[body_id].reshape(3, 3)
                self.model.site_pos[site_id] = rotation.T @ (anchor_world - self.data.xpos[body_id])
        mujoco.mj_forward(self.model, self.data)

    def _configure_dynamics(self) -> None:
        # Match enabled_self_collisions=False while retaining ground collision for every USD collider.
        for geom_id in range(self.model.ngeom):
            self.model.geom_friction[geom_id] = np.asarray(self.sim_cfg.ground_friction)
            if geom_id == self.floor_geom_id:
                self.model.geom_contype[geom_id] = 1
                self.model.geom_conaffinity[geom_id] = 2
            else:
                self.model.geom_contype[geom_id] = 2
                self.model.geom_conaffinity[geom_id] = 1

        for name in PASSIVE_JOINTS:
            self.model.dof_damping[self.dof_adrs[name]] = self.sim_cfg.passive_joint_damping

        if self.sim_cfg.analytic_wheel_collisions:
            wheel_specs = {
                "wheel_right": np.array([-0.000002555, 0.000007725, -0.005]),
                "wheel_left": np.array([0.0, 0.0, 0.005]),
            }
            for body_name, center in wheel_specs.items():
                body_id = _id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
                geom_ids = np.flatnonzero(self.model.geom_bodyid == body_id)
                if len(geom_ids) != 1:
                    raise RuntimeError(f"Expected one collision geom on {body_name}, got {geom_ids}")
                geom_id = int(geom_ids[0])
                self.model.geom_type[geom_id] = mujoco.mjtGeom.mjGEOM_CYLINDER
                self.model.geom_pos[geom_id] = center
                self.model.geom_quat[geom_id] = np.array([1.0, 0.0, 0.0, 0.0])
                self.model.geom_size[geom_id] = np.array([0.077, 0.012, 0.0])

        source_total_mass = float(np.sum(self.model.body_mass))
        mass_delta = self.sim_cfg.total_mass - source_total_mass
        old_base_mass = float(self.model.body_mass[self.base_body_id])
        new_base_mass = old_base_mass + mass_delta
        if new_base_mass <= 0.0:
            raise ValueError(
                f"Requested total mass {self.sim_cfg.total_mass:.3f} kg makes base mass non-positive"
            )
        # Match Isaac Lab's randomize_rigid_body_mass with recompute_inertia=True:
        # the base inertia tensor scales by the same ratio as the mass.
        self.model.body_mass[self.base_body_id] = new_base_mass
        self.model.body_inertia[self.base_body_id] *= new_base_mass / old_base_mass

    def _joint_pos(self) -> dict[str, float]:
        return {name: float(self.data.qpos[address]) for name, address in self.qpos_adrs.items()}

    def _joint_vel(self) -> dict[str, float]:
        return {name: float(self.data.qvel[address]) for name, address in self.dof_adrs.items()}

    def _apply_vmc(self) -> np.ndarray:
        self.last_controls = self.vmc.compute_controls(self._joint_pos(), self._joint_vel())
        self.data.ctrl[:] = self.last_controls
        return self.last_controls

    def _stack_frame(self) -> np.ndarray:
        return self.obs_builder.stack_frame(
            self.model, self.data, self.base_body_id, self.vmc, self.last_action
        )

    def loop_error(self) -> float:
        return max(
            float(np.linalg.norm(self.data.site_xpos[first] - self.data.site_xpos[second]))
            for first, second in self.loop_site_ids
        )

    def _update_contact_diagnostics(self) -> None:
        base_force = 0.0
        leg_force = 0.0
        self.last_ncon = int(self.data.ncon)
        for contact_index in range(self.data.ncon):
            contact = self.data.contact[contact_index]
            force = np.zeros(6, dtype=np.float64)
            mujoco.mj_contactForce(self.model, self.data, contact_index, force)
            contact_force = float(np.linalg.norm(force[:3]))
            body1 = int(self.model.geom_bodyid[contact.geom1])
            body2 = int(self.model.geom_bodyid[contact.geom2])
            bodies = {body1, body2}
            if self.base_body_id in bodies and 0 in bodies:
                base_force = max(base_force, contact_force)
            robot_body = body2 if body1 == 0 else body1 if body2 == 0 else -1
            if robot_body > 0:
                body_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, robot_body)
                if body_name not in WHEEL_BODIES and robot_body != self.base_body_id:
                    leg_force = max(leg_force, contact_force)

        current_loop_error = self.loop_error()
        self.max_loop_error = max(self.max_loop_error, current_loop_error)
        self.max_base_contact_force = max(self.max_base_contact_force, base_force)
        self.max_leg_contact_force = max(self.max_leg_contact_force, leg_force)
        if (
            self.sim_cfg.terminate_on_base_contact
            and base_force > self.sim_cfg.base_contact_force_threshold
        ):
            self.termination_reason = f"base_contact({base_force:.2f}N)"

    def reset(self) -> np.ndarray:
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[self.free_qpos_adr : self.free_qpos_adr + 7] = np.array(
            [0.0, 0.0, self.sim_cfg.root_height, 1.0, 0.0, 0.0, 0.0]
        )
        self.data.qvel[self.free_dof_adr : self.free_dof_adr + 6] = 0.0
        for name in STANDARD_WL_JOINTS:
            self.data.qpos[self.qpos_adrs[name]] = self.init_joint_pos[name]
            self.data.qvel[self.dof_adrs[name]] = 0.0
        self.last_action[:] = 0.0
        self.vmc.process_action(self.last_action)
        self.termination_reason = None
        self.max_loop_error = 0.0
        self.max_base_contact_force = 0.0
        self.max_leg_contact_force = 0.0
        mujoco.mj_forward(self.model, self.data)
        self._apply_vmc()
        initial_loop_error = self.loop_error()
        if initial_loop_error > 1.0e-4:
            raise RuntimeError(f"Initial closed-chain error is too large: {initial_loop_error:.6g} m")
        return self.obs_builder.reset(self._stack_frame())

    def step_policy(self, obs: np.ndarray) -> np.ndarray:
        self.last_action = self.vmc.process_action(self.policy(obs))
        for _ in range(self.sim_cfg.policy_decimation):
            self._apply_vmc()
            for _ in range(self.sim_cfg.vmc_substeps):
                mujoco.mj_step(self.model, self.data)
                if not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all():
                    self.termination_reason = "non_finite_state"
                    break
                self._update_contact_diagnostics()
                if self.termination_reason is not None:
                    break
            if self.termination_reason is not None:
                break
        return self.obs_builder.update(self._stack_frame())

    def base_state(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        angular_velocity_b, linear_velocity_b = ObservationBuilder.body_velocity(
            self.model, self.data, self.base_body_id
        )
        return (
            self.data.xpos[self.base_body_id].copy(),
            self.data.xquat[self.base_body_id].copy(),
            linear_velocity_b,
            angular_velocity_b,
        )

    @staticmethod
    def _log_header() -> list[str]:
        return [
            "time",
            "base_x",
            "base_y",
            "base_z",
            "base_qw",
            "base_qx",
            "base_qy",
            "base_qz",
            "base_vx_b",
            "base_vy_b",
            "base_vz_b",
            "base_wx_b",
            "base_wy_b",
            "base_wz_b",
            "cmd_vx",
            "cmd_vy",
            "cmd_yaw",
            "cmd_height",
            *[f"action_{index}" for index in range(6)],
            "theta_ref_l",
            "theta_ref_r",
            "l0_ref_l",
            "l0_ref_r",
            "wheel_ref_l",
            "wheel_ref_r",
            "theta_l",
            "theta_r",
            "theta_dot_l",
            "theta_dot_r",
            "l0_l",
            "l0_r",
            "l0_dot_l",
            "l0_dot_r",
            "wheel_vel_l",
            "wheel_vel_r",
            "force_l",
            "force_r",
            "torque_l",
            "torque_r",
            *[f"ctrl_{name}" for name in ("jAB", "jAG", "jIJ", "jIO", "right_wheel", "left_wheel")],
            *[f"q_{name}" for name in ACTIVE_JOINTS],
            *[f"qd_{name}" for name in ACTIVE_JOINTS],
            "ncon",
            "max_loop_error",
            "max_base_contact_force",
            "max_leg_contact_force",
            "termination_reason",
        ]

    def _log_row(self) -> list[object]:
        pos, quat, linear_velocity_b, angular_velocity_b = self.base_state()
        q = self._joint_pos()
        qd = self._joint_vel()
        return [
            self.data.time,
            *pos,
            *quat,
            *linear_velocity_b,
            *angular_velocity_b,
            self.command.lin_vel_x,
            self.command.lin_vel_y,
            self.command.ang_vel_z,
            self.command.pos_z,
            *self.last_action,
            *self.vmc.theta0_ref,
            *self.vmc.l0_ref,
            *self.vmc.wheel_vel_ref,
            *self.vmc.theta0,
            *self.vmc.theta0_dot,
            *self.vmc.l0,
            *self.vmc.l0_dot,
            *self.vmc.wheel_vel,
            *self.vmc.virtual_force,
            *self.vmc.virtual_torque,
            *self.last_controls,
            *[q[name] for name in ACTIVE_JOINTS],
            *[qd[name] for name in ACTIVE_JOINTS],
            self.last_ncon,
            self.max_loop_error,
            self.max_base_contact_force,
            self.max_leg_contact_force,
            self.termination_reason or "",
        ]

    def run(self, log_path: Path | None, viewer: bool) -> None:
        obs = self.reset()
        writer = None
        log_file = None
        if log_path is not None:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_file = log_path.open("w", newline="", encoding="utf-8")
            writer = csv.writer(log_file)
            writer.writerow(self._log_header())

        viewer_ctx = None
        if viewer:
            import mujoco.viewer

            viewer_ctx = mujoco.viewer.launch_passive(self.model, self.data)

        velocity_errors: list[float] = []
        yaw_errors: list[float] = []
        start_time = time.monotonic()
        try:
            while self.data.time < self.sim_cfg.duration and self.termination_reason is None:
                if viewer_ctx is not None and not viewer_ctx.is_running():
                    self.termination_reason = "viewer_closed"
                    break
                obs = self.step_policy(obs)
                _, _, linear_velocity_b, angular_velocity_b = self.base_state()
                velocity_errors.append(float(linear_velocity_b[0] - self.command.lin_vel_x))
                yaw_errors.append(float(angular_velocity_b[2] - self.command.ang_vel_z))
                if writer is not None:
                    writer.writerow(self._log_row())
                if viewer_ctx is not None:
                    viewer_ctx.sync()
                    target_wall_time = start_time + self.data.time
                    delay = target_wall_time - time.monotonic()
                    if delay > 0.0:
                        time.sleep(delay)
        finally:
            if viewer_ctx is not None:
                viewer_ctx.close()
            if log_file is not None:
                log_file.close()

        vx_rmse = math.sqrt(float(np.mean(np.square(velocity_errors)))) if velocity_errors else float("nan")
        yaw_rmse = math.sqrt(float(np.mean(np.square(yaw_errors)))) if yaw_errors else float("nan")
        pos, _, linear_velocity_b, _ = self.base_state()
        status = self.termination_reason or "duration_reached"
        print(
            "[RESULT] "
            f"status={status}, sim_time={self.data.time:.3f}s, "
            f"base_z={pos[2]:.3f}m, vx={linear_velocity_b[0]:.3f}m/s, "
            f"vx_rmse={vx_rmse:.3f}m/s, yaw_rmse={yaw_rmse:.3f}rad/s, "
            f"max_loop_error={self.max_loop_error:.3e}m, "
            f"max_base_contact={self.max_base_contact_force:.2f}N, "
            f"max_leg_contact={self.max_leg_contact_force:.2f}N"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a Standard WL PPO policy in the original closed-chain MuJoCo model.")
    parser.add_argument(
        "--policy",
        type=Path,
        default=None,
        help="Run directory, exported policy.pt, or raw model_*.pt checkpoint.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Checkpoint name/path when --policy is a run directory (for example model_2900.pt).",
    )
    parser.add_argument("--no-policy", action="store_true", help="Use zero actions for a VMC/model smoke test.")
    parser.add_argument("--mjcf", type=Path, default=STANDARD_WL_MJCF, help="Standard WL environment MJCF.")
    parser.add_argument("--duration", type=float, default=20.0, help="Simulation duration in seconds.")
    parser.add_argument("--vx", type=float, default=0.0, help="Commanded x velocity in m/s.")
    parser.add_argument("--vy", type=float, default=0.0, help="Commanded y velocity in m/s.")
    parser.add_argument("--yaw", type=float, default=0.0, help="Commanded yaw rate in rad/s.")
    parser.add_argument("--height", type=float, default=0.31, help="Commanded base height in metres.")
    parser.add_argument("--root-height", type=float, default=None, help="Override reset base height from metadata.")
    parser.add_argument("--total-mass", type=float, default=18.0, help="Target total vehicle mass in kg.")
    parser.add_argument("--physics-dt", type=float, default=0.001, help="MuJoCo internal physics step in seconds.")
    parser.add_argument("--num-policy-stacks", type=int, default=None, help="Override stack count from metadata.")
    parser.add_argument(
        "--terminate-on-base-contact",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Stop when base-floor contact exceeds the training termination threshold.",
    )
    parser.add_argument(
        "--mesh-wheel-collisions",
        action="store_true",
        help="Use source wheel meshes instead of the analytic cylinders used by the training USD.",
    )
    parser.add_argument(
        "--log",
        type=Path,
        default=Path("sim2sim/logs/standard_wl_mujoco.csv"),
        help="CSV diagnostic log path.",
    )
    parser.add_argument("--viewer", action="store_true", help="Open the MuJoCo passive viewer.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.no_policy:
        if args.policy is not None or args.checkpoint is not None:
            raise SystemExit("--no-policy cannot be combined with --policy or --checkpoint")
        policy_path = None
        run_dir = None
    elif args.policy is None:
        raise SystemExit("Provide --policy RUN_OR_CHECKPOINT, or use --no-policy for a smoke test")
    else:
        policy_path, run_dir = resolve_policy_source(args.policy, args.checkpoint)

    sim_cfg = StandardSimConfig(
        physics_dt=args.physics_dt,
        duration=args.duration,
        total_mass=args.total_mass,
        terminate_on_base_contact=args.terminate_on_base_contact,
        analytic_wheel_collisions=not args.mesh_wheel_collisions,
    )
    vmc_cfg = StandardVmcConfig()
    obs_cfg = StandardObsConfig()
    sim_cfg, vmc_cfg, obs_cfg, init_joint_pos = load_training_metadata(
        run_dir, sim_cfg, vmc_cfg, obs_cfg
    )
    if args.root_height is not None:
        sim_cfg = replace(sim_cfg, root_height=args.root_height)
    if args.num_policy_stacks is not None:
        obs_cfg = replace(obs_cfg, num_policy_stacks=args.num_policy_stacks)

    ratio = sim_cfg.vmc_dt / sim_cfg.physics_dt
    if not math.isclose(ratio, round(ratio), rel_tol=0.0, abs_tol=1.0e-9):
        raise SystemExit(
            f"vmc_dt/physics_dt must be an integer, got {sim_cfg.vmc_dt}/{sim_cfg.physics_dt}"
        )
    if sim_cfg.vmc_substeps < 1:
        raise SystemExit("physics_dt must not be larger than vmc_dt")

    policy = InferencePolicy(policy_path, run_dir)
    if policy.input_dim is not None and policy.input_dim != obs_cfg.policy_dim:
        raise SystemExit(
            f"Policy expects {policy.input_dim} observations, builder produces {obs_cfg.policy_dim}"
        )
    command = StandardCommand(args.vx, args.vy, args.yaw, args.height)

    if run_dir is not None:
        print(f"[INFO] Loaded training metadata from: {run_dir}")
    if policy_path is not None:
        print(f"[INFO] Policy: {policy_path} ({policy.backend})")
    print(
        "[INFO] Standard WL sim2sim: "
        f"physics_dt={sim_cfg.physics_dt}, vmc_dt={sim_cfg.vmc_dt}, "
        f"policy_dt={sim_cfg.policy_dt}, stacks={obs_cfg.num_policy_stacks}, "
        f"mass={sim_cfg.total_mass:.3f}kg, action_scale_vel={vmc_cfg.action_scale_vel}, "
        f"command_scale={obs_cfg.command_scale}, "
        f"wheel_collision={'cylinder' if sim_cfg.analytic_wheel_collisions else 'mesh'}"
    )

    runner = StandardWlMujocoRunner(
        args.mjcf,
        policy,
        command,
        sim_cfg,
        vmc_cfg,
        obs_cfg,
        init_joint_pos,
    )
    print(
        f"[INFO] MuJoCo model: nq={runner.model.nq}, nv={runner.model.nv}, "
        f"nu={runner.model.nu}, equality={runner.model.neq}, total_mass={runner.model_total_mass:.3f}kg"
    )
    runner.run(args.log, args.viewer)


if __name__ == "__main__":
    main()
