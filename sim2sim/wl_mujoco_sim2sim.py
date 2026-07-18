#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import tempfile
import time
import xml.etree.ElementTree as ET
from collections import deque
from dataclasses import fields, replace
from pathlib import Path

import mujoco
import numpy as np
import torch
import yaml

from sim2sim.wl_config import Command, ObsConfig, SimConfig, VmcConfig, WL_URDF


JOINT_NAMES = ("lf0_Joint", "lf1_Joint", "l_wheel_Joint", "rf0_Joint", "rf1_Joint", "r_wheel_Joint")
INIT_JOINT_POS = {
    "lf0_Joint": 0.5,
    "lf1_Joint": 0.35,
    "l_wheel_Joint": 0.0,
    "rf0_Joint": -0.5,
    "rf1_Joint": -0.35,
    "r_wheel_Joint": 0.0,
}


def _id(model: mujoco.MjModel, obj_type: mujoco.mjtObj, name: str) -> int:
    obj_id = mujoco.mj_name2id(model, obj_type, name)
    if obj_id < 0:
        raise RuntimeError(f"MuJoCo object not found: {name}")
    return obj_id


def _make_mujoco_urdf(src_urdf: Path, sim_cfg: SimConfig) -> tempfile.TemporaryDirectory[str]:
    """Create a temporary URDF that MuJoCo can load with a free base and ground."""
    tmp = tempfile.TemporaryDirectory(prefix="wl_mujoco_")
    tmp_path = Path(tmp.name)

    tree = ET.parse(src_urdf)
    root = tree.getroot()

    world = ET.Element("link", {"name": "world"})
    collision = ET.SubElement(world, "collision", {"name": "ground_collision"})
    ET.SubElement(collision, "origin", {"xyz": f"0 0 {-sim_cfg.ground_size[2] / 2.0}", "rpy": "0 0 0"})
    geometry = ET.SubElement(collision, "geometry")
    ET.SubElement(geometry, "box", {"size": " ".join(str(v) for v in sim_cfg.ground_size)})

    floating = ET.Element("joint", {"name": "world_to_base", "type": "floating"})
    ET.SubElement(floating, "parent", {"link": "world"})
    ET.SubElement(floating, "child", {"link": "base_link"})
    ET.SubElement(floating, "origin", {"xyz": "0 0 0", "rpy": "0 0 0"})

    root.insert(0, floating)
    root.insert(0, world)

    for idx, collision in enumerate(root.findall(".//collision")):
        if not collision.attrib.get("name"):
            collision.attrib["name"] = f"collision_{idx}"

    for mesh in root.findall(".//mesh"):
        mesh_path = (src_urdf.parent / mesh.attrib["filename"]).resolve()
        local_mesh = tmp_path / mesh_path.name
        if not local_mesh.exists():
            os.symlink(mesh_path, local_mesh)
        mesh.attrib["filename"] = mesh_path.name

    out_urdf = tmp_path / "wl_mujoco.urdf"
    tree.write(out_urdf, encoding="utf-8", xml_declaration=True)
    return tmp


class TorchPolicy:
    def __init__(self, path: Path | None):
        self.path = path
        self.model = None
        if path is not None:
            if not path.exists():
                raise FileNotFoundError(path)
            self.model = torch.jit.load(str(path), map_location="cpu")
            self.model.eval()
            if hasattr(self.model, "reset"):
                self.model.reset()

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        if self.model is None:
            return np.zeros(6, dtype=np.float32)
        with torch.inference_mode():
            obs_t = torch.from_numpy(obs.astype(np.float32, copy=False)).unsqueeze(0)
            action = self.model(obs_t).squeeze(0).cpu().numpy()
        return np.clip(action, -1.0, 1.0).astype(np.float32)


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


def resolve_policy_path(path: Path | None) -> tuple[Path | None, Path | None]:
    if path is None:
        return None, None
    path = path.expanduser().resolve()
    if path.is_dir():
        candidates = [(path / "policy.pt", path.parent if path.name == "exported" else None), (path / "exported/policy.pt", path)]
        for candidate in candidates:
            if candidate[0].exists():
                return candidate
        raise FileNotFoundError(f"No policy.pt found in {path} or {path / 'exported'}")
    run_dir = path.parent.parent if path.name == "policy.pt" and path.parent.name == "exported" else None
    return path, run_dir


def load_training_metadata(
    run_dir: Path | None, sim_cfg: SimConfig, vmc_cfg: VmcConfig, obs_cfg: ObsConfig
) -> tuple[SimConfig, VmcConfig, ObsConfig]:
    if run_dir is None:
        return sim_cfg, vmc_cfg, obs_cfg

    env_yaml = run_dir / "params/env.yaml"
    if env_yaml.exists():
        env_cfg = _load_isaac_yaml(env_yaml)
        sim_cfg = replace(
            sim_cfg,
            sim_dt=float(env_cfg.get("sim", {}).get("dt", sim_cfg.sim_dt)),
            decimation=int(env_cfg.get("decimation", sim_cfg.decimation)),
        )
        vmc_data = env_cfg.get("actions", {}).get("vmc", {})
        vmc_updates = {}
        for field in fields(VmcConfig):
            if field.name in vmc_data:
                vmc_updates[field.name] = float(vmc_data[field.name])
        if vmc_updates:
            vmc_cfg = replace(vmc_cfg, **vmc_updates)

    agent_yaml = run_dir / "params/agent.yaml"
    if agent_yaml.exists():
        agent_cfg = _load_isaac_yaml(agent_yaml)
        if "num_policy_stacks" in agent_cfg:
            obs_cfg = replace(obs_cfg, num_policy_stacks=int(agent_cfg["num_policy_stacks"]))

    return sim_cfg, vmc_cfg, obs_cfg


class WlVmc:
    def __init__(self, cfg: VmcConfig):
        self.cfg = cfg
        self.theta0_ref = np.zeros(2, dtype=np.float64)
        self.l0_ref = np.full(2, cfg.l0_offset, dtype=np.float64)
        self.wheel_vel_ref = np.zeros(2, dtype=np.float64)
        self.l0 = np.zeros(2, dtype=np.float64)
        self.theta0 = np.zeros(2, dtype=np.float64)
        self.l0_dot = np.zeros(2, dtype=np.float64)
        self.theta0_dot = np.zeros(2, dtype=np.float64)

    def process_action(self, action: np.ndarray) -> np.ndarray:
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        self.theta0_ref = np.array([action[0], action[3]]) * self.cfg.action_scale_theta
        self.l0_ref = np.array([action[1], action[4]]) * self.cfg.action_scale_l0 + self.cfg.l0_offset
        self.wheel_vel_ref = np.array([action[2], action[5]]) * self.cfg.action_scale_vel
        return action.astype(np.float32)

    def _forward_kinematics(self, theta1: np.ndarray, theta2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        end_x = self.cfg.offset + self.cfg.l1 * np.cos(theta1) + self.cfg.l2 * np.cos(theta1 + theta2)
        end_y = self.cfg.l1 * np.sin(theta1) + self.cfg.l2 * np.sin(theta1 + theta2)
        l0 = np.sqrt(end_x**2 + end_y**2)
        theta0 = np.arctan2(end_y, end_x) - np.pi / 2.0
        return l0, theta0

    def _vmc(
        self, theta1: np.ndarray, theta2: np.ndarray, l0: np.ndarray, theta0: np.ndarray, force: np.ndarray, torque: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        theta0v = theta0 + np.pi / 2.0
        t11 = self.cfg.l1 * np.sin(theta0v - theta1) - self.cfg.l2 * np.sin(theta1 + theta2 - theta0v)
        # Upstream Wheel-Legged-Gym formula (kept for reference; its l1 sign is incorrect):
        # t12 = (self.cfg.l1 * np.cos(theta0v - theta1) - self.cfg.l2 * np.cos(theta1 + theta2 - theta0v)) / l0
        t12 = (-self.cfg.l1 * np.cos(theta0v - theta1) - self.cfg.l2 * np.cos(theta1 + theta2 - theta0v)) / l0
        # t12 = -d(theta0)/d(theta1): the l1 term is NEGATIVE (must match vmc_action.py).
        t21 = -self.cfg.l2 * np.sin(theta1 + theta2 - theta0v)
        t22 = (-self.cfg.l2 * np.cos(theta1 + theta2 - theta0v)) / l0
        return t11 * force - t12 * torque, t21 * force - t22 * torque

    def compute_efforts(self, q: dict[str, float], qd: dict[str, float]) -> np.ndarray:
        theta1 = np.array([q["lf0_Joint"], -q["rf0_Joint"]])
        theta2 = np.array([q["lf1_Joint"] + np.pi / 2.0, -q["rf1_Joint"] + np.pi / 2.0])
        theta1_dot = np.array([qd["lf0_Joint"], -qd["rf0_Joint"]])
        theta2_dot = np.array([qd["lf1_Joint"], -qd["rf1_Joint"]])

        self.l0, self.theta0 = self._forward_kinematics(theta1, theta2)
        dt = 1.0e-3
        l0_t, theta0_t = self._forward_kinematics(theta1 + theta1_dot * dt, theta2 + theta2_dot * dt)
        self.l0_dot = (l0_t - self.l0) / dt
        self.theta0_dot = (theta0_t - self.theta0) / dt

        torque_leg = self.cfg.kp_theta * (self.theta0_ref - self.theta0) - self.cfg.kd_theta * self.theta0_dot
        force_leg = self.cfg.kp_l0 * (self.l0_ref - self.l0) - self.cfg.kd_l0 * self.l0_dot
        wheel_vel = np.array([qd["l_wheel_Joint"], qd["r_wheel_Joint"]])
        torque_wheel = self.cfg.wheel_damping * (self.wheel_vel_ref - wheel_vel)
        t1, t2 = self._vmc(theta1, theta2, self.l0, self.theta0, force_leg + self.cfg.feedforward_force, torque_leg)

        efforts = np.array([t1[0], t2[0], torque_wheel[0], -t1[1], -t2[1], torque_wheel[1]], dtype=np.float64)
        limits = np.array(
            [
                self.cfg.leg_effort_limit,
                self.cfg.leg_effort_limit,
                self.cfg.wheel_effort_limit,
                self.cfg.leg_effort_limit,
                self.cfg.leg_effort_limit,
                self.cfg.wheel_effort_limit,
            ]
        )
        return np.clip(efforts, -limits, limits)


class ObservationBuilder:
    def __init__(self, obs_cfg: ObsConfig, command: Command):
        self.cfg = obs_cfg
        self.command = command
        self.frames: deque[np.ndarray] = deque(maxlen=obs_cfg.total_frames)

    @staticmethod
    def _projected_gravity(model: mujoco.MjModel, data: mujoco.MjData, body_id: int) -> np.ndarray:
        quat = data.xquat[body_id]
        rot = np.empty(9, dtype=np.float64)
        mujoco.mju_quat2Mat(rot, quat)
        rot = rot.reshape(3, 3)
        return rot.T @ np.array([0.0, 0.0, -1.0])

    @staticmethod
    def _body_velocity(model: mujoco.MjModel, data: mujoco.MjData, body_id: int) -> tuple[np.ndarray, np.ndarray]:
        vel = np.zeros(6, dtype=np.float64)
        mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, body_id, vel, 1)
        return vel[:3].copy(), vel[3:].copy()

    def stack_obs(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        base_body_id: int,
        vmc: WlVmc,
        joint_vel: dict[str, float],
        last_action: np.ndarray,
    ) -> np.ndarray:
        ang_vel_b, _ = self._body_velocity(model, data, base_body_id)
        frame = np.concatenate(
            [
                ang_vel_b * 0.25,
                self._projected_gravity(model, data, base_body_id),
                vmc.theta0,
                vmc.theta0_dot * 0.05,
                vmc.l0 * 5.0,
                vmc.l0_dot * 0.25,
                np.array([joint_vel["l_wheel_Joint"], joint_vel["r_wheel_Joint"]]) * 0.05,
                last_action,
            ]
        ).astype(np.float32)
        if frame.shape[0] != self.cfg.stack_dim:
            raise RuntimeError(f"Unexpected stack obs dim: {frame.shape[0]} != {self.cfg.stack_dim}")
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
        command = np.asarray(self.command.scaled(), dtype=np.float32)
        return np.concatenate([*self.frames, command]).astype(np.float32)


class WlMujocoRunner:
    def __init__(self, policy_path: Path | None, command: Command, sim_cfg: SimConfig, vmc_cfg: VmcConfig, obs_cfg: ObsConfig):
        self.sim_cfg = sim_cfg
        self.command = command
        self.vmc = WlVmc(vmc_cfg)
        self.policy = TorchPolicy(policy_path)
        self.obs_builder = ObservationBuilder(obs_cfg, command)
        self._tmp_urdf = _make_mujoco_urdf(WL_URDF, sim_cfg)
        self.model = mujoco.MjModel.from_xml_path(str(Path(self._tmp_urdf.name) / "wl_mujoco.urdf"))
        self.model.opt.timestep = sim_cfg.sim_dt
        self.data = mujoco.MjData(self.model)
        self.base_body_id = _id(self.model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
        self.joint_ids = {name: _id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in JOINT_NAMES}
        self.qpos_adrs = {name: int(self.model.jnt_qposadr[jid]) for name, jid in self.joint_ids.items()}
        self.dof_adrs = {name: int(self.model.jnt_dofadr[jid]) for name, jid in self.joint_ids.items()}
        self.free_qpos_adr = int(self.model.jnt_qposadr[_id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "world_to_base")])
        self.free_dof_adr = int(self.model.jnt_dofadr[_id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "world_to_base")])
        self.last_action = np.zeros(6, dtype=np.float32)

        for geom_id in range(self.model.ngeom):
            self.model.geom_friction[geom_id] = np.asarray(sim_cfg.ground_friction)
        self._configure_collision_masks()

    def _configure_collision_masks(self) -> None:
        """Match Isaac's enabled_self_collisions=False while preserving ground contact."""
        for geom_id in range(self.model.ngeom):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
            if name == "ground_collision":
                self.model.geom_contype[geom_id] = 1
                self.model.geom_conaffinity[geom_id] = 2
            else:
                self.model.geom_contype[geom_id] = 2
                self.model.geom_conaffinity[geom_id] = 1

    def close(self) -> None:
        self._tmp_urdf.cleanup()

    def reset(self) -> np.ndarray:
        mujoco.mj_resetData(self.model, self.data)
        qadr = self.free_qpos_adr
        self.data.qpos[qadr : qadr + 7] = np.array([0.0, 0.0, self.sim_cfg.root_height, 1.0, 0.0, 0.0, 0.0])
        self.data.qvel[self.free_dof_adr : self.free_dof_adr + 6] = 0.0
        for name, value in INIT_JOINT_POS.items():
            self.data.qpos[self.qpos_adrs[name]] = value
            self.data.qvel[self.dof_adrs[name]] = 0.0
        self.last_action[:] = 0.0
        self.vmc.process_action(self.last_action)
        mujoco.mj_forward(self.model, self.data)
        self._apply_vmc()
        frame = self._stack_frame()
        return self.obs_builder.reset(frame)

    def _joint_pos(self) -> dict[str, float]:
        return {name: float(self.data.qpos[adr]) for name, adr in self.qpos_adrs.items()}

    def _joint_vel(self) -> dict[str, float]:
        return {name: float(self.data.qvel[adr]) for name, adr in self.dof_adrs.items()}

    def _apply_vmc(self) -> np.ndarray:
        efforts = self.vmc.compute_efforts(self._joint_pos(), self._joint_vel())
        self.data.qfrc_applied[:] = 0.0
        for effort, name in zip(efforts, JOINT_NAMES, strict=True):
            self.data.qfrc_applied[self.dof_adrs[name]] = effort
        return efforts

    def _stack_frame(self) -> np.ndarray:
        return self.obs_builder.stack_obs(
            self.model, self.data, self.base_body_id, self.vmc, self._joint_vel(), self.last_action
        )

    def _base_state(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        qadr = self.free_qpos_adr
        pos = self.data.qpos[qadr : qadr + 3].copy()
        quat = self.data.qpos[qadr + 3 : qadr + 7].copy()
        _, lin_vel_b = ObservationBuilder._body_velocity(self.model, self.data, self.base_body_id)
        return pos, quat, lin_vel_b

    def step_control(self, obs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        self.last_action = self.vmc.process_action(self.policy(obs))
        efforts = np.zeros(6, dtype=np.float64)
        for _ in range(self.sim_cfg.decimation):
            efforts = self._apply_vmc()
            mujoco.mj_step(self.model, self.data)
        next_frame = self._stack_frame()
        return self.obs_builder.update(next_frame), efforts

    def run(self, log_path: Path | None = None, viewer: bool = False) -> None:
        obs = self.reset()
        steps = int(self.sim_cfg.duration / self.sim_cfg.control_dt)
        writer = None
        log_file = None
        if log_path is not None:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_file = log_path.open("w", newline="")
            writer = csv.writer(log_file)
            writer.writerow(
                [
                    "time",
                    "base_x",
                    "base_y",
                    "base_z",
                    "base_vx_b",
                    "base_vy_b",
                    "base_vz_b",
                    "cmd_vx",
                    "cmd_yaw",
                    *[f"action_{i}" for i in range(6)],
                    *[f"effort_{name}" for name in JOINT_NAMES],
                    "theta0_l",
                    "theta0_r",
                    "l0_l",
                    "l0_r",
                ]
            )

        viewer_ctx = None
        if viewer:
            import mujoco.viewer

            viewer_ctx = mujoco.viewer.launch_passive(self.model, self.data)

        try:
            start = time.time()
            for step in range(steps):
                obs, efforts = self.step_control(obs)
                sim_time = (step + 1) * self.sim_cfg.control_dt
                if writer is not None:
                    pos, _, lin_vel_b = self._base_state()
                    writer.writerow(
                        [
                            sim_time,
                            *pos,
                            *lin_vel_b,
                            self.command.lin_vel_x,
                            self.command.ang_vel_z,
                            *self.last_action.tolist(),
                            *efforts.tolist(),
                            *self.vmc.theta0.tolist(),
                            *self.vmc.l0.tolist(),
                        ]
                    )
                if viewer_ctx is not None:
                    viewer_ctx.sync()
                    wall_target = start + sim_time
                    if wall_target > time.time():
                        time.sleep(wall_target - time.time())
        finally:
            if viewer_ctx is not None:
                viewer_ctx.close()
            if log_file is not None:
                log_file.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run WL TorchScript policy in MuJoCo sim2sim.")
    parser.add_argument(
        "--policy",
        type=Path,
        default=None,
        help="Path to exported policy.pt, an exported/ directory, or a run directory. Omit with --no-policy.",
    )
    parser.add_argument("--no-policy", action="store_true", help="Run zero policy actions for smoke testing.")
    parser.add_argument("--duration", type=float, default=20.0, help="Simulation duration in seconds.")
    parser.add_argument("--vx", type=float, default=0.0, help="Commanded x velocity in m/s.")
    parser.add_argument("--vy", type=float, default=0.0, help="Commanded y velocity in m/s.")
    parser.add_argument("--yaw", type=float, default=0.0, help="Commanded yaw rate in rad/s.")
    parser.add_argument("--height", type=float, default=0.25, help="Commanded base height.")
    parser.add_argument("--num-policy-stacks", type=int, default=None, help="Override policy stack count from metadata.")
    parser.add_argument("--log", type=Path, default=Path("sim2sim/logs/wl_mujoco.csv"), help="CSV log path.")
    parser.add_argument("--viewer", action="store_true", help="Open MuJoCo passive viewer.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.no_policy:
        policy_path = None
        run_dir = None
    elif args.policy is not None:
        policy_path, run_dir = resolve_policy_path(args.policy)
    else:
        raise SystemExit("Provide --policy path/to/exported/policy.pt, or use --no-policy for a smoke test.")

    sim_cfg = SimConfig(duration=args.duration)
    vmc_cfg = VmcConfig()
    command = Command(args.vx, args.vy, args.yaw, args.height)
    obs_cfg = ObsConfig()
    sim_cfg, vmc_cfg, obs_cfg = load_training_metadata(run_dir, sim_cfg, vmc_cfg, obs_cfg)
    if args.num_policy_stacks is not None:
        obs_cfg = replace(obs_cfg, num_policy_stacks=args.num_policy_stacks)
    if obs_cfg.policy_dim != 70:
        print(f"[WARN] policy obs dim is {obs_cfg.policy_dim}; current WL PPO exports normally expect 70.")

    if run_dir is not None:
        print(f"[INFO] Loaded training metadata from: {run_dir}")
    print(
        "[INFO] Sim2sim config: "
        f"dt={sim_cfg.sim_dt}, decimation={sim_cfg.decimation}, stacks={obs_cfg.num_policy_stacks}, "
        f"action_scale_vel={vmc_cfg.action_scale_vel}, kd_theta={vmc_cfg.kd_theta}, "
        f"wheel_damping={vmc_cfg.wheel_damping}"
    )

    runner = WlMujocoRunner(policy_path, command, sim_cfg, vmc_cfg, obs_cfg)
    try:
        runner.run(args.log, args.viewer)
    finally:
        runner.close()


if __name__ == "__main__":
    main()
