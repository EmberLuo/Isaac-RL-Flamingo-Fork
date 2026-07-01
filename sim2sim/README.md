# WL MuJoCo Sim2Sim

This directory runs the exported WL PPO policy in MuJoCo using the same high-level
interface as Isaac Lab:

- policy input: `stack_policy` frames plus the scaled velocity/height command
- action: `[theta0_L, l0_L, wheel_vel_L, theta0_R, l0_R, wheel_vel_R]`
- low-level control: the same VMC torque law as `mdp/vmc_action.py`

## Export policy

Run Isaac `play.py` once for the checkpoint you want to test. It writes:

```bash
logs/co_rl/WL_Flat_Stand_Drive/ppo/<run>/exported/policy.pt
```

## Smoke test without policy

```bash
python -m sim2sim.wl_mujoco_sim2sim --no-policy --duration 0.2
```

## Run policy

```bash
python -m sim2sim.wl_mujoco_sim2sim \
  --policy logs/co_rl/WL_Flat_Stand_Drive/ppo/<run> \
  --vx 0.5 \
  --yaw 0.0 \
  --height 0.25 \
  --duration 20 \
  --log sim2sim/logs/wl_mujoco.csv
```

Add `--viewer` to open MuJoCo's passive viewer. The `--policy` argument may be
the run directory, its `exported/` directory, or the `exported/policy.pt` file.

## Notes

- Only TorchScript `policy.pt` is supported by default; `onnxruntime` is not installed in the current environment.
- The runner generates a temporary URDF with a floating base, a fixed ground box, and MuJoCo-readable mesh paths.
- Observation order intentionally mirrors `wl_env/velocity_env_cfg.py`; the default input dimension is `70`.
- When a run directory can be inferred, `params/env.yaml` and `params/agent.yaml` are loaded so VMC gains,
  action scales, sim `dt`, decimation, and policy stack count match the policy's training run.
