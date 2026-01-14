# Isaac LAB for Flamingo (Fork)

[![IsaacSim](https://img.shields.io/badge/IsaacSim-5.1.0-silver.svg)](https://docs.omniverse.nvidia.com/isaacsim/latest/overview.html)
[![IsaacLab](https://img.shields.io/badge/Lab-2.3.0-silver)](https://isaac-sim.github.io/IsaacLab/)
[![Python](https://img.shields.io/badge/python-3.10-blue.svg)](https://docs.python.org/3/whatsnew/3.10.html)
[![Linux platform](https://img.shields.io/badge/platform-linux--64-orange.svg)](https://releases.ubuntu.com/20.04/)
[![License](https://img.shields.io/badge/license-MIT-yellow.svg)](https://opensource.org/license/mit)

> **Note**: This repository is forked from [jaykorea/Isaac-RL-Two-wheel-Legged-Bot](https://github.com/jaykorea/Isaac-RL-Two-wheel-Legged-Bot) and has been modified for personal use and experimentation.

## **✨ Key Features**
✔️ **Flamingo Robot Support**: Multiple Flamingo variants (rev.0.1.4, Edu v1, Light v1, 4W4L, Humanoid)  
✔️ **Stack Environment**: Observations can be stacked for temporal information  
✔️ **Constraint Manager**: [Constraints as Termination (CaT)](https://arxiv.org/abs/2403.18765) implementation  
✔️ **CoRL Framework**: PPO, SRMPPO, SAC, TQC, TACO algorithms support  
✔️ **Sim2Real Transfer**: Zero-shot transfer capabilities demonstrated  

## **🔧 My Modifications**
- ✅ **Updated to Isaac Sim 5.1.0 + Isaac Lab 2.3.0** (from 4.5 + 2.0.0)
- ✅ **API Migration**: Migrated deprecated APIs
  - `attach_yaw_only=True` → `ray_alignment="yaw"`
  - `quat_rotate_inverse` → `quat_apply_inverse`
- ✅ **System Configuration**: Fixed inotify watches limit issue for Isaac Sim 5.1.0
- ✅ **Documentation**: Added detailed configuration parameters and setup guide  

## Sim2Real - ZeroShot Transfer
<table>
    <td><img src="https://github.com/user-attachments/assets/bb14612c-85c2-43ce-a7df-8b09ee4d3f69" width="800" height="400"/></td>
</table>
<table>
  <tr>
    <td><img src="https://github.com/user-attachments/assets/8f9f990d-e8e9-400a-82b2-1131ff73f891" width="385" height="170"/></td>
    <td><img src="https://github.com/user-attachments/assets/93c6b187-4680-435e-800a-9e6d3d570d13" width="385" height="170"/></td>
  </tr>
  <tr>
    <td><img src="https://github.com/user-attachments/assets/9991ff73-5b3e-4d10-9b63-548197f18e54" width="385" height="170"/></td>
    <td><img src="https://github.com/user-attachments/assets/545fd258-1add-499a-8c62-520e113a951b" width="385" height="170"/></td>
  </tr>
</table>


## Isaac Lab Flamingo
<table>
  <tr>
    <td><img src="https://github.com/user-attachments/assets/0037889b-bab7-4686-a9a5-46ea9bbe6ac2" width="385" height="240"/></td>
    <td><img src="https://github.com/user-attachments/assets/16d8d025-7e57-479a-80d4-9cfef2cf9b6b" width="385" height="240"/></td>
  </tr>
</table>

## Sim 2 Sim framework - Lab to MuJoCo
<table>
  <tr>
    <td><img src="https://github.com/user-attachments/assets/edcc4077-e082-4fce-90a6-b10c94869aad" width="385" height="240"/></td>
    <td><img src="https://github.com/user-attachments/assets/df58b2db-00c6-4228-a953-eb605dee2797" width="385" height="240"/></td>
  </tr>
</table>

- Simulation to Simulation framework is available on sim2sim_onnx branch (Currently on migration update)
- You can simply inference trained policy (basically export as .onnx from isaac lab)

## 📋 Requirements
- **OS**: Ubuntu 20.04 or 22.04 (tested on 24.04)
- **Python**: 3.10
- **Isaac Sim**: 5.1.0
- **Isaac Lab**: 2.3.0
- **GPU**: NVIDIA GPU with CUDA support

## 🚀 Setup

### 1. System Configuration (Important for Isaac Sim 5.1.0)

Increase inotify watches limit to prevent file monitoring errors:

```bash
echo fs.inotify.max_user_watches=524288 | sudo tee -a /etc/sysctl.conf
sudo sysctl -p
```

### 2. Install Isaac Sim 5.1.0

Follow the official installation guide:
```
https://docs.omniverse.nvidia.com/isaacsim/latest/installation/index.html
```

### 3. Install Isaac Lab 2.3.0

```bash
git clone https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab
./isaaclab.sh --install
```

### 4. Install This Package

**i. Clone repository**
```bash
git clone https://github.com/EmberLuo/Isaac-RL-Flamingo-Fork
cd Isaac-RL-Flamingo-Fork
```

**ii. Install package**
```bash
conda activate Isaac-RL-Two-wheel-Legged-Bot  # or your Isaac Lab conda env
pip install -e .
```

**iii. Unzip USD assets**

Since git doesn't correctly upload '.usd' files, manually unzip the USD files:
```bash
# Example path: lab/flamingo/assets/data/Robots/Flamingo/flamingo_rev01_4_1/
cd lab/flamingo/assets/data/Robots/Flamingo/
# Unzip all .zip files containing USD assets
```

## 🎮 Usage

### Training

**Basic command:**
```bash
python scripts/co_rl/train.py --task {TASK_NAME} --algo {ALGORITHM} \
    --num_envs {NUM_ENVS} --headless \
    --num_policy_stacks {POLICY_STACK} --num_critic_stacks {CRITIC_STACK}
```

**Example - Velocity tracking (Flat terrain):**
```bash
python scripts/co_rl/train.py --task Isaac-Velocity-Flat-Flamingo-v1-ppo \
    --algo ppo --num_envs 4096 --headless \
    --num_policy_stacks 2 --num_critic_stacks 2
```

**Example - Position tracking (Rough terrain):**
```bash
python scripts/co_rl/train.py --task Isaac-Position-Rough-Flamingo-v1-ppo \
    --algo ppo --num_envs 4096 --headless \
    --num_policy_stacks 2 --num_critic_stacks 2 \
    --max_iterations 10000
```

### Evaluation/Play

**Basic command:**
```bash
python scripts/co_rl/play.py --task {TASK_NAME} --algo {ALGORITHM} \
    --num_envs 64 \
    --num_policy_stacks {POLICY_STACK} --num_critic_stacks {CRITIC_STACK} \
    --load_run {RUN_FOLDER} --plot False
```

**Example:**
```bash
python scripts/co_rl/play.py --task Isaac-Velocity-Flat-Flamingo-Play-v1-ppo \
    --algo ppo --num_envs 64 \
    --num_policy_stacks 2 --num_critic_stacks 2 \
    --load_run 2025-03-16_17-09-35 --plot False
```

## ⚙️ Configuration Parameters

### Default Values

| Parameter | Default Value | Description |
|-----------|--------------|-------------|
| `--num_envs` | 4096 | Number of parallel environments |
| `--num_policy_stacks` | 2 | Observation stack size for policy network |
| `--num_critic_stacks` | 2 | Observation stack size for critic network |
| `--max_iterations` | Varies by task | Maximum training iterations |
| `--algo` | ppo | Algorithm (ppo, srmppo, sac, tqc, taco) |

### Task-Specific Iterations

| Task Type | Default Iterations | Notes |
|-----------|-------------------|-------|
| **Velocity - Flat** | 5,000 | Stand Drive, Track Z, Jump, etc. |
| **Velocity - Rough** | 10,000 | Stand Drive, Stand Walk |
| **Position - Flat** | 10,000 | Position tracking |
| **Position - Rough** | 10,000 | Rough terrain position |
| **Humanoid - Flat** | 5,000 | Humanoid locomotion |
| **Humanoid - Rough** | 10,000 | Challenging terrain |
| **Constraint-based** | 15,000 | With CaT implementation |
| **Special (Back Flip)** | 50,000 | Complex acrobatic tasks |
| **SAC/TQC (Off-policy)** | 200,000 | Off-policy algorithms |

### Available Algorithms

- **PPO**: Proximal Policy Optimization (on-policy)
- **SRMPPO**: State Representation Model PPO (on-policy with recurrent module)
- **SAC**: Soft Actor-Critic (off-policy)
- **TQC**: Truncated Quantile Critics (off-policy)
- **TACO**: Temporal Action Chunking with Optimistic update (off-policy)

## 🤖 Available Tasks

### Manager-Based Tasks

**Velocity Tracking:**
- `Isaac-Velocity-Flat-Flamingo-v1-ppo` - Flat terrain velocity control
- `Isaac-Velocity-Rough-Flamingo-v1-ppo` - Rough terrain velocity control
- `Isaac-Velocity-Flat-Flamingo-Light-v1-ppo` - Light version
- `Isaac-Velocity-Flat-Flamingo-4W4L-v1-ppo` - 4-wheel-4-leg variant

**Position Tracking:**
- `Isaac-Position-Flat-Flamingo-v1-ppo` - Flat terrain position control
- `Isaac-Position-Rough-Flamingo-v1-ppo` - Rough terrain position control

**Humanoid:**
- `Isaac-Velocity-Flat-Humanoid-v1-ppo` - Humanoid locomotion
- `Isaac-Velocity-Rough-Humanoid-v1-ppo` - Rough terrain humanoid

### Constraint-Based Tasks
- `Isaac-Velocity-Flat-Flamingo-Constraint-v1-ppo` - With CaT constraints
- `Isaac-Velocity-Rough-Flamingo-Constraint-v1-ppo` - Rough terrain with constraints

## 📊 Training Tips

1. **Start with flat terrain** tasks to verify setup
2. **Use fewer environments** (`--num_envs 1024`) for initial testing
3. **Adjust stack numbers** based on task complexity:
   - Simple tasks: `--num_policy_stacks 1 --num_critic_stacks 1`
   - Complex tasks: `--num_policy_stacks 3 --num_critic_stacks 3`
4. **Monitor training** in logs directory: `logs/co_rl/{experiment_name}/`

## 📁 Project Structure

```
Isaac-RL-Two-wheel-Legged-Bot/
├── lab/flamingo/              # Main package
│   ├── assets/                # Robot URDF/USD files
│   ├── tasks/                 # Task definitions
│   │   ├── manager_based/     # Manager-based RL tasks
│   │   └── constraint_based/  # Constraint RL tasks
│   └── isaaclab/              # Extended Isaac Lab modules
├── scripts/co_rl/             # Training/evaluation scripts
│   ├── train.py               # Training script
│   ├── play.py                # Evaluation script
│   └── core/                  # Algorithm implementations
└── logs/                      # Training logs (generated)
```

## 🙏 Credits

This repository is based on:
- Original repository: [jaykorea/Isaac-RL-Two-wheel-Legged-Bot](https://github.com/jaykorea/Isaac-RL-Two-wheel-Legged-Bot)
- [Isaac Sim](https://docs.omniverse.nvidia.com/isaacsim/) by NVIDIA
- [Isaac Lab](https://isaac-sim.github.io/IsaacLab/) by NVIDIA
- [rsl_rl](https://github.com/leggedrobotics/rsl_rl) by Robotic Systems Lab, ETH Zurich

## 📄 License

This project is licensed under the MIT License - see the [LICENCE](LICENCE) file for details.
