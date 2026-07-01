# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import (
    agents,
    flat_env,
)

##
# Register Gym environments.
##

#########################################CoRL###################################################
################################################################################################
gym.register(
    id="Isaac-Velocity-Flat-WL-v1-ppo",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env.flat_env_stand_drive_cfg.WLFlatEnvCfg,
        "co_rl_cfg_entry_point": agents.co_rl_cfg.WLFlatPPORunnerCfg_Stand_Drive,
    },
)

gym.register(
    id="Isaac-Velocity-Flat-WL-v1-ppo-Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env.flat_env_stand_drive_cfg.WLFlatEnvCfg_PLAY,
        "co_rl_cfg_entry_point": agents.co_rl_cfg.WLFlatPPORunnerCfg_Stand_Drive,
    },
)
