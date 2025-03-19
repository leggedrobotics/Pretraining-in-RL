# Copyright (c) 2022-2024, The IsaacLab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import blind

from . import agents
##
# Register Gym environments.
##

### blind ###

# flat

gym.register(
    id="P4RL-Pedipulation-Flat-Blind-Anymal-D-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": blind.PedipulationPositionBlindFlatEnvCfg,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.AnymalDFlatPPORunnerCfg,
    },
)

gym.register(
    id="P4RL-Pedipulation-Flat-Blind-Anymal-D-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": blind.PedipulationPositionBlindFlatEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.AnymalDFlatPPORunnerCfg,
    },
)

# rough

gym.register(
    id="P4RL-Pedipulation-Rough-Blind-Anymal-D-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": blind.PedipulationPositionBlindRoughEnvCfg,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.AnymalDRoughPPORunnerCfg,
    },
)

gym.register(
    id="P4RL-Pedipulation-Rough-Blind-Anymal-D-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": blind.PedipulationPositionBlindRoughEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.AnymalDRoughPPORunnerCfg,
    },
)

gym.register(
    id="P4RL-Pedipulation-Rough-Blind-On-Flat-Anymal-D-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": blind.PedipulationPositionBlindRoughOnFlatEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.AnymalDRoughPPORunnerCfg,
    },
)

### perceptive ###

# TODO: Implement perceptive environments