# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.terrains import TerrainImporter
from isaaclab.managers import CommandManager
from .commands.foot_position_command import FootPositionCommand

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def foot_command_space(env: ManagerBasedRLEnv, env_ids: Sequence[int], tracking_threshhold: float) -> None:

    command_manager: CommandManager = env.command_manager
    command_term: FootPositionCommand = command_manager.get_term("foot_position")

    avg_tracking_error: torch.tensor = torch.mean(torch.linalg.norm(command_term.tracking_error_sum[env_ids, :], dim=1) / command_term.log_step_counter[env_ids])

    if (avg_tracking_error < tracking_threshhold):
        command_term.increase_difficulty()