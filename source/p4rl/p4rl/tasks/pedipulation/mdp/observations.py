# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

# from omni.isaac.contrib_tasks.pedipulation.assets.legged_robot import LeggedRobot
from isaaclab.managers import SceneEntityCfg
# from omni.isaac.orbit.sensors import RayCaster
from isaaclab.assets import RigidObject
from isaaclab.utils.math import quat_rotate_inverse
from isaaclab.terrains import TerrainImporter
from .commands.commands_cfg import FootPositionCommand
from isaaclab.managers.command_manager import CommandManager

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

def foot_tracking_commands(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """
        Commands are given to the policy in an inertial (non moving) frame, here the environment origin. To transform 
        that vector into the robot base frame, we first subtract the vector that points from the env_origin to the robot
        base and then rotate into the robot frame. The asset_cfg is defined in pedipulation environment config file.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    command_manager: CommandManager = env.command_manager
    command_w: torch.tensor = command_manager.get_command("foot_position") # commands given in world frame

    robot_base_pos_w: torch.tensor = asset.data.root_pos_w
    robot_base_quat_q: torch.tensor = asset.data.root_quat_w
    # Transform desired foot position from env_origin in current robot base frame 
    desired_foot_pos_b: torch.tensor = quat_rotate_inverse(robot_base_quat_q, command_w - robot_base_pos_w)
    return desired_foot_pos_b
