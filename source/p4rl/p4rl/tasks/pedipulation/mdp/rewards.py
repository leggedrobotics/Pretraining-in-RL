# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_rotate_inverse, quat_rotate  # , wrap_to_pi, yaw_quat
from isaaclab.assets import RigidObject
# from omni.isaac.contrib_tasks.pedipulation.assets.legged_robot import LeggedRobots
from .commands.foot_position_command import FootPositionCommand
from isaaclab.terrains import TerrainImporter
from isaaclab.managers.command_manager import CommandManager
from isaaclab.sensors import ContactSensor
from isaaclab.assets import Articulation

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from ..config.anymal_d.pedipulation_base import PedipulationPositionEnvCfg

def foot_tracking(env: ManagerBasedRLEnv, sigma: float, asset_cfg: SceneEntityCfg):
    """
        Commands are given in an inertial (non moving) frame, here the environment origin. To transform that vector into
        the robot base frame, we subtract the vector that points from the env_origin to the robot base. The asset_cfg is 
        defined in pedipulation environment config file.
    """
    env_cfg: PedipulationPositionEnvCfg = env.cfg
    asset: RigidObject = env.scene[asset_cfg.name]

    command_manager: CommandManager = env.command_manager  # commands given in world frame
    if asset_cfg.body_ids is None:
        raise ValueError("The body_ids of the robot are not defined in the environment config.")
    foot_pos_w = asset.data.body_state_w[:, asset_cfg.body_ids[env_cfg.foot_index], :3]
    desired_foot_pos_w = command_manager.get_command("foot_position")  # commands given in world frame  
    return torch.exp(-torch.norm(foot_pos_w - desired_foot_pos_w, dim=1) / sigma)
