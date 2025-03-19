# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from p4rl.tasks.pedipulation.config.anymal_d.pedipulation_base import PedipulationPositionEnvCfg, PedipulationPositionEnvCfg_PLAY
from p4rl.tasks.pedipulation.config.anymal_d.pedipulation_base import PedipulationBaseSceneCfg
import p4rl.tasks.pedipulation.mdp as mdp


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # -- task
    track_foot_position_xyz_exp = RewTerm(func=mdp.foot_tracking, weight=15, params={"asset_cfg": SceneEntityCfg("robot", body_names=".*FOOT"), "sigma": 0.8})
    # -- penalties
    dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-2.0e-5)
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-5e-6)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    dof_vel = RewTerm(func=mdp.joint_vel_l2, weight=-0.05)
    collisions = RewTerm(
        func=mdp.undesired_contacts,
        weight=-2.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*(THIGH|SHANK|RF_FOOT)"), "threshold": 0.1},
    )
    terminations = RewTerm(func=mdp.is_terminated, weight=-80.0)


@configclass
class PedipulationFlatSceneCfg(PedipulationBaseSceneCfg):
    """Configuration for the flat scene."""
    # Set the terrain to be a flat plane
    terrain: TerrainImporterCfg = TerrainImporterCfg(
            prim_path="/World/ground",
            terrain_type="plane", 
            terrain_generator=None,
            max_init_terrain_level=5,
            collision_group=-1,
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="average",
                restitution_combine_mode="average",
                static_friction=1.0,
                dynamic_friction=1.0),
            debug_vis=False)


@configclass
class PedipulationPositionBlindFlatEnvCfg(PedipulationPositionEnvCfg):
    """Configuration for the pedipulation foot position tracking environment in flat terrain."""

    # We inherit from the pedipulation base environment configuration and modify as needed
    scene: PedipulationFlatSceneCfg = PedipulationFlatSceneCfg()
    rewards: RewardsCfg = RewardsCfg()

    def __post_init__(self) -> None:
        super().__post_init__() # call parent post init (PedipulationPositionEnvCfg)

        self.sim.physics_material = self.scene.terrain.physics_material


@configclass
class PedipulationPositionBlindFlatEnvCfg_PLAY(PedipulationPositionBlindFlatEnvCfg, PedipulationPositionEnvCfg_PLAY):
    def __post_init__(self) -> None:
        # call post init of parents (first PedipulationPositionBlindFlatEnvCfg, then PedipulationPositionEnvCfg_PLAY)
        for parent in self.__class__.__bases__:
            parent.__post_init__(self)
