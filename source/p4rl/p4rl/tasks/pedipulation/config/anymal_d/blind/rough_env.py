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
from isaaclab.terrains import TerrainGeneratorCfg, TerrainImporterCfg, HfRandomUniformTerrainCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from p4rl.tasks.pedipulation.config.anymal_d.pedipulation_base import PedipulationPositionEnvCfg, PedipulationPositionEnvCfg_PLAY
from p4rl.tasks.pedipulation.config.anymal_d.pedipulation_base import PedipulationBaseSceneCfg
from p4rl.tasks.pedipulation.config.anymal_d.blind.flat_env import PedipulationFlatSceneCfg
import p4rl.tasks.pedipulation.mdp as mdp


UNIFORM_ROUGH_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "random_rough": HfRandomUniformTerrainCfg(
            proportion=0.2, noise_range=(-0.04, 0.04), noise_step=0.005, downsampled_scale=0.2, border_width=0.25
        ),
    },
)

@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # -- task
    track_foot_position_xyz_exp = RewTerm(func=mdp.foot_tracking, weight=15, params={"asset_cfg": SceneEntityCfg("robot", body_names=".*FOOT"), "sigma": 0.8})
    # -- penalties
    dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-1.0e-4)
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-5.0e-6)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.025)
    dof_vel = RewTerm(func=mdp.joint_vel_l2, weight=-0.05)
    collisions = RewTerm(
        func=mdp.undesired_contacts,
        weight=-2.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*(THIGH|SHANK)"), "threshold": 0.1},
    )
    terminations = RewTerm(func=mdp.is_terminated, weight=-80.0)


@configclass
class PedipulationRoughSceneCfg(PedipulationBaseSceneCfg):
    """Configuration for the flat scene."""
    # Set the terrain to be a flat plane
    terrain: TerrainImporterCfg = TerrainImporterCfg(
            prim_path="/World/ground",
            terrain_type="generator",
            terrain_generator=UNIFORM_ROUGH_TERRAINS_CFG,
            max_init_terrain_level=5,
            collision_group=-1,
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="average",
                restitution_combine_mode="average",
                static_friction=1.0,
                dynamic_friction=1.0,
            ),
            debug_vis=False,
        )


@configclass
class PedipulationPositionBlindRoughEnvCfg(PedipulationPositionEnvCfg):
    """Configuration for the pedipulation foot position tracking environment in flat terrain."""

    scene: PedipulationRoughSceneCfg = PedipulationRoughSceneCfg() # overwrite the scene configuration from the parent class
    rewards: RewardsCfg = RewardsCfg()

    def __post_init__(self) -> None:
        super().__post_init__() # call parent post init (PedipulationPositionEnvCfg)

        self.sim.physics_material = self.scene.terrain.physics_material


@configclass
class PedipulationPositionBlindRoughEnvCfg_PLAY(PedipulationPositionBlindRoughEnvCfg, PedipulationPositionEnvCfg_PLAY):
    def __post_init__(self) -> None:
        # call post init of parents (first PedipulationPositionBlindRoughEnvCfg, then PedipulationPositionEnvCfg_PLAY)
        for parent in self.__class__.__bases__:
            parent.__post_init__(self)


@configclass
class PedipulationPositionBlindRoughOnFlatEnvCfg_PLAY(PedipulationPositionBlindRoughEnvCfg_PLAY):
    """Configuration to deploy the rough ground trained pedipulation policy on flat ground."""

    scene: PedipulationFlatSceneCfg = PedipulationFlatSceneCfg()
