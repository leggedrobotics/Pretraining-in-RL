from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import sample_uniform


if TYPE_CHECKING:
    from ..config.anymal_d.pedipulation_base import PedipulationPositionEnvCfg

def apply_external_force_torque_to_foot(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    force_range: tuple[float, float],
    torque_range: tuple[float, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Randomize the external forces and torques applied to the bodies.

    This function creates a set of random forces and torques sampled from the given ranges. The number of forces
    and torques is equal to the number of bodies times the number of environments. The forces and torques are
    applied to the bodies by calling ``asset.set_external_force_and_torque``. The forces and torques are only
    applied when ``asset.write_data_to_sim()`` is called.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    num_envs = env.scene.num_envs
    env_cfg: PedipulationPositionEnvCfg = env.cfg
    # resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(num_envs)
    if asset_cfg.body_ids is None:
        return
    foot_id = asset_cfg.body_ids[env_cfg.foot_index]

    # sample random forces and torques
    size = (len(env_ids), 1, 3)
    forces = sample_uniform(*force_range, size, asset.device)
    torques = sample_uniform(*torque_range, size, asset.device)
    # set the forces and torques into the buffers
    # note: these are only applied when you call: `asset.write_data_to_sim()`
    asset.set_external_force_and_torque(forces, torques, env_ids=env_ids, body_ids=foot_id)