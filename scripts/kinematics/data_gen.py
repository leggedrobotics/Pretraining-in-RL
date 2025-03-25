# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script demonstrates the environment concept that combines a scene with an action,
observation and randomization manager for a quadruped robot.

A locomotion policy is loaded and used to control the robot. This shows how to use the
environment with a policy.
"""

from __future__ import annotations

"""Launch Isaac Sim Simulator first."""


import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="This script demonstrates how to use the concept of an Environment.")
parser.add_argument("--num_envs", type=int, default=2000, help="Number of environments to spawn.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""
import os
import torch
import numpy as np

# from omni.isaac.orbit_assets import ORBIT_ASSETS_DATA_DIR
# from omni.isaac.orbit.envs import BaseEnv, BaseEnvCfg
# import omni.isaac.orbit_tasks.loco_manipulation.wbc_teacher_student.mdp as mdp

from isaaclab.utils.math import subtract_frame_transforms
from isaaclab.terrains import TerrainImporterCfg
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedEnvCfg, ManagerBasedEnv
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, ContactSensor
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab_assets.robots.anymal import ANYDRIVE_4_MLP_ACTUATOR_CFG
import os
from isaaclab.markers import VisualizationMarkersCfg, VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG
from einops import rearrange, repeat, reduce
import h5py
from tqdm import tqdm


from isaacsim.core.utils.viewports import set_camera_view



# from p4rl import P4RL_EXT_DIR # cause "can not import from partially initialized module" error
P4RL_EXT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../.."))
import p4rl.tasks.pedipulation.mdp as mdp
##
# Pre-defined configs
##
from p4rl.tasks.pedipulation.config.anymal_d.pedipulation_base import ANYMAL_D_CFG  # isort: skip

# JOINT_NAMES = ["base", "*HIP", "*THIGH", "*SHANK", "*FOOT"]

BODY_NAMES = ['base', 'LF_HIP', 'LF_THIGH', 'LF_SHANK', 'LF_FOOT', 'LH_HIP', 'LH_THIGH', 'LH_SHANK', 'LH_FOOT', 'RF_HIP', 'RF_THIGH', 'RF_SHANK', 'RF_FOOT', 'RH_HIP', 'RH_THIGH', 'RH_SHANK', 'RH_FOOT']
JOINT_NAMES = ['LF_HAA', 'LH_HAA', 'RF_HAA', 'RH_HAA', 'LF_HFE', 'LH_HFE', 'RF_HFE', 'RH_HFE', 'LF_KFE', 'LH_KFE', 'RF_KFE', 'RH_KFE']



def obs_undesired_contacts(env: ManagerBasedEnvCfg, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize undesired contacts as the number of violations that are above a threshold."""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # check if contact force is above threshold
    contact_forces_norm = torch.norm(contact_sensor.data.net_forces_w_history[:, 0], dim=-1) # [num_envs, num_bodies]
    return contact_forces_norm # [num_envs, num_bodies]
##
# Scene definition
##


@configclass
class MySceneCfg(InteractiveSceneCfg):
    """Example scene configuration."""

    # add terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        terrain_generator=None,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        debug_vis=False,
    )

    # add robot
    robot: ArticulationCfg = ANYMAL_D_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    robot.init_state.pos = (0, 0, 2.0)

    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )

    # contact sensor
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=5, track_air_time=True)


##
# MDP settings
##


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""
    leg_joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=[".*HAA", ".*HFE", ".*KFE"], scale=0.5, use_default_offset=True
    )                


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        undesired_contacts = ObsTerm(
            func=obs_undesired_contacts,
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=BODY_NAMES), "threshold": 100.0}, #TODO what is the good threshold for collision detection?
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class RandomizationCfg:
    """Configuration for randomization."""


##
# Environment configuration
##

@configclass
class QuadrupedEnvCfg(ManagerBasedEnvCfg):
    """Configuration for the locomotion velocity-tracking environment."""

    # Scene settings
    scene: MySceneCfg = MySceneCfg(num_envs=10, env_spacing=2.5, replicate_physics=True)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    randomization: RandomizationCfg = RandomizationCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 4
        self.episode_length_s = 20.0
        # simulation settings
        self.sim.dt = 0.005
        # update sensor update periods


def main():
    """Main function."""

    vis_size = 0.1
    visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/CartesionSpaceMarkers")
    visualizer_cfg.markers["frame"].scale = (vis_size, vis_size, vis_size)
    marker = VisualizationMarkers(visualizer_cfg)

    # setup base environment
    env = ManagerBasedEnv(cfg=QuadrupedEnvCfg(scene=MySceneCfg(num_envs=args_cli.num_envs, env_spacing=2.5, replicate_physics=False)))
    obs, _ = env.reset()

    assert env.num_envs == 1, "This script only supports a single environment."

    set_camera_view(eye=[1.5, 1.5, 4.0], target=[0, 0, 2.5], camera_prim_path="/OmniverseKit_Persp")


    # extracted joint limits from the USD file for the original locomotion task.
    # the USD file for pedipulation does not include the joint limits. 

    joint_limits = torch.tensor([[-0.7854,  0.6109],
                                [-0.7854,  0.6109],
                                [-0.6109,  0.7854],
                                [-0.6109,  0.7854],
                                [-9.4248,  9.4248],
                                [-9.4248,  9.4248],
                                [-9.4248,  9.4248],
                                [-9.4248,  9.4248],
                                [-9.4248,  9.4248],
                                [-9.4248,  9.4248],
                                [-9.4248,  9.4248],
                                [-9.4248,  9.4248]], device='cuda:0') 
 
    soft_joint_limits = torch.tensor([[-0.7505,  0.5760],
            [-0.7505,  0.5760],
            [-0.5760,  0.7505],
            [-0.5760,  0.7505],
            [-8.9535,  8.9535],
            [-8.9535,  8.9535],
            [-8.9535,  8.9535],
            [-8.9535,  8.9535],
            [-8.9535,  8.9535],
            [-8.9535,  8.9535],
            [-8.9535,  8.9535],
            [-8.9535,  8.9535]], device='cuda:0')
    # joint_names = ['LF_HAA', 'LH_HAA', 'RF_HAA', 'RH_HAA', 'LF_HFE', 'LH_HFE', 'RF_HFE', 'RH_HFE', 'LF_KFE', 'LH_KFE', 'RF_KFE', 'RH_KFE']
    
    # uniformly sample 10 samples from the joint limits
    num_samples = 1000000
    joint_pos_samples = torch.rand((num_samples, 12), device='cuda:0') * (joint_limits[:, 1] - joint_limits[:, 0]) + joint_limits[:, 0] # [10, 12]

    num_samples = joint_pos_samples.shape[0]

    joint_pos_valid = []
    link_states_valid = []

    # simulate physics
    count = 0
    sample_idx = 0

    self_collision_count = 0

    while simulation_app.is_running():

        # if count%1000 == 0:
        # if True:
        for sample_idx in tqdm(range(num_samples)):
                
            with torch.inference_mode():

                # if sample_idx >= num_samples:
                #     break
                
                obs, _ = env.reset()
                joint_pos_sample = joint_pos_samples[sample_idx].unsqueeze(0)

                robot = env.scene.articulations["robot"]

                # root_state = robot.data.default_root_state.clone()
                # root_state[:, 2] += 2.0 
                # robot.write_root_state_to_sim(root_state)
                
                # joint_pos, joint_vel = robot.data.default_joint_pos.clone(), robot.data.default_joint_vel.clone()

                # directly apply the joint positions to the simulation
                zero_joint_vel = torch.zeros_like(joint_pos_sample)
                robot.write_joint_state_to_sim(joint_pos_sample, zero_joint_vel)
                # just make one step to get new observations of whether there is collision
                action = env.scene.articulations["robot"].data.default_joint_pos
                obs, _ = env.step(action)

                n_body = len(BODY_NAMES)

                # read the link states of the robot, then subtract the base position and orientation to get 
                # the pose in the base frame
                ee_pos_w = robot._data.body_pos_w[:, robot.find_bodies(BODY_NAMES)[0]].squeeze(0)
                ee_quat_w = robot._data.body_quat_w[:, robot.find_bodies(BODY_NAMES)[0]].squeeze(0)
                base_pos_w = repeat(robot._data.root_pos_w, 'b d -> b n d', n=n_body).squeeze(0)
                base_quat_w = repeat(robot._data.root_quat_w, 'b d -> b n d', n=n_body).squeeze(0)

                # visualize the pose
                marker.visualize(marker_indices=[0]*n_body*1, translations=ee_pos_w, orientations=ee_quat_w)
                # marker.visualize(marker_indices=[0]*n_body*{num_envs}, translations=rearrange(ee_pos_w, 'b n d -> (b n) d'), orientations=rearrange(ee_quat_w, 'b n d -> (b n) d'))

                # Convert to base frame
                ee_pos_b, ee_quat_b = subtract_frame_transforms(base_pos_w, base_quat_w, ee_pos_w, ee_quat_w)

                single_data_frame = torch.cat((ee_pos_b, ee_quat_b, obs['policy'].squeeze(0).unsqueeze(-1)), dim=-1)

                # check for collisions, if any, do not record this sample to dataframe. 
                # # ?? maybe we do not need to filter out samples with collisions

                if obs['policy'].sum() > 0:
                    self_collision_count += 1
                    # [BODY_NAMES[idx] for idx in obs['policy'][0].nonzero().flatten().cpu().tolist()]

                joint_pos_valid.append(joint_pos_sample)
                link_states_valid.append(single_data_frame)

                # sample_idx += 1

                if not args_cli.headless:
                    for _ in range(100):
                        env.sim.render()

        break

    X = torch.stack(joint_pos_valid, dim=0)
    Y = torch.stack(link_states_valid, dim=0)

    print("Among all {} samples, there are {} samples with self-collisions.".format(num_samples, self_collision_count))

    # Save dataset
    with h5py.File("./logs/datasets/mock_kinematic_dataset.h5", "w") as f:
        f.create_dataset("X", data=X.cpu().numpy())
        f.create_dataset("Y", data=Y.cpu().numpy())
        f.attrs["input_joint_names"] = ",".join(JOINT_NAMES)  # Store as a single string
        f.attrs["output_body_names"] = ",".join(BODY_NAMES)  # Store as a single string
        f.attrs["output_feature_dimensions"] = "translation(x,y,z),quaternion(x,y,z,w),contact_force(f)"

    print("Dataset saved successfully!")
 
    env.close()
        
    # Save the data
    print("[INFO]: Saving data!")
    # np.save("data_loco_manipulation/data_ee_poses.npy", data_ee_pose.detach().cpu().numpy())
    # np.save("data_loco_manipulation/data_ee_joint_pos.npy", data_ee_joint_pos.detach().cpu().numpy())
    
    print("[INFO]: Script has finished!")
            
            
    
if __name__ == "__main__":
    main()
    simulation_app.close()