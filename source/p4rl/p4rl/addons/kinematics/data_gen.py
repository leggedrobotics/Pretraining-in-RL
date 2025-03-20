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

from omni.isaac.orbit.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="This script demonstrates how to use the concept of an Environment.")
parser.add_argument("--num_envs", type=int, default=2000, help="Number of environments to spawn.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

args_cli.headless = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""
import os
import torch
import numpy as np

from omni.isaac.orbit_assets import ORBIT_ASSETS_DATA_DIR

from omni.isaac.orbit.envs import BaseEnv, BaseEnvCfg
from omni.isaac.orbit.terrains import TerrainImporterCfg
from omni.isaac.orbit.utils.math import subtract_frame_transforms
import omni.isaac.orbit_tasks.loco_manipulation.wbc_teacher_student.mdp as mdp

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg, BaseEnv, BaseEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab_assets.robots.anymal import ANYDRIVE_4_MLP_ACTUATOR_CFG
import os
# from p4rl import P4RL_EXT_DIR # cause "can not import from partially initialized module" error
P4RL_EXT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../.."))
import p4rl.tasks.pedipulation.mdp as mdp
##
# Pre-defined configs
##
from isaaclab_assets.robots.alma import ALMA_D_WHEELS_NO_TOOL_CFG, LEG_JOINT_NAMES, ARM_JOINT_NAMES, WHEEL_JOINT_NAMES, ARM_BODY_NAMES


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
    robot: ArticulationCfg = ALMA_D_WHEELS_NO_TOOL_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

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
    wheel_joint_vel = mdp.JointVelocityActionCfg(
        asset_name="robot", joint_names=[".*WHEEL"], scale=5.0, use_default_offset=True
    )
    manipulator_joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=ARM_JOINT_NAMES, scale=0.0, use_default_offset=True)


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        undesired_contacts = ObsTerm(
            func=mdp.obs_undesired_contacts,
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=ARM_BODY_NAMES), "threshold": 1.0},
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
class QuadrupedEnvCfg(BaseEnvCfg):
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

    # setup base environment
    env = BaseEnv(cfg=QuadrupedEnvCfg(scene=MySceneCfg(num_envs=args_cli.num_envs, env_spacing=2.5, replicate_physics=False)))
    obs, _ = env.reset()
    
    # initalize data struct to save EE pose in base frame (position & orientation)
    data_ee_pose = None
    data_ee_joint_pos = None
    number_of_samples = 100000

    # simulate physics
    count = 0
    while simulation_app.is_running():
        with torch.inference_mode():
            
            # reset base state (alma should stand still)
            obs, _ = env.reset()

            robot = env.scene.articulations["robot"]
            root_state = robot.data.default_root_state.clone()
            root_state[:, :3] += env.scene.terrain.env_origins
            robot.write_root_state_to_sim(root_state)
            joint_pos, joint_vel = robot.data.default_joint_pos.clone(), robot.data.default_joint_vel.clone()

            # Randomize the joint positions of the manipulator
            if count % 10 == 0:
                # Get transforms of the end effector in world frame
                ee_pos_w = robot._data.body_pos_w[:, robot.find_bodies(["dynaarm_END_EFFECTOR"])[0]].squeeze(1)
                ee_quat_w = robot._data.body_quat_w[:, robot.find_bodies(["dynaarm_END_EFFECTOR"])[0]].squeeze(1)
                base_pos_w = robot._data.root_pos_w
                base_quat_w = robot._data.root_quat_w
                # Convert to base frame
                ee_pos_b, ee_quat_b = subtract_frame_transforms(base_pos_w, base_quat_w, ee_pos_w, ee_quat_w)
                ee_pose = torch.cat((ee_pos_b, ee_quat_b), dim=1)
                # Joint positions
                ee_joint_pos = robot._data.joint_pos[:, robot.find_joints(ARM_JOINT_NAMES)[0]]
                # Save the data
                if data_ee_pose is None:
                    if not count == 0:
                        data_ee_pose = ee_pose
                        data_ee_joint_pos = ee_joint_pos
                else:
                    # check for collisions and remove these commands
                    if torch.nonzero(obs['policy'][:,0]).any():
                        # remove the commands that have collisions
                        ee_pose = ee_pose[obs['policy'][:,0] == 0]
                        ee_joint_pos = ee_joint_pos[obs['policy'][:,0] == 0]

                    data_ee_pose = torch.cat((data_ee_pose, ee_pose), dim=0)
                    data_ee_joint_pos = torch.cat((data_ee_joint_pos, ee_joint_pos), dim=0)
                
                # Generate random joint positions for the manipulator
                joints_manipulator = robot.find_joints(ARM_JOINT_NAMES)[0]
                limits = robot._data.soft_joint_pos_limits[:,joints_manipulator]
                
                # Generate random numbers in the range [0, 1]
                random_numbers = torch.rand_like(limits[..., 0])

                # Calculate random values within each specified [lower, upper] pair
                random_values = limits[..., 0] + (limits[..., 1] - limits[..., 0]) * random_numbers
                
            # Overwrite the joint positions of the manipulator with the random values
            joint_pos[:, joints_manipulator] = random_values

            # Write the joint positions to the simulator
            robot.write_joint_state_to_sim(joint_pos, joint_vel)
            # reset the internal state
            robot.reset()

            # infer action - default joint position
            action = env.scene.articulations["robot"].data.default_joint_pos
            
            # step env
            obs, _ = env.step(action)
            # update counter
            count += 1
            
            # Stop after a certain number of samples
            if data_ee_pose is not None:
                print(f"Number of samples: {data_ee_pose.shape[0]}")
                if data_ee_pose.shape[0] >= number_of_samples:
                    break
    env.close()
        
    # Save the data
    print("[INFO]: Saving data!")
    np.save("data_loco_manipulation/data_ee_poses.npy", data_ee_pose.detach().cpu().numpy())
    np.save("data_loco_manipulation/data_ee_joint_pos.npy", data_ee_joint_pos.detach().cpu().numpy())
    
    print("[INFO]: Script has finished!")
            
            
    
if __name__ == "__main__":
    main()
    simulation_app.close()