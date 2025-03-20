# Copyright (c) 2022-2024, The IsaacLab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
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
# isort: off

# Load custom usd for the robot.
ANYMAL_D_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        # This models the shoulder bodies as capsules, which the locomotion focused default model does not.
        # It is in git lfs and needs to be downloaded.
        usd_path=f"{P4RL_EXT_DIR}/p4rl/assets/anymal-d-pedipulation/anymal_d.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=4, solver_velocity_iteration_count=0
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.6),
        joint_pos={
            ".*HAA": 0.0,  # all HAA
            ".*F_HFE": 0.4,  # both front HFE
            ".*H_HFE": -0.4,  # both hind HFE
            ".*F_KFE": -0.8,  # both front KFE
            ".*H_KFE": 0.8,  # both hind KFE
        },
    ),
    actuators={"legs": ANYDRIVE_4_MLP_ACTUATOR_CFG},
    soft_joint_pos_limit_factor=0.95,
)
"""Configuration of ANYmal-D robot using actuator-net."""


##
# Scene definition
##

@configclass
class PedipulationBaseSceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    # ground terrain
    terrain = None # defined in child classes
    # robots
    robot: ArticulationCfg = ANYMAL_D_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    # contact sensors
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)
    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(
            intensity=1000,
            exposure=1.5,
            color=(1.0, 1.0, 1.0)
        )
    )
    env_spacing = 5
    replicate_physics = False
    num_envs = 4096


##
# MDP settings
##

@configclass
class CommandsCfg:
    """Configuration for the foot position command generator."""

    foot_position = mdp.FootPositionCommandCfg(
        asset_name="robot",
        resampling_time_range=(6.0, 6.0),
    )

@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1)) # indices 0:3
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2)) # indices 3:6
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        ) # indices 6:9
        
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01)) # length 12, indices 12:24
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5)) # length 12, indices 24:36
        actions = ObsTerm(func=mdp.last_action, clip=(-100.00, 100.00)) # length 12, indices 36:48

        # P4RL needs the command to be at last
        foot_tracking_commands = ObsTerm(func=mdp.foot_tracking_commands, params={"asset_cfg": SceneEntityCfg("robot", body_names=".*FOOT")}) # indices 9:12

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    
    
# NOTE: This is how we can do inheritance in nested config classes 
@configclass
class ImprovedObservationsCfg(ObservationsCfg): # .. they just need a different name
    @configclass
    class PolicyCfg(ObservationsCfg.PolicyCfg):
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        
        
@configclass
class EventCfg:
    """Configuration for randomization."""

    # startup
    physics_material: EventTerm | None = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.0, 1.5),  # TODO: legged_gym only has one friction termn, how to choose?
            "dynamic_friction_range": (0.0, 1.5),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    add_base_mass: EventTerm | None = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={"asset_cfg": SceneEntityCfg("robot", body_names="base"), "mass_distribution_params": (-5.0, 5.0), "operation": "add"},
    )

    push_foot_constant: EventTerm | None = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*FOOT"),
            "force_range": (0.0, 12.0),
            "torque_range": (-0.0, 0.0),
        },
    )

    reset_base: EventTerm = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)},
            "velocity_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0), "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0)},
        },
    )

    reset_robot_joints: EventTerm = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (1.0, 1.0),
            "velocity_range": (0.0, 0.0),
        },
    )

    # interval
    push_robot: EventTerm | None = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(2.0, 4.0),
        params={"velocity_range": {"x": (-1.5, 1.5),
                                    "y": (-1.5, 1.5),
                                    "z": (-1.5, 1.5),
                                    "roll": (-1.5, 1.5),
                                    "pitch": (-1.5, 1.5),
                                    "yaw": (-1.5, 1.5),
                                    }},
    )

    push_foot_interval: EventTerm | None = EventTerm(
        func=mdp.apply_external_force_torque_to_foot,
        mode="interval",
        interval_range_s=(2.0, 4.0),
        params={"asset_cfg": SceneEntityCfg("robot", body_names=".*FOOT"),
                "force_range": (-50, 50),
                "torque_range": (-0.0, 0.0),
                },
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=0.5, use_default_offset=True)


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm( # terminate if the base contacts the ground hard enough
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="base"), "threshold": 1.0},
    )


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    foot_command_space = CurrTerm(func=mdp.foot_command_space, params={"tracking_threshhold": 0.06})


##
# Environment configuration
##

@configclass
class PedipulationPositionEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the pedipulation environment."""

    # Scene settings
    scene: PedipulationBaseSceneCfg = PedipulationBaseSceneCfg() # Note: 3070 8GB doesn't have enough memory for 4096 envs
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings (rewards are defined in the child classes)
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()
    # Simulation settings
    decimation: int = 4
    episode_length_s: float = 12.0 # time between resets. There may be multiple command resamplings within this time
    # Pedipulation settings
    foot_index: int = 2

    def __post_init__(self):
        """Post initialization."""

        # simulation settings
        self.sim.dt = 0.005
        self.sim.disable_contact_processing = True

@configclass
class PedipulationPositionEnvCfg_PLAY(PedipulationPositionEnvCfg):
    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 20
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing
        # self.events.push_robot = None
        self.events.push_foot_interval = None
        # remove randomization
        self.events.physics_material = None
        self.events.add_base_mass = None
        # enable debug visualization
        self.commands.foot_position.debug_vis = True
        # make curriculum easier
        self.curriculum.foot_command_space.params["tracking_threshhold"] = 1.0
        # reduce the number of curriculum steps
        self.commands.foot_position.difficulty_steps = 2
        # decrease resample interval
        self.commands.foot_position.resampling_time_range = (2.0, 2.0)
        # decrease episode lenght
        self.episode_length_s = 6.0

        # increase command domain
        # self.commands.foot_position.ranges.pos_x = (0.0, 2.0)
        # self.commands.foot_position.ranges.pos_y = (-1.4, 1.2)