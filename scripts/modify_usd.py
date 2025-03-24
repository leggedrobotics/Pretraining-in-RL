
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
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

from pxr import Usd
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR



# Load an existing USD file
stage = Usd.Stage.Open(f"{ISAACLAB_NUCLEUS_DIR}/Robots/ANYbotics/ANYmal-D/anymal_d.usd")

if not stage:
    print(f"Failed to open {usd_file}")
    exit()

pass




# # Save the changes
# stage.GetRootLayer().Save()
# print("Changes saved successfully!")
