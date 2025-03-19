from __future__ import annotations

from isaaclab.managers import SceneEntityCfg

import torch
from typing import TYPE_CHECKING, Sequence

import carb

from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import CUBOID_MARKER_CFG
from isaaclab.terrains import TerrainImporter
import isaaclab.sim as sim_utils
from isaaclab.assets.articulation import Articulation
from isaaclab.managers import CommandTerm
from isaaclab.assets import RigidObject
from isaaclab.utils.math import quat_rotate, quat_from_angle_axis
from isaaclab.markers.visualization_markers import VisualizationMarkersCfg
import omni.usd.commands

import itertools


if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from pedipulation.mdp.commands.commands_cfg import FootPositionCommandCfg
    from pedipulation.config.anymal_d.pedipulation_base import PedipulationPositionEnvCfg


CYLINDER_MARKER_CFG = VisualizationMarkersCfg(
    markers={
        "cylinder": sim_utils.CylinderCfg(
            radius=0.02,
            height=1.0, # needs to be adjusted to the sizes of edges of the command space
            axis="X",
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.2, 1.0)),
        )
    }
)


class FootPositionCommand(CommandTerm):
    """ TODO: Description
    """

    cfg: FootPositionCommandCfg
    """Configuration for the command generator."""


    def __init__(self, cfg: FootPositionCommandCfg, env: ManagerBasedRLEnv):
        """Initialize the command generator class.

        Args:
            cfg: The configuration parameters for the command generator.
            env: The environment object.
        """
        
        # command space to sample from
        self.difficulty = 0 # determines the size of the command space. Integers from 0 to max_difficulty
        self.max_difficulty = cfg.difficulty_steps
        self.init_ranges_e = torch.tensor(cfg.init_ranges_e, device=env.device).repeat(env.num_envs, 1, 1) # shape (num_envs, 3, 2); 3 dimensions xyz, min and max
        self.final_ranges_e = torch.tensor(cfg.final_ranges_e, device=env.device).repeat(env.num_envs, 1, 1)  # shape (num_envs, 3, 2); 3 dimensions xyz, min and max
        self.current_command_space = self.init_ranges_e.clone()  # shape (num_envs, 3, 2); 3 dimensions xyz, min and max
        self.env_origins = env.scene.env_origins

        # initialize the base class
        super().__init__(cfg, env)

        self.env_cfg: PedipulationPositionEnvCfg = env.cfg # for type hinting

        # obtain the robot and terrain assets
        # -- robot
        self.robot: Articulation = env.scene[cfg.asset_name]

        # # -- terrain
        self.terrain: TerrainImporter = env.scene.terrain

        # -- metrics
        self.metrics["command_space_difficulty [0,1]"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["average_tracking_error [m]"] = torch.zeros(self.num_envs, device=self.device)

        # -- command: x pos, y pos, z pos in environment frame
        self.foot_pos_commands_e = torch.zeros(self.num_envs, 3, device=self.device)

        # -- command: x pos, y pos, z pos in world frame
        self.foot_pos_commands_w = torch.zeros(self.num_envs, 3, device=self.device)
        ''' As the visualization markers may be recomputed after the command is updated, we need to store the command in 
         the world frame as otherwise the ever changin robot base position and orientations would need to be used to
         transform the command_b to command_w, which would lead to wrong results.'''



        # predefined commands
        self.predefined_commands_e = torch.tensor(cfg.predetermined_commands_e, device=self.device).repeat(self.num_envs, 1, 1) # shape (n_envs, n_commands, 3) [x, y, z] [m]
        self.num_predefined_commands = self.predefined_commands_e.shape[1]

        # -- command: x pos, y pos, z pos
        self.tracking_error_sum = torch.zeros(self.num_envs, 3, device=self.device)
        self.log_step_counter = torch.zeros(self.num_envs, device=self.device)
        self.feet_ids, _ = self._env.scene["robot"].find_bodies(".*FOOT")

    def __str__(self) -> str:
        msg = "FootPositionCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}\n"
        # TODO: what foot command specific info to print?
        return msg


    """
    Properties
    """

    @property
    def command(self) -> torch.Tensor:
        """The desired foot position in world frame. Shape is (num_envs, 3)."""
        return self.foot_pos_commands_w


    """
    Implementation specific functions.
    """

    def increase_difficulty(self):
        # make sure the difficulty does not exceed the maximum difficulty
        if self.difficulty >= self.max_difficulty:
            return
        self.difficulty += 1
        self._update_command_space()


    def _update_command_space(self):
        # update the command space based on the current difficulty
        self.current_command_space = self.init_ranges_e + (self.final_ranges_e - self.init_ranges_e) * (self.difficulty / self.max_difficulty)
        # assume that the command space is the same for all environments
        self.current_command_space = self.current_command_space.repeat(self.num_envs, 1, 1)
        
        # delete existing command space visualization markers
        if hasattr(self, "command_space_visualizers"):
            for marker in self.command_space_visualizers:
                # turn off the visibility of the visualization marker
                marker.set_visibility(False)
                # delete the visualization marker
                omni.usd.commands.DeletePrimsCommand(paths=[marker.prim_path])
            del self.command_space_visualizers
        # visualize the updated command space
        self._create_command_space_visualizers()


    def _resample_command(self, env_ids: Sequence[int]):
        """Randomly select commands of some environments."""
        if len(env_ids) == 0:
            return

        # set tracking error to zero
        self.tracking_error_sum[env_ids] = 0.0
        self.log_step_counter[env_ids] = 0.0

        if not self.cfg.use_predefined_commands:
            # sample a random command from the current command space
            r = torch.rand(len(env_ids), 3, device=self.device)

            # interpolate between the min and max values of the current command space
            self.foot_pos_commands_e = self.current_command_space[env_ids, :, 0] + r * (self.current_command_space[env_ids, :, 1] - self.current_command_space[env_ids, :, 0])

        else:
            # choose a predefined command based on the current command counter (each resample goeas to the next predefined command)
            command_indices = self.command_counter[env_ids] % self.num_predefined_commands
            self.foot_pos_commands_e = self.predefined_commands_e[env_ids, command_indices]
            
        # transform the command to world frame
        self.foot_pos_commands_w[env_ids] = self.env_origins[env_ids] + self.foot_pos_commands_e # convert to world frame


    def _update_command(self): # called right after the command is resampled
        self._log_data()


    def _log_data(self):
        # log data used in the curriculum
        asset: RigidObject = self._env.scene["robot"]
        foot_pos_w = asset.data.body_state_w[:, self.feet_ids[self.env_cfg.foot_index], :3]
        self.tracking_error_sum += torch.abs(self.foot_pos_commands_w - foot_pos_w)
        self.log_step_counter += 1.0


    def _update_metrics(self):
        self.metrics["command_space_difficulty [0,1]"] = torch.tensor([self.difficulty / self.max_difficulty], device=self.device).repeat(self.num_envs, 1)
        self.metrics["average_tracking_error [m]"] = torch.linalg.norm(self.tracking_error_sum, dim=1) / self.log_step_counter


    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            # create the visualization markers if they do not exist
            if not hasattr(self, "command_space_visualizers"):
                self._create_command_space_visualizers()
            if not hasattr(self, "foot_position_command_visualizer"):
                self._create_foot_position_command_visualizer()
            if not hasattr(self, "foot_position_visualizer"):
                self._create_foot_position_visualizer()

            # make the visualization markers visible
            self.foot_position_command_visualizer.set_visibility(True)
            self.foot_position_visualizer.set_visibility(True)
            for marker in self.command_space_visualizers:
                marker.set_visibility(True)
        else:
            # make all visualization markers to invisible
            if hasattr(self, "command_space_visualizers"):
                for marker in self.command_space_visualizers:
                    marker.set_visibility(False)
            if hasattr(self, "foot_position_command_visualizer"):
                self.foot_position_command_visualizer.set_visibility(False)
            if hasattr(self, "foot_position_visualizer"):
                self.foot_position_visualizer.set_visibility(False)


    def _create_command_space_visualizers(self):
        # a cuboid has 12 edges, which we need to find the length, center position and orientation of
        self.command_space_visualizers = []
        self.edge_positions = torch.zeros(12, self.num_envs, 3, device=self.device)
        self.edge_orientations = torch.zeros(12, self.num_envs, 4, device=self.device)

        # compute the 8 corners of the command space
        corners: torch.tensor = torch.zeros([2, 2, 2, 3], device=self.device) # min and max values for x, y, z
        for i in [0, 1]:
            for j in [0, 1]:
                for k in [0, 1]:
                    # Use only the first environment for now (current_command_space[0, ...])
                    corners[i, j, k, :] = torch.tensor([self.current_command_space[0, 0, i], self.current_command_space[0, 1, j], self.current_command_space[0, 2, k]], device=self.device)

        # compute the 12 edges of the command space
        edge_lengths = torch.zeros(12, device=self.device)
        edge_positions = torch.zeros(12, 3, device=self.device)
        edge_orientations = torch.zeros(12, 4, device=self.device) # quaternions

        # compute the 4 edges along x
        for i in range(2):
            for j in range(2):
                edge_lengths[i * 2 + j] = torch.linalg.norm(corners[i, j, 0] - corners[i, j, 1], dim=-1)
                edge_positions[i * 2 + j] = (corners[i, j, 0] + corners[i, j, 1]) / 2.0
                angle = torch.acos(torch.dot(corners[i, j, 0] - corners[i, j, 1], torch.tensor([1.0, 0.0, 0.0], device=self.device)) / edge_lengths[i * 2 + j])
                axis = torch.linalg.cross(corners[i, j, 0] - corners[i, j, 1], torch.tensor([1.0, 0.0, 0.0], device=self.device))
                edge_orientations[i * 2 + j] = quat_from_angle_axis(angle, axis)
        # compute the 4 edges along y
        for i in range(2):
            for j in range(2):
                edge_lengths[4 + i * 2 + j] = torch.linalg.norm(corners[i, 0, j] - corners[i, 1, j], dim=-1)
                edge_positions[4 + i * 2 + j] = (corners[i, 0, j] + corners[i, 1, j]) / 2.0
                angle = torch.acos(torch.dot(corners[i, 0, j] - corners[i, 1, j], torch.tensor([1.0, 0.0, 0.0], device=self.device)) / edge_lengths[4 + i * 2 + j])
                axis = torch.linalg.cross(corners[i, 0, j] - corners[i, 1, j], torch.tensor([1.0, 0.0, 0.0], device=self.device))
                edge_orientations[4 + i * 2 + j] = quat_from_angle_axis(angle, axis)
        # compute the 4 edges along z
        for i in range(2):
            for j in range(2):
                edge_lengths[8 + i * 2 + j] = torch.linalg.norm(corners[0, i, j] - corners[1, i, j], dim=-1)
                edge_positions[8 + i * 2 + j] = (corners[0, i, j] + corners[1, i, j]) / 2.0
                angle = torch.acos(torch.dot(corners[0, i, j] - corners[1, i, j], torch.tensor([1.0, 0.0, 0.0], device=self.device)) / edge_lengths[8 + i * 2 + j])
                axis = torch.linalg.cross(corners[0, i, j] - corners[1, i, j], torch.tensor([1.0, 0.0, 0.0], device=self.device))
                edge_orientations[8 + i * 2 + j] = quat_from_angle_axis(angle, axis)
        
        # we assume that the command space is the same for all environments
        edge_positions = edge_positions.repeat(self.num_envs, 1, 1) + self.env_origins.repeat(12, 1, 1).swapaxes(0, 1) # shape (num_envs, 12, 3)
        edge_orientations = edge_orientations.repeat(self.num_envs, 1, 1) # shape (num_envs, 12, 4)

        # create and visualize the 12 markers
        marker_cfg = CYLINDER_MARKER_CFG.copy()
        marker_cfg.markers["cylinder"].visual_material = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0))
        for i in range(12):
            marker_cfg.prim_path = f"/Visuals/Command/Command_space_edge_{i}"
            marker_cfg.markers["cylinder"].height = edge_lengths[i].item()
            marker = VisualizationMarkers(marker_cfg)
            marker.visualize(edge_positions[:, i], edge_orientations[:, i])
            self.command_space_visualizers.append(marker)


    def _create_foot_position_command_visualizer(self):
        marker_cfg = CUBOID_MARKER_CFG.copy()
        marker_cfg.prim_path = "/Visuals/Command/position_goal"
        marker_cfg.markers["cuboid"].scale = (0.1, 0.1, 0.1)
        self.foot_position_command_visualizer = VisualizationMarkers(marker_cfg)


    def _create_foot_position_visualizer(self):
        marker_cfg = CUBOID_MARKER_CFG.copy()
        marker_cfg.prim_path = "/Visuals/Command/foot_position"
        marker_cfg.markers["cuboid"].scale = (0.1, 0.1, 0.1)
        marker_cfg.markers["cuboid"].visual_material = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0))
        self.foot_position_visualizer = VisualizationMarkers(marker_cfg)


    def _debug_vis_callback(self, event):
        # visualize the position of the pedipulating foot
        asset: RigidObject = self._env.scene["robot"]
        foot_pos_w = asset.data.body_state_w[:, self.feet_ids[self.env_cfg.foot_index], :3]
        self.foot_position_visualizer.visualize(foot_pos_w)

        # visualize the actual foot position
        self.foot_position_command_visualizer.visualize(self.foot_pos_commands_w)

        # the command space visualization only needs to be updated when the command space is updated