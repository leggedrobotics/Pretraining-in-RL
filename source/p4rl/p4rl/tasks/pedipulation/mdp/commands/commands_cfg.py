from dataclasses import MISSING

from isaaclab.managers import CommandTermCfg
from isaaclab.utils import configclass

from .foot_position_command import FootPositionCommand


@configclass
class FootPositionCommandCfg(CommandTermCfg):
    """Configuration for the foot position command generator."""

    class_type: type = FootPositionCommand

    asset_name: str = MISSING

    # Randomly sampled commands (in environment frame)
    init_ranges_e: list[list] = [[0.0, 1.0], [-0.4, 0.2], [-0.1, 1.3]]  # dim 0: [x, y, z]; dim 1: [min, max] [m]
    final_ranges_e: list[list] = [[0.0, 2.2], [-1.2, 1.2], [-0.1, 1.3]]  # dim 0: [x, y, z]; dim 1: [min, max] [m]
    difficulty_steps: int = 10
    
    # Predefined commands
    use_predefined_commands: bool = False
    predetermined_commands_e: list[list] = [[0.5, -0.2, 0.5], [0.4, -0.4, 0.8], [0.6, 0.0, 0.2]] # shape (n_commands, 3) [x, y, z] [m]
    
    # We set these to true by default s.t. the user is aware of the visualization options
    command_space_debug_vis: bool = True
    commanded_foot_position_debug_vis: bool = True
    foot_position_debug_vis: bool = True