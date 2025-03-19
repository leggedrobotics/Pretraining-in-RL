"""
Python module serving as a project/extension template.
"""

# Register Gym environments.
from .tasks import *

# Register UI extensions.
from .ui_extension_example import *

P4RL_EXT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
