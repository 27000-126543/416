"""
Global Climate Simulation Platform
A large-scale parallel computing platform for global climate research.
"""

__version__ = "1.0.0"
__author__ = "Climate Research Platform Team"

from .config import PlatformConfig, load_config
from .main import ClimateSimulationPlatform, create_default_config

__all__ = [
    "PlatformConfig",
    "load_config",
    "ClimateSimulationPlatform",
    "create_default_config",
]
