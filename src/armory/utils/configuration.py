"""
Utility functions for loading and accessing configuration information.
"""

import json
import os
from pathlib import Path


def get_armory_home() -> str:
    env_var_value = os.getenv("ARMORY_HOME")

    if env_var_value is None:
        env_var_value = str(Path.home() / ".armory")

    return env_var_value


def get_verify_ssl():
    return os.getenv("VERIFY_SSL") == "true" or os.getenv("VERIFY_SSL") is None


def load_config(filepath: str) -> dict:
    """
    Loads and validates a config file
    """
    with open(filepath) as f:
        config = json.load(f)

    return config
