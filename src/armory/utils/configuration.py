"""
Utility functions for loading and accessing configuration information.
"""

import json
import os
from pathlib import Path


def get_configured_path(env_var: str, default_subdir: str) -> str:
    # Retrieve the value of the environment variable
    env_var_value = os.getenv(env_var)

    # If the environment variable does not exist,
    # construct a default path using the home directory, '.armory', and the provided default subdirectory
    if env_var_value is None:
        default_path = str(Path.home() / ".armory" / default_subdir)
        return default_path

    # If the environment variable exists, return its value
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
