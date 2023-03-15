"""
Reference objects for armory paths
"""

import os
import warnings  # armory.logs initialization depends on this module, use warnings instead

from armory import configuration


class ArmoryDefaultPaths:
    def __init__(self):
        self.cwd = os.getcwd()
        self.user_dir = os.path.expanduser("~")
        self.armory_dir = os.path.join(self.user_dir, ".armory")
        self.armory_config = os.path.join(self.armory_dir, "config.json")
        self.dataset_dir = os.path.join(self.armory_dir, "datasets")
        self.local_git_dir = os.path.join(self.armory_dir, "git")
        self.saved_model_dir = os.path.join(self.armory_dir, "saved_models")
        self.pytorch_dir = os.path.join(self.armory_dir, "saved_models", "pytorch")
        self.tmp_dir = os.path.join(self.armory_dir, "tmp")
        self.output_dir = os.path.join(self.armory_dir, "outputs")
        self.external_repo_dir = os.path.join(self.tmp_dir, "external")


class HostPaths(ArmoryDefaultPaths):
    def __init__(self):
        super().__init__()
        if os.path.isfile(self.armory_config):
            # Parse paths from config
            config = configuration.load_global_config(self.armory_config)
            for k in (
                "dataset_dir",
                "local_git_dir",
                "saved_model_dir",
                # pytorch_dir should not be found in the config file
                # because it is not configurable (yet)
                "output_dir",
                "tmp_dir",
            ):
                setattr(self, k, config[k])
        else:
            warnings.warn(f"No {self.armory_config} file. Using default paths.")
            warnings.warn("Please run `armory configure`")

        os.makedirs(self.dataset_dir, exist_ok=True)
        os.makedirs(self.local_git_dir, exist_ok=True)
        os.makedirs(self.saved_model_dir, exist_ok=True)
        os.makedirs(self.pytorch_dir, exist_ok=True)
        os.makedirs(self.tmp_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)


class DockerPaths(ArmoryDefaultPaths):
    def __init__(self):
        super().__init__()
