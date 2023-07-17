"""
Reference objects for armory paths
"""

import json
from pathlib import Path
import warnings


class HostPaths:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize_paths()
            cls._instance._load_config_and_update_paths()
            cls._instance._create_directories()
        return cls._instance

    def _initialize_paths(self):
        self.armory_dir = Path.home() / ".armory"
        self.armory_config = self.armory_dir / "config.json"
        self.dataset_dir = self.armory_dir / "datasets"
        self.local_git_dir = self.armory_dir / "git"
        self.saved_model_dir = self.armory_dir / "saved_models"
        self.pytorch_dir = self.saved_model_dir / "pytorch"
        self.tmp_dir = self.armory_dir / "tmp"
        self.output_dir = self.armory_dir / "outputs"
        self.external_repo_dir = self.tmp_dir / "external"

    def _load_config_and_update_paths(self):
        if self.armory_config.exists():
            _config = self._read_config()
            for k in (
                "dataset_dir",
                "local_git_dir",
                "saved_model_dir",
                "output_dir",
                "tmp_dir",
            ):
                setattr(self, k, self.armory_dir / _config[k])

    def _read_config(self):
        try:
            return json.loads(self.armory_config.read_text())
        except (json.decoder.JSONDecodeError, OSError) as e:
            warnings.warn(
                f"Armory config file {self.armory_config} could not be read/decoded"
            )
            raise e

    def _create_directories(self):
        directory_tree = (
            self.dataset_dir,
            self.local_git_dir,
            self.saved_model_dir,
            self.pytorch_dir,
            self.tmp_dir,
            self.output_dir,
        )

        for directory in directory_tree:
            directory.mkdir(parents=True, exist_ok=True)
