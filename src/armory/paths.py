"""
Reference objects for armory paths
"""

import json
from pathlib import Path
import warnings


class HostPaths:
    def __init__(self):
        self._initialize_paths()
        self._load_config_and_update_paths()
        self._create_directories()

    def __getattr__(self, name):
        """Intercept attribute access and return a string representation of the Path object if the attribute exists."""
        if name.startswith("_"):
            raise AttributeError(f"{name} is not a valid attribute")

        if hasattr(self, name):
            return str(getattr(self, name))
        else:
            raise AttributeError(f"{name} not found")

    def _initialize_paths(self):
        self.cwd = Path.cwd()
        self.user_dir = Path.home()
        self.armory_dir = Path(self.user_dir / ".armory")
        self.armory_config = Path(self.armory_dir / "config.json")
        self.dataset_dir = Path(self.armory_dir / "datasets")
        self.local_git_dir = Path(self.armory_dir / "git")
        self.saved_model_dir = Path(self.armory_dir / "saved_models")
        self.pytorch_dir = Path(self.armory_dir / "saved_models", "pytorch")
        self.tmp_dir = Path(self.armory_dir / "tmp")
        self.output_dir = Path(self.armory_dir / "outputs")
        self.external_repo_dir = Path(self.tmp_dir / "external")

    def _load_config_and_update_paths(self):
        """Load config from file(`/tmp/armory/config.json`) and update paths if config exists."""
        if Path(self.armory_config).exists():
            print("Reading armory config file")

            _config = self._read_config()
            for k in (
                # pytorch_dir should not be found in the config file
                # because it is not configurable (yet)
                "dataset_dir",
                "local_git_dir",
                "saved_model_dir",
                "output_dir",
                "tmp_dir",
            ):
                setattr(self, k, _config[k])

    def _read_config(self):
        try:
            return json.loads(Path(self.armory_config).read_text())
        except json.decoder.JSONDecodeError:
            warnings.warn(
                f"Armory config file {self.armory_config} could not be decoded"
            )
            raise
        except OSError:
            warnings.warn(f"Armory config file {self.armory_config} could not be read")
            raise

    def _create_directories(self):
        """Create all directories in the armory directory tree"""
        directory_tree = (
            self.dataset_dir,
            self.local_git_dir,
            self.saved_model_dir,
            self.pytorch_dir,
            self.tmp_dir,
            self.output_dir,
        )

        for directory in directory_tree:
            Path(directory).mkdir(parents=True, exist_ok=True)
