"""Armory Experiment Configuration Classes"""
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional

from art.estimators import BaseEstimator

from armory.data.datasets import ArmoryDataGenerator

MethodName = Callable[
    ..., Any
]  # reference to a python method e.g. "armory.attacks.weakest"


@dataclass
class Attack:
    function: MethodName
    kwargs: Dict[str, Any]
    knowledge: Literal["white", "black"]
    use_label: bool = False
    type: Optional[str] = None
    generate_kwargs: Optional[Dict[str, Any]] = field(default_factory=dict)
    sweep_params: Optional[Dict[str, Any]] = field(default_factory=dict)
    targeted: Optional[bool] = False
    targeted_labels: Optional[Dict[str, Any]] = field(default_factory=dict)


@dataclass
class Dataset:
    name: str
    test_dataset: ArmoryDataGenerator
    train_dataset: Optional[ArmoryDataGenerator] = None

    def __post_init__(self):
        assert isinstance(
            self.test_dataset, ArmoryDataGenerator
        ), "Evaluation dataset's test_dataset is not an instance of ArmoryDataGenerator"
        if self.train_dataset is not None:
            assert isinstance(
                self.train_dataset, ArmoryDataGenerator
            ), "Evaluation dataset's train_dataset is not an instance of ArmoryDataGenerator"


@dataclass
class Defense:
    function: MethodName
    kwargs: Dict[str, Any]
    type: Literal[
        "Preprocessor",
        "Postprocessor",
        "Trainer",
        "Transformer",
        "PoisonFilteringDefense",
    ]


@dataclass
class Metric:
    profiler_type: Literal["basic", "deterministic"]
    supported_metrics: List[str]
    perturbation: List[str]
    task: List[str]
    means: bool
    record_metric_per_sample: bool


@dataclass
class Model:
    name: str
    model: BaseEstimator
    predict_kwargs: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        assert isinstance(
            self.model, BaseEstimator
        ), "Evaluation model is not an instance of BaseEstimator"


@dataclass
class Scenario:
    function: MethodName
    kwargs: Dict[str, Any]
    export_batches: Optional[bool] = False


@dataclass
class SysConfig:
    """Class for handling system configurations.

    Attributes:
        gpus: A list of GPU devices.
        use_gpu: A boolean indicating whether to use a GPU.
        paths: A dictionary of paths for system directories.
        armory_home: The home directory for armory.
    """

    gpus: List[str]
    use_gpu: bool = False
    paths: Dict[str, str] = field(default_factory=dict)
    armory_home: str = os.getenv("ARMORY_HOME", str(Path.home() / ".armory"))

    def __post_init__(self):
        self.armory_home = Path(self.armory_home)
        self._initialize_paths()
        self._load_config_and_update_paths()
        self._create_directories_and_update_env_vars()

    def _initialize_paths(self):
        """Construct the paths for each directory and file"""
        self.paths = {
            "armory_home": str(self.armory_home),
            "armory_config": str(self.armory_home / "config.json"),
            "dataset_dir": str(self.armory_home / "datasets"),
            "local_git_dir": str(self.armory_home / "git"),
            "saved_model_dir": str(self.armory_home / "saved_models"),
            "pytorch_dir": str(self.armory_home / "saved_models" / "pytorch"),
            "tmp_dir": str(self.armory_home / "tmp"),
            "output_dir": str(self.armory_home / "outputs"),
            "external_repo_dir": str(self.armory_home / "tmp" / "external"),
        }

    def _load_config_and_update_paths(self):
        """Load the configuration file and update the paths accordingly."""
        config_path = Path(self.paths["armory_config"])
        if config_path.exists():
            with open(config_path, "r") as f:
                config = json.load(f)
            # Update paths based on the configuration
            for key in (
                "dataset_dir",
                "local_git_dir",
                "saved_model_dir",
                "output_dir",
                "tmp_dir",
            ):
                self.paths[key] = str(self.armory_home / config[key])

    def _create_directories_and_update_env_vars(self):
        """Create directories if they do not exist and update environment variables."""
        for key, path_str in self.paths.items():
            path = Path(path_str)
            # Do not create directories for .json files
            if path.suffix != ".json":
                # Set environment variable
                os.environ[key.upper()] = path_str
                # Create directory if it does not exist
                path.mkdir(parents=True, exist_ok=True)


@dataclass
class Evaluation:
    name: str
    description: str
    model: Model
    scenario: Scenario
    dataset: Dataset
    author: Optional[str]
    attack: Optional[Attack] = None
    defense: Optional[Defense] = None
    metric: Optional[Metric] = None
    sysconfig: Optional[SysConfig] = None
