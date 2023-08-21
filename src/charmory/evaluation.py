"""Armory Experiment Configuration Classes"""
from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional

from art.attacks import EvasionAttack
from art.data_generators import DataGenerator
from art.estimators import BaseEstimator

from charmory.labels import LabelTargeter

MethodName = Callable[
    ..., Any
]  # reference to a python method e.g. "armory.attacks.weakest"


@dataclass
class Attack:
    name: str
    attack: EvasionAttack
    generate_kwargs: Dict[str, Any] = field(default_factory=dict)
    use_label_for_untargeted: bool = False
    label_targeter: Optional[LabelTargeter] = None

    def __post_init__(self):
        assert isinstance(
            self.attack, EvasionAttack
        ), "Evaluation attack is not an instance of EvasionAttack"

        if self.targeted:
            assert isinstance(
                self.label_targeter, LabelTargeter
            ), "Evaluation attack's label_targeter is not an instance of LabelTargeter"
            assert (
                not self.use_label_for_untargeted
            ), "Evaluation attack is targeted, use_label_for_targeted cannot be True"
        else:
            assert (
                not self.label_targeter
            ), "Evaluation attack is untargeted, cannot use a label_targeter"

    @property
    def targeted(self) -> bool:
        """Whether the attack is targeted"""
        return self.attack.targeted


@dataclass
class Dataset:
    name: str
    test_dataset: DataGenerator
    train_dataset: Optional[DataGenerator] = None

    def __post_init__(self):
        assert isinstance(
            self.test_dataset, DataGenerator
        ), "Evaluation dataset's test_dataset is not an instance of DataGenerator"
        if self.train_dataset is not None:
            assert isinstance(
                self.train_dataset, DataGenerator
            ), "Evaluation dataset's train_dataset is not an instance of DataGenerator"


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
    paths: Dict[str, Path] = field(init=False)
    armory_home: Path = Path(os.getenv("ARMORY_HOME", Path.home() / ".armory"))

    def __post_init__(self):
        self._initialize_paths()
        self._create_directories_and_update_env_vars()

    def _initialize_paths(self):
        """Construct the paths for each directory. Some of these are old keys used in armory version <= 0.19.0
        and will be deprecated in the future. Please use `armory_home` instead."""
        self.paths = {
            "armory_home": self.armory_home,
            "dataset_dir": self.armory_home / "datasets",
            "saved_model_dir": self.armory_home / "saved_models",
            "output_dir": self.armory_home / "outputs",
            "external_repo_dir": self.armory_home / "tmp" / "external",
        }

    def _create_directories_and_update_env_vars(self):
        """Create directories if they do not exist and update environment variables."""
        for key, config_path in self.paths.items():
            # Set environment variable
            os.environ[key.upper()] = str(config_path)
            # Create directory if it does not exist
            config_path.mkdir(parents=True, exist_ok=True)


@dataclass
class Evaluation:
    name: str
    description: str
    model: Model
    scenario: Scenario
    dataset: Dataset
    author: Optional[str]
    attack: Optional[Attack] = None
    metric: Optional[Metric] = None
    sysconfig: Optional[SysConfig] = None
