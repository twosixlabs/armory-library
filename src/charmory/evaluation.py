"""Armory Experiment Configuration Classes"""
from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Any, Dict, Optional

from art.attacks import EvasionAttack
from art.estimators import BaseEstimator
from torch.utils.data.dataloader import DataLoader

from armory.metrics.compute import NullProfiler, Profiler
from charmory.labels import LabelTargeter


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
    x_key: str
    y_key: str
    test_dataloader: DataLoader
    train_dataloader: Optional[DataLoader] = None

    def __post_init__(self):
        assert isinstance(
            self.test_dataloader, DataLoader
        ), "Evaluation dataset's test_dataloader is not an instance of DataLoader"
        if self.train_dataloader is not None:
            assert isinstance(
                self.train_dataloader, DataLoader
            ), "Evaluation dataset's train_dataloader is not an instance of DataLoader"


@dataclass
class Metric:
    profiler: Profiler = field(default_factory=NullProfiler)


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
class SysConfig:
    """Class for handling system configurations.

    Attributes:
        armory_home: The home directory for armory.
    """

    armory_home: Path = Path(os.getenv("ARMORY_HOME", Path.home() / ".armory"))
    # When using torchvision the user must specify a directory to download the dataset into.
    dataset_cache: Path = Path(
        os.getenv(
            "ARMORY_DATASET_CACHE", Path.home() / ".cache" / "armory" / "dataset_cache"
        )
    )

    def __post_init__(self):
        self.armory_home.mkdir(parents=True, exist_ok=True)
        self.dataset_cache.mkdir(parents=True, exist_ok=True)

        os.environ["ARMORY_HOME"] = self.armory_home.as_posix()
        os.environ["ARMORY_DATASET_CACHE"] = self.dataset_cache.as_posix()


@dataclass
class Evaluation:
    name: str
    description: str
    model: Model
    dataset: Dataset
    author: Optional[str]
    attack: Optional[Attack] = None
    metric: Metric = field(default_factory=Metric)
    sysconfig: SysConfig = field(default_factory=SysConfig)
