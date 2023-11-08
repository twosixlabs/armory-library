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
    """Configuration for the attack to be applied during model evaluation"""

    name: str
    """Descriptive name of the attack"""
    attack: EvasionAttack
    """Evasion attack instance"""
    generate_kwargs: Dict[str, Any] = field(default_factory=dict)
    """
    Optional, additional keyword arguments to be used with the evasion attack's
    `generate` method
    """
    use_label_for_untargeted: bool = False
    """
    When the attack is untargeted, set to `True` to use the natural labels as
    the `y` argument to the evasion attack's `generate` method. When `False`,
    the `y` argument will be `None`.
    """
    label_targeter: Optional[LabelTargeter] = None
    """
    Required when the attack is targeted, the label targeter generates the
    target label that is used as the `y` argument to the evasion attack's
    `generate` method.
    """

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
        """
        Whether the attack is targeted. When an attack is targeted, it attempts
        to optimize the perturbation such that the model's prediction of the
        perturbed input matches a desired (targeted) result. When untargeted,
        the attack may use the natural label as a hint of the prediction result
        to optimize _away from_.
        """
        return self.attack.targeted


@dataclass
class Dataset:
    """Configuration for the dataset to be used for model evaluation"""

    name: str
    """Descriptive name of the dataset"""
    x_key: str
    """
    Key in the sample dictionaries containing the data to be used for model
    inference
    """
    y_key: str
    """Key in the sample dictionaries containing the natural labels"""
    test_dataloader: DataLoader
    """Data loader for evaluation data"""
    train_dataloader: Optional[DataLoader] = None
    """Optional, data loader for training data"""

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
    """Configuration for the metrics collected during model evaluation"""

    profiler: Profiler = field(default_factory=NullProfiler)
    """Computational performance profiler instance"""


@dataclass
class Model:
    """Configuration for the model being evaluated"""

    name: str
    """Descriptive name of the model"""
    model: BaseEstimator
    """Model, wrapped in an ART estimator"""
    predict_kwargs: Dict[str, Any] = field(default_factory=dict)
    """
    Optional, additional keyword arguments to be used with the estimator's
    `predict` method
    """

    def __post_init__(self):
        assert isinstance(
            self.model, BaseEstimator
        ), "Evaluation model is not an instance of BaseEstimator"


@dataclass
class SysConfig:
    """
    Host system configuration. This should only need to be instantiated to
    customize the default system configuration.
    """

    armory_home: Path = Path(os.getenv("ARMORY_HOME", Path.home() / ".armory"))
    """
    Path to the user-specific folder where Armory will store evaluation results,
    cached data, etc.
    """
    dataset_cache: Path = Path(
        os.getenv(
            "ARMORY_DATASET_CACHE", Path.home() / ".cache" / "armory" / "dataset_cache"
        )
    )
    """Path to folder used as a dataset cache"""

    def __post_init__(self):
        """Ensures all directories exist"""
        self.armory_home.mkdir(parents=True, exist_ok=True)
        self.dataset_cache.mkdir(parents=True, exist_ok=True)

        # These are being set for legacy code that does not have access to a
        # SysConfig object
        os.environ["ARMORY_HOME"] = self.armory_home.as_posix()
        os.environ["ARMORY_DATASET_CACHE"] = self.dataset_cache.as_posix()


@dataclass
class Evaluation:
    """Configuration of an Armory model evaluation"""

    name: str
    """
    Short name for the evaluation. This will be used for the experiment name in
    MLflow
    """
    description: str
    """Full description of the evaluation"""
    model: Model
    """Configuration for the model being evaluated"""
    dataset: Dataset
    """Configuration for the dataset to be used for evaluation"""
    author: Optional[str]
    """Optional, author to which to attribute evaluation results"""
    attack: Optional[Attack] = None
    """Optional, configuration for the attack to be applied during evaluation"""
    metric: Metric = field(default_factory=Metric)
    """Optional, configuration for the metrics collected during model evaluation"""
    sysconfig: SysConfig = field(default_factory=SysConfig)
    """Optional, host system configuration"""
