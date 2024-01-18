"""Armory Experiment Configuration Classes"""
from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Dict,
    Iterable,
    Mapping,
    Optional,
    Protocol,
    runtime_checkable,
)

from armory.data import Batch
from armory.export import Exporter, NullExporter
from armory.metric import Metric
from armory.metrics.compute import NullProfiler, Profiler

if TYPE_CHECKING:
    from torch.utils.data.dataloader import DataLoader


@dataclass
class Dataset:
    """Configuration for the dataset to be used for model evaluation"""

    name: str
    """Descriptive name of the dataset"""

    dataloader: "DataLoader"
    """Data loader for evaluation data"""


@runtime_checkable
class ModelProtocol(Protocol):
    """Model being evaluated"""

    name: str
    """Descriptive name of the model"""

    def predict(self, batch: Batch):
        """Executes the model to generate predictions from the given batch"""
        ...


@runtime_checkable
class PerturbationProtocol(Protocol):
    """A perturbation that can be applied to dataset batches"""

    name: str
    """Descriptive name of the perturbation"""

    def apply(self, batch: Batch) -> None:
        """Applies a perturbation to the given batch"""
        ...


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
    model: ModelProtocol
    """Configuration for the model being evaluated"""
    dataset: Dataset
    """Configuration for the dataset to be used for evaluation"""
    author: Optional[str]
    """Optional, author to which to attribute evaluation results"""
    perturbations: Dict[str, Iterable[PerturbationProtocol]] = field(
        default_factory=dict
    )
    """Optional, perturbation chains to be applied during evaluation"""
    metrics: Mapping[str, Metric] = field(default_factory=dict)
    """Optional, dictionary of metric names to metric collection objects"""
    exporter: Exporter = field(default_factory=NullExporter)
    """Optional, sample exporter"""
    profiler: Profiler = field(default_factory=NullProfiler)
    """Optional, computational performance profiler instance"""
    sysconfig: SysConfig = field(default_factory=SysConfig)
    """Optional, host system configuration"""
