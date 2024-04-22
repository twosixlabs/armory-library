"""Armory Experiment Configuration Classes"""

from contextlib import contextmanager
from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Iterable,
    Mapping,
    Optional,
    Protocol,
    runtime_checkable,
)

from armory.data import Batch
from armory.export import Exporter
from armory.metric import Metric
from armory.metrics.compute import NullProfiler, Profiler
from armory.track import Trackable, track_call, trackable_context

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
    exporters: Iterable[Exporter] = field(default_factory=list)
    """Optional, sample exporter"""
    profiler: Profiler = field(default_factory=NullProfiler)
    """Optional, computational performance profiler instance"""
    sysconfig: SysConfig = field(default_factory=SysConfig)
    """Optional, host system configuration"""


class Chain(Trackable):
    """
    A single chain of the following:
    - dataset as input
    - perturbations to be applied to input samples
    - model to generate predictions
    - metrics to be calculated from model predictions
    - exporters to save samples
    """

    def __init__(
        self,
        name: str,
        dataset: Optional[Dataset] = None,
        perturbations: Optional[Iterable[PerturbationProtocol]] = None,
        model: Optional[ModelProtocol] = None,
        metrics: Optional[Mapping[str, Metric]] = None,
        exporters: Optional[Iterable[Exporter]] = None,
    ):
        super().__init__()
        self.name = name
        self.dataset = dataset
        self.perturbations = list(perturbations) if perturbations is not None else None
        self.model = model
        self.metrics = dict(metrics) if metrics is not None else None
        self.exporters = list(exporters) if exporters is not None else None
        self.track_call = track_call

    def use_dataset(self, dataset: Dataset) -> None:
        """Set the dataset to use for this evaluation chain"""
        self.dataset = dataset

    def use_perturbations(self, perturbations: Iterable[PerturbationProtocol]) -> None:
        """Set the perturbations to use for this evaluation chain"""
        self.perturbations = list(perturbations)

    def add_perturbation(self, perturbation: PerturbationProtocol) -> None:
        """Add the perturbation to this evaluation chain"""
        if self.perturbations is None:
            self.perturbations = [perturbation]
        else:
            self.perturbations.append(perturbation)

    def use_model(self, model: ModelProtocol) -> None:
        """Set the model to use for this evaluation chain"""
        self.model = model

    def use_metrics(self, metrics: Mapping[str, Metric]) -> None:
        """Set the metrics to use for this evaluation chain"""
        self.metrics = dict(metrics)

    def add_metric(self, name: str, metric: Metric) -> None:
        """Add a metric to this evaluation chain"""
        if self.metrics is None:
            self.metrics = {name: metric}
        else:
            self.metrics[name] = metric

    def use_exporters(self, exporters: Iterable[Exporter]) -> None:
        """Set the exporters to use for this evaluation chain"""
        self.exporters = list(exporters)

    def add_exporter(self, exporter: Exporter) -> None:
        """Add an exporter to this evaluation chain"""
        if self.exporters is None:
            self.exporters = [exporter]
        else:
            self.exporters.append(exporter)

    def validate(self) -> None:
        """Ensure all required components of this chain have been defined"""
        if self.dataset is None:
            raise ValueError(f"No dataset has been defined for chain {self.name}")
        if self.model is None:
            raise ValueError(f"No model has been defined for chain {self.name}")

    def get_tracked_params(self) -> Dict[str, Any]:
        """
        Return the tracked parameters for this chain and all trackable
        components of the chain
        """
        params = {}
        params.update(self.tracked_params)
        if isinstance(self.dataset, Trackable):
            params.update(self.dataset.tracked_params)
        if self.perturbations is not None:
            for perturbation in self.perturbations:
                if isinstance(perturbation, Trackable):
                    params.update(perturbation.tracked_params)
        if isinstance(self.model, Trackable):
            params.update(self.model.tracked_params)
        if self.metrics is not None:
            for metric in self.metrics.values():
                if isinstance(metric, Trackable):
                    params.update(metric.tracked_params)
        if self.exporters is not None:
            for exporter in self.exporters:
                if isinstance(exporter, Trackable):
                    params.update(exporter.tracked_params)
        return params


class NewEvaluation:
    """
    A collection of datasets, perturbations, models, and outputs defining an
    Armory evaluation.

    Each combination of dataset, optional perturbations, model, and outputs
    represents a distinct evaluation chain. Chains are executed sequentially.
    """

    def __init__(self, name: str, description: str, author: str):
        self.name = name
        self.description = description
        self.author = author
        self.default_dataset: Optional[Dataset] = None
        self.default_perturbations: Optional[Iterable[PerturbationProtocol]] = None
        self.default_model: Optional[ModelProtocol] = None
        self.default_metrics: Optional[Mapping[str, Metric]] = None
        self.default_exporters: Optional[Iterable[Exporter]] = None
        self.chains: Dict[str, Chain] = {}

    def use_dataset(self, dataset: Dataset) -> None:
        """
        Set the default dataset to use for evaluation chains that do not specify
        a dataset
        """
        self.default_dataset = dataset

    def use_perturbations(self, perturbations: Iterable[PerturbationProtocol]) -> None:
        """
        Set the default perturbations to use for evaluation chains that do not
        specify perturbations
        """
        self.default_perturbations = perturbations

    def use_model(self, model: ModelProtocol) -> None:
        """
        Set the default model to use for evaluation chains that do not specify a
        model
        """
        self.default_model = model

    def use_metrics(self, metrics: Mapping[str, Metric]) -> None:
        """
        Set the default metrics to use for evaluation chains that do not specify
        metrics
        """
        self.default_metrics = metrics

    def use_exporters(self, exporters: Iterable[Exporter]) -> None:
        """
        Set the default exporters to use for evaluation chains that do not specify
        exporters
        """
        self.default_exporters = exporters

    @contextmanager
    def autotrack(self):
        """
        Creates a context for automatically tracking parameters. Any trackable
        objects created while the context is active will be associated with the
        parameters recorded during the context.

        Example::

            with evaluation.autotrack() as track:
                model = track(load_model, split="test")

        Return:
            Utility function to invoke functions or classes while recording
            keyword arguments as parameters
        """
        with trackable_context():
            yield track_call

    @contextmanager
    def add_chain(self, name: str):
        """
        Add a new evaluation chain to the evaluation

        Args:
            name: Name of the evaluation chain

        Return:
            Chain: The evaluation chain to be configured
        """
        with trackable_context():
            chain = Chain(
                name=name,
                dataset=self.default_dataset,
                perturbations=self.default_perturbations,
                model=self.default_model,
                metrics=self.default_metrics,
                exporters=self.default_exporters,
            )
            yield chain
            chain.validate()
            self.chains[name] = chain
