"""Armory Experiment Configuration Classes"""

from dataclasses import dataclass, field
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
    fit: bool = False
    fit_kwargs: Dict[str, Any] = field(default_factory=dict)
    predict_kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Scenario:
    function: MethodName
    kwargs: Dict[str, Any]
    export_batches: Optional[bool] = False


@dataclass
class SysConfig:
    # TODO: should get ArmoryControls (e.g. num_eval_batches, num_epochs, etc.)
    gpus: List[str]
    use_gpu: bool = False


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
