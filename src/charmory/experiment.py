"""Armory Experiment Configuration Classes"""

# TODO: review the Optionals with @woodall

from dataclasses import dataclass
from typing import Literal, Any, Optional


MethodName = str  # reference to a python method e.g. "armory.attacks.weakest"
StrDict = dict[str, Any]  # dictionary of string keys and any values


@dataclass
class Attack:
    knowledge: Literal["white", "black"]
    # TODO: make function first arg
    function: MethodName
    kwargs: StrDict
    use_label: bool = False
    type: Optional[str] = None


@dataclass
class Dataset:
    function: MethodName
    framework: Literal["tf", "torch", "numpy"]
    batch_size: int


@dataclass
class Defense:
    function: MethodName
    kwargs: StrDict
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
    supported_metrics: list[str]
    perturbation: list[str]
    task: list[str]
    means: bool
    record_metric_per_sample: bool


@dataclass
class Model:
    function: MethodName
    weights_file: Optional[list[str]]
    wrapper_kwargs: StrDict
    model_kwargs: StrDict
    fit: bool
    fit_kwargs: StrDict


@dataclass
class Scenario:
    function: MethodName
    kwargs: StrDict


@dataclass
class SysConfig:
    # TODO: should get ArmoryControls (e.g. num_eval_batches, num_epochs, etc.)
    gpus: list[str]
    use_gpu: bool = False


@dataclass
class MetaData:
    name: str
    description: str
    author: Optional[str]


@dataclass
class Experiment:
    metadata: MetaData
    model: Model
    scenario: Scenario
    dataset: Dataset
    attack: Optional[Attack] = None
    defense: Optional[Defense] = None
    metric: Optional[Metric] = None
    sysconfig: Optional[SysConfig] = None

    # def save(self, filename):
    #     with open(filename, "w") as f:
    #         f.write(self.json())
