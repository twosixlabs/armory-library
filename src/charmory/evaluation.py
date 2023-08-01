"""Armory Experiment Configuration Classes"""

from dataclasses import dataclass, field
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
    metric: Optional[Metric] = None
    sysconfig: Optional[SysConfig] = None
