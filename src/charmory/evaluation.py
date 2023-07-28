"""Armory Experiment Configuration Classes"""

from dataclasses import dataclass, field
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
    gpus: List[str]
    use_gpu: bool = False
    paths: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        # TODO: Discuss/refactor the following
        import json

        armory_dir = Path.home() / ".armory"
        self.paths = {
            "armory_dir": str(armory_dir),
            "armory_config": str(armory_dir / "config.json"),
            "dataset_dir": str(armory_dir / "datasets"),
            "local_git_dir": str(armory_dir / "git"),
            "saved_model_dir": str(armory_dir / "saved_models"),
            "pytorch_dir": str(armory_dir / "saved_models" / "pytorch"),
            "tmp_dir": str(armory_dir / "tmp"),
            "output_dir": str(armory_dir / "outputs"),
            "external_repo_dir": str(armory_dir / "tmp" / "external"),
        }

        # Load config and update paths
        if Path(self.paths["armory_config"]).exists():
            _config = json.loads(Path(self.paths["armory_config"]).read_text())
            for k in (
                "dataset_dir",
                "local_git_dir",
                "saved_model_dir",
                "output_dir",
                "tmp_dir",
            ):
                setattr(self, k, armory_dir / _config[k])

        # Create directories
        for d in self.paths.values():
            if not d.endswith(".json"):
                Path(d).mkdir(parents=True, exist_ok=True)


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
