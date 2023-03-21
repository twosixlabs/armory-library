"""Armory Experiment Configuration Classes"""

# TODO: review the Optionals with @woodall

from dataclasses import asdict, dataclass
from typing import Any, Literal, Optional

MethodName = str  # reference to a python method e.g. "armory.attacks.weakest"
StrDict = dict[str, Any]  # dictionary of string keys and any values


@dataclass
class Attack:
    function: MethodName
    kwargs: StrDict
    knowledge: Literal["white", "black"]
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
    model_kwargs: StrDict
    wrapper_kwargs: StrDict
    weights_file: Optional[list[str]]
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
class Evaluation:
    _metadata: MetaData
    model: Model
    scenario: Scenario
    dataset: Dataset
    attack: Optional[Attack] = None
    defense: Optional[Defense] = None
    metric: Optional[Metric] = None
    sysconfig: Optional[SysConfig] = None

    def asdict(self) -> dict:
        return asdict(self)

    def flatten(self):
        """return all parameters as (dot.path, value) pairs for externalization"""

        def flatten_dict(root, path):
            for key, value in root.items():
                if isinstance(value, dict):
                    yield from flatten_dict(value, path + [key])
                else:
                    yield ".".join(path + [key]), value

        return [x for x in flatten_dict(self.asdict(), [])]


# List of old armory environmental variables used in evaluations
# self.config.update({
#   "ARMORY_GITHUB_TOKEN": os.getenv("ARMORY_GITHUB_TOKEN", default=""),
#   "ARMORY_PRIVATE_S3_ID": os.getenv("ARMORY_PRIVATE_S3_ID", default=""),
#   "ARMORY_PRIVATE_S3_KEY": os.getenv("ARMORY_PRIVATE_S3_KEY", default=""),
#   "ARMORY_INCLUDE_SUBMISSION_BUCKETS": os.getenv(
#     "ARMORY_INCLUDE_SUBMISSION_BUCKETS", default=""
#   ),
#   "VERIFY_SSL": self.armory_global_config["verify_ssl"] or False,
#   "NVIDIA_VISIBLE_DEVICES": self.config["sysconfig"].get("gpus", None),
#   "PYTHONHASHSEED": self.config["sysconfig"].get("set_pythonhashseed", "0"),
#   "TORCH_HOME": paths.HostPaths().pytorch_dir,
#   environment.ARMORY_VERSION: armory.__version__,
#   # "HOME": "/tmp",
# })
