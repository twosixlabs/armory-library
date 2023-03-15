# flake8: noqa
# psuedocode from the tiga days showing the intent of the Experiment class

from importlib import import_module
import json
import os

import yaml

from armory.logs import log
from armory.utils import parse_overrides


class Attack:
    name: str
    module: str
    knowledge: str
    kwargs: dict
    type: str = None


class Dataset:
    name: str
    module: str
    framework: str
    batch_size: int


class Defense:
    name: str
    type: str
    module: str
    kwargs: dict


class Metric:
    means: bool
    perturbation: str
    record_metric_per_sample: bool
    task: list


class Model:
    name: str
    module: str
    weights_file: str = None
    wrapper_kwargs: dict
    model_kwargs: dict
    fit_kwargs: dict
    fit: bool


class Scenario:
    function_name: str
    module_name: str
    kwargs: dict


class SystemConfiguration:
    docker_image: str = None
    gpus: str = None
    external_github_repo: str = None
    output_dir: str = None
    output_filename: str = None
    use_gpu: bool = False


class MetaData:
    name: str
    author: str
    description: str


class Poison:
    pass


class Experiment:
    _meta: MetaData
    poison: Poison = None
    attack: Attack = None
    dataset: Dataset
    defense: Defense = None
    metric: Metric = None
    model: Model
    scenario: Scenario
    # sysconfig: SystemConfiguration = None

    # def save(self, filename):
    #     with open(filename, "w") as f:
    #         f.write(self.json())


class Experiment(object):
    """Execution Class to `run` armory experiments"""

    def __init__(self, experiment_, environment_):
        log.info(f"Constructing Experiment using : \n{experiment_}")
        self.exp_pars = experiment_
        self.env_pars = environment_
        log.info(f"Importing Scenario Module: {self.exp_pars.scenario.module_name}")
        self.scenario_module = import_module(self.exp_pars.scenario.module_name)
        log.info(f"Loading Scenario Function: {self.exp_pars.scenario.function_name}")
        self.scenario_fn = getattr(
            self.scenario_module, self.exp_pars.scenario.function_name
        )
