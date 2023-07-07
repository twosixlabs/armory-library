"""Evaluation components for MNIST baseline."""
import art.attacks.evasion

import armory.baseline_models.keras.mnist
import armory.data.datasets
import armory.scenarios.image_classification
from charmory.evaluation import (
    Attack,
    Dataset,
    Evaluation,
    Metric,
    ModelConfig,
    Scenario,
    SysConfig,
)
from functools import partial

"""These pieces have fully qualified names which allows them to be used
like

    from charmory.blocks import mnist
    print(mnist.dataset)
"""

dataset = Dataset(
    function=armory.data.datasets.mnist, framework="numpy", batch_size=128
)

model = ModelConfig(
    name="keras mnist",
    load_model=partial(
        armory.baseline_models.keras.mnist.get_art_model,
        model_kwargs={},
        wrapper_kwargs={},
        weights_path=None,
    )
    fit=True,
    fit_kwargs={"nb_epochs": 20},
)

attack = Attack(
    function=art.attacks.evasion.FastGradientMethod,
    kwargs={
        "batch_size": 1,
        "eps": 0.2,
        "eps_step": 0.1,
        "minimal": False,
        "num_random_init": 0,
        "targeted": False,
    },
    knowledge="white",
    use_label=True,
    type=None,
)


scenario = Scenario(
    function=armory.scenarios.image_classification.ImageClassificationTask,
    kwargs={},
)

metric = Metric(
    profiler_type="basic",
    supported_metrics=["accuracy"],
    perturbation=["linf"],
    task=["categorical_accuracy"],
    means=True,
    record_metric_per_sample=False,
)

sysconfig = SysConfig(gpus=["all"], use_gpu=True)


baseline = Evaluation(
    name="mnist_baseline",
    description="derived from mnist_baseline.json",
    author="msw@example.com",
    dataset=dataset,
    model=model,
    attack=attack,
    scenario=scenario,
    defense=None,
    metric=metric,
    sysconfig=sysconfig,
)
