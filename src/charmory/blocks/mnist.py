"""Evaluation components for MNIST baseline."""
import art.attacks.evasion

import armory.baseline_models.keras.mnist
import armory.data.datasets
from armory.instrument.config import MetricsLogger
from charmory.evaluation import (
    Attack,
    Dataset,
    Evaluation,
    Metric,
    Model,
    Scenario,
    SysConfig,
)
import charmory.scenarios.image_classification

"""These pieces have fully qualified names which allows them to be used
like

    from charmory.blocks import mnist
    print(mnist.dataset)
"""

dataset = Dataset(
    name="MNIST",
    train_dataset=armory.data.datasets.mnist(
        split="train",
        epochs=20,
        batch_size=128,
        shuffle_files=True,
    ),
    test_dataset=armory.data.datasets.mnist(
        split="test",
        epochs=1,
        batch_size=128,
        shuffle_files=False,
    ),
)

classifier = armory.baseline_models.keras.mnist.get_art_model(
    model_kwargs={},
    wrapper_kwargs={},
    weights_path=None,
)

model = Model(
    name="keras mnist",
    model=classifier,
)

attack = Attack(
    name="fast gradient",
    attack=art.attacks.evasion.FastGradientMethod(
        classifier,
        batch_size=1,
        eps=0.2,
        eps_step=0.1,
        minimal=False,
        num_random_init=0,
        targeted=False,
    ),
    use_label_for_untargeted=True,
)


scenario = Scenario(
    function=charmory.scenarios.image_classification.ImageClassificationTask,
    kwargs={},
)

metric = Metric(
    logger=MetricsLogger(
        supported_metrics=["accuracy"],
        perturbation=["linf"],
        task=["categorical_accuracy"],
        means=True,
        record_metric_per_sample=False,
    ),
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
    metric=metric,
    sysconfig=sysconfig,
)
