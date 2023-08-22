"""Evaluation components for CIFAR10 baseline."""
import art.attacks.evasion

import armory.baseline_models.pytorch.cifar
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

dataset = Dataset(
    name="CIFAR10",
    train_dataset=armory.data.datasets.cifar10(
        split="train",
        epochs=20,
        batch_size=64,
        shuffle_files=True,
    ),
    test_dataset=armory.data.datasets.cifar10(
        split="test",
        epochs=1,
        batch_size=64,
        shuffle_files=False,
    ),
)

classifier = armory.baseline_models.pytorch.cifar.get_art_model(
    model_kwargs={},
    wrapper_kwargs={},
    weights_path=None,
)

model = Model(
    name="pytorch cifar",
    model=classifier,
)

attack = Attack(
    name="PGD",
    attack=art.attacks.evasion.ProjectedGradientDescent(
        classifier,
        batch_size=1,
        eps=0.031,
        eps_step=0.007,
        max_iter=20,
        num_random_init=1,
        random_eps=False,
        targeted=False,
        verbose=False,
    ),
    use_label_for_untargeted=True,
)


scenario = Scenario(
    function=charmory.scenarios.image_classification.ImageClassificationTask,
    kwargs={},
    export_batches=True,
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
    name="cifar_baseline",
    description="Baseline cifar10 image classification",
    author="msw@example.com",
    dataset=dataset,
    model=model,
    attack=attack,
    scenario=scenario,
    metric=metric,
    sysconfig=sysconfig,
)
