import pytest

import charmory.evaluation as evaluation
from charmory.blocks import mnist


def test_initializers():
    """Instantiate all the classes and check that they don't fault."""
    attack = evaluation.Attack("art.FGSM", {"eps": 0.3}, "white", use_label=True)
    assert attack.knowledge == "white"
    assert attack.function == "art.FGSM"
    assert attack.kwargs["eps"] == 0.3
    assert attack.use_label is True

    dataset = evaluation.Dataset("armory.load_mnist", "tf", 128)
    assert dataset.framework == "tf"

    with pytest.raises(TypeError):
        defense = evaluation.Defense("Preprocessor", "armory.some_defense")  # type: ignore

    defense = evaluation.Defense("armory.some_defense", kwargs={}, type="Preprocessor")
    assert defense.type == "Preprocessor"

    metric = evaluation.Metric(
        "basic",
        supported_metrics=["accuracy"],
        perturbation=["clean"],
        means=True,
        task=["image"],
        record_metric_per_sample=False,
    )
    assert metric.profiler_type == "basic"

    model = evaluation.Model(
        "msw.a.model",
        weights_file=None,
        wrapper_kwargs={},
        model_kwargs={},
        fit=True,
        fit_kwargs={},
    )
    assert model.function == "msw.a.model"

    scenario = evaluation.Scenario("armory.scenarios.image_classification", {})
    assert scenario.function == "armory.scenarios.image_classification"

    sysconfig = evaluation.SysConfig(gpus=["0", "2"], use_gpu=True)
    assert "2" in sysconfig.gpus

    metadata = evaluation.MetaData("null experiment", "test", "msw <msw@example.com>")
    assert metadata.name == "null experiment"

    with pytest.raises(TypeError):
        bad = evaluation.Evaluation()  # type: ignore
        assert bad._metadata.name == "null experiment"

    exp = evaluation.Evaluation(
        metadata,
        model,
        scenario,
        dataset,
        attack,
        defense,
        metric,
        sysconfig,
    )
    assert exp._metadata.name == "null experiment"


def test_mnist_experiment():
    exp = mnist.baseline

    assert exp.asdict() == {
        "_metadata": {
            "name": "mnist_baseline",
            "description": "derived from mnist_baseline.json",
            "author": "msw@example.com",
        },
        "model": {
            "function": "armory.baseline_models.keras.mnist:get_art_model",
            "model_kwargs": {},
            "wrapper_kwargs": {},
            "weights_file": None,
            "fit": True,
            "fit_kwargs": {"nb_epochs": 20},
        },
        "scenario": {
            "function": "armory.scenarios.image_classification:ImageClassificationTask",
            "kwargs": {},
        },
        "dataset": {
            "function": "armory.data.datasets:mnist",
            "framework": "numpy",
            "batch_size": 128,
        },
        "attack": {
            "function": "art.attacks.evasion:FastGradientMethod",
            "kwargs": {
                "batch_size": 1,
                "eps": 0.2,
                "eps_step": 0.1,
                "minimal": False,
                "num_random_init": 0,
                "targeted": False,
            },
            "knowledge": "white",
            "use_label": True,
            "type": None,
        },
        "defense": None,
        "metric": {
            "profiler_type": "basic",
            "supported_metrics": ["accuracy"],
            "perturbation": ["linf"],
            "task": ["categorical_accuracy"],
            "means": True,
            "record_metric_per_sample": False,
        },
        "sysconfig": {"gpus": ["all"], "use_gpu": True},
    }
