import pytest

import charmory.experiment as experiment


def test_initializers():
    """Instantiate all the classes and check that they don't fault."""
    attack = experiment.Attack("art.FGSM", {"eps": 0.3}, "white", use_label=True)
    assert attack.knowledge == "white"
    assert attack.function == "art.FGSM"
    assert attack.kwargs["eps"] == 0.3
    assert attack.use_label is True

    dataset = experiment.Dataset("armory.load_mnist", "tf", 128)
    assert dataset.framework == "tf"

    with pytest.raises(TypeError):
        defense = experiment.Defense("Preprocessor", "armory.some_defense")  # type: ignore

    defense = experiment.Defense("armory.some_defense", kwargs={}, type="Preprocessor")
    assert defense.type == "Preprocessor"

    metric = experiment.Metric(
        "basic",
        supported_metrics=["accuracy"],
        perturbation=["clean"],
        means=True,
        task=["image"],
        record_metric_per_sample=False,
    )
    assert metric.profiler_type == "basic"

    model = experiment.Model(
        "msw.a.model",
        weights_file=None,
        wrapper_kwargs={},
        model_kwargs={},
        fit=True,
        fit_kwargs={},
    )
    assert model.function == "msw.a.model"

    scenario = experiment.Scenario("armory.scenarios.image_classification", {})
    assert scenario.function == "armory.scenarios.image_classification"

    sysconfig = experiment.SysConfig(gpus=["0", "2"], use_gpu=True)
    assert "2" in sysconfig.gpus

    metadata = experiment.MetaData("null experiment", "test", "msw <msw@example.com>")
    assert metadata.name == "null experiment"

    with pytest.raises(TypeError):
        bad = experiment.Experiment()  # type: ignore
        assert bad._metadata.name == "null experiment"

    exp = experiment.Experiment(
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
    """Instantiate a full experiment as in mnist_baseline.json"""
    metadata = experiment.MetaData(
        "mnist experiment", "derived from mnist_baseline.json", "msw@example.com"
    )
    exp = experiment.Experiment(
        metadata,
        model=experiment.Model(
            "armory.baseline_models.keras.mnist.get_art_model",
            weights_file=None,
            wrapper_kwargs={},
            model_kwargs={},
            fit=True,
            fit_kwargs={"nb_epochs": 20},
        ),
        dataset=experiment.Dataset(
            "armory.data.datasets.mnist",
            framework="numpy",
            batch_size=128,
        ),
        scenario=experiment.Scenario(
            "armory.scenarios.image_classification.ImageClassificationTask", kwargs={}
        ),
        attack=experiment.Attack(
            "art.attacks.evasion.FastGradientMethod",
            {
                "batch_size": 1,
                "eps": 0.2,
                "eps_step": 0.1,
                "minimal": False,
                "num_random_init": 0,
                "targeted": False,
            },
            knowledge="white",
            use_label=True,
        ),
        defense=None,
        metric=experiment.Metric(
            profiler_type="basic",
            task=["categorical_accuracy"],
            supported_metrics=["accuracy"],
            perturbation=["linf"],
            means=True,
            record_metric_per_sample=False,
        ),
        sysconfig=experiment.SysConfig(gpus=["all"], use_gpu=True),
    )

    assert exp.asdict() == {
        "_metadata": {
            "name": "mnist experiment",
            "description": "derived from mnist_baseline.json",
            "author": "msw@example.com",
        },
        "model": {
            "function": "armory.baseline_models.keras.mnist.get_art_model",
            "model_kwargs": {},
            "wrapper_kwargs": {},
            "weights_file": None,
            "fit": True,
            "fit_kwargs": {"nb_epochs": 20},
        },
        "scenario": {
            "function": "armory.scenarios.image_classification.ImageClassificationTask",
            "kwargs": {},
        },
        "dataset": {
            "function": "armory.data.datasets.mnist",
            "framework": "numpy",
            "batch_size": 128,
        },
        "attack": {
            "function": "art.attacks.evasion.FastGradientMethod",
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

    return str(exp)
