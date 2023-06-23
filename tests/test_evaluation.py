import art
import art.attacks.evasion

# import msw.a
import pytest

import armory
import armory.baseline_models.keras.mnist
import armory.data.datasets
import armory.scenarios
from charmory.blocks import mnist
import charmory.evaluation as evaluation


def test_initializers():
    """Instantiate all the classes and check that they don't fault."""

    # Changed the function param from art.FGSM to art.attacks.__path__ as the earlier version was not a valid function and the purpose of this unit test is independent
    # of the function itself and is simply trying to see whether the classes fault upon instantiation -RN
    attack = evaluation.Attack(
        art.attacks.__path__, {"eps": 0.3}, "white", use_label=True
    )
    assert attack.knowledge == "white"
    assert attack.function == art.attacks.__path__
    assert attack.kwargs["eps"] == 0.3
    assert attack.use_label is True

    # Changed the function given in the dataset instantiation from armory.load_mnist (which doesnt exist) to armory.data.datasets.mnist and given that all the classes
    # are simply being instantiated to verify that they dont fault upon instantation this should be fine - RN
    dataset = evaluation.Dataset(armory.data.datasets.mnist, "tf", 128)
    assert dataset.framework == "tf"

    with pytest.raises(TypeError):
        # I changed the function given in the defense object instantiation from armory.some_defense (which doesnt exist) to armory.__path__ which does exist - RN
        defense = evaluation.Defense("Preprocessor", armory.__path__)  # type: ignore

    defense = evaluation.Defense(armory.__path__, kwargs={}, type="Preprocessor")
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
        armory.baseline_models.__path__,
        weights_file=None,
        wrapper_kwargs={},
        model_kwargs={},
        fit=True,
        fit_kwargs={},
    )
    # Changed the model.function param from msw.a.model (which doesnt exist) to armory.baseline_models.__path__ as the whole purpose of the function is to test
    # whether instantiating the objects works at all.
    assert model.function == armory.baseline_models.__path__

    scenario = evaluation.Scenario(armory.scenarios.__path__, {})
    # Changed the scenario.function param from armory.scenarios.image_classification (which doesnt exist) to armory.scenario.__path__ as the whole purpose of the function
    # is to test whether instantiating the objects works at all
    assert scenario.function == armory.scenarios.__path__

    sysconfig = evaluation.SysConfig(gpus=["0", "2"], use_gpu=True)
    assert "2" in sysconfig.gpus

    with pytest.raises(TypeError):
        bad = evaluation.Evaluation()  # type: ignore
        assert bad.name == "null experiment"

    exp = evaluation.Evaluation(
        name="null experiment",
        description="test",
        author="msw <msw@example.com>",
        dataset=dataset,
        model=model,
        scenario=scenario,
        attack=attack,
        defense=defense,
        metric=metric,
        sysconfig=sysconfig,
    )
    assert exp.name == "null experiment"


def test_mnist_experiment():
    exp = mnist.baseline

    assert exp.asdict() == {
        "name": "mnist_baseline",
        "description": "derived from mnist_baseline.json",
        "author": "msw@example.com",
        "model": {
            "function": armory.baseline_models.keras.mnist.get_art_model,
            "model_kwargs": {},
            "wrapper_kwargs": {},
            "weights_file": None,
            "fit": True,
            "fit_kwargs": {"nb_epochs": 20},
        },
        "scenario": {
            "function": armory.scenarios.image_classification.ImageClassificationTask,
            "kwargs": {},
        },
        "dataset": {
            "function": armory.data.datasets.mnist,
            "framework": "numpy",
            "batch_size": 128,
        },
        "attack": {
            "function": art.attacks.evasion.FastGradientMethod,
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
