from unittest.mock import MagicMock

import art
import art.attacks.evasion
import pytest

import armory
import armory.baseline_models.keras.mnist
import armory.data.datasets
import armory.scenarios
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
    dataset = evaluation.Dataset(
        name="test",
        test_dataset=armory.data.datasets.mnist(
            batch_size=128,
        ),
    )

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
        name="test",
        model=MagicMock(),
        fit=True,
        fit_kwargs={},
    )

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
