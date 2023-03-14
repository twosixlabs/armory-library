import pytest
import charmory.experiment as experiment


def test_initializers():
    attack = experiment.Attack("white", "art.FGSM", {"eps": 0.3}, use_label=True)
    assert attack.knowledge == "white"
    assert attack.function == "art.FGSM"
    assert attack.kwargs["eps"] == 0.3
    assert attack.use_label is True

    dataset = experiment.Dataset("armory.load_mnist", "tf", 128)
    assert dataset.framework == "tf"

    with pytest.raises(TypeError):
        defense = experiment.Defense("Preprocessor", "armory.some_defense")  # type: ignore

    defense = experiment.Defense("Preprocessor", "armory.some_defense", {})
    assert defense.type == "Preprocessor"

    metric = experiment.Metric(
        "basic",
        False,
        supported_metrics=["accuracy"],
        perturbation=["clean"],
        means=True,
        task=["image"],
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
        assert bad.metadata.name == "null experiment"

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
    assert exp.metadata.name == "null experiment"
