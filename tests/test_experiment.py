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


def test_mnist_experiment():
    metadata = experiment.MetaData("mnist experiment", "test", "msw@example.com")
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
        sysconfig=experiment.SysConfig(gpus=["all", "2"], use_gpu=True),
    )

    print(exp)
