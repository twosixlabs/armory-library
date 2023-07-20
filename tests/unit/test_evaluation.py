from unittest.mock import MagicMock

from art.estimators import BaseEstimator
import pytest

from armory.data.datasets import ArmoryDataGenerator
import charmory.evaluation as evaluation

pytestmark = pytest.mark.unit


###
# Fixtures
###


@pytest.fixture
def model():
    return evaluation.Model(
        name="test",
        model=MagicMock(spec=BaseEstimator),
    )


@pytest.fixture
def generator():
    return MagicMock(spec=ArmoryDataGenerator)


@pytest.fixture
def scenario():
    return evaluation.Scenario(function=str, kwargs={})


@pytest.fixture
def attack():
    return evaluation.Attack(function=str, kwargs={}, knowledge="white")


@pytest.fixture
def metric():
    return evaluation.Metric(
        profiler_type="basic",
        supported_metrics=[],
        perturbation=[],
        task=[],
        means=False,
        record_metric_per_sample=False,
    )


###
# Tests
###


def test_model_init_raises_on_invalid_model():
    with pytest.raises(AssertionError, match=r"model.*instance of"):
        evaluation.Model(name="bad model", model=42)  # type: ignore


def test_dataset_init_raises_on_invalid_test_dataset():
    with pytest.raises(AssertionError, match=r"test_dataset.*instance of"):
        evaluation.Dataset(name="bad test dataset", test_dataset=42)  # type: ignore


def test_dataset_init_raises_on_invalid_train_dataset(generator):
    with pytest.raises(AssertionError, match=r"train_dataset.*instance of"):
        evaluation.Dataset(
            name="bad train dataset",
            test_dataset=generator,
            train_dataset=42,  # type: ignore
        )


def test_dataset_init_when_no_train_dataset(generator):
    evaluation.Dataset(
        name="bad train dataset",
        test_dataset=generator,
    )


def test_dataset_init_when_train_dataset(generator):
    evaluation.Dataset(
        name="bad train dataset",
        test_dataset=generator,
        train_dataset=generator,
    )


def test_evaluation_init_raises_on_missing_train_dataset(
    model, generator, scenario, attack, metric
):
    model.fit = True
    with pytest.raises(AssertionError, match=r"not provide.*train_dataset"):
        evaluation.Evaluation(
            name="missing train dataset",
            description="test evaluation",
            author=None,
            model=model,
            dataset=evaluation.Dataset(
                name="dataset",
                test_dataset=generator,
            ),
            scenario=scenario,
            attack=attack,
            metric=metric,
        )


def test_evaluation_init_when_train_dataset(model, generator, scenario, attack, metric):
    model.fit = True
    evaluation.Evaluation(
        name="missing train dataset",
        description="test evaluation",
        author=None,
        model=model,
        dataset=evaluation.Dataset(
            name="dataset",
            test_dataset=generator,
            train_dataset=generator,
        ),
        scenario=scenario,
        attack=attack,
        metric=metric,
    )


def test_evaluation_init_when_no_train_dataset(
    model, generator, scenario, attack, metric
):
    evaluation.Evaluation(
        name="missing train dataset",
        description="test evaluation",
        author=None,
        model=model,
        dataset=evaluation.Dataset(
            name="dataset",
            test_dataset=generator,
        ),
        scenario=scenario,
        attack=attack,
        metric=metric,
    )
