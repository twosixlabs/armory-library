from armory.data.datasets import ArmoryDataGenerator
from art.estimators import BaseEstimator
from charmory.evaluation import (
    Attack,
    Dataset,
    Evaluation,
    Metric,
    Model,
    Scenario as EvalScenario,
)
from charmory.scenario import Scenario
import pytest
from unittest.mock import MagicMock, patch


pytestmark = pytest.mark.unit


###
# Fixtures
###


@pytest.fixture
def model():
    return Model(
        name="test",
        model=MagicMock(spec=BaseEstimator),
    )


@pytest.fixture
def test_dataset():
    return MagicMock(spec=ArmoryDataGenerator)


@pytest.fixture
def train_dataset():
    return MagicMock(spec=ArmoryDataGenerator)


@pytest.fixture
def dataset(test_dataset):
    return Dataset(
        name="test",
        test_dataset=test_dataset,
    )


@pytest.fixture
def attack():
    return Attack(function=str, kwargs={}, knowledge="white")


@pytest.fixture
def metric():
    return Metric(
        profiler_type="basic",
        supported_metrics=[],
        perturbation=[],
        task=[],
        means=False,
        record_metric_per_sample=False,
    )


@pytest.fixture
def evaluation(model, dataset, attack, metric):
    return Evaluation(
        name="test",
        description="test evaluation",
        author=None,
        model=model,
        dataset=dataset,
        scenario=EvalScenario(function=str, kwargs={}),
        attack=attack,
        metric=metric,
    )


class TestScenario(Scenario):
    def _load_sample_exporter(self):
        return MagicMock()


###
# Tests
###


def test_init_raises_on_invalid_model(evaluation):
    evaluation.model.model = 42
    with pytest.raises(AssertionError, match=r"model.*instance of"):
        TestScenario(evaluation)


def test_init_raises_on_invalid_test_dataset(evaluation):
    evaluation.dataset.test_dataset = 42
    with pytest.raises(AssertionError, match=r"test_dataset.*instance of"):
        TestScenario(evaluation)


def test_init_raises_on_missing_train_dataset(evaluation):
    evaluation.model.fit = True
    with pytest.raises(AssertionError, match=r"not provide.*train_dataset"):
        TestScenario(evaluation)


def test_init_raises_on_invalid_train_dataset(evaluation):
    evaluation.model.fit = True
    evaluation.dataset.train_dataset = 42
    with pytest.raises(AssertionError, match=r"train_dataset.*instance of"):
        TestScenario(evaluation)


def test_init_does_not_train_model(evaluation):
    with patch.object(Scenario, "fit") as mock_fit:
        TestScenario(evaluation)
        mock_fit.assert_not_called()


def test_init_trains_model(evaluation, train_dataset):
    evaluation.model.fit = True
    evaluation.dataset.train_dataset = train_dataset
    with patch.object(Scenario, "fit") as mock_fit:
        TestScenario(evaluation)
        mock_fit.assert_called_once()
