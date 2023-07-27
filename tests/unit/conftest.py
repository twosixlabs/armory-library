from unittest.mock import MagicMock

from art.attacks import Attack as ArtAttack
from art.estimators import BaseEstimator
import pytest

from armory.data.datasets import ArmoryDataGenerator
import charmory.evaluation
from charmory.scenario import Scenario


@pytest.fixture
def evaluation_model():
    return charmory.evaluation.Model(
        name="test",
        model=MagicMock(spec=BaseEstimator),
    )


@pytest.fixture
def data_generator():
    return MagicMock(spec=ArmoryDataGenerator)


@pytest.fixture
def evaluation_dataset(data_generator):
    return charmory.evaluation.Dataset(
        name="test",
        test_dataset=data_generator,
    )


@pytest.fixture
def evaluation_scenario():
    return charmory.evaluation.Scenario(
        function=lambda _: MagicMock(spec=Scenario), kwargs={}
    )


@pytest.fixture
def evaluation_attack():
    return charmory.evaluation.Attack(name="test", attack=MagicMock(spec=ArtAttack))


@pytest.fixture
def evaluation_metric():
    return charmory.evaluation.Metric(
        profiler_type="basic",
        supported_metrics=[],
        perturbation=[],
        task=[],
        means=False,
        record_metric_per_sample=False,
    )


@pytest.fixture
def evaluation(
    evaluation_model,
    evaluation_dataset,
    evaluation_scenario,
    evaluation_attack,
    evaluation_metric,
):
    return charmory.evaluation.Evaluation(
        name="test",
        description="test evaluation",
        author=None,
        model=evaluation_model,
        dataset=evaluation_dataset,
        scenario=evaluation_scenario,
        attack=evaluation_attack,
        metric=evaluation_metric,
    )
