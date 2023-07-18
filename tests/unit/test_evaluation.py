from unittest.mock import MagicMock

from art.estimators import BaseEstimator
import pytest
from unittest.mock import MagicMock

from armory.data.datasets import ArmoryDataGenerator
import charmory.evaluation as evaluation

pytestmark = pytest.mark.unit


def test_model_init_raises_on_invalid_model():
    with pytest.raises(AssertionError, match=r"model.*instance of"):
        evaluation.Model(name="bad model", model=42)  # type: ignore


def test_dataset_init_raises_on_invalid_test_dataset():
    with pytest.raises(AssertionError, match=r"test_dataset.*instance of"):
        evaluation.Dataset(name="bad test dataset", test_dataset=42)  # type: ignore


def test_evaluation_init():
    evaluation.Evaluation(
        name="missing train dataset",
        description="test evaluation",
        author=None,
        model=evaluation.Model(
            name="test",
            model=MagicMock(spec=BaseEstimator),
        ),
        dataset=evaluation.Dataset(
            name="dataset",
            test_dataset=MagicMock(spec=ArmoryDataGenerator),
        ),
        scenario=evaluation.Scenario(function=str, kwargs={}),
        attack=evaluation.Attack(function=str, kwargs={}, knowledge="white"),
        metric=evaluation.Metric(
            profiler_type="basic",
            supported_metrics=[],
            perturbation=[],
            task=[],
            means=False,
            record_metric_per_sample=False,
        ),
    )
