import pytest

import charmory.evaluation as evaluation

# These tests use fixtures from conftest.py


pytestmark = pytest.mark.unit


def test_model_init_raises_on_invalid_model():
    with pytest.raises(AssertionError, match=r"model.*instance of"):
        evaluation.Model(name="bad model", model=42)  # type: ignore


def test_dataset_init_raises_on_invalid_test_dataset():
    with pytest.raises(AssertionError, match=r"test_dataset.*instance of"):
        evaluation.Dataset(name="bad test dataset", test_dataset=42)  # type: ignore


def test_dataset_init_raises_on_invalid_train_dataset(data_generator):
    with pytest.raises(AssertionError, match=r"train_dataset.*instance of"):
        evaluation.Dataset(
            name="bad train dataset",
            test_dataset=data_generator,
            train_dataset=42,  # type: ignore
        )


def test_dataset_init_when_no_train_dataset(data_generator):
    evaluation.Dataset(
        name="bad train dataset",
        test_dataset=data_generator,
    )


def test_dataset_init_when_train_dataset(data_generator):
    evaluation.Dataset(
        name="bad train dataset",
        test_dataset=data_generator,
        train_dataset=data_generator,
    )


def test_evaluation_init(
    evaluation_model,
    evaluation_dataset,
    evaluation_scenario,
    evaluation_attack,
    evaluation_metric,
):
    evaluation.Evaluation(
        name="test evaluation",
        description="test evaluation",
        author=None,
        model=evaluation_model,
        dataset=evaluation_dataset,
        scenario=evaluation_scenario,
        attack=evaluation_attack,
        metric=evaluation_metric,
    )
