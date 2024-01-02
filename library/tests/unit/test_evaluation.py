import pytest

import charmory.evaluation as evaluation

# These tests use fixtures from conftest.py


pytestmark = pytest.mark.unit


def test_model_init_raises_on_invalid_model():
    with pytest.raises(AssertionError, match=r"model.*instance of"):
        evaluation.Model(name="bad model", model=42)  # type: ignore


def test_dataset_init_raises_on_invalid_test_dataset():
    with pytest.raises(AssertionError, match=r"test_dataloader.*instance of"):
        evaluation.Dataset(
            name="bad test dataset",
            test_dataloader=42,  # type: ignore
            x_key="data",
            y_key="target",
        )


def test_dataset_init_raises_on_invalid_train_dataset(data_loader):
    with pytest.raises(AssertionError, match=r"train_dataloader.*instance of"):
        evaluation.Dataset(
            name="bad train dataset",
            test_dataloader=data_loader,
            train_dataloader=42,  # type: ignore
            x_key="data",
            y_key="target",
        )


def test_dataset_init_when_no_train_dataset(data_loader):
    evaluation.Dataset(
        name="bad train dataset",
        test_dataloader=data_loader,
        x_key="data",
        y_key="target",
    )


def test_dataset_init_when_train_dataset(data_loader):
    evaluation.Dataset(
        name="bad train dataset",
        test_dataloader=data_loader,
        train_dataloader=data_loader,
        x_key="data",
        y_key="target",
    )


def test_evaluation_init(
    evaluation_model,
    evaluation_dataset,
    evaluation_perturbation,
    evaluation_metric,
):
    evaluation.Evaluation(
        name="test evaluation",
        description="test evaluation",
        author=None,
        model=evaluation_model,
        dataset=evaluation_dataset,
        perturbations={"test": [evaluation_perturbation]},
        metric=evaluation_metric,
    )
