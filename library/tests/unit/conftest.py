from unittest.mock import MagicMock

from art.estimators import BaseEstimator
import pytest
from torch.utils.data.dataloader import DataLoader

import charmory.evaluation
from charmory.perturbation import Perturbation


@pytest.fixture
def evaluation_model():
    return charmory.evaluation.Model(
        name="test",
        model=MagicMock(spec=BaseEstimator),
    )


@pytest.fixture
def data_loader():
    return MagicMock(spec=DataLoader)


@pytest.fixture
def evaluation_dataset(data_loader):
    return charmory.evaluation.Dataset(
        name="test",
        test_dataloader=data_loader,
        x_key="data",
        y_key="target",
    )


@pytest.fixture
def evaluation_perturbation():
    return MagicMock(spec=Perturbation)


@pytest.fixture
def evaluation_metric():
    return charmory.evaluation.Metric()


@pytest.fixture
def evaluation_sysconfig():
    return charmory.evaluation.SysConfig(gpus=["all"], use_gpu=True)


@pytest.fixture
def evaluation(
    evaluation_model,
    evaluation_dataset,
    evaluation_perturbation,
    evaluation_metric,
    evaluation_sysconfig,
):
    return charmory.evaluation.Evaluation(
        name="test",
        description="test evaluation",
        author=None,
        model=evaluation_model,
        dataset=evaluation_dataset,
        perturbations={"test": [evaluation_perturbation]},
        metric=evaluation_metric,
        sysconfig=evaluation_sysconfig,
    )
