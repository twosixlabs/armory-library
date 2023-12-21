from unittest.mock import MagicMock

import pytest
from torch.utils.data.dataloader import DataLoader

import charmory.evaluation


@pytest.fixture
def evaluation_model():
    return MagicMock(spec=charmory.evaluation.ModelProtocol)


@pytest.fixture
def data_loader():
    return MagicMock(spec=DataLoader)


@pytest.fixture
def evaluation_dataset(data_loader):
    return charmory.evaluation.Dataset(
        name="test",
        dataloader=data_loader,
    )


@pytest.fixture
def evaluation_perturbation():
    return MagicMock(spec=charmory.evaluation.PerturbationProtocol)


@pytest.fixture
def evaluation_metric():
    return MagicMock(spec=charmory.evaluation.Metric)


@pytest.fixture
def evaluation_sysconfig():
    return charmory.evaluation.SysConfig()


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
        metrics={"test": evaluation_metric},
        sysconfig=evaluation_sysconfig,
    )
