from unittest.mock import MagicMock

import pytest

import armory.evaluation

pytestmark = pytest.mark.unit


@pytest.fixture
def evaluation():
    return armory.evaluation.NewEvaluation(
        name="test", description="test", author="test"
    )


@pytest.fixture
def dataset():
    return MagicMock(spec=armory.evaluation.Dataset)


@pytest.fixture
def perturbation():
    return MagicMock(spec=armory.evaluation.PerturbationProtocol)


@pytest.fixture
def model():
    return MagicMock(spec=armory.evaluation.ModelProtocol)


@pytest.fixture
def metric():
    return MagicMock(spec=armory.evaluation.Metric)


@pytest.fixture
def exporter():
    return MagicMock(spec=armory.evaluation.Exporter)


def test_chain_uses_default_dataset(evaluation, dataset, model):
    evaluation.use_dataset(dataset)
    with evaluation.add_chain("test") as chain:
        chain.use_model(model)
    assert evaluation.chains["test"].dataset == dataset


def test_chain_uses_specific_dataset(evaluation, dataset, model):
    evaluation.use_dataset(MagicMock())
    with evaluation.add_chain("test") as chain:
        chain.use_dataset(dataset)
        chain.use_model(model)
    assert evaluation.chains["test"].dataset == dataset


def test_chain_uses_default_perturbations(evaluation, dataset, model, perturbation):
    evaluation.use_perturbations([perturbation])
    with evaluation.add_chain("test") as chain:
        chain.use_dataset(dataset)
        chain.use_model(model)
    assert evaluation.chains["test"].perturbations == [perturbation]


def test_chain_uses_specific_perturbations(evaluation, dataset, model, perturbation):
    evaluation.use_perturbations([MagicMock()])
    with evaluation.add_chain("test") as chain:
        chain.use_dataset(dataset)
        chain.use_model(model)
        chain.use_perturbations([perturbation])
    assert evaluation.chains["test"].perturbations == [perturbation]


def test_chain_adds_to_default_perturbations(
    evaluation,
    dataset,
    model,
    perturbation,
):
    evaluation.use_perturbations([perturbation])
    with evaluation.add_chain("test") as chain:
        chain.use_dataset(dataset)
        chain.use_model(model)
        chain.add_perturbation(perturbation)
    assert evaluation.chains["test"].perturbations == [perturbation, perturbation]


def test_chain_uses_default_model(evaluation, dataset, model):
    evaluation.use_model(model)
    with evaluation.add_chain("test") as chain:
        chain.use_dataset(dataset)
    assert evaluation.chains["test"].model == model


def test_chain_uses_specific_model(evaluation, dataset, model):
    evaluation.use_model(MagicMock())
    with evaluation.add_chain("test") as chain:
        chain.use_model(model)
        chain.use_dataset(dataset)
    assert evaluation.chains["test"].model == model


def test_chain_uses_default_metrics(evaluation, dataset, model, metric):
    evaluation.use_metrics({"test": metric})
    with evaluation.add_chain("test") as chain:
        chain.use_dataset(dataset)
        chain.use_model(model)
    assert evaluation.chains["test"].metrics == {"test": metric}


def test_chain_uses_specific_metrics(evaluation, dataset, model, metric):
    evaluation.use_metrics({"test": MagicMock()})
    with evaluation.add_chain("test") as chain:
        chain.use_dataset(dataset)
        chain.use_model(model)
        chain.use_metrics({"test": metric})
    assert evaluation.chains["test"].metrics == {"test": metric}


def test_chain_adds_to_default_metrics(evaluation, dataset, model, metric):
    evaluation.use_metrics({"default": metric})
    with evaluation.add_chain("test") as chain:
        chain.use_dataset(dataset)
        chain.use_model(model)
        chain.add_metric("specific", metric)
    assert evaluation.chains["test"].metrics == {"default": metric, "specific": metric}


def test_chain_uses_default_exporters(evaluation, dataset, model, exporter):
    evaluation.use_exporters([exporter])
    with evaluation.add_chain("test") as chain:
        chain.use_dataset(dataset)
        chain.use_model(model)
    assert evaluation.chains["test"].exporters == [exporter]


def test_chain_uses_specific_exporters(evaluation, dataset, model, exporter):
    evaluation.use_exporters([MagicMock()])
    with evaluation.add_chain("test") as chain:
        chain.use_dataset(dataset)
        chain.use_model(model)
        chain.use_exporters([exporter])
    assert evaluation.chains["test"].exporters == [exporter]


def test_chain_adds_to_default_exporters(evaluation, dataset, model, exporter):
    evaluation.use_exporters([exporter])
    with evaluation.add_chain("test") as chain:
        chain.use_dataset(dataset)
        chain.use_model(model)
        chain.add_exporter(exporter)
    assert evaluation.chains["test"].exporters == [exporter, exporter]


def test_chain_is_invalid_with_no_dataset(evaluation, model):
    with pytest.raises(ValueError):
        with evaluation.add_chain("test") as chain:
            chain.use_model(model)


def test_chain_is_invalid_with_no_model(evaluation, dataset):
    with pytest.raises(ValueError):
        with evaluation.add_chain("test") as chain:
            chain.use_dataset(dataset)
