from unittest.mock import MagicMock

import pytest

import armory.evaluation

pytestmark = pytest.mark.unit


def drop_func_params(params):
    return {k: v for k, v in params.items() if not k.endswith("._func")}


@pytest.fixture
def evaluation():
    return armory.evaluation.Evaluation(name="test", description="test", author="test")


@pytest.fixture
def dataset():
    ds = MagicMock(spec=armory.evaluation.Dataset)
    ds.tracked_params = {}
    return ds


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


def test_chain_has_tracked_params_from_self(evaluation, dataset, model):
    with evaluation.add_chain("test") as chain:
        chain.track_call(lambda a, b: None, a=1, b=2)

        chain.use_dataset(dataset)
        chain.use_model(model)

    params = drop_func_params(evaluation.chains["test"].get_tracked_params())
    assert params == {"<lambda>.a": 1, "<lambda>.b": 2}


class MockTrackable(armory.evaluation.Trackable):
    pass


def test_chain_has_tracked_params_from_dataset(evaluation, model):
    with evaluation.autotrack() as track:
        track(lambda a, b: None, a=1, b=2)
        dataset = MockTrackable()

    with evaluation.add_chain("test") as chain:
        chain.use_dataset(dataset)
        chain.use_model(model)

    params = drop_func_params(evaluation.chains["test"].get_tracked_params())
    assert params == {"<lambda>.a": 1, "<lambda>.b": 2}


def test_chain_has_tracked_params_from_perturbation(evaluation, dataset, model):
    with evaluation.autotrack() as track:
        track(lambda a, b: None, a=1, b=2)
        perturbation = MockTrackable()

    with evaluation.add_chain("test") as chain:
        chain.use_dataset(dataset)
        chain.use_model(model)
        chain.use_perturbations([perturbation])

    params = drop_func_params(evaluation.chains["test"].get_tracked_params())
    assert params == {"<lambda>.a": 1, "<lambda>.b": 2}


def test_chain_has_tracked_params_from_model(evaluation, dataset):
    with evaluation.autotrack() as track:
        track(lambda a, b: None, a=1, b=2)
        model = MockTrackable()

    with evaluation.add_chain("test") as chain:
        chain.use_dataset(dataset)
        chain.use_model(model)

    params = drop_func_params(evaluation.chains["test"].get_tracked_params())
    assert params == {"<lambda>.a": 1, "<lambda>.b": 2}


def test_chain_has_tracked_params_from_metric(evaluation, dataset, model):
    with evaluation.autotrack() as track:
        track(lambda a, b: None, a=1, b=2)
        metric = MockTrackable()

    with evaluation.add_chain("test") as chain:
        chain.use_dataset(dataset)
        chain.use_model(model)
        chain.use_metrics({"test": metric})

    params = drop_func_params(evaluation.chains["test"].get_tracked_params())
    assert params == {"<lambda>.a": 1, "<lambda>.b": 2}


def test_chain_has_tracked_params_from_exporter(evaluation, dataset, model):
    with evaluation.autotrack() as track:
        track(lambda a, b: None, a=1, b=2)
        exporter = MockTrackable()

    with evaluation.add_chain("test") as chain:
        chain.use_dataset(dataset)
        chain.use_model(model)
        chain.use_exporters([exporter])

    params = drop_func_params(evaluation.chains["test"].get_tracked_params())
    assert params == {"<lambda>.a": 1, "<lambda>.b": 2}
