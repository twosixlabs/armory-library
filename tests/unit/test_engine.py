from unittest.mock import Mock

import pytest

from charmory.engine import Engine

# These tests use fixtures from conftest.py


pytestmark = pytest.mark.unit


def test_engine_train_raises_when_missing_train_dataset(evaluation):
    evaluation.dataset.train_dataset = None
    engine = Engine(evaluation)
    with pytest.raises(AssertionError, match=r"not provide.*train_dataset"):
        engine.train()


def test_engine_train_invokes_model(data_generator, evaluation_model, evaluation):
    evaluation.dataset.train_dataset = data_generator
    evaluation_model.fit_generator = Mock()
    evaluation.model.model = evaluation_model
    engine = Engine(evaluation)
    engine.train()
    evaluation_model.fit_generator.assert_called_with(data_generator, nb_epochs=1)
    engine.train(nb_epochs=20)
    evaluation_model.fit_generator.assert_called_with(data_generator, nb_epochs=20)
