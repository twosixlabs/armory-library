from unittest.mock import MagicMock, call

from art.attacks import Attack as ArtAttack
import numpy as np
import numpy.testing as nptest
import pytest

from charmory.scenario import Scenario

# These tests use fixtures from conftest.py


pytestmark = pytest.mark.unit


###
# Fixtures
###


class TestScenario(Scenario):
    def _load_sample_exporter(self):
        return MagicMock()


###
# Tests
###


def test_run_benign_when_skip_misclassified_disabled(evaluation):
    x = np.array([0, 1, 2])
    batch = Scenario.Batch(i=0, x=x, y=np.array([0.0, 0.0, 1.0]))
    y_pred = np.array([0.1, 0.8, 0.1])

    evaluation.model.model.predict = MagicMock(return_value=y_pred)

    scenario = TestScenario(evaluation)
    scenario.run_benign(batch)

    nptest.assert_array_equal(batch.y_pred, y_pred)
    assert batch.misclassified is None


def test_run_benign_when_skip_misclassified_enabled(evaluation):
    x = np.array([0, 1, 2])
    batch = Scenario.Batch(i=0, x=x, y=np.array([0.0, 0.0, 1.0]))
    y_pred = np.array([0.1, 0.8, 0.1])

    evaluation.model.model.predict = MagicMock(return_value=y_pred)

    scenario = TestScenario(evaluation, skip_misclassified=True)
    scenario.run_benign(batch)

    nptest.assert_array_equal(batch.y_pred, y_pred)
    assert batch.misclassified is True


def test_run_attack_when_skip_misclassified_enabled(evaluation):
    batch = Scenario.Batch(
        i=0,
        x=np.array([0, 1, 2]),
        y=np.array([0.0, 0.0, 1.0]),
        y_pred=np.array([0.1, 0.8, 0.1]),
        misclassified=True,
    )
    evaluation.model.model.predict = MagicMock()

    scenario = TestScenario(evaluation, skip_misclassified=True)
    scenario.hub.set_context(batch=0)
    scenario.run_attack(batch)

    evaluation.model.model.predict.assert_not_called()
    nptest.assert_array_equal(batch.x_adv, batch.x)
    nptest.assert_array_equal(batch.y_pred_adv, batch.y_pred)
    assert batch.y_target is None


def test_run_attack(evaluation):
    batch = Scenario.Batch(
        i=0,
        x=np.array([0, 1, 2]),
        y=np.array([0.0, 0.0, 1.0]),
        y_pred=np.array([0.1, 0.1, 0.8]),
        misclassified=True,
    )
    x_adv = np.array([0.0, 0.9, 2.1])
    y_pred_adv = np.array([0.1, 0.8, 0.1])
    evaluation.model.model.predict = MagicMock(return_value=y_pred_adv)

    attack = MagicMock(spec=ArtAttack)
    attack.generate = MagicMock(return_value=x_adv)
    evaluation.attack.function = MagicMock(return_value=attack)

    scenario = TestScenario(evaluation)
    scenario.hub.set_context(batch=0)
    scenario.run_attack(batch)

    scenario.attack.generate.assert_called_with(x=batch.x, y=None)
    evaluation.model.model.predict.assert_called_with(x_adv)
    nptest.assert_array_equal(batch.x_adv, x_adv)
    nptest.assert_array_equal(batch.y_pred_adv, y_pred_adv)
    assert batch.y_target is None


# TODO write more tests for run_attack after attack refactor


def test_evaluate_current_when_skip_benign_enabled(evaluation):
    scenario = TestScenario(evaluation, skip_benign=True)
    scenario.run_benign = MagicMock()
    scenario.run_attack = MagicMock()

    scenario.evaluate_current(Scenario.Batch(i=0, x=0, y=0))

    scenario.run_benign.assert_not_called()
    scenario.run_attack.assert_called()


def test_evaluate_current_when_skip_attack_enabled(evaluation):
    scenario = TestScenario(evaluation, skip_attack=True)
    scenario.run_benign = MagicMock()
    scenario.run_attack = MagicMock()

    scenario.evaluate_current(Scenario.Batch(i=0, x=0, y=0))

    scenario.run_benign.assert_called()
    scenario.run_attack.assert_not_called()


def test_evaluate_all(evaluation):
    scenario = TestScenario(evaluation)
    scenario.test_dataset = MagicMock()
    scenario.test_dataset.__next__ = MagicMock()
    scenario.test_dataset.__next__.side_effect = [("x1", "y1"), ("x2", "y2")]
    scenario.test_dataset.__len__ = MagicMock(return_value=2)

    scenario.evaluate_current = MagicMock()
    scenario.evaluate_all()

    scenario.evaluate_current.assert_has_calls(
        [
            call(Scenario.Batch(i=0, x="x1", y="y1")),
            call(Scenario.Batch(i=1, x="x2", y="y2")),
        ]
    )
