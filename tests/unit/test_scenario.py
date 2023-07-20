from unittest.mock import MagicMock

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


def test_scenario_init(evaluation):
    TestScenario(evaluation)
