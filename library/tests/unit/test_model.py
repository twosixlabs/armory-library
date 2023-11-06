from unittest.mock import Mock

import numpy as np
from numpy.testing import assert_array_equal
import pytest

from charmory.model import ArmoryModel
from charmory.model.image_classification import JaticImageClassificationModel

pytestmark = pytest.mark.unit


def test_ArmoryModel_with_preadapter():
    model = Mock(return_value=[0, 0.9, 0])

    def preadapter(x):
        return (x * 2,), dict()

    wrapper = ArmoryModel(model, preadapter=preadapter)
    assert wrapper(np.array([1, 2, 3])) == [0, 0.9, 0]

    model.assert_called_once()
    assert_array_equal(np.array([2, 4, 6]), model.call_args.args[0])


def test_ArmoryModel_with_preadapter_kwargs():
    model = Mock(return_value=[0.7, 0.2, 0])

    def preadapter(*, data, target):
        return tuple(), {"data": data * 2, "target": target}

    wrapper = ArmoryModel(model, preadapter=preadapter)
    assert wrapper(data=np.array([1, 2, 3]), target=[1, 0, 0]) == [0.7, 0.2, 0]

    model.assert_called_once()
    assert_array_equal(np.array([2, 4, 6]), model.call_args.kwargs["data"])
    assert [1, 0, 0] == model.call_args.kwargs["target"]


def test_ArmoryModel_with_postadapter():
    model = Mock(return_value={"scores": [0, 0, 0.8]})

    def postadapter(output):
        return output["scores"]

    wrapper = ArmoryModel(model, postadapter=postadapter)
    assert wrapper(np.array([4, 5, 6])) == [0, 0, 0.8]

    model.assert_called_once()
    assert_array_equal(np.array([4, 5, 6]), model.call_args.args[0])


@pytest.mark.parametrize("prop", ["logits", "probs", "scores"])
def test_JaticImageClassificationModel(prop):
    output = type("", (), {})
    setattr(output, prop, [0.1, 0.6, 0.3])

    model = Mock(return_value=output)

    wrapper = JaticImageClassificationModel(model)
    assert wrapper([1, 2, 3]) == [0.1, 0.6, 0.3]
