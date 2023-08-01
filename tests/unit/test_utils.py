from art.defences.postprocessor import GaussianNoise
from art.defences.preprocessor import JpegCompression
from art.estimators import BaseEstimator
import pytest

import charmory.utils as utils

pytestmark = pytest.mark.unit


###
# Fixtures
###


class Estimator(BaseEstimator):
    def fit(self, *args, **kwargs):
        pass

    def input_shape(self):
        pass

    def predict(self, *args, **kwargs):
        pass


@pytest.fixture
def estimator():
    return Estimator(model={}, clip_values=None)


###
# Tests
###


def test_apply_art_preprocessor_defense(estimator):
    assert not utils.is_defended(estimator)
    utils.apply_art_preprocessor_defense(
        estimator, JpegCompression(clip_values=(0.0, 1.0))
    )
    assert utils.is_defended(estimator)

    # make sure we can append to the list
    assert len(estimator.preprocessing_defences) == 1
    utils.apply_art_preprocessor_defense(
        estimator, JpegCompression(clip_values=(0.0, 1.0))
    )
    assert len(estimator.preprocessing_defences) == 2


def test_apply_art_postprocessor_defense(estimator):
    assert not utils.is_defended(estimator)
    utils.apply_art_postprocessor_defense(estimator, GaussianNoise())
    assert utils.is_defended(estimator)

    # make sure we can append to the list
    assert len(estimator.postprocessing_defences) == 1
    utils.apply_art_postprocessor_defense(estimator, GaussianNoise())
    assert len(estimator.postprocessing_defences) == 2
