from unittest.mock import MagicMock

from art.attacks import EvasionAttack
import pytest

from charmory.labels import LabelTargeter
from charmory.perturbation import ArtEvasionAttack

pytestmark = pytest.mark.unit


def test_ArtEvasionAttack_init_raises_on_invalid_attack():
    with pytest.raises(AttributeError, match=r"targeted"):
        ArtEvasionAttack(
            name="test",
            attack=42,  # type: ignore
        )


def test_ArtEvasionAttack_init_raises_when_targeted_and_use_label_for_untargeted():
    with pytest.raises(AssertionError, match=r"targeted.*use_label_for_targeted"):
        attack = MagicMock(spec=EvasionAttack)
        attack.targeted = True
        ArtEvasionAttack(
            name="test",
            attack=attack,
            label_targeter=MagicMock(spec=LabelTargeter),
            use_label_for_untargeted=True,
        )


def test_ArtEvasionAttack_init_when_targeted_and_label_targeter_provided():
    attack = MagicMock(spec=EvasionAttack)
    attack.targeted = True
    ArtEvasionAttack(
        name="test",
        attack=attack,
        label_targeter=MagicMock(spec=LabelTargeter),
    )


def test_ArtEvasionAttack_init_raises_when_untargeted_and_label_targeter_provided():
    with pytest.raises(AssertionError, match=r"untargeted.*label_targeter"):
        attack = MagicMock(spec=EvasionAttack)
        attack.targeted = False
        ArtEvasionAttack(
            name="test",
            attack=attack,
            label_targeter=MagicMock(spec=LabelTargeter),
        )
