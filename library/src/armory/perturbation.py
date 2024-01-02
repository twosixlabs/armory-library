"""Perturbation adaption APIs"""

from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Mapping,
    Optional,
    Protocol,
    Tuple,
    runtime_checkable,
)

import numpy as np
import torch

if TYPE_CHECKING:
    from art.attacks import EvasionAttack

    from armory.labels import LabelTargeter


@runtime_checkable
class Perturbation(Protocol):
    """A perturbation that can be applied to dataset batches"""

    name: str
    """Descriptive name of the perturbation"""

    def apply(
        self, data: np.ndarray, batch
    ) -> Tuple[np.ndarray, Optional[Mapping[str, Any]]]:
        """
        Applies a perturbation to the given batch of sample data.

        Args:
            data: (N,*) data array where N is the batch size
            batch: Additional batch metadata
                (see :class:`charmory.tasks.base.BaseEvaluationTask.Batch`)

        Returns: Tuple of perturbed data and optional perturbation metadata to
            be recorded with exported batches
        """
        ...


@dataclass
class CallablePerturbation(Perturbation):
    """
    A generic perturbation for a simple callable (e.g., a transform or
    augmentation).

    Example::

        from charmory.perturbation import CallablePerturbation
        from torchvision.transforms.v2 import GaussianBlur

        perturb = CallablePerturbation(
            name="blur",
            perturbation=GaussianBlur(kernel_size=5),
        )
    """

    name: str
    """Descriptive name of the perturbation"""
    perturbation: Callable[[np.ndarray], np.ndarray]
    """Callable accepting the input data and returning the perturbed data"""

    def apply(
        self, data: np.ndarray, batch
    ) -> Tuple[np.ndarray, Optional[Mapping[str, Any]]]:
        return self.perturbation(data), None


@dataclass
class TorchTransformPerturbation(Perturbation):
    """
    A generic perturbation for a torchvision transform.

    Example::

        from charmory.perturbation import TorchPerturbation
        from torchvision.transforms.v2 import GaussianBlur

        perturb = TorchPerturbation(
            name="blur",
            perturbation=GaussianBlur(kernel_size=5),
        )
    """

    name: str
    """Descriptive name of the perturbation"""
    perturbation: Callable[[torch.Tensor], torch.Tensor]
    """Callable accepting the input data and returning the perturbed data"""

    def apply(
        self, data: np.ndarray, batch
    ) -> Tuple[np.ndarray, Optional[Mapping[str, Any]]]:
        return self.perturbation(torch.Tensor(data)).numpy(), None


@dataclass
class ArtEvasionAttack(Perturbation):
    """
    A perturbation using an evasion attack from the Adversarial Robustness
    Toolbox (ART).

    Example::

        from art.attacks.evasion import ProjectedGradientDescent
        from charmory.perturbation import ArtEvasionAttack

        perturb = ArtEvasionAttack(
            name="PGD",
            perturbation=ProjectedGradientDescent(classifier),
            use_label_for_untargeted=False,
        )
    """

    name: str
    """Descriptive name of the attack"""
    attack: "EvasionAttack"
    """Evasion attack instance"""
    generate_kwargs: Dict[str, Any] = field(default_factory=dict)
    """
    Optional, additional keyword arguments to be used with the evasion attack's
    `generate` method
    """
    use_label_for_untargeted: bool = False
    """
    When the attack is untargeted, set to `True` to use the natural labels as
    the `y` argument to the evasion attack's `generate` method. When `False`,
    the `y` argument will be `None`.
    """
    label_targeter: Optional["LabelTargeter"] = None
    """
    Required when the attack is targeted, the label targeter generates the
    target label that is used as the `y` argument to the evasion attack's
    `generate` method.
    """

    def __post_init__(self):
        if self.targeted:
            assert (
                self.label_targeter is not None
            ), "Evaluation attack is targeted, must provide a label_targeter"
            assert (
                not self.use_label_for_untargeted
            ), "Evaluation attack is targeted, use_label_for_targeted cannot be True"
        else:
            assert (
                not self.label_targeter
            ), "Evaluation attack is untargeted, cannot use a label_targeter"

    @property
    def targeted(self) -> bool:
        """
        Whether the attack is targeted. When an attack is targeted, it attempts
        to optimize the perturbation such that the model's prediction of the
        perturbed input matches a desired (targeted) result. When untargeted,
        the attack may use the natural label as a hint of the prediction result
        to optimize _away from_.
        """
        return self.attack.targeted

    def apply(self, x: np.ndarray, batch) -> Tuple[np.ndarray, Mapping[str, Any]]:
        # If targeted, use the label targeter to generate the target label
        if self.targeted:
            if TYPE_CHECKING:
                assert self.label_targeter
            y_target = self.label_targeter.generate(batch.y)
        else:
            # If untargeted, use either the natural or benign labels
            # (when set to None, the ART attack handles the benign label)
            y_target = batch.y if self.use_label_for_untargeted else None

        return (
            self.attack.generate(x=x, y=y_target, **self.generate_kwargs),
            dict(y_target=y_target),
        )
