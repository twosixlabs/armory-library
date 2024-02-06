"""Perturbation adaption APIs"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Dict, Generic, Optional, TypeVar

from armory.data import Accessor, Batch, DefaultNumpyAccessor
from armory.evaluation import PerturbationProtocol

if TYPE_CHECKING:
    from art.attacks import EvasionAttack
    import numpy as np

    from armory.labels import LabelTargeter


T = TypeVar("T")


@dataclass
class CallablePerturbation(PerturbationProtocol, Generic[T]):
    name: str
    perturbation: Callable[[T], T]
    inputs_accessor: Accessor[T]

    def apply(self, batch: "Batch"):
        perturbed = self.perturbation(self.inputs_accessor.get(batch.inputs))
        self.inputs_accessor.set(batch.inputs, perturbed)


@dataclass
class ArtEvasionAttack(PerturbationProtocol):
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
    inputs_accessor: Accessor["np.ndarray"] = field(
        default_factory=DefaultNumpyAccessor
    )
    targets_accessor: Accessor["np.ndarray"] = field(
        default_factory=DefaultNumpyAccessor
    )
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

    def _generate_y_target(self, batch: Batch) -> Optional["np.ndarray"]:
        # If targeted, use the label targeter to generate the target label
        if self.targeted:
            if TYPE_CHECKING:
                assert self.label_targeter
            return self.label_targeter.generate(
                self.targets_accessor.get(batch.targets)
            )

        # If untargeted, use either the natural or benign labels
        # (when set to None, the ART attack handles the benign label)
        return (
            self.targets_accessor.get(batch.targets)
            if self.use_label_for_untargeted
            else None
        )

    def apply(self, batch: Batch):
        y_target = self._generate_y_target(batch)
        perturbed = self.attack.generate(
            x=self.inputs_accessor.get(batch.inputs),
            y=y_target,
            **self.generate_kwargs,
        )
        self.inputs_accessor.set(batch.inputs, perturbed)
        batch.metadata["perturbations"][self.name] = dict(y_target=y_target)


@dataclass
class ArtPatchAttack(ArtEvasionAttack):
    """
    A perturbation using a patch evasion attack from the Adversarial Robustness
    Toolbox (ART).

    Example::

        from art.attacks.evasion import AdversarialPatch
        from charmory.perturbation import ArtPatchAttack

        perturb = ArtPatchAttack(
            name="Patch",
            perturbation=AdversarialPatch(classifier),
            use_label_for_untargeted=False,
        )
    """

    generate_every_batch: bool = True
    """Optional, whether to generate the patch for each batch """
    apply_patch_kwargs: Dict[str, Any] = field(default_factory=dict)
    """
    Optional, additional keyword arguments to be used with the patch attack's
    `apply_patch` method
    """

    def _generate(self, x: "np.ndarray", batch: Batch):
        y_target = self._generate_y_target(batch)
        self.patch = self.attack.generate(
            x=x,
            y=y_target,
            **self.generate_kwargs,
        )
        batch.metadata["perturbations"][self.name] = dict(y_target=y_target)

    def generate(self, batch: Batch):
        self._generate(
            self.inputs_accessor.get(batch.inputs),
            batch,
        )

    def apply(self, batch: Batch):
        x = self.inputs_accessor.get(batch.inputs)
        if self.generate_every_batch:
            self._generate(x, batch)
        perturbed = self.attack.apply_patch(x=x, **self.apply_patch_kwargs)
        self.inputs_accessor.set(batch.inputs, perturbed)
        batch.metadata["perturbations"][self.name] = dict(patch=self.patch)
