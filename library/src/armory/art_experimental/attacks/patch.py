"""
Support for patch attacks using same interface.
"""


from typing import Any, Mapping, Optional, Sequence

from art.attacks.attack import EvasionAttack

from charmory.typing import autocoerce


class AttackWrapper(EvasionAttack):
    def __init__(
        self,
        attack,
        apply_patch_args: Optional[Sequence[Any]] = None,
        apply_patch_kwargs: Optional[Mapping[str, Any]] = None,
    ):
        self._attack = attack
        self._attack.generate = autocoerce(self._attack.generate)
        self._attack.apply_patch = autocoerce(self._attack.apply_patch)
        self._targeted = attack.targeted
        self.args = apply_patch_args or []
        self.kwargs = apply_patch_kwargs or {}

    def generate(self, x, y=None, **kwargs):
        self._attack.generate(x, y=y, **kwargs)
        return self._attack.apply_patch(x, *self.args, **self.kwargs)
