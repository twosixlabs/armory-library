"""
Support for patch attacks using same interface.
"""

from typing import Any, Mapping, Optional, Sequence

from art.attacks.attack import EvasionAttack


class AttackWrapper(EvasionAttack):
    def __init__(
        self,
        attack,
        apply_patch_args: Optional[Sequence[Any]] = None,
        apply_patch_kwargs: Optional[Mapping[str, Any]] = None,
    ):
        """
        AttackWrapper init

        :param attack: attack
        :type attack: ?
        :param apply_patch_args: patch args, defaults to None
        :type apply_patch_args: Sequence[Any], optional
        :param apply_patch_kwargs: patch kwargs, defaults to None
        :type apply_patch_kwargs: Mapping[str, Any], optional
        """
        self._attack = attack
        self._targeted = attack.targeted
        self.args = apply_patch_args or []
        self.kwargs = apply_patch_kwargs or {}

    def generate(self, x, y=None, **kwargs):
        """
        _summary_

        :param x: x
        :type x: ?
        :param y: y, defaults to None
        :type y: ?, optional
        """
        self._attack.generate(x, y=y, **kwargs)
        return self._attack.apply_patch(x, *self.args, **self.kwargs)
