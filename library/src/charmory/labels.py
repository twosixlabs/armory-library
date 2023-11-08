"""
Label-related utilties
"""

from typing import Any, Protocol, Sequence, Union, runtime_checkable

import numpy as np


@runtime_checkable
class LabelTargeter(Protocol):
    """A generator of target labels for an attack"""

    def generate(self, y) -> Any:
        """Generates target label for an attack given the original label"""
        pass


# Targeters assume a numpy 1D array as input to generate


class FixedLabelTargeter:
    """
    Label targeter that returns a single, fixed integer value for each input label
    """

    def __init__(self, *, value: int):
        """
        Initializes the label targeter.

        Args:
            value: Fixed integer value
        """
        if not isinstance(value, int) or value < 0:
            raise ValueError(f"value {value} must be a nonnegative int")
        self.value = value

    def generate(self, y):
        return np.ones_like(y) * self.value


class FixedStringTargeter:
    """
    Label targeter that returns a single, fixed string value for each input label
    """

    def __init__(self, *, value: str):
        """
        Initializes the label targeter.

        Args:
            value: Fixed string value
        """
        if not isinstance(value, str):
            raise ValueError(f"target value {value} is not a string")
        self.value = value

    def generate(self, y):
        return [self.value] * len(y)


class RandomLabelTargeter:
    """
    Label targeter that returns a random value from the given range of values
    for each input label
    """

    def __init__(self, *, num_classes):
        """
        Initializes the label targeter.

        Args:
            num_classes: Total number of classes, the returned value will be a
                random value in the range of [0,num_classes)
        """
        if not isinstance(num_classes, int) or num_classes < 2:
            raise ValueError(f"num_classes {num_classes} must be an int >= 2")
        self.num_classes = num_classes

    def generate(self, y):
        y = y.astype(int)
        return (y + np.random.randint(1, self.num_classes, len(y))) % self.num_classes


class RoundRobinTargeter:
    """
    Label targeter that applies a fixed integer offset to each input label
    """

    def __init__(self, *, num_classes: int, offset: int = 1):
        """
        Initializes the label targeter.

        Args:
            num_classes: Total number of classes, used to determine when an
                offset input value wraps back to 0
            offset: Fixed integer offset value to be added to input label values
        """
        if not isinstance(num_classes, int) or num_classes < 1:
            raise ValueError(f"num_classes {num_classes} must be a positive int")
        if not isinstance(offset, int) or offset % num_classes == 0:
            raise ValueError(f"offset {offset} must be an int with % num_classes != 0")
        self.num_classes = num_classes
        self.offset = offset

    def generate(self, y):
        y = y.astype(int)
        return (y + self.offset) % self.num_classes


class ManualTargeter:
    """
    Label targeter that returns fixed values as specified in an ordered list
    """

    def __init__(
        self, *, values: Sequence[Any], repeat: bool = False, dtype: type = int
    ):
        """
        Initializes the label targeter.

        Args:
            values: Ordered list of fixed values to return. Values are consumed
                from the list with each call to `generate`.
            repeat: Whether to wrap back to the beginning of the `values` list
                after reaching the end, defaults to `False`
            dtype: The type of the values, defaults to `int`
        """
        if not values:
            raise ValueError('"values" cannot be an empty list')
        self.values = values
        self.repeat = bool(repeat)
        self.current = 0
        self.dtype = dtype

    def _generate(self, y_i):
        if self.current == len(self.values):
            if self.repeat:
                self.current = 0
            else:
                raise ValueError("Ran out of target labels. Consider repeat=True")

        y_target_i = self.values[self.current]
        self.current += 1
        return y_target_i

    def generate(self, y):
        y_target = []
        for y_i in y:
            y_target.append(self._generate(y_i))
        return np.array(y_target, dtype=self.dtype)


class IdentityTargeter:
    """Label targeter that returns unmodified copies of the input labels"""

    def generate(self, y):
        return y.copy().astype(int)


class ObjectDetectionFixedLabelTargeter:
    """
    Label targeter that replaces the ground truth labels with the specified
    fixed integer value. Does not modify the number of boxes or location of
    boxes.
    """

    def __init__(self, *, value: int, score: float = 1.0):
        """
        Initializes the label targeter.

        Args:
            value: Fixed integer value for the label
            score: Fixed score floating point value
        """
        if not isinstance(value, int) or value < 0:
            raise ValueError(f"value {value} must be a nonnegative int")
        self.value = value
        self.score = score

    def generate(self, y):
        targeted_y = []
        for y_i in y:
            target_y_i = y_i.copy()
            target_y_i["labels"] = (
                np.ones_like(y_i["labels"]).reshape((-1,)) * self.value
            )
            target_y_i["scores"] = (
                np.ones_like(y_i["labels"]).reshape((-1,)) * self.score
            ).astype(np.float32)
            target_y_i["boxes"] = y_i["boxes"].reshape((-1, 4))
            targeted_y.append(target_y_i)
        return targeted_y


class MatchedTranscriptLengthTargeter:
    """
    Label targeter that returns a transcript from a fixed list with a length
    closest to that of the input labels.

    If two transcripts are tied in length, then it pseudorandomly picks one.
    """

    def __init__(self, *, transcripts: Sequence[Union[bytes, str]]):
        """
        Initializes the label targeter.

        Args:
            transcripts: List of replacement transcripts from which to choose
        """
        if not transcripts:
            raise ValueError('"transcripts" cannot be None or an empty list')
        for t in transcripts:
            if type(t) not in (bytes, str):
                raise ValueError(f"transcript type {type(t)} not in (bytes, str)")
        self.transcripts = transcripts
        self.count = 0

    def _generate(self, y):
        distances = [
            (np.abs(len(y) - len(t)), i) for (i, t) in enumerate(self.transcripts)
        ]
        distances.sort()
        min_dist, i = distances[0]
        pool = [i]
        for dist, i in distances[1:]:
            if dist == min_dist:
                pool.append(i)

        chosen_index = pool[self.count % len(pool)]
        y_target = self.transcripts[chosen_index]
        self.count += 1

        return y_target

    def generate(self, y):
        y_target = [self._generate(y_i) for y_i in y]
        if type(y) != list:
            y_target = np.array(y_target)
        return y_target
