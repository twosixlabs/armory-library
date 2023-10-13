"""Armory engine to create adversarial datasets"""
from typing import TYPE_CHECKING, Any, Callable, Mapping, Optional

import datasets

from charmory.evaluation import Evaluation
from charmory.track import get_current_params

SampleAdapter = Callable[[Mapping[str, Any]], Mapping[str, Any]]
"""
An adapter for generated samples. The input argument and return types are a
dictionary of column names to values for a single sample.
"""


class AdversarialDatasetEngine:
    """
    Armory engine to create adversarial datasets. An adversarial dataset has
    an adversarial attack already applied to every sample in the dataset.

    Example:

        from charmory.engine import AdversarialDatasetEngine

        # assuming `evaluation` has been defined using `charmory.evaluation` classes
        engine = AdversarialDatasetEngine(
            evaluation,
            output_dir="dataset_output_dir",
        )
        engine.generate()

        # to load the generated dataset...
        import datasets
        ds = datasets.load_from_disk("dataset_output_dir")
    """

    def __init__(
        self,
        evaluation: Evaluation,
        output_dir: Optional[str] = None,
        adapter: Optional[SampleAdapter] = None,
        features: Optional[datasets.Features] = None,
        num_batches: Optional[int] = None,
    ):
        """
        Initializes the engine.

        Args:
            evaluation: Armory evaluation from which to generate the dataset
            output_dir: Optional, directory to which to write the generated dataset
            adapter: Optional, adapter to perform additional modifications to samples
            features: Optional, dataset features
            num_batches: Optional, number of batches from the original dataset to
                attack and include in the generated dataset
        """
        self.evaluation = evaluation
        self.output_dir = output_dir
        self.features = features
        self.adapter: SampleAdapter = (
            adapter if adapter is not None else self._default_adapter
        )
        self.num_batches = num_batches

    @staticmethod
    def _default_adapter(sample: Mapping[str, Any]):
        # do nothing
        return sample

    def generate(self) -> datasets.Dataset:
        """Create the adversarial dataset"""
        dataset = datasets.Dataset.from_generator(
            self._generator, features=self.features
        )
        assert isinstance(dataset, datasets.Dataset)

        if self.output_dir is not None:
            dsdict = datasets.DatasetDict({"test": dataset})
            dsdict.save_to_disk(self.output_dir)

        return dataset

    def _generator(self):
        """
        Iterates over every batch in the source dataset, applies the adversarial
        attack, and yields the pre-attacked samples.
        """
        if TYPE_CHECKING:
            assert self.evaluation.attack

        batch_idx = 0
        for batch in iter(self.evaluation.dataset.test_dataloader):
            x = batch[self.evaluation.dataset.x_key]
            y = batch[self.evaluation.dataset.y_key]
            if self.evaluation.attack.targeted:
                if TYPE_CHECKING:
                    assert self.evaluation.attack.label_targeter
                y_target = self.evaluation.attack.label_targeter.generate(y)
            else:
                y_target = (
                    y if self.evaluation.attack.use_label_for_untargeted else None
                )

            x_adv = self.evaluation.attack.attack.generate(
                x=x, y=y_target, **self.evaluation.attack.generate_kwargs
            )

            for idx in range(len(x)):
                sample = {key: val[idx] for key, val in batch.items()}
                sample[self.evaluation.dataset.x_key] = x_adv[idx]
                yield self.adapter(sample)

            batch_idx += 1
            if self.num_batches is not None and batch_idx >= self.num_batches:
                return

    def __getstate__(self):
        """
        Return the mapping of tracked params from the evaluation as the engine
        state for pickling. We do this because `datasets` relies on the hashed
        pickled state of the generator (this class, because it is a bounded method)
        to determine the cache of the dataset. Thus, we want a reproducible,
        deterministic state in contrast to the default behavior for Python objects.
        """
        return get_current_params()
