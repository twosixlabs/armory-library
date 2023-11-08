"""Armory engine to create adversarial datasets"""
from typing import TYPE_CHECKING, Any, Callable, Generator, Mapping, Optional

import numpy as np

from charmory.tasks.base import BaseEvaluationTask
from charmory.track import get_current_params

if TYPE_CHECKING:
    import datasets

SampleAdapter = Callable[[Mapping[str, Any]], Mapping[str, Any]]
"""
An adapter for generated samples. The input argument and return types are a
dictionary of column names to values for a single sample.
"""


class AdversarialDatasetEngine:
    """
    Armory engine to create adversarial datasets. An adversarial dataset has
    an adversarial attack already applied to every sample in the dataset.

    Example::

        from charmory.engine import AdversarialDatasetEngine

        # assuming `task` has been defined using a `charmory.tasks` class
        engine = AdversarialDatasetEngine(
            task,
            output_dir="dataset_output_dir",
        )
        engine.generate()

        # to load the generated dataset...
        import datasets
        ds = datasets.load_from_disk("dataset_output_dir")
    """

    def __init__(
        self,
        task: BaseEvaluationTask,
        output_dir: Optional[str] = None,
        adapter: Optional[SampleAdapter] = None,
        features: Optional["datasets.Features"] = None,
        num_batches: Optional[int] = None,
    ):
        """
        Initializes the engine.

        Args:
            task: Armory evaluation task from which to generate the dataset
            output_dir: Optional, directory to which to write the generated dataset
            adapter: Optional, adapter to perform additional modifications to samples
            features: Optional, dataset features
            num_batches: Optional, number of batches from the original dataset to
                attack and include in the generated dataset
        """
        self.task = task
        self.output_dir = output_dir
        self.features = features
        self.adapter: SampleAdapter = (
            adapter if adapter is not None else self._default_adapter
        )
        self.num_batches = num_batches

    @staticmethod
    def _default_adapter(sample: Mapping[str, Any]) -> Mapping[str, Any]:
        # do nothing
        return sample

    def generate(self) -> "datasets.Dataset":
        """Create the adversarial dataset"""
        import datasets

        dataset = datasets.Dataset.from_generator(
            self._generator, features=self.features
        )
        assert isinstance(dataset, datasets.Dataset)

        if self.output_dir is not None:
            dsdict = datasets.DatasetDict({"test": dataset})
            dsdict.save_to_disk(self.output_dir)

        return dataset

    def _generator(self) -> Generator[Mapping[str, Any], None, None]:
        """
        Iterates over every batch in the source dataset, applies the adversarial
        attack, and yields the pre-attacked samples.
        """
        batch_idx = 0
        for ds_batch in iter(self.task.evaluation.dataset.test_dataloader):
            task_batch = self.task.create_batch(ds_batch, batch_idx)
            self.task.apply_attack(task_batch)
            assert isinstance(task_batch.x_adv, np.ndarray)

            for idx in range(len(task_batch.x_adv)):
                sample = {key: val[idx] for key, val in ds_batch.items()}
                sample[self.task.evaluation.dataset.x_key] = task_batch.x_adv[idx]
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
