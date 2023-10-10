"""Armory engine to create adversarial datasets"""
from typing import TYPE_CHECKING, Any, Callable, Mapping, Optional

import datasets

from charmory.evaluation import Evaluation

SampleAdapter = Callable[[Mapping[str, Any]], Mapping[str, Any]]
"""
An adapter for generated samples. The input argument and return types are a
dictionary of column names to values for a single sample.
"""


class AdversarialDatasetEngine:
    def __init__(
        self,
        evaluation: Evaluation,
        output_dir: str,
        adapter: Optional[SampleAdapter] = None,
        features: Optional[datasets.Features] = None,
        num_batches: Optional[int] = None,
    ):
        self.evaluation = evaluation
        self.output_dir = output_dir
        self.features = features
        self.adapter: SampleAdapter = (
            adapter if adapter is not None else self._default_adapter
        )
        self.num_batches = num_batches

    def generate(self) -> None:
        """Create the adversarial dataset"""
        dataset = datasets.Dataset.from_generator(
            self._generator, features=self.features
        )
        assert isinstance(dataset, datasets.Dataset)

        dataset.to_parquet(self.output_dir)

    def _generator(self):
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

    @staticmethod
    def _default_adapter(sample: Mapping[str, Any]):
        # do nothing
        return sample
