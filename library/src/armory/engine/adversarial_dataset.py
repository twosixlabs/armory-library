"""Armory engine to create adversarial datasets"""

from typing import TYPE_CHECKING, Any, Callable, Generator, Mapping, Optional

from armory.data import DataSpecification, NumpySpec
from armory.engine.evaluation_module import EvaluationModule
from armory.evaluation import Chain
from armory.metrics.compute import NullProfiler, Profiler
from armory.track import get_current_params

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
        chain: Chain,
        inputs_key: str,
        targets_key: str,
        inputs_spec: Optional[DataSpecification] = None,
        targets_spec: Optional[DataSpecification] = None,
        output_dir: Optional[str] = None,
        adapter: Optional[SampleAdapter] = None,
        features: Optional["datasets.Features"] = None,
        num_batches: Optional[int] = None,
        profiler: Optional[Profiler] = None,
    ):
        """
        Initializes the engine.

        :param chain: Evaluation chain
        :type chain: Chain
        :param inputs_key: Inputs key
        :type inputs_key: str
        :param targets_key: Targets key
        :type targets_key: str
        :param inputs_spec: Inputs specification, defaults to None
        :type inputs_spec: DataSpecification, optional
        :param targets_spec: Targets specification, defaults to None
        :type targets_spec: DataSpecification, optional
        :param output_dir: directory to which to write the generated dataset, defaults to None
        :type output_dir: str, optional
        :param adapter: adapter to perform additional modifications to samples, defaults to None
        :type adapter: SampleAdapter, optional
        :param features: dataset features, defaults to None
        :type features: "datasets.Features", optional
        :param num_batches: number of batches from the original dataset to
                attack and include in the generated dataset, defaults to None
        :type num_batches: int, optional
        :param profiler: profiler to collect computational metrics. By
                default, no computational metrics will be collected.
        :type profiler: Profiler, optional
        """
        self.chain = chain
        self.module = EvaluationModule(chain, profiler or NullProfiler())
        self.inputs_key = inputs_key
        self.inputs_spec = inputs_spec or NumpySpec()
        self.targets_key = targets_key or NumpySpec()
        self.targets_spec = targets_spec
        self.output_dir = output_dir
        self.features = features
        self.adapter: SampleAdapter = (
            adapter if adapter is not None else self._default_adapter
        )
        self.num_batches = num_batches

    @staticmethod
    def _default_adapter(sample: Mapping[str, Any]) -> Mapping[str, Any]:
        """
        Do nothing

        :param sample: Sample
        :type sample: Mapping[str, Any]
        :return: Sample
        :rtype: Mapping[str, Any]
        """
        # do nothing
        return sample

    def generate(self) -> "datasets.Dataset":
        """
        Create the adversarial dataset

        :return: Adversarial dataset
        :rtype: datasets.Dataset
        """
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

        :rtype: Generator[Mapping[str, Any], None, None]
        """
        assert self.chain.dataset

        batch_idx = 0
        for batch in iter(self.chain.dataset.dataloader):
            self.module.apply_perturbations(batch)

            inputs = batch.inputs.get(self.inputs_spec)
            targets = batch.targets.get(self.targets_spec)

            for idx in range(len(batch)):
                sample = {key: val[idx] for key, val in batch.metadata["data"].items()}
                sample[self.inputs_key] = inputs[idx]
                sample[self.targets_key] = targets[idx]
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
