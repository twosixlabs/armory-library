"""Armory Dataset Classes"""

import logging
import multiprocessing
from typing import Any, Callable, Iterable, List, Mapping, Optional, Sequence, cast

import numpy as np
import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset

import armory.data as data
from armory.track import track_init_params

DatasetOutputAdapter = Callable[..., Mapping[str, Any]]
"""
An adapter for dataset samples. The output must be a dictionary of column names
to values.
"""


_logger = logging.getLogger(__name__)


class ArmoryDataset(Dataset):
    """
    Wrapper around a PyTorch dataset to apply an adapter to all samples obtained
    from the dataset.

    Example::

        from armory.dataset import ArmoryDataset

        def rename_fields(sample):
            # Rename the 'data' field in the sample to 'image'
            sample["image"] = sample.pop("data")
            return sample

        # assuming `dataset` has been defined elsewhere
        renamed_dataset = ArmoryDataset(dataset, rename_fields)
    """

    def __init__(self, dataset, adapter: DatasetOutputAdapter):
        """
        Initializes the dataset.

        :param dataset: Source dataset to be wrapped. It must be subscriptable and
                support the `len` operator.
        :type dataset: _type_
        :param adapter: Dataset sample adapter
        :type adapter: DatasetOutputAdapter
        """
        self._dataset = dataset
        self._adapter = adapter

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, index):
        return self._adapter(self._dataset[index])


class TupleDataset(ArmoryDataset):
    """
    Dataset wrapper with a pre-applied adapter to adapt tuples to map-like
    samples.

    Example::

        from armory.dataset import TupleDataset

        # assuming `dataset` has been defined elsewhere
        print(dataset[0])
        # output: [[0, 0, 0], [0, 0, 0]], [5]

        tuple_ds = TupleDataset(dataset, ("image", "label"))
        print(tuple_ds[0])
        # output: {'image': [[0, 0, 0], [0, 0, 0]], 'label': [5]}
    """

    def __init__(
        self,
        dataset,
        keys: Iterable[str],
    ):
        """
        Initializes the dataset.

        :param dataset: Source dataset where samples are a tuples of data
        :type dataset: _type_
        :param keys: List of key names to use for each element in the sample tuple
                when converted to a dictionary
        :type keys: Iterable[str]
        """
        super().__init__(dataset, self._adapt)
        self._keys = keys

    def _adapt(self, sample):
        return dict(zip(self._keys, sample))


def _collate_by_type(values: List):
    if len(values) == 0:
        return []
    if isinstance(values[0], np.ndarray):
        return np.asarray(values)
    if isinstance(values[0], torch.Tensor):
        return torch.stack(values)
    return values


def _pop_and_cast(values, key):
    value = values.pop(key)
    return _cast(key, value)


def _cast(key, value):
    if isinstance(value, np.ndarray) or isinstance(value, torch.Tensor):
        return value
    if isinstance(value, list):
        return np.asarray(value)
    raise ValueError(f"Dataset {key} is unsupported type: {type(value)}")


def _tolist(value) -> list:
    if not isinstance(value, list):
        raise ValueError(f"Expected list, got {type(value)}")
    return value


class ShuffleableDataLoader(DataLoader):
    """
    Extension to the PyTorch data loader that enables determinstic shuffling
    when the data loader is used multiple times.

    Example::

        from armory.dataset import ShuffleableDataLoader

        # assuming `dataset` has been defined elsewhere
        dataloader = ShuffleableDataLoader(
            dataset,
            shuffle=True,
            seed=8675309,
        )

        batch1 = next(iter(dataloader))
        batch2 = next(iter(dataloader))
        assert batch1 == batch2
    """

    def __init__(
        self,
        *args,
        seed: Optional[int] = None,
        shuffle: bool = False,
        **kwargs,
    ):
        """
        Initializes the data loader.

        :param seed: Optional, explicit seed to use for shuffling. If not provided,
                a random seed will be generated.
        :type seed: int, optional
        :param shuffle: Whether to shuffle the dataset, defaults to False
        :type shuffle: bool, optional
        :param *args: All positional arguments will be forwarded to the
                `torch.utils.data.dataloader.DataLoader` class.
        :param **kwargs: All other keyword arguments will be forwarded to the
                `torch.utils.data.dataloader.DataLoader` class.
        """
        super().__init__(*args, shuffle=shuffle, **kwargs)
        # We require a seed when shuffling so that we get the same sequence of
        # samples from the dataset each time we use the dataloader
        if shuffle and seed is None:
            seed = np.random.randint(0, 2**32)
        self.seed = seed

    def __iter__(self):
        """
        Applies the seed to the random number generator before creating the
        iterator
        """
        if self.seed is not None:
            torch.manual_seed(self.seed)
        return super().__iter__()


@track_init_params
class ImageClassificationDataLoader(ShuffleableDataLoader):
    """
    Data loader for image classification datasets.

    Example::

        import armory.data
        from armory.dataset import ImageClassificationDataLoader

        # assuming `dataset` has been defined elsewhere
        dataloader = ImageClassificationDataLoader(
            dataset,
            dim=armory.data.ImageDimensions.CHW,
            image_key="image",
            label_key="label",
            scale=armory.data.Scale(dtype=armory.data.DataType.FLOAT, max=1.0),
        )
    """

    def __init__(
        self,
        *args,
        dim: data.ImageDimensions,
        image_key: str,
        label_key: str,
        scale: data.Scale,
        **kwargs,
    ):
        """
        Initializes the data loader.

        :param *args: All positional arguments will be forwarded to the
                `ShuffleableDataLoader` class.
        :param dim: Image dimensions format (either CHW or HWC) of the image data
                in the samples returned by the dataset.
        :type dim: data.ImageDimensions
        :param image_key: Key in the dataset sample dictionary for the image data.
        :type image_key: str
        :param label_key: Key in the dataset sample dictionary for the natural labels.
        :type label_key: str
        :param scale: Scale (i.e., data type, max value, normalization parameters)
                of the image data values in the samples returned by the dataset.
        :type scale: data.Scale
        :param **kwargs: All other keyword arguments will be forwarded to the
                `ShuffleableDataLoader` class.
        """
        kwargs.pop("collate_fn", None)
        if "num_workers" not in kwargs:
            kwargs["num_workers"] = multiprocessing.cpu_count() - 1
            _logger.debug(
                f"Defaulting dataloader num_workers to {kwargs['num_workers']}"
            )
        super().__init__(*args, collate_fn=self._collate, **kwargs)
        self.dim = dim
        self.image_key = image_key
        self.label_key = label_key
        self.scale = scale

    def _collate(
        self, samples: Sequence[Mapping[str, Any]]
    ) -> data.ImageClassificationBatch:
        collated = {
            key: _collate_by_type([s[key] for s in samples])
            for key in samples[0].keys()
        }
        images = data.Images(
            images=_pop_and_cast(collated, self.image_key),
            spec=data.ImageSpec(
                dim=self.dim,
                scale=self.scale,
            ),
        )
        labels = data.NDimArray(_pop_and_cast(collated, self.label_key))
        return data.ImageClassificationBatch(
            inputs=images,
            targets=labels,
            metadata=data.Metadata(data=collated, perturbations=dict()),
        )


@track_init_params
class ObjectDetectionDataLoader(ShuffleableDataLoader):
    """
    Data loader for object detection datasets.

    Example::

        import armory.data
        from armory.dataset import ObjectDetectionDataLoader

        # assuming `dataset` has been defined elsewhere
        dataloader = ObjectDetectionDataLoader(
            dataset,
            boxes_key="boxes",
            dim=armory.data.ImageDimensions.HWC,
            format=armory.data.BBoxFormat.XYWH,
            image_key="image",
            labels_key="category",
            objects_key="objects",
            scale=armory.data.Scale(dtype=armory.data.DataType.FLOAT, max=1.0),
        )
    """

    def __init__(
        self,
        *args,
        boxes_key: str,
        dim: data.ImageDimensions,
        format: data.BBoxFormat,
        image_key: str,
        labels_key: str,
        objects_key: str,
        scale: data.Scale,
        **kwargs,
    ):
        """
        Initializes the data loader.

        :param *args: All positional arguments will be forwarded to the
                `ShuffleableDataLoader` class.
        :param boxes_key: Key in each object dictionary for the bounding boxes.
        :type boxes_key: str
        :param dim: Image dimensions format (either CHW or HWC) of the image data
                in the samples returned by the dataset.
        :type dim: data.ImageDimensions
        :param format: Bounding box format (e.g., XYXY, XYWH) of the boxes for the
                objects in the samples returned by the dataset.
        :type format: data.BBoxFormat
        :param image_key: Key in the dataset sample dictionary for the image data.
        :type image_key: str
        :param labels_key: Key in each object dictionary for the natural labels.
        :type labels_key: str
        :param objects_key: Key in the dataset sample dictionary for the objects.
        :type objects_key: str
        :param scale: Scale (i.e., data type, max value, normalization parameters)
                of the image data values in the samples returned by the dataset.
        :type scale: data.Scale
        :param **kwargs: All other keyword arguments will be forwarded to the
                `ShuffleableDataLoader` class.
        """
        kwargs.pop("collate_fn", None)
        if "num_workers" not in kwargs:
            kwargs["num_workers"] = multiprocessing.cpu_count() - 1
            _logger.debug(
                f"Defaulting dataloader num_workers to {kwargs['num_workers']}"
            )
        super().__init__(*args, collate_fn=self._collate, **kwargs)
        self.boxes_key = boxes_key
        self.dim = dim
        self.format = format
        self.image_key = image_key
        self.labels_key = labels_key
        self.objects_key = objects_key
        self.scale = scale

    def _to_bbox(self, obj):
        boxes = {
            "boxes": _pop_and_cast(obj, self.boxes_key),
            "labels": _pop_and_cast(obj, self.labels_key),
        }
        # Add the remaining properties from the object
        boxes.update({key: _cast(key, value) for key, value in obj.items()})
        return cast(data.BoundingBoxes.BoxesNumpy, boxes)

    def _collate(
        self, samples: Sequence[Mapping[str, Any]]
    ) -> data.ObjectDetectionBatch:
        collated = {
            key: _collate_by_type([s[key] for s in samples])
            for key in samples[0].keys()
        }
        images = data.Images(
            images=_pop_and_cast(collated, self.image_key),
            spec=data.ImageSpec(
                dim=self.dim,
                scale=self.scale,
            ),
        )
        boxes = data.BoundingBoxes(
            boxes=[self._to_bbox(obj) for obj in collated.pop(self.objects_key)],
            spec=data.BoundingBoxSpec(
                format=self.format,
            ),
        )
        return data.ObjectDetectionBatch(
            inputs=images,
            targets=boxes,
            metadata=data.Metadata(data=collated, perturbations=dict()),
        )


@track_init_params
class TextPromptDataLoader(ShuffleableDataLoader):

    def __init__(
        self,
        *args,
        inputs_key: str,
        context_key: Optional[str] = None,
        targets_key: Optional[str] = None,
        **kwargs,
    ):
        collate_fn = kwargs.pop("collate_fn", self._collate)
        super().__init__(*args, collate_fn=collate_fn, **kwargs)
        self.inputs_key = inputs_key
        self.context_key = context_key
        self.targets_key = targets_key

    def _collate(self, samples: Sequence[Mapping[str, Any]]):
        collated = {
            key: _collate_by_type([s[key] for s in samples])
            for key in samples[0].keys()
        }
        inputs = data.Text(text=_tolist(collated.pop(self.inputs_key)))
        contexts = (
            data.Text(text=_tolist(collated.pop(self.context_key)))
            if self.context_key
            else None
        )
        targets = (
            data.Text(text=_tolist(collated.pop(self.targets_key)))
            if self.targets_key
            else None
        )
        return data.TextPromptBatch(
            inputs=inputs,
            contexts=contexts,
            targets=targets,
            metadata=data.Metadata(data=collated, perturbations=dict()),
        )
