"""Armory Dataset Classes"""

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

        Args:
            dataset: Source dataset to be wrapped. It must be subscriptable and
                support the `len` operator.
            adapter: Dataset sample adapter
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

        Args:
            dataset: Source dataset where samples are a tuples of data
            keys: List of key names to use for each element in the sample tuple
                when converted to a dictionary
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

        Args:
            *args: All positional arguments will be forwarded to the
                `torch.utils.data.dataloader.DataLoader` class.
            seed: Optional, explicit seed to use for shuffling. If not provided,
                a random seed will be generated.
            shuffle: Whether to shuffle the dataset.
            **kwargs: All other keyword arguments will be forwarded to the
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

        Args:
            *args: All positional arguments will be forwarded to the
                `ShuffleableDataLoader` class.
            dim: Image dimensions format (either CHW or HWC) of the image data
                in the samples returned by the dataset.
            image_key: Key in the dataset sample dictionary for the image data.
            label_key: Key in the dataset sample dictionary for the natural labels.
            scale: Scale (i.e., data type, max value, normalization parameters)
                of the image data values in the samples returned by the dataset.
            **kwargs: All other keyword arguments will be forwarded to the
                `ShuffleableDataLoader` class.
        """
        kwargs.pop("collate_fn", None)
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

        Args:
            *args: All positional arguments will be forwarded to the
                `ShuffleableDataLoader` class.
            boxes_key: Key in each object dictionary for the bounding boxes.
            dim: Image dimensions format (either CHW or HWC) of the image data
                in the samples returned by the dataset.
            format: Bounding box format (e.g., XYXY, XYWH) of the boxes for the
                objects in the samples returned by the dataset.
            image_key: Key in the dataset sample dictionary for the image data.
            label_key: Key in each object dictionary for the natural labels.
            objects_key: Key in the dataset sample dictionary for the objects.
            scale: Scale (i.e., data type, max value, normalization parameters)
                of the image data values in the samples returned by the dataset.
            **kwargs: All other keyword arguments will be forwarded to the
                `ShuffleableDataLoader` class.
        """
        kwargs.pop("collate_fn", None)
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
