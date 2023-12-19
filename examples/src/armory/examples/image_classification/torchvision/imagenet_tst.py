"""
imagenet-tst is an adaptation from the HuggingFace downloadable parquet files
representing a supset of the ImageNet dataset. The original dataset is thought
to be no longer distributed by the original authors [citation needed].

The interface needed to match the Torchvision protocol is documented at
https://pytorch.org/vision/stable/generated/torchvision.datasets.ImageNet.html
and is quite narrow requiring only a __init__ and __getitem__ and possibly
__len__ method. The __getitem__ method is expected to return a tuple of the
image and the label.

Most of this adaptation is concerned with unpacking the parquet types into
expected Image and integer label types.

Because it is derived from a competition, the train dataset is claimed to have
only -1 for its labels. I've not confirmed this yet.
"""

import functools
import io
import json
from pathlib import Path
import timeit

from PIL import Image
import pyarrow.parquet as pq
import torchvision.datasets

from charmory.evaluation import SysConfig


class ImageNetTST(torchvision.datasets.VisionDataset):
    def __init__(self, root, split: str, transform=None, target_transform=None):
        assert split in [
            "train",
            "val",
            "test",
        ], "split must be one of train, val, test"

        self.labels: list[str] = json.load(open(Path(root) / "imagenet_classes.json"))[
            "imagenet_classes"
        ]

        self.split = split
        self.root = str(Path(root) / self.split)
        self.transform = transform
        self.target_transform = target_transform

        super(ImageNetTST, self).__init__(
            self.root, transform=transform, target_transform=target_transform
        )

        parquet = pq.ParquetDataset(self.root)
        self.table = parquet.read()

        # it appears that a pyarrow table is not intended to be accessed
        # directly, but rather through a pyarrow.Table.to_pandas() dataframe.
        self.df = self.table.to_pandas()

    def __len__(self):
        return len(self.table)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        image = Image.open(io.BytesIO(row["image"]["bytes"])).convert("RGB")
        if self.transform:
            image = self.transform(image)
        target = row["label"]
        return image, target

    def label(self, index: int) -> str:
        return self.labels[index]


@functools.cache
def get_local_imagenettst(split: str = "val", transform=None):
    root = SysConfig().dataset_cache / "imagenet-tst"
    return ImageNetTST(root, split=split, transform=transform)


def test_intst_dataset_images(dataset):
    print(f"{len(dataset)=}")

    for image, label in dataset:
        assert isinstance(image, Image.Image)


if __name__ == "__main__":
    print(timeit.timeit(lambda: get_local_imagenettst(split="val")))
