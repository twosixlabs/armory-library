"""Utilities to load the VisDrone 2019 dataset."""

import csv
import io
from pathlib import Path
from pprint import pprint
from typing import Any, Dict, Iterator, List, Tuple

import albumentations as A
import albumentations.pytorch
import datasets
import numpy as np

import armory.data
import armory.dataset


def create_dataloader(
    dataset: datasets.Dataset, max_size: int, **kwargs
) -> armory.dataset.ObjectDetectionDataLoader:
    """
    Create an Armory object detection dataloader for the given VisDrone2019 dataset split.

    Args:
        dataset: VisDrone2019 dataset split
        max_size: Maximum image size to which to resize and pad image samples
        **kwargs: Additional keyword arguments to pass to the dataloader constructor

    Return:
        Armory object detection dataloader
    """
    resize = A.Compose(
        [
            A.LongestMaxSize(max_size=max_size),
            A.PadIfNeeded(
                min_height=max_size,
                min_width=max_size,
                border_mode=0,
                value=(0, 0, 0),
            ),
            A.ToFloat(max_value=255),
            albumentations.pytorch.ToTensorV2(),
        ],
        bbox_params=A.BboxParams(
            format="coco",
            label_fields=["id", "category", "occlusion", "truncation"],
        ),
    )

    def transform(sample):
        tmp = dict(**sample)
        tmp["image"] = []
        tmp["objects"] = []
        for image, objects in zip(sample["image"], sample["objects"]):
            res = resize(
                image=np.asarray(image),
                bboxes=objects["bbox"],
                id=objects["id"],
                category=objects["category"],
                occlusion=objects["occlusion"],
                truncation=objects["truncation"],
            )
            tmp["image"].append(res.pop("image"))
            tmp["objects"].append(res)
        return tmp

    dataset.set_transform(transform)

    return armory.dataset.ObjectDetectionDataLoader(
        dataset,
        image_key="image",
        dim=armory.data.ImageDimensions.CHW,
        scale=armory.data.Scale(
            dtype=armory.data.DataType.FLOAT,
            max=1.0,
        ),
        objects_key="objects",
        boxes_key="bboxes",
        format=armory.data.BBoxFormat.XYWH,
        labels_key="category",
        **kwargs,
    )


TRAIN_URL = "https://github.com/ultralytics/yolov5/releases/download/v1.0/VisDrone2019-DET-train.zip"
VAL_URL = "https://github.com/ultralytics/yolov5/releases/download/v1.0/VisDrone2019-DET-val.zip"
TEST_URL = "https://github.com/ultralytics/yolov5/releases/download/v1.0/VisDrone2019-DET-test-dev.zip"


def load_dataset() -> datasets.DatasetDict:
    """
    Load the train and validation splits of the VisDrone2019 dataset.

    Return:
        Dictionary containing the train and validation splits
    """
    dl_manager = datasets.DownloadManager(dataset_name="VisDrone2019")
    ds_features = features()
    paths = dl_manager.download({"train": TRAIN_URL, "val": VAL_URL, "test": TEST_URL})
    train_files = dl_manager.iter_archive(paths["train"])
    val_files = dl_manager.iter_archive(paths["val"])
    test_files = dl_manager.iter_archive(paths["test"])
    return datasets.DatasetDict(
        {
            datasets.Split.TRAIN: datasets.Dataset.from_generator(
                generate_samples,
                gen_kwargs={"files": train_files},
                features=ds_features,
            ),
            datasets.Split.VALIDATION: datasets.Dataset.from_generator(
                generate_samples,
                gen_kwargs={"files": val_files},
                features=ds_features,
            ),
            datasets.Split.TEST: datasets.Dataset.from_generator(
                generate_samples,
                gen_kwargs={"files": test_files},
                features=ds_features,
            ),
        }
    )


CATEGORIES = [
    # The YOLOv5 model removed this class and shifted all others down by 1 when
    # it trained on the VisDrone data
    # "ignored",
    "pedestrian",
    "people",
    "bicycle",
    "car",
    "van",
    "truck",
    "tricycle",
    "awning-tricycle",
    "bus",
    "motor",
    # The YOLOv5 model also ignored/removed this class
    # "other",
]


def features() -> datasets.Features:
    """Create VisDrone2019 dataset features"""
    return datasets.Features(
        {
            "image_id": datasets.Value("int64"),
            "file_name": datasets.Value("string"),
            "image": datasets.Image(),
            "objects": datasets.Sequence(
                {
                    "id": datasets.Value("int64"),
                    "bbox": datasets.Sequence(datasets.Value("float32"), length=4),
                    "category": datasets.ClassLabel(
                        num_classes=len(CATEGORIES), names=CATEGORIES
                    ),
                    "truncation": datasets.Value("int32"),
                    "occlusion": datasets.Value("int32"),
                }
            ),
        }
    )


ANNOTATION_FIELDS = [
    "x",
    "y",
    "width",
    "height",
    "score",
    "category_id",
    "truncation",
    "occlusion",
]


def load_annotations(file: io.BufferedReader) -> List[Dict[str, Any]]:
    """Load annotations/objects from the given file"""
    reader = csv.DictReader(
        io.StringIO(file.read().decode("utf-8")), fieldnames=ANNOTATION_FIELDS
    )
    annotations = []
    for idx, row in enumerate(reader):
        score = int(row["score"])
        category = int(row["category_id"])
        if score != 0:  # Drop annotations with score of 0 (class-0 & class-11)
            category -= 1  # The model was trained with 0-indexed categories starting at pedestrian
            bbox = list(map(float, [row[k] for k in ANNOTATION_FIELDS[:4]]))
            if bbox[2] == 0 or bbox[3] == 0:
                continue
            annotations.append(
                {
                    "id": idx,
                    "bbox": bbox,
                    "category": category,
                    "truncation": row["truncation"],
                    "occlusion": row["occlusion"],
                }
            )
    return annotations


def generate_samples(
    files: Iterator[Tuple[str, io.BufferedReader]], annotation_file_ext: str = ".txt"
) -> Iterator[Dict[str, Any]]:
    """Generate dataset samples from the given files in a VisDrone2019 archive"""
    annotations = {}
    images = {}
    for path, file in files:
        file_name = Path(path).stem
        if Path(path).suffix == annotation_file_ext:
            annotations[file_name] = load_annotations(file)
        else:
            images[file_name] = {"path": path, "bytes": file.read()}

    for idx, (file_name, annotation) in enumerate(annotations.items()):
        yield {
            "image_id": idx,
            "file_name": file_name,
            "image": images[file_name],
            "objects": annotation,
        }


if __name__ == "__main__":
    pprint(load_dataset())
