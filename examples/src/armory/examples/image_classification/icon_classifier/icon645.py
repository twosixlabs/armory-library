"""Icon645 dataset from IconQA"""
import os
from pprint import pprint
import shutil
import zipfile

import datasets
import requests
from tqdm.auto import tqdm
from tqdm.utils import CallbackIOWrapper

from armory.evaluation import SysConfig

DOWNLOAD_URL = "https://iconqa2021.s3.us-west-1.amazonaws.com/icon645.zip"
IMAGES_SUBDIR = "colored_icons_final"


class CachePaths:
    def __init__(self, sysconfig: SysConfig = SysConfig()):
        self.base_path = sysconfig.dataset_cache / "icon645"
        self.zip_file = self.base_path / "icon645.zip"
        self.images_path = self.base_path / "images"
        self.dataset_path = self.base_path / "dataset"


def download_zip(paths: CachePaths):
    if paths.zip_file.exists():
        print(f"{paths.zip_file} already exists")
        return

    print(f"Downloading {DOWNLOAD_URL}...")
    req = requests.get(DOWNLOAD_URL, stream=True)
    if req.status_code != 200:
        req.raise_for_status()  # if a 4xx error
        raise RuntimeError(
            f"Request to {DOWNLOAD_URL} returned status {req.status_code}"
        )

    file_size = int(req.headers.get("Content-Length", 0))
    desc = "(Unknown total file size)" if file_size == 0 else ""

    with tqdm.wrapattr(req.raw, "read", total=file_size, desc=desc) as raw:
        with paths.zip_file.open("wb") as out:
            shutil.copyfileobj(raw, out)


def extract_zip(paths: CachePaths):
    if paths.images_path.exists():
        print(f"{paths.images_path} already exists")
        return

    paths.images_path.mkdir(parents=True, exist_ok=True)

    print(f"Extracting {paths.zip_file}...")
    with zipfile.ZipFile(paths.zip_file, "r") as zip:
        # zip.extractall(paths.images_path)
        with tqdm(
            desc="Extracting",
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            total=sum(getattr(i, "file_size", 0) for i in zip.infolist()),
        ) as pbar:
            for i in zip.infolist():
                if not getattr(i, "file_size", 0):  # directory
                    zip.extract(i, os.fspath(paths.images_path))
                else:
                    with zip.open(i) as infile, open(
                        paths.images_path / i.filename, "wb"
                    ) as outfile:
                        shutil.copyfileobj(
                            CallbackIOWrapper(pbar.update, infile), outfile
                        )


def rename_split_folder(paths: CachePaths):
    train_path = paths.images_path / "train"
    if train_path.exists():
        print(f"{train_path} already exists")
        return

    shutil.move(paths.images_path / IMAGES_SUBDIR, train_path)


def load_images_dataset(paths: CachePaths) -> datasets.Dataset:
    print(f"Loading {paths.images_path}...")
    dataset = datasets.load_dataset(
        "imagefolder", data_dir=str(paths.images_path), split="train"
    )
    assert isinstance(dataset, datasets.Dataset)
    return dataset


def split_dataset(dataset: datasets.Dataset) -> datasets.DatasetDict:
    print("Creating train/test/validation splits...")
    dsdict = dataset.train_test_split(test_size=0.2, stratify_by_column="label")
    val_train = dsdict["train"].train_test_split(
        test_size=0.25, stratify_by_column="label"
    )
    dsdict["train"] = val_train["train"]
    dsdict["validation"] = val_train["test"]
    return dsdict


def prepare_dataset(paths: CachePaths = CachePaths()):
    if paths.dataset_path.exists():
        print(f"{paths.dataset_path} already exists")
        return

    paths.base_path.mkdir(parents=True, exist_ok=True)
    download_zip(paths)
    extract_zip(paths)
    rename_split_folder(paths)
    img_ds = load_images_dataset(paths)
    ds_splits = split_dataset(img_ds)
    ds_splits.save_to_disk(paths.dataset_path)
    print("Done creating dataset cache")


def load_dataset(paths: CachePaths = CachePaths()) -> datasets.DatasetDict:
    if not paths.dataset_path.exists():
        raise RuntimeError(
            "Dataset has not been cached, run `prepare_dataset` to download and cache the dataset first"
        )
    return datasets.load_from_disk(str(paths.dataset_path))


if __name__ == "__main__":
    prepare_dataset()
    ds = load_dataset()
    pprint(ds)
