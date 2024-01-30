"""Icon645 dataset from IconQA"""
import os
import shutil
import zipfile

import requests
from tqdm.auto import tqdm
from tqdm.utils import CallbackIOWrapper

from armory.evaluation import SysConfig

DOWNLOAD_URL = "https://iconqa2021.s3.us-west-1.amazonaws.com/icon645.zip"


class CachePaths:
    def __init__(self, sysconfig: SysConfig):
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


def prepare_dataset(paths: CachePaths):
    paths.base_path.mkdir(parents=True, exist_ok=True)
    download_zip(paths)
    extract_zip(paths)


if __name__ == "__main__":
    prepare_dataset(CachePaths(SysConfig()))
