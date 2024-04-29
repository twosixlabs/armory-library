"""Utilities to load the VisDrone 2019 dataset."""

from pathlib import Path
import zipfile

import requests
from tqdm import tqdm

from armory.evaluation import SysConfig


def download_from_gdrive(fileid: str, filepath: Path) -> None:
    url = "https://drive.usercontent.google.com/download"
    response = requests.get(url, params={"id": fileid, "confirm": "1"}, stream=True)

    total_size = int(response.headers.get("content-length", 0))
    block_size = 1024

    with tqdm(total=total_size, unit="B", unit_scale=True) as pbar:
        with open(filepath, "wb") as f:
            for data in response.iter_content(block_size):
                f.write(data)
                pbar.update(len(data))

    if total_size != 0 and pbar.n != total_size:
        raise RuntimeError("Download failed")


def download_visdrone(cachedir: Path):
    train_fileid = "1a2oHjcEcwXP8oUF95qiwrqzACb2YlUhn"
    train_filename = "VisDrone2019-DET-train.zip"
    train_filepath = cachedir / train_filename
    if not train_filepath.exists():
        print(f"Downloading to {train_filepath}")
        download_from_gdrive(train_fileid, train_filepath)
    else:
        print(f"Using cached {train_filepath}")
    train_dir = cachedir / "train" / "VisDrone2019-DET-train"
    if not train_dir.exists():
        if zipfile.is_zipfile(train_filepath):
            with zipfile.ZipFile(train_filepath, "r") as zip_ref:
                zip_ref.extractall(cachedir / "train")
        else:
            raise RuntimeError(
                f"Download of training data failed, {train_filepath} not a zip file"
            )
        train_dir = cachedir / "train" / "VisDrone2019-DET-train"
        if not train_dir.exists():
            raise RuntimeError(
                f"Extraction of training data failed, {train_dir} not found"
            )
    else:
        print(f"Using cached {train_dir}")

    test_fileid = "1bxK5zgLn0_L8x276eKkuYA_FzwCIjb59"
    test_filename = "VisDrone2019-DET-val.zip"
    test_filepath = cachedir / test_filename
    if not test_filepath.exists():
        print(f"Downloading to {test_filepath}")
        download_from_gdrive(test_fileid, test_filepath)
    else:
        print(f"Using cached {test_filepath}")
    test_dir = cachedir / "test" / "VisDrone2019-DET-val"
    if not test_dir.exists():
        if zipfile.is_zipfile(test_filepath):
            with zipfile.ZipFile(test_filepath, "r") as zip_ref:
                zip_ref.extractall(cachedir / "test")
        else:
            raise RuntimeError(
                f"Download of test data failed, {test_filepath} not a zip file"
            )
        if not test_dir.exists():
            raise RuntimeError(f"Extraction of test data failed, {test_dir} not found")
    else:
        print(f"Using cached {test_dir}")

    return train_dir, test_dir


if __name__ == "__main__":
    sysconfig = SysConfig()
    cachedir = sysconfig.dataset_cache / "visdrone2019"
    cachedir.mkdir(parents=True, exist_ok=True)

    train_dir, test_dir = download_visdrone(cachedir)
