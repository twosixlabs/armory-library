import json
import pathlib
from typing import TYPE_CHECKING, Optional

import importlib_resources

import armory.report.common as common

if TYPE_CHECKING:
    import argparse

    from importlib_resources.abc import Traversable


BLACKLIST = ["tsconfig.json", "tailwind.config.js"]


def configure_args(parser: "argparse.ArgumentParser"):
    parser.description = "Produce HTML report for an Armory evaluation"
    parser.add_argument(
        "--out",
        default="./public",
        help="Directory in which to place generated HTML files",
    )
    parser.add_argument(
        "--experiment",
        help="ID of the MLFlow experiment for which to report",
    )


def copy_resources(srcdir: "Traversable", outdir: pathlib.Path):
    for entry in srcdir.iterdir():
        if entry.name in BLACKLIST:
            continue
        if entry.is_dir():
            subdir = outdir / entry.name
            subdir.mkdir(parents=True, exist_ok=True)
            copy_resources(entry, subdir)
        else:
            with open(outdir / entry.name, "wb") as outfile:
                outfile.write(entry.read_bytes())


def generate(out: str, experiment: Optional[str], **kwargs):
    outpath = pathlib.Path(out)
    outpath.mkdir(parents=True, exist_ok=True)

    if experiment:
        data = common.dump_experiment(experiment)
        with open(outpath / "data.json", "w") as outfile:
            json.dump(data, outfile, indent=2, sort_keys=True)
        return

    print(f"Producing HTML output in {out}...")
    copy_resources(importlib_resources.files(__package__) / "www", outpath)
    print("Done")
