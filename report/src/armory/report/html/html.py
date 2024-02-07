import pathlib
from typing import TYPE_CHECKING

import importlib_resources

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


def generate(out: str, **kwargs):
    print(f"Producing HTML output in {out}...")
    copy_resources(importlib_resources.files(__package__) / "www", pathlib.Path(out))
    print("Done")
