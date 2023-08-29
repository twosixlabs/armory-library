"""
Example programmatic entrypoint for scenario execution
"""
import sys

from jatic_toolbox import __version__ as jatic_version
from jatic_toolbox import load_dataset as load_jatic_dataset

import armory.baseline_models.pytorch.pokemon
import armory.version


def load_huggingface_dataset():
    train_dataset = load_jatic_dataset(
        provider="huggingface",
        dataset_name="Honaker/xview_dataset",
        task="object-detection",
        split="train",
    )

    return train_dataset


def main(argv: list = sys.argv[1:]):
    if len(argv) > 0:
        if "--version" in argv:
            print(f"armory: {armory.version.__version__}")
            print(f"JATIC-toolbox: {jatic_version}")
            sys.exit(0)

    print("Armory: Example Programmatic Entrypoint for Scenario Execution")

    train_dataset = load_huggingface_dataset()
    print(train_dataset)


if __name__ == "__main__":
    sys.exit(main())
