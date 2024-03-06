"""Script argument parsing utilities for armory-library examples"""

import argparse
from typing import Optional


def create_parser(
    description: str,
    batch_size: int = 1,
    export_every_n_batches: int = 0,
    num_batches: Optional[int] = None,
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--batch-size",
        default=batch_size,
        help="Number of samples per batch",
        type=int,
    )
    parser.add_argument(
        "--export-every-n-batches",
        default=export_every_n_batches,
        help="Frequency at which batches will be exported to MLflow",
        type=int,
    )
    parser.add_argument(
        "--num-batches",
        default=num_batches,
        help="Number of batches to process",
        type=int,
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle the dataset",
    )
    parser.add_argument(
        "--seed",
        default=None,
        help="Randomization seed (when shuffling the dataset)",
        type=int,
    )
    return parser
