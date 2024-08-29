"""Utilities to autoconfigure logging for Armory."""

import logging
import sys


def configure_logging(
    armory_level: int = logging.INFO,
    datasets_level: int = logging.WARNING,
    lightning_level: int = logging.WARNING,
    mlflow_level: int = logging.WARNING,
    transformers_level: int = logging.WARNING,
    enable_progress_bars: bool = sys.stdout.isatty(),
):
    """
    Configure log levels for Armory and its dependencies.
    """

    armory = logging.getLogger("armory")
    armory.setLevel(armory_level)

    lightning = logging.getLogger("lightning.pytorch")
    lightning.setLevel(lightning_level)

    mlflow = logging.getLogger("mlflow")
    mlflow.setLevel(mlflow_level)

    try:
        import datasets

        datasets.logging.set_verbosity(datasets_level)
        if not enable_progress_bars:
            datasets.logging.disable_progress_bar()
    except ModuleNotFoundError:
        pass

    try:
        import transformers

        transformers.logging.set_verbosity(transformers_level)
        if not enable_progress_bars:
            transformers.logging.disable_progress_bar()
    except ModuleNotFoundError:
        pass
