"""Utilities to autoconfigure logging for Armory."""

import logging
import os
import sys


def configure_logging(
    armory_level: int = logging.INFO,
    datasets_level: int = logging.WARNING,
    lightning_level: int = logging.WARNING,
    mlflow_level: int = logging.WARNING,
    transformers_level: int = logging.WARNING,
    enable_basic_config: bool = True,
    enable_progress_bars: bool = sys.stdout.isatty(),
    **kwargs,
):
    """
    Configure log levels for Armory and its dependencies.
    """
    if enable_basic_config:
        logging.basicConfig(**kwargs)

    armory = logging.getLogger("armory")
    armory.setLevel(armory_level)

    lightning = logging.getLogger("lightning.pytorch")
    lightning.setLevel(lightning_level)

    mlflow = logging.getLogger("mlflow")
    mlflow.setLevel(mlflow_level)

    if not enable_progress_bars:
        os.environ["TQDM_DISABLE"] = "1"

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
        transformers.logging.disable_default_handler()
        if not enable_progress_bars:
            transformers.logging.disable_progress_bar()
    except ModuleNotFoundError:
        pass
