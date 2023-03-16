"""
Based on `track.py`
"""
import os
import sys

import charmory.canned
from charmory.track import Evaluator


def configure_environment():
    """
    Setup a general machine learning development environment.
    """
    print("Delayed imports and dependency configuration.")

    try:
        print("Importing and configuring torch, tensorflow, and art, if available. ")
        print("This may take some time.")

        # import torch before tensorflow to ensure torch.utils.data.DataLoader can utilize
        # all CPU resources when num_workers > 1
        import art
        import tensorflow as tf
        import torch  # noqa: F401

        from armory.paths import HostPaths

        # Handle ART configuration by setting the art data
        # path if art can be imported in the current environment
        art.config.set_data_path(os.path.join(HostPaths().saved_model_dir, "art"))

        if gpus := tf.config.list_physical_devices("GPU"):
            # Currently, memory growth needs to be the same across GPUs
            # From: https://www.tensorflow.org/guide/gpu#limiting_gpu_memory_growth
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(
                "Setting tf.config.experimental.set_memory_growth to True on all GPUs"
            )

    except RuntimeError:
        print("Import armory before initializing GPU tensors")
        raise
    except ImportError:
        pass


def main():
    print("Armory: Example Programmatic Entrypoint for Scenario Execution")
    # configure_environment()

    print("Starting demo")
    mnist = charmory.canned.mnist_baseline()
    evaluator = Evaluator(mnist)
    evaluator.run()

    print("mnist experiment results tracked")


if __name__ == "__main__":
    sys.exit(main())
