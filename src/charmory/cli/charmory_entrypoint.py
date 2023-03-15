"""
Based on `track.py`
"""
import os
import sys

from mlflow import log_metric, log_param

from charmory.experiment import (
    Attack,
    Dataset,
    Experiment,
    MetaData,
    Metric,
    Model,
    Scenario,
    SysConfig,
)


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


class Evaluator:
    def __init__(self, experiment: Experiment):
        self.experiment = experiment

    def run(self):
        print(f"Running experiment {self.experiment._metadata.name}")
        result = {}
        result["benign"] = id(self.experiment.model)
        if self.experiment.attack:
            result["attack"] = id(self.experiment.attack)
        return result


def mnist_experiment() -> Experiment:
    return Experiment(
        _metadata=MetaData(
            name="mnist experiment",
            description="derived from mnist_baseline.json",
            author="msw@example.com",
        ),
        model=Model(
            function="armory.baseline_models.keras.mnist.get_art_model",
            model_kwargs={},
            wrapper_kwargs={},
            weights_file=None,
            fit=True,
            fit_kwargs={"nb_epochs": 20},
        ),
        scenario=Scenario(
            function="armory.scenarios.image_classification.ImageClassificationTask",
            kwargs={},
        ),
        dataset=Dataset(
            function="armory.data.datasets.mnist", framework="numpy", batch_size=128
        ),
        attack=Attack(
            function="art.attacks.evasion.FastGradientMethod",
            kwargs={
                "batch_size": 1,
                "eps": 0.2,
                "eps_step": 0.1,
                "minimal": False,
                "num_random_init": 0,
                "targeted": False,
            },
            knowledge="white",
            use_label=True,
            type=None,
        ),
        defense=None,
        metric=Metric(
            profiler_type="basic",
            supported_metrics=["accuracy"],
            perturbation=["linf"],
            task=["categorical_accuracy"],
            means=True,
            record_metric_per_sample=False,
        ),
        sysconfig=SysConfig(gpus=["all"], use_gpu=True),
    )


def main():
    print("Armory: Example Programmatic Entrypoint for Scenario Execution")
    # configure_environment()

    print("Starting demo")
    experiment = mnist_experiment()
    evaluator = Evaluator(experiment)
    exp_dict = experiment.asdict()

    for block in exp_dict:
        log_param(block, exp_dict[block])
    print("mnist experiment tracked")

    results = evaluator.run()
    for item in results:
        log_metric(item, results[item])
    print("mnist experiment results tracked")


if __name__ == "__main__":
    sys.exit(main())
