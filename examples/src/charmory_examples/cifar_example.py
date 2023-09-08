"""
Example programmatic entrypoint for scenario execution
"""
import sys

import art.attacks.evasion

import armory.baseline_models.pytorch.cifar
import armory.data.datasets
from armory.instrument.config import MetricsLogger
from armory.metrics.compute import BasicProfiler
import armory.version
from charmory.evaluation import Attack, Dataset, Evaluation, Metric, Model, SysConfig
from charmory.experimental.lightning_execution import execute_lightning, print_outputs
from charmory.tasks.image_classification import ImageClassificationTask
from charmory.track import track_init_params, track_params
from charmory.utils import apply_art_preprocessor_defense


def main(argv: list = sys.argv[1:]):
    if len(argv) > 0:
        if "--version" in argv:
            print(armory.version.__version__)
            sys.exit(0)

    print("Armory: Example Programmatic Entrypoint for Scenario Execution")

    dataset = Dataset(
        name="CIFAR10",
        train_dataset=track_params(armory.data.datasets.cifar10)(
            split="train",
            epochs=20,
            batch_size=64,
            shuffle_files=True,
        ),
        test_dataset=track_params(armory.data.datasets.cifar10)(
            split="test",
            epochs=1,
            batch_size=64,
            shuffle_files=False,
        ),
    )

    classifier = track_params(prefix="model")(
        armory.baseline_models.pytorch.cifar.get_art_model
    )(
        model_kwargs={},
        wrapper_kwargs={},
        weights_path=None,
    )

    model = Model(
        name="cifar",
        model=classifier,
    )

    defense = track_init_params(art.defences.preprocessor.JpegCompression)(
        apply_fit=False,
        apply_predict=True,
        clip_values=(0.0, 1.0),
        quality=50,
    )
    apply_art_preprocessor_defense(model.model, defense)

    attack = Attack(
        name="PGD",
        attack=track_init_params(art.attacks.evasion.ProjectedGradientDescent)(
            classifier,
            batch_size=1,
            eps=0.031,
            eps_step=0.007,
            max_iter=20,
            num_random_init=1,
            random_eps=False,
            targeted=False,
            verbose=False,
        ),
        use_label_for_untargeted=True,
    )

    metric = Metric(
        profiler=BasicProfiler(),
        logger=MetricsLogger(
            supported_metrics=["accuracy"],
            perturbation=["linf"],
            task=["categorical_accuracy"],
            means=True,
            record_metric_per_sample=False,
        ),
    )

    sysconfig = SysConfig(gpus=["all"], use_gpu=True)

    evaluation = Evaluation(
        name="cifar_baseline",
        description="Baseline cifar10 image classification",
        author="msw@example.com",
        dataset=dataset,
        model=model,
        attack=attack,
        scenario=None,
        metric=metric,
        sysconfig=sysconfig,
    )

    task = ImageClassificationTask(evaluation, num_classes=10, export_every_n_batches=5)

    results = execute_lightning(task, limit_test_batches=5)
    print_outputs(dataset, model, results)

    print("CIFAR10 Experiment Complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
