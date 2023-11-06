import sys

import art.attacks.evasion
from jatic_toolbox import load_dataset as load_jatic_dataset
from torchvision import transforms as T

import armory.baseline_models.pytorch.food101
import armory.data.datasets
from armory.metrics.compute import BasicProfiler
import armory.version
from charmory.data import ArmoryDataLoader
from charmory.engine import EvaluationEngine
from charmory.evaluation import Attack, Dataset, Evaluation, Metric, Model, SysConfig
from charmory.experimental.example_results import print_outputs
from charmory.tasks.image_classification import ImageClassificationTask
from charmory.utils import PILtoNumpy

BATCH_SIZE = 16


def load_torchvision_dataset(sysconfig: SysConfig):
    print("Loading torchvision dataset from jatic_toolbox")
    train_dataset = load_jatic_dataset(
        provider="torchvision",
        dataset_name="Food101",
        task="image-classification",
        split="train",
        root=sysconfig.dataset_cache,
        download=True,
        transform=T.Compose(
            [
                T.Resize(size=(512, 512)),
                PILtoNumpy(),
            ]
        ),
    )
    train_dataloader = ArmoryDataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    test_dataset = load_jatic_dataset(
        provider="torchvision",
        dataset_name="Food101",
        task="image-classification",
        split="test",
        root=sysconfig.dataset_cache,
        download=True,
        transform=T.Compose(
            [
                T.Resize(size=(512, 512)),
                PILtoNumpy(),
            ]
        ),
    )
    test_dataloader = ArmoryDataLoader(
        dataset=test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    return train_dataloader, test_dataloader


def main():
    sysconfig = SysConfig()
    train_dataset, test_dataset = load_torchvision_dataset(sysconfig)

    dataset = Dataset(
        name="Food101",
        x_key="image",
        y_key="label",
        train_dataloader=train_dataset,
        test_dataloader=test_dataset,
    )
    classifier = armory.baseline_models.pytorch.food101.get_art_model(
        model_kwargs={},
        wrapper_kwargs={},
        weights_path=None,
    )
    model = Model(
        name="Food101",
        model=classifier,
    )

    ###
    # The rest of this file was directly copied from the existing cifar example
    ###

    attack = Attack(
        name="PGD",
        attack=art.attacks.evasion.ProjectedGradientDescent(
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
    )

    evaluation = Evaluation(
        name="food101_baseline",
        description="Baseline food101 image classification",
        author="msw@example.com",
        dataset=dataset,
        model=model,
        attack=attack,
        metric=metric,
        sysconfig=sysconfig,
    )

    task = ImageClassificationTask(
        evaluation, num_classes=101, export_every_n_batches=5
    )

    engine = EvaluationEngine(task, limit_test_batches=5)
    results = engine.run()
    print_outputs(dataset, model, results)

    print("JATIC Experiment Complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
