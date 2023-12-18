""" run_matrix - Run torchvision image classification models trained on
ImageNet against the TwoSix ImageNet dataset.

Usage:
    run_matrix.py [options]

Options:
    --batch=N       Batch size [default: 1]
    --num=N         Number of batches to run [default: 5]
    --export=N      Export results every N batches [default: 5]
"""


from copy import deepcopy
from pprint import pprint

import albumentations as A
from art.attacks.evasion import ProjectedGradientDescent
from art.estimators.classification import PyTorchClassifier
from imagenet_tst import get_local_imagenettst
import numpy as np
import torch.nn as nn
from torchvision.models import (
    alexnet,
    convnext_base,
    convnext_tiny,
    densenet121,
    densenet169,
    efficientnet_b0,
)

from armory.matrix import matrix
from armory.matrix.range import frange
from armory.metrics.compute import BasicProfiler
from charmory.data import ArmoryDataLoader
from charmory.engine import EvaluationEngine
from charmory.evaluation import Attack, Dataset, Evaluation, Metric, Model
from charmory.tasks.image_classification import ImageClassificationTask
from charmory.track import track_init_params, track_param


def _load_model(factory):
    model = factory(weights="DEFAULT")
    classifier = track_init_params(PyTorchClassifier)(
        model,
        loss=nn.CrossEntropyLoss(),
        # TODO: is optimizer even used in PGD?
        # optimizer=torch.optim.Adam(model.parameters(), lr=0.003),
        input_shape=(224, 224, 3),
        channels_first=False,
        nb_classes=1000,
        clip_values=(0.0, 1.0),
    )

    eval_model = Model(
        name=type(model).__name__,
        model=classifier,
    )

    return eval_model, classifier


def _load_dataset(batch_size: int):
    transforms = A.Compose(
        [
            A.LongestMaxSize(max_size=224),
            A.PadIfNeeded(
                min_height=224,
                min_width=224,
                border_mode=0,
                value=(0, 0, 0),
            ),
            A.ToFloat(max_value=255),  # Scale to [0, 1]
        ],
    )

    def transform_func(sample):
        transformed = deepcopy(sample)
        for i in range(len(sample["image"])):
            transformed["image"] = [
                transforms(image=np.asarray(image))["image"].transpose(2, 0, 1)
                for image in sample["image"]
            ]
        return transformed

    dataset = get_local_imagenettst(split="val", transform=transform_func)

    dataloader = ArmoryDataLoader(dataset, batch_size=batch_size)

    eval_dataset = Dataset(
        name="ImageNet1k",
        x_key="image",
        y_key="label",
        test_dataloader=dataloader,
    )

    return eval_dataset


def _create_metric():
    return Metric(profiler=BasicProfiler())


def _create_attack(classifier: PyTorchClassifier, eps: float, max_iter: int):
    return Attack(
        name="PGD",
        attack=track_init_params(ProjectedGradientDescent)(
            classifier,
            verbose=False,
            eps=eps,
            max_iter=max_iter,
            batch_size=1,
            eps_step=0.007,
            num_random_init=1,
            random_eps=False,
            targeted=False,
        ),
        use_label_for_untargeted=True,
    )


def create_evaluation(
    model_factory,
    batch_size: int,
    eps: float,
    max_iter: int,
) -> Evaluation:
    model, classifier = _load_model(model_factory)
    dataset = _load_dataset(batch_size=batch_size)
    attack = _create_attack(classifier, eps=eps, max_iter=max_iter)

    evaluation = Evaluation(
        name=f"mwartell-imagenet-{model_factory.__name__}-pgd",
        description=f"Imagenet classification using {model_factory.__name__} model with PGD attack",
        author="TwoSix",
        attack=attack,
        dataset=dataset,
        metric=_create_metric(),
        model=model,
    )

    return evaluation


def create_evaluation_task(
    export_every_n_batches: int, **kwargs
) -> ImageClassificationTask:
    evaluation = create_evaluation(**kwargs)

    task = ImageClassificationTask(
        evaluation,
        num_classes=1000,
        export_every_n_batches=export_every_n_batches,
    )

    return task


@matrix(
    model_factory=(
        alexnet,
        convnext_base,
        convnext_tiny,
        densenet169,
        densenet121,
        efficientnet_b0,
    ),
    eps=frange(0.01, 0.03, 0.005),
    max_iter=range(10, 26, 5),
)
def run_evaluation(num_batches, **kwargs):
    task = create_evaluation_task(**kwargs)
    track_param("main.num_batches", num_batches)
    engine = EvaluationEngine(task, limit_test_batches=num_batches)
    results = engine.run()
    pprint(results)


if __name__ == "__main__":
    import docopt

    args = docopt.docopt(__doc__)

    res = run_evaluation(
        num_batches=args["--batch"],
        batch_size=args["--num"],
        export_every_n_batches=args["--export"],
    )
    pprint(res)
