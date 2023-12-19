""" run_matrix - Run torchvision image classification models trained on
ImageNet against the TwoSix ImageNet dataset.

Usage:
    run_matrix.py [options]
    run_matrix.py --script

Options:
    -b --batches=N         Number of batches to run [default: 20]
    -s --size=N  i         Batch size [default: 1]
    -e --export=N          Export results every N batches [default: 5]
    -m --model=NAME        Model to run [default: alexnet]
"""


# from copy import deepcopy  # TODO: why is this unref, what was it doing in run_matrix?
import sys

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
    efficientnet_v2_m,
    efficientnet_v2_s,
    googlenet,
    inception_v3,
    maxvit_t,
    mobilenet_v2,
    mobilenet_v3_small,
    regnet_x_400mf,
    resnet18,
    resnet50,
    resnext50_32x4d,
    shufflenet_v2_x1_0,
    vgg11,
    vit_b_16,
    wide_resnet50_2,
)

# TODO this selection of models is arbitrary and incomplete
_MODELS = {
    "alexnet": alexnet,
    "convnext_base": convnext_base,
    "convnext_tiny": convnext_tiny,
    "densenet121": densenet121,
    "densenet169": densenet169,
    "efficientnet_b0": efficientnet_b0,
    "efficientnet_v2_s": efficientnet_v2_s,
    "efficientnet_v2_m": efficientnet_v2_m,
    "googlenet": googlenet,
    "inception_v3": inception_v3,
    "maxvit_t": maxvit_t,
    "mobilenet_v2": mobilenet_v2,
    "mobilenet_v3_small": mobilenet_v3_small,
    "regnet_x_400mf": regnet_x_400mf,
    "resnet18": resnet18,
    "resnet50": resnet50,
    "resnext50_32x4d": resnext50_32x4d,
    "shufflenet_v2_x1_0": shufflenet_v2_x1_0,
    "vgg11": vgg11,
    "vit_b_16": vit_b_16,
    "wide_resnet50_2": wide_resnet50_2,
}

from armory.metrics.compute import BasicProfiler
from charmory.data import ArmoryDataLoader, TupleDataset
from charmory.engine import EvaluationEngine
from charmory.evaluation import Attack, Dataset, Evaluation, Metric, Model
from charmory.tasks.image_classification import ImageClassificationTask
from charmory.track import track_init_params, track_param


def _load_model(name: str):
    factory = _MODELS[name]
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
    track_param("model.name", name)

    eval_model = Model(
        name=name,
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

    def transform_func(image):
        return transforms(image=np.asarray(image))["image"].transpose(2, 0, 1)

    dataset = get_local_imagenettst(split="val", transform=transform_func)

    dataloader = ArmoryDataLoader(
        TupleDataset(dataset, "image", "label"), batch_size=batch_size
    )

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


def run_evaluation(
    experiment: str,
    model_name: str,
    batches: int,
    size: int,
    export: int,
    eps: float,
    max_iter: int,
):
    dataset = _load_dataset(batch_size=size)
    model, classifier = _load_model(model_name)
    attack = _create_attack(classifier, eps=eps, max_iter=max_iter)

    evaluation = Evaluation(
        name=experiment,
        description=f"Imagenet classification using {model_name} model with PGD attack",
        author="TwoSix",
        attack=attack,
        dataset=dataset,
        metric=_create_metric(),
        model=model,
    )

    task = ImageClassificationTask(
        evaluation,
        num_classes=1000,
        export_every_n_batches=export,
    )

    track_param("main.num_batches", batches)
    engine = EvaluationEngine(task, limit_test_batches=batches)
    return engine.run()


if __name__ == "__main__":
    import docopt

    args = docopt.docopt(__doc__)
    args = {k.replace("--", ""): v for k, v in args.items()}

    # emit a shell script to run all registered models
    if args["script"]:
        for model in _MODELS:
            print(
                f"python run_matrix.py --model {model} "
                f"--batches {args['batches']} --size {args['size']} "
                f"--export {args['export']}"
            )
        sys.exit(0)

    res = run_evaluation(
        experiment="mwartell-imagenet-nomatrix",
        model_name=args["model"],
        batches=int(args["batches"]),
        size=int(args["size"]),
        export=int(args["export"]),
        eps=0.003,
        max_iter=10,
    )
    print(res)
