"""Definition for the EuroSAT classification evaluation"""

import argparse
from copy import deepcopy
import os

import albumentations as A
from art.attacks.evasion import ProjectedGradientDescent
from art.estimators.classification import PyTorchClassifier
import jatic_toolbox
import numpy as np
import torch.nn as nn

from armory.metrics.compute import BasicProfiler
from charmory.data import ArmoryDataLoader, JaticImageClassificationDataset
from charmory.evaluation import Attack, Dataset, Evaluation, Metric, Model
from charmory.model.image_classification import JaticImageClassificationModel
from charmory.tasks.image_classification import ImageClassificationTask
from charmory.track import track_init_params, track_params

_MODELS = {
    "untrained": f"{os.path.dirname(__file__)}/finetuned_eurosat_final/",
    "convnext": "mrm8488/convnext-tiny-finetuned-eurosat",
    "swin": "nielsr/swin-tiny-patch4-window7-224-finetuned-eurosat",
    "vit": "nielsr/vit-finetuned-eurosat-kornia",
}


def get_cli_args(with_attack: bool):
    """Get CLI-specified arguments to configure the evaluation."""
    parser = argparse.ArgumentParser(
        description="Run EuroSAT image classification evaluation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "model_name",
        choices=_MODELS.keys(),
    )
    parser.add_argument(
        "--batch-size",
        default=4,
        type=int,
    )
    parser.add_argument(
        "--num-batches",
        default=20,
        type=int,
    )

    if with_attack:
        attack_args = parser.add_argument_group("attack", "PGD attack parameters")
        attack_args.add_argument(
            "--attack-batch-size",
            default=1,
            type=int,
        )
        attack_args.add_argument(
            "--attack-eps",
            default=0.031,
            type=float,
        )
        attack_args.add_argument(
            "--attack-eps-step",
            default=0.007,
            type=float,
        )
        attack_args.add_argument(
            "--attack-max-iter",
            default=20,
            type=int,
        )
        attack_args.add_argument(
            "--attack-num-random-init",
            default=1,
            type=int,
        )
        attack_args.add_argument(
            "--attack-random-eps",
            action="store_true",
        )
        attack_args.add_argument(
            "--attack-targeted",
            action="store_true",
        )

    return parser.parse_args()


def _load_model(name: str):
    model = track_params(jatic_toolbox.load_model)(
        provider="huggingface",
        model_name=name,
        task="image-classification",
    )

    classifier = track_init_params(PyTorchClassifier)(
        JaticImageClassificationModel(model),
        loss=nn.CrossEntropyLoss(),
        input_shape=(224, 224, 3),
        channels_first=False,
        nb_classes=10,
        clip_values=(0.0, 1.0),
    )

    eval_model = Model(
        name=name,
        model=classifier,
    )

    return eval_model, classifier


def _load_dataset(batch_size: int):
    dataset = track_params(jatic_toolbox.load_dataset)(
        provider="huggingface",
        dataset_name="tanganke/EuroSAT",
        task="image-classification",
        split="test",
    )

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

    dataset.set_transform(transform_func)

    dataloader = ArmoryDataLoader(
        JaticImageClassificationDataset(dataset), batch_size=batch_size
    )

    eval_dataset = Dataset(
        name="EuroSAT",
        test_dataset=dataloader,
    )

    return eval_dataset


def _create_metric():
    return Metric(profiler=BasicProfiler())


def _create_attack(classifier: PyTorchClassifier, verbose: bool = False, **kwargs):
    return Attack(
        name="PGD",
        attack=track_init_params(ProjectedGradientDescent)(
            classifier,
            verbose=False,
            **kwargs,
        ),
        use_label_for_untargeted=True,
    )


def _get_attack_kwargs(**kwargs):
    """Filter kwargs down to those starting with "attack_" and remove that prefix"""
    attack_kwargs = {}
    for key, value in kwargs.items():
        if key.startswith("attack_"):
            attack_kwargs[key[7:]] = value
    return attack_kwargs


def create_evaluation(
    model_name: str, batch_size: int, with_attack: bool, **kwargs
) -> Evaluation:
    model, classifier = _load_model(_MODELS[model_name])
    dataset = _load_dataset(batch_size=batch_size)
    attack = (
        _create_attack(classifier, **_get_attack_kwargs(**kwargs))
        if with_attack
        else None
    )

    evaluation = Evaluation(
        name=f"eurosat-{model_name}-pgd-generation",
        # name=f"eurosat-{model_name}-precomputed-pgd",
        description=f"EuroSAT classification using {model_name} model with PGD attack generation",
        # description=f"EuroSAT classification using {model_name} model with precomputed PGD attack",
        author="TwoSix",
        attack=attack,
        dataset=dataset,
        metric=_create_metric(),
        model=model,
    )

    return evaluation


def create_attack_evaluation_task(**kwargs) -> ImageClassificationTask:
    evaluation = create_evaluation(with_attack=True, **kwargs)

    task = ImageClassificationTask(
        evaluation,
        num_classes=10,
        export_every_n_batches=1,  # Export all batches
    )

    return task


def create_pregenerated_evaluation_task(**kwargs) -> ImageClassificationTask:
    evaluation = create_evaluation(with_attack=False, **kwargs)

    task = ImageClassificationTask(
        evaluation,
        num_classes=10,
        export_every_n_batches=0,
        skip_attack=True,
    )

    return task
