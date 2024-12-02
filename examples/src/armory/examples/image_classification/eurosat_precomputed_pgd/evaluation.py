"""Definition for the EuroSAT classification evaluation"""

from copy import deepcopy
import os
from typing import Optional

import albumentations as A
from art.attacks.evasion import ProjectedGradientDescent
from art.estimators.classification import PyTorchClassifier
import datasets
# import jatic_toolbox
# from jatic_toolbox.interop.huggingface import HuggingFaceVisionDataset
import numpy as np
import torch
import torch.nn as nn

from armory.examples.utils.args import create_parser
from armory.metrics.compute import BasicProfiler
from armory.dataset import DataLoader
from armory.evaluation import Attack, Dataset, Evaluation, Metric, Model
# from armory.model.image_classification import JaticImageClassificationModel
# from armory.tasks.image_classification import ImageClassificationTask
from armory.track import track_init_params, track_params

from armory.model.image_classification import ImageClassifier
import transformers
import armory

_MODELS = {
    "untrained": f"{os.path.dirname(__file__)}/finetuned_eurosat_final/",
    "convnext": "mrm8488/convnext-tiny-finetuned-eurosat",
    "swin": "nielsr/swin-tiny-patch4-window7-224-finetuned-eurosat",
    "vit": "nielsr/vit-finetuned-eurosat-kornia",
}

normalized_scale = armory.data.Scale(
    dtype=armory.data.DataType.FLOAT,
    max=1.0,
    mean=(0.5, 0.5, 0.5),
    std=(0.5, 0.5, 0.5),
)

def get_cli_args(with_attack: bool):
    """Get CLI-specified arguments to configure the evaluation."""
    parser = create_parser(
        description="Run EuroSAT image classification evaluation",
        batch_size=4,
        export_every_n_batches=5,
    )
    parser.add_argument(
        "model_name",
        choices=_MODELS.keys(),
    )
    if not with_attack:
        parser.add_argument(
            "dataset_path",
            type=str,
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
    hf_model = track_params(
        transformers.AutoModelForObjectDetection.from_pretrained
    )(pretrained_model_name_or_path=name)

    armory_model = ImageClassifier(
        name="ViT-finetuned-food101",
        model=hf_model,
        inputs_spec=armory.data.TorchImageSpec(
            dim=armory.data.ImageDimensions.CHW, scale=normalized_scale
        ),
    )

    classifier = track_init_params(PyTorchClassifier)(
        model=armory_model,
        loss=torch.nn.CrossEntropyLoss(),
        optimizer=torch.optim.Adam(armory_model.parameters(), lr=0.003),
        input_shape=(3, 224, 224),
        channels_first=True,
        nb_classes=101,
        clip_values=(-1, 1),
    )

    eval_model = Model(
        name=name,
        model=classifier,
    )

    return eval_model, classifier


def _load_dataset(batch_size: int, dataset_path: Optional[str] = None):
    if dataset_path is not None:
    #     ds = track_params(datasets.load_from_disk)(dataset_path=dataset_path)
    #     if isinstance(ds, datasets.DatasetDict):
    #         ds = ds["test"]
    #     dataset = HuggingFaceVisionDataset(ds)
        dataset = datasets.load_dataset(dataset_path, split="test", trust_remote_code=True)
    else:
        # dataset = track_params(jatic_toolbox.load_dataset)(
        #     provider="huggingface",
        #     dataset_name="tanganke/EuroSAT",
        #     task="image-classification",
        #     split="test",
        # )
        dataset = datasets.load_dataset("tanganke/EuroSAT", split="test", trust_remote_code=True)


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

    dataloader = DataLoader(dataset, batch_size=batch_size)

    eval_dataset = Dataset(
        name="EuroSAT",
        x_key="image",
        y_key="label",
        test_dataloader=dataloader,
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
    model_name: str,
    batch_size: int,
    with_attack: bool,
    dataset_path: Optional[str] = None,
    **kwargs,
) -> Evaluation:
    model, classifier = _load_model(_MODELS[model_name])
    dataset = _load_dataset(batch_size=batch_size, dataset_path=dataset_path)
    attack = (
        _create_attack(classifier, **_get_attack_kwargs(**kwargs))
        if with_attack
        else None
    )

    attack_type = "generated" if with_attack else "precomputed"

    evaluation = Evaluation(
        name=f"eurosat-{model_name}-{attack_type}-pgd",
        description=f"EuroSAT classification using {model_name} model with {attack_type} PGD attack",
        author="TwoSix",
        attack=attack,
        dataset=dataset,
        metric=_create_metric(),
        model=model,
    )

    return evaluation

def create_evaluation_task(
    export_every_n_batches: int, with_attack: bool, **kwargs
) -> ImageClassificationTask:
    evaluation = create_evaluation(with_attack=with_attack, **kwargs)

    task = ImageClassificationTask(
        evaluation,
        num_classes=10,
        export_every_n_batches=export_every_n_batches,
        skip_attack=not with_attack,
    )

    return task
