"""
Example Armory evaluation using a pre-perturbed food-101 dataset against a
collection of image classification models
"""

import argparse
import inspect
import sys
from typing import Any, Dict, Optional

import albumentations as A
import albumentations.pytorch.transforms
import art.estimators.classification
import datasets
import numpy as np
import torch
import transformers

import armory.data
import armory.dataset
import armory.engine
import armory.evaluation
from armory.examples.image_classification.food101 import (
    create_exporters,
    create_metrics,
    create_pgd_attack,
    load_huggingface_dataset,
    normalized_scale,
    unnormalized_scale,
)
import armory.model.image_classification
import armory.track

models = {
    "vit": {
        "name": "ViT-finetuned-food101",
        "hf_path": "nateraw/food",
        "inputs_spec": armory.data.TorchImageSpec(
            dim=armory.data.ImageDimensions.CHW, scale=normalized_scale
        ),
    },
    "swin": {
        "name": "swin-finetuned-food101",
        "hf_path": "skylord/swin-finetuned-food101",
        "inputs_spec": armory.data.TorchImageSpec(
            dim=armory.data.ImageDimensions.CHW,
            scale=armory.data.Scale(
                dtype=armory.data.DataType.FLOAT,
                max=1.0,
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ),
    },
}


def parse_cli_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        description="Perform food-101 PGD transferability evaluation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Generation arguments
    parser.add_argument(
        "--generate",
        action="store_true",
        help="Generate PGD-perturbed dataset",
    )
    parser.add_argument(
        "--gen-batch-size",
        default=16,
        help="Number of samples per batch when generated the PGD perturbations",
        type=int,
    )
    parser.add_argument(
        "--gen-num-batches",
        default=1000,
        help="Number of batches to generate for the PGD-perturbed dataset",
        type=int,
    )
    parser.add_argument(
        "--gen-shuffle",
        action="store_true",
        help="Shuffle the source dataset when generating PGD perturbations",
    )
    parser.add_argument(
        "--gen-seed",
        default=None,
        help="Randomization seed when shuffling the source dataset",
        type=int,
    )

    # Common arguments
    parser.add_argument(
        "--pgd-model",
        choices=models.keys(),
        required=True,
    )

    # Evaluation arguments
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Evaluate models on PGD-perturbed dataset",
    )
    parser.add_argument(
        "--eval-batch-size",
        default=16,
        help="Number of samples per batch when evaluating models",
        type=int,
    )
    parser.add_argument(
        "--eval-num-batches",
        default=1000,
        help="Number of batches for which to evaluate",
        type=int,
    )
    parser.add_argument(
        "--eval-shuffle",
        action="store_true",
        help="Shuffle the perturbed dataset when evaluating",
    )
    parser.add_argument(
        "--eval-seed",
        default=None,
        help="Randomization seed when shuffling the perturbed dataset",
        type=int,
    )
    parser.add_argument(
        "--eval-export-every-n-batches",
        default=5,
        help="Frequency at which batches will be exported to MLflow",
        type=int,
    )

    return parser.parse_args()


def filter_args(func, args: argparse.Namespace) -> Dict[str, Any]:
    """Filter arguments to keyword arguments supported by the given function"""
    sig = inspect.signature(func)
    # if the function supports **kwargs, return all arguments
    if any([p.kind == p.VAR_KEYWORD for p in sig.parameters.values()]):
        return vars(args)
    return {k: v for k, v in vars(args).items() if k in sig.parameters}


def ds_dir_name(pgd_model: str) -> str:
    """Create dataset output directory based on the model used to generate the perturbations"""
    sysconfig = armory.evaluation.SysConfig()
    return str(sysconfig.dataset_cache / "food-101-with-pgd" / pgd_model)


def load_model(model_name: str):
    """Load model from HuggingFace"""
    if model_name not in models:
        raise ValueError(f"Model {model_name} not supported")

    model = models[model_name]

    hf_model = armory.track.track_params(
        transformers.AutoModelForImageClassification.from_pretrained
    )(pretrained_model_name_or_path=model["hf_path"])

    armory_model = armory.model.image_classification.ImageClassifier(
        name=model["name"],
        model=hf_model,
        inputs_spec=model["inputs_spec"],
    )

    art_classifier = armory.track.track_init_params(
        art.estimators.classification.PyTorchClassifier
    )(
        armory_model,
        loss=torch.nn.CrossEntropyLoss(),
        optimizer=torch.optim.Adam(armory_model.parameters(), lr=0.003),
        input_shape=(3, 224, 224),
        channels_first=True,
        nb_classes=101,
        clip_values=(-1.0, 1.0),
    )

    return armory_model, art_classifier


def load_dataset(
    track, pgd_model: str, batch_size: int, shuffle: bool, seed: Optional[int]
):
    hf_dataset = track(datasets.load_from_disk, dataset_path=ds_dir_name(pgd_model))[
        "test"
    ]

    resize = A.Compose(
        [
            A.LongestMaxSize(224),
            A.PadIfNeeded(
                min_height=224,
                min_width=224,
                border_mode=0,
                value=(0, 0, 0),
            ),
            A.ToFloat(max_value=255),
            albumentations.pytorch.ToTensorV2(),
        ],
    )

    def transform(sample):
        tmp = dict(**sample)
        tmp["image"] = [
            resize(image=np.asarray(image.convert("RGB")))["image"]
            for image in sample["image"]
        ]
        return tmp

    hf_dataset.set_transform(transform)

    dataloader = armory.dataset.ImageClassificationDataLoader(
        hf_dataset,
        dim=armory.data.ImageDimensions.CHW,
        scale=unnormalized_scale,
        image_key="image",
        label_key="label",
        batch_size=batch_size,
        shuffle=shuffle,
        seed=seed,
    )

    dataset = armory.evaluation.Dataset(
        name=f"food-101-pgd-{pgd_model}",
        dataloader=dataloader,
    )
    return dataset


def generate_dataset(
    pgd_model: str,
    gen_batch_size: int,
    gen_num_batches: int,
    gen_shuffle: bool,
    gen_seed: Optional[int],
):
    """Generate PGD-perturbed dataset"""
    print("Generating PGD-perturbed dataset")

    evaluation = armory.evaluation.Evaluation(
        name=f"food-101-pgd-{pgd_model}-generation",
        description=f"PGD-perturbation of food-101 using {pgd_model} model",
        author="TwoSix",
    )

    with evaluation.add_chain("pgd") as chain:
        dataset, labels = load_huggingface_dataset(
            batch_size=gen_batch_size, shuffle=gen_shuffle, seed=gen_seed
        )
        chain.use_dataset(dataset)

        model, art_classifier = load_model(pgd_model)
        chain.use_model(model)

        pgd = create_pgd_attack(art_classifier)
        chain.add_perturbation(pgd)

    engine = armory.engine.AdversarialDatasetEngine(
        chain=evaluation.chains["pgd"],
        inputs_key="image",
        inputs_spec=armory.data.NumpyImageSpec(
            dim=armory.data.ImageDimensions.HWC,
            scale=armory.data.Scale(dtype=armory.data.DataType.UINT8, max=255),
            dtype=np.uint8,
        ),
        targets_key="label",
        output_dir=ds_dir_name(pgd_model),
        features=dataset.dataloader.dataset.features,
        num_batches=gen_num_batches,
    )
    engine.generate()


def evaluate_models(
    pgd_model: str,
    eval_batch_size: int,
    eval_num_batches: int,
    eval_shuffle: bool,
    eval_seed: bool,
    eval_export_every_n_batches: int,
):
    """Evaluate models on PGD-perturbed dataset"""
    print("Evaluating models on PGD-perturbed dataset")

    evaluation = armory.evaluation.Evaluation(
        name=f"food-101-pgd-{pgd_model}-evaluation",
        description=f"Evaluation of food-101 with PGD using {pgd_model} model",
        author="TwoSix",
    )

    with evaluation.autotrack() as track:
        dataset = load_dataset(
            track,
            pgd_model=pgd_model,
            batch_size=eval_batch_size,
            shuffle=eval_shuffle,
            seed=eval_seed,
        )
    evaluation.use_dataset(dataset)

    evaluation.use_metrics(create_metrics())

    for model_name in models.keys():
        with evaluation.add_chain(model_name) as chain:
            model, _ = load_model(model_name)
            chain.use_model(model)
            chain.use_exporters(create_exporters(model, eval_export_every_n_batches))

    engine = armory.engine.EvaluationEngine(
        evaluation,
        limit_test_batches=eval_num_batches,
    )
    results = engine.run()

    if results:
        for chain_name, chain_results in results.children.items():
            chain_results.metrics.table(title=f"{chain_name} Metrics")


if __name__ == "__main__":
    args = parse_cli_args()
    if not args.generate and not args.evaluate:
        print("No action specified. Exiting.")
        sys.exit(1)
    if args.generate:
        generate_dataset(**filter_args(generate_dataset, args))
    if args.evaluate:
        evaluate_models(**filter_args(evaluate_models, args))
