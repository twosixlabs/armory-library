"""
Example Armory evaluation of food-101 image classification against projected
gradient descent (PGD) adversarial perturbation
"""

from pprint import pprint

import art.attacks.evasion
import art.estimators.classification
import numpy as np
import torch
import torch.nn
import torchmetrics.classification
import transformers

import armory.data
import armory.engine
import armory.evaluation
import armory.metrics.compute
import armory.metrics.perturbation
import armory.model.image_classification
import armory.perturbation
import armory.tasks.image_classification
import armory.track
import armory.utils


def parse_cli_args():
    """Parse command-line arguments"""
    from armory.examples.utils.args import create_parser

    parser = create_parser(
        description="Perform image classification of food-101",
        batch_size=16,
        export_every_n_batches=5,
        num_batches=5,
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
    )
    parser.add_argument(
        "--seed",
        default=None,
        type=int,
    )
    parser.add_argument(
        "--dataset",
        dest="dataset_src",
        choices=["huggingface", "torchvision"],
        default="huggingface",
    )
    return parser.parse_args()


def load_model():
    """Load model from HuggingFace"""
    hf_model = armory.track.track_params(
        transformers.AutoModelForImageClassification.from_pretrained
    )(pretrained_model_name_or_path="nateraw/food")

    armory_model = armory.model.image_classification.JaticImageClassificationModel(
        hf_model
    )

    art_classifier = armory.track.track_init_params(
        art.estimators.classification.PyTorchClassifier
    )(
        armory_model,
        loss=torch.nn.CrossEntropyLoss(),
        optimizer=torch.optim.Adam(armory_model.parameters(), lr=0.003),
        input_shape=(224, 224, 3),
        channels_first=False,
        nb_classes=101,
        clip_values=(0.0, 1.0),
    )

    evaluation_model = armory.evaluation.Model(
        name="ViT-finetuned-food101",
        model=art_classifier,
    )

    return evaluation_model, art_classifier


def transform(processor, sample):
    """Use the HF image processor and convert from BW To RGB"""
    sample["image"] = processor([img.convert("RGB") for img in sample["image"]])[
        "pixel_values"
    ]
    return sample


def load_huggingface_dataset(batch_size: int, shuffle: bool):
    """Load food-101 dataset from HuggingFace"""
    import functools

    import datasets

    hf_dataset = datasets.load_dataset("food101", split="validation")
    assert isinstance(hf_dataset, datasets.Dataset)

    labels = hf_dataset.features["label"].names

    hf_processor = transformers.AutoImageProcessor.from_pretrained("nateraw/food")
    hf_dataset.set_transform(functools.partial(transform, hf_processor))

    dataloader = armory.data.ArmoryDataLoader(
        hf_dataset, batch_size=batch_size, shuffle=shuffle
    )

    evaluation_dataset = armory.evaluation.Dataset(
        name="food-101",
        x_key="image",
        y_key="label",
        test_dataloader=dataloader,
    )

    return evaluation_dataset, labels


def load_torchvision_dataset(
    batch_size: int, shuffle: bool, sysconfig: armory.evaluation.SysConfig
):
    """Load food-101 dataset from TorchVision"""
    from torchvision import datasets
    from torchvision import transforms as T

    tv_dataset = datasets.Food101(
        root=str(sysconfig.dataset_cache),
        split="test",
        download=True,
        transform=T.Compose(
            [
                T.Resize(size=(224, 224)),
                T.ToTensor(),  # HWC->CHW and scales to 0-1
                T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
                T.Lambda(np.asarray),
            ]
        ),
    )

    labels = tv_dataset.classes

    armory_dataset = armory.data.TupleDataset(
        tv_dataset,
        x_key="image",
        y_key="label",
    )

    dataloader = armory.data.ArmoryDataLoader(
        armory_dataset, batch_size=batch_size, shuffle=shuffle
    )

    evaluation_dataset = armory.evaluation.Dataset(
        name="food-101",
        x_key="image",
        y_key="label",
        test_dataloader=dataloader,
    )

    return evaluation_dataset, labels


def create_attack(classifier: art.estimators.classification.PyTorchClassifier):
    """Creates the PGD attack"""
    pgd = armory.track.track_init_params(art.attacks.evasion.ProjectedGradientDescent)(
        classifier,
        batch_size=1,
        eps=0.031,
        eps_step=0.007,
        max_iter=20,
        num_random_init=1,
        random_eps=False,
        targeted=False,
        verbose=False,
    )

    evaluation_attack = armory.perturbation.ArtEvasionAttack(
        name="PGD",
        attack=pgd,
        use_label_for_untargeted=True,
    )

    return evaluation_attack


def create_metric():
    """Create evaluation metrics"""
    evaluation_metric = armory.evaluation.Metric(
        profiler=armory.metrics.compute.BasicProfiler(),
        perturbation={
            "linf_norm": armory.metrics.perturbation.PerturbationNormMetric(
                ord=torch.inf
            ),
        },
        prediction={
            "accuracy": torchmetrics.classification.Accuracy(
                task="multiclass", num_classes=101
            ),
        },
    )

    return evaluation_metric


@armory.track.track_params(prefix="main")
def main(batch_size, export_every_n_batches, num_batches, dataset_src, seed, shuffle):
    """Perform evaluation"""
    if seed is not None:
        torch.manual_seed(seed)

    sysconfig = armory.evaluation.SysConfig()
    model, art_classifier = load_model()
    dataset, _ = (
        load_huggingface_dataset(batch_size, shuffle)
        if dataset_src == "huggingface"
        else load_torchvision_dataset(batch_size, shuffle, sysconfig)
    )
    attack = create_attack(art_classifier)
    metric = create_metric()

    evaluation = armory.evaluation.Evaluation(
        name=f"food101-classification-{dataset_src}",
        description=f"Image classification of food-101 from {dataset_src}",
        author="TwoSix",
        dataset=dataset,
        model=model,
        perturbations={
            "benign": [],
            "attack": [attack],
        },
        metric=metric,
        sysconfig=sysconfig,
    )

    task = armory.tasks.image_classification.ImageClassificationTask(
        evaluation,
        export_every_n_batches=export_every_n_batches,
        export_adapter=armory.utils.Unnormalize(
            mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)
        ),
    )
    engine = armory.engine.EvaluationEngine(task, limit_test_batches=num_batches)
    results = engine.run()

    pprint(results)


if __name__ == "__main__":
    main(**vars(parse_cli_args()))
