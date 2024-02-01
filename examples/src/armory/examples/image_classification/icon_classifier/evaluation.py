from pprint import pprint
from typing import Optional

import art.attacks.evasion
import art.defences.preprocessor
import art.estimators.classification
import numpy as np
import torch
import torch.nn
import torchmetrics.classification

import armory.data
import armory.dataset
import armory.engine
import armory.evaluation
import armory.examples.image_classification.icon_classifier.icon645 as icon645
import armory.examples.image_classification.icon_classifier.vit as vit
import armory.experimental.patch
import armory.export.image_classification
import armory.metric
import armory.metrics.compute
import armory.metrics.perturbation
import armory.model.image_classification
import armory.perturbation
import armory.track
import armory.utils

normalized_scale = armory.data.Scale(
    dtype=armory.data.DataType.FLOAT,
    max=1.0,
    mean=vit.image_processor.image_mean,
    std=vit.image_processor.image_std,
)

unnormalized_scale = armory.data.Scale(
    dtype=armory.data.DataType.FLOAT,
    max=1.0,
)


def parse_cli_args():
    """Parse command-line arguments"""
    from armory.examples.utils.args import create_parser

    parser = create_parser(
        description="Perform image classification of icon645",
        batch_size=16,
        export_every_n_batches=5,
        num_batches=5,
    )
    return parser.parse_args()


def load_model():
    """Load model"""
    hf_model = vit.load_model()

    armory_model = armory.model.image_classification.ImageClassifier(
        name="ViT-finetuned-icon645",
        model=hf_model,
        accessor=armory.data.Images.as_torch(scale=normalized_scale),
    )

    art_classifier = armory.track.track_init_params(
        art.estimators.classification.PyTorchClassifier
    )(
        armory_model,
        loss=torch.nn.CrossEntropyLoss(),
        # optimizer=torch.optim.Adam(armory_model.parameters(), lr=0.003),
        input_shape=(3, 224, 224),
        channels_first=True,
        nb_classes=377,
        clip_values=(0.0, 1.0),
        preprocessing=(
            np.array(vit.image_processor.image_mean),
            np.array(vit.image_processor.image_std),
        ),
    )

    return armory_model, art_classifier


def load_dataset(batch_size: int, shuffle: bool):
    """Load icon645 dataset"""

    hf_dataset = icon645.load_dataset()["test"]
    hf_dataset.set_transform(vit.eval_transform)

    labels = hf_dataset.features["label"].names

    dataloader = armory.dataset.ImageClassificationDataLoader(
        hf_dataset,
        dim=armory.data.ImageDimensions.CHW,
        scale=normalized_scale,
        image_key="image",
        label_key="label",
        batch_size=batch_size,
        shuffle=shuffle,
    )

    evaluation_dataset = armory.evaluation.Dataset(
        name="icon645",
        dataloader=dataloader,
    )

    return evaluation_dataset, labels


def create_pgd_attack(classifier: art.estimators.classification.PyTorchClassifier):
    """Creates the PGD attack"""
    pgd = armory.track.track_init_params(art.attacks.evasion.ProjectedGradientDescent)(
        classifier,
        batch_size=1,
        eps=0.003,
        eps_step=0.0005,
        max_iter=20,  # 100,
        # norm=1,
        num_random_init=1,
        random_eps=False,
        targeted=False,
        verbose=False,
    )

    evaluation_attack = armory.perturbation.ArtEvasionAttack(
        name="PGD",
        attack=pgd,
        use_label_for_untargeted=True,
        inputs_accessor=armory.data.Images.as_numpy(scale=unnormalized_scale),
    )

    return evaluation_attack


def create_compression_defence(
    classifier: Optional[art.estimators.classification.PyTorchClassifier] = None,
):
    jpeg_compression = armory.track.track_init_params(
        art.defences.preprocessor.JpegCompression
    )(
        clip_values=(0.0, 1.0),
        quality=50,
        channels_first=True,
    )

    if classifier is not None:
        armory.utils.apply_art_preprocessor_defense(classifier, jpeg_compression)

    perturbation = armory.perturbation.ArtPreprocessorDefence(
        name="JPEG_compression",
        defence=jpeg_compression,
        inputs_accessor=armory.data.Images.as_numpy(scale=unnormalized_scale),
    )

    return perturbation


def create_metrics():
    """Create evaluation metrics"""
    return {
        "linf_norm": armory.metric.PerturbationMetric(
            armory.metrics.perturbation.PerturbationNormMetric(ord=torch.inf),
        ),
        "accuracy": armory.metric.PredictionMetric(
            torchmetrics.classification.Accuracy(task="multiclass", num_classes=377),
        ),
    }


@armory.track.track_params(prefix="main")
def main(
    batch_size,
    export_every_n_batches,
    num_batches,
    seed,
    shuffle,
):
    """Perform evaluation"""
    if seed is not None:
        torch.manual_seed(seed)

    sysconfig = armory.evaluation.SysConfig()
    model, art_classifier = load_model()
    dataset, _ = load_dataset(batch_size, shuffle)
    perturbations = dict()
    metrics = create_metrics()
    profiler = armory.metrics.compute.BasicProfiler()

    perturbations["benign_no_defense"] = []
    # perturbations["benign_with_defense"] = [create_compression_defence()]
    # perturbations["pgd_undefended"] = [create_pgd_attack(art_classifier)]
    # perturbations["pgd_no_feedback_defended"] = [
    #     create_pgd_attack(art_classifier),
    #     create_compression_defence(),
    # ]
    # _, classifier2 = load_model()
    # defense = create_compression_defence(classifier2)
    # perturbations["pgd_with_feedback_defended"] = [
    #     create_pgd_attack(classifier2),
    #     defense,
    # ]

    evaluation = armory.evaluation.Evaluation(
        name="icon645-classification-defense-analysis",
        description="Image classification of icon645",
        author="TwoSix",
        dataset=dataset,
        model=model,
        perturbations=perturbations,
        metrics=metrics,
        exporter=armory.export.image_classification.ImageClassificationExporter(),
        profiler=profiler,
        sysconfig=sysconfig,
    )

    engine = armory.engine.EvaluationEngine(
        evaluation,
        export_every_n_batches=export_every_n_batches,
        limit_test_batches=num_batches,
    )
    results = engine.run()

    pprint(results)


if __name__ == "__main__":
    main(**vars(parse_cli_args()))
