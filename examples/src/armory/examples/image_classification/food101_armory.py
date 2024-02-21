"""
Example Armory evaluation of food-101 image classification against projected
gradient descent (PGD) adversarial perturbation
"""

import functools
from pprint import pprint

import art.attacks.evasion
import art.defences.preprocessor
import art.estimators.classification
import datasets
import torch
import torch.nn
import torchmetrics.classification
import transformers

import armory.data
import armory.dataset
import armory.engine
import armory.evaluation
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
    mean=(0.5, 0.5, 0.5),
    std=(0.5, 0.5, 0.5),
)

unnormalized_scale = armory.data.Scale(
    dtype=armory.data.DataType.FLOAT,
    max=1.0,
)


# Model


hf_model = armory.track.track_params(
    transformers.AutoModelForImageClassification.from_pretrained
)(pretrained_model_name_or_path="nateraw/food")

armory_model = armory.model.image_classification.ImageClassifier(
    name="ViT-finetuned-food101",
    model=hf_model,
    accessor=armory.data.Images.as_torch(scale=normalized_scale),
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


# Dataset


def transform(processor, sample):
    """Use the HF image processor and convert from BW To RGB"""
    sample["image"] = processor([img.convert("RGB") for img in sample["image"]])[
        "pixel_values"
    ]
    return sample


hf_dataset = datasets.load_dataset("food101", split="validation")
assert isinstance(hf_dataset, datasets.Dataset)

labels = hf_dataset.features["label"].names

hf_processor = transformers.AutoImageProcessor.from_pretrained("nateraw/food")
hf_dataset.set_transform(functools.partial(transform, hf_processor))

dataloader = armory.dataset.ImageClassificationDataLoader(
    hf_dataset,
    dim=armory.data.ImageDimensions.CHW,
    scale=normalized_scale,
    image_key="image",
    label_key="label",
    batch_size=16,
)

evaluation_dataset = armory.evaluation.Dataset(
    name="food-101",
    dataloader=dataloader,
)


# Attack


pgd = armory.track.track_init_params(art.attacks.evasion.ProjectedGradientDescent)(
    art_classifier,
    batch_size=1,
    eps=0.003,
    eps_step=0.0007,
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
    inputs_accessor=armory.data.Images.as_numpy(scale=normalized_scale),
)


# Defense


jpeg_compression = armory.track.track_init_params(
    art.defences.preprocessor.JpegCompression
)(
    clip_values=(0.0, 1.0),
    quality=50,
    channels_first=True,
)

evaluation_defense = armory.perturbation.ArtPreprocessorDefence(
    name="JPEG_compression",
    defence=jpeg_compression,
    inputs_accessor=armory.data.Images.as_numpy(scale=unnormalized_scale),
)


# Metrics


metrics = {
    "linf_norm": armory.metric.PerturbationMetric(
        armory.metrics.perturbation.PerturbationNormMetric(ord=torch.inf),
    ),
    "accuracy": armory.metric.PredictionMetric(
        torchmetrics.classification.Accuracy(task="multiclass", num_classes=101),
    ),
}


# Evaluation


evaluation = armory.evaluation.Evaluation(
    name="food101-classification",
    description="Image classification of food-101",
    author="TwoSix",
    dataset=evaluation_dataset,
    model=armory_model,
    perturbations=dict(
        benign=[],
        attacked=[evaluation_attack],
        defended=[evaluation_attack, evaluation_defense],
    ),
    metrics=metrics,
    exporter=armory.export.image_classification.ImageClassificationExporter(),
    profiler=armory.metrics.compute.BasicProfiler(),
    sysconfig=armory.evaluation.SysConfig(),
)

engine = armory.engine.EvaluationEngine(
    evaluation,
    export_every_n_batches=5,
    limit_test_batches=5,
)
results = engine.run()

pprint(results)
