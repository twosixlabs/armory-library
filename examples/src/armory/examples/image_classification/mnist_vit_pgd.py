import functools
from pprint import pprint

from art.attacks.evasion import ProjectedGradientDescent
from art.estimators.classification import PyTorchClassifier
import datasets
import torch
import torch.nn
import torch.utils.data.dataloader
import torchmetrics.classification
from torchvision.transforms.v2 import GaussianBlur
from transformers import AutoImageProcessor, AutoModelForImageClassification

from armory.examples.utils.args import create_parser
from armory.metrics.compute import BasicProfiler
from charmory.batch import BatchedImages, DataType, ImageDimensions, NDimArray, Scale
from charmory.data import ImageClassificationDataLoader
from charmory.engine import EvaluationEngine
import charmory.evaluation as ev
from charmory.metric import PerturbationMetric, PredictionMetric
from charmory.metrics.perturbation import PerturbationNormMetric
from charmory.model.image_classification import (
    ImageClassifier,
    JaticImageClassificationModel,
)
from charmory.perturbation import ArtEvasionAttack, CallablePerturbation
from charmory.tasks.image_classification import ImageClassificationTask
from charmory.track import track_init_params, track_params
from charmory.utils import Unnormalize


def get_cli_args():
    parser = create_parser(
        description="MNIST image classification using a ViT model and PGD attack",
        batch_size=16,
        export_every_n_batches=5,
        num_batches=10,
    )
    return parser.parse_args()


###
# Transform
###
def transform(processor, sample):
    # Use the HF image processor and convert from BW To RGB
    sample["image"] = processor([img.convert("RGB") for img in sample["image"]])[
        "pixel_values"
    ]
    return sample


@track_params
def main(batch_size, export_every_n_batches, num_batches):
    ###
    # Model
    ###
    model = JaticImageClassificationModel(
        "vit",
        track_params(AutoModelForImageClassification.from_pretrained)(
            "farleyknight-org-username/vit-base-mnist"
        ),
    )
    classifier = track_init_params(PyTorchClassifier)(
        model,
        loss=torch.nn.CrossEntropyLoss(),
        optimizer=torch.optim.Adam(model.parameters(), lr=0.003),
        input_shape=(3, 224, 224),
        channels_first=True,
        nb_classes=10,
        clip_values=(-1, 1),
    )

    ###
    # Dataset
    ###
    dataset = datasets.load_dataset("mnist", split="test")
    processor = AutoImageProcessor.from_pretrained(
        "farleyknight-org-username/vit-base-mnist"
    )

    dataset.set_transform(functools.partial(transform, processor))
    dataloader = ImageClassificationDataLoader(
        dataset,
        dim=ImageDimensions.CHW,
        scale=Scale(dtype=DataType.FLOAT, max=1.0),
        image_key="image",
        label_key="label",
        batch_size=batch_size,
        num_workers=5,
        shuffle=True,
    )

    ###
    # Attack
    ###
    blur = track_init_params(GaussianBlur)(
        kernel_size=5,
    )

    blur_perturb = CallablePerturbation(
        name="blur",
        perturbation=blur,
        inputs_accessor=BatchedImages.as_torch(),
    )

    pgd = track_init_params(ProjectedGradientDescent)(
        classifier,
        batch_size=batch_size,
        eps=0.031,
        eps_step=0.007,
        max_iter=20,
        num_random_init=1,
        random_eps=False,
        targeted=False,
        verbose=False,
    )

    pgd_attack = ArtEvasionAttack(
        name="PGD",
        attack=pgd,
        inputs_accessor=BatchedImages.as_numpy(),
        use_label_for_untargeted=False,
    )

    ###
    # Metrics
    ###
    metrics = {
        "linf_norm": PerturbationMetric(
            PerturbationNormMetric(ord=torch.inf),
            BatchedImages.as_torch(),
        ),
        "accuracy_avg": PredictionMetric(
            torchmetrics.classification.Accuracy(task="multiclass", num_classes=10),
            NDimArray.as_torch(device=torch.device("cuda")),
        ),
        "accuracy_by_class": PredictionMetric(
            torchmetrics.classification.Accuracy(
                task="multiclass", num_classes=10, average=None
            ),
            NDimArray.as_torch(device=torch.device("cuda")),
        ),
        "precision_avg": PredictionMetric(
            torchmetrics.classification.Precision(task="multiclass", num_classes=10),
            NDimArray.as_torch(device=torch.device("cuda")),
        ),
        "precision_by_class": PredictionMetric(
            torchmetrics.classification.Precision(
                task="multiclass", num_classes=10, average=None
            ),
            NDimArray.as_torch(device=torch.device("cuda")),
        ),
        "recall_avg": PredictionMetric(
            torchmetrics.classification.Recall(task="multiclass", num_classes=10),
            NDimArray.as_torch(device=torch.device("cuda")),
        ),
        "recall_by_class": PredictionMetric(
            torchmetrics.classification.Recall(
                task="multiclass", num_classes=10, average=None
            ),
            NDimArray.as_torch(device=torch.device("cuda")),
        ),
        "f1_score_avg": PredictionMetric(
            torchmetrics.classification.F1Score(task="multiclass", num_classes=10),
            NDimArray.as_torch(device=torch.device("cuda")),
        ),
        "f1_score_by_class": PredictionMetric(
            torchmetrics.classification.F1Score(
                task="multiclass", num_classes=10, average=None
            ),
            NDimArray.as_torch(device=torch.device("cuda")),
        ),
        "confusion": PredictionMetric(
            torchmetrics.classification.ConfusionMatrix(
                task="multiclass", num_classes=10
            ),
            NDimArray.as_torch(device=torch.device("cuda")),
        ),
    }

    ###
    # Evaluation
    ###
    evaluation = ev.Evaluation(
        name="mnist-vit-pgd",
        description="MNIST image classification using a ViT model and PGD attack",
        author="TwoSix",
        dataset=ev.Dataset(
            name="MNIST",
            dataloader=dataloader,
        ),
        model=ImageClassifier(
            name="ViT",
            model=model,
            inputs_accessor=BatchedImages.as_torch(device=torch.device("cuda")),
        ),
        perturbations={
            "benign": [],
            "attack": [pgd_attack],
            "blur": [blur_perturb],
        },
        metrics=metrics,
        profiler=BasicProfiler(),
    )

    ###
    # Engine
    ###
    task = ImageClassificationTask(
        evaluation,
        export_adapter=Unnormalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        export_every_n_batches=export_every_n_batches,
    )
    engine = EvaluationEngine(task, limit_test_batches=num_batches)

    ###
    # Execute
    ###
    pprint(engine.run())
    pprint(task.metrics.compute())
    print("benign")
    pprint(task.metrics["benign"]["confusion"].compute())
    print("attack")
    pprint(task.metrics["attack"]["confusion"].compute())


if __name__ == "__main__":
    main(**vars(get_cli_args()))
