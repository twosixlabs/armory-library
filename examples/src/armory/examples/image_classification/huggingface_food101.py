from pprint import pprint

import art.attacks.evasion
from art.estimators.classification import PyTorchClassifier
import datasets
import torch
import torch.nn
import torchmetrics.classification
from transformers import AutoImageProcessor, AutoModelForImageClassification

from armory.data import ArmoryDataLoader
from armory.engine import EvaluationEngine
from armory.evaluation import Dataset, Evaluation, Metric, Model
from armory.examples.utils.args import create_parser
from armory.metrics.compute import BasicProfiler
from armory.metrics.perturbation import PerturbationNormMetric
from armory.model.image_classification import JaticImageClassificationModel
from armory.perturbation import ArtEvasionAttack
from armory.tasks.image_classification import ImageClassificationTask
from armory.track import track_init_params, track_params


def get_cli_args():
    parser = create_parser(
        description="Perform image classification of food-101 from HuggingFace",
        batch_size=16,
        export_every_n_batches=5,
        num_batches=5,
    )
    return parser.parse_args()


def main(args):
    ###
    # Model
    ###
    model = JaticImageClassificationModel(
        track_params(AutoModelForImageClassification.from_pretrained)("nateraw/food")
    )

    classifier = track_init_params(PyTorchClassifier)(
        model,
        loss=torch.nn.CrossEntropyLoss(),
        optimizer=torch.optim.Adam(model.parameters(), lr=0.003),
        input_shape=(224, 224, 3),
        channels_first=False,
        nb_classes=101,
        clip_values=(0.0, 1.0),
    )

    ###
    # Dataset
    ###
    dataset = datasets.load_dataset("food101", split="validation")
    processor = AutoImageProcessor.from_pretrained("nateraw/food")

    def transform(sample):
        # Use the HF image processor and convert from BW To RGB
        sample["image"] = processor([img.convert("RGB") for img in sample["image"]])[
            "pixel_values"
        ]
        return sample

    dataset.set_transform(transform)

    dataloader = ArmoryDataLoader(dataset, batch_size=args.batch_size)

    ###
    # Evaluation
    ###
    eval_dataset = Dataset(
        name="food-category-classification",
        x_key="image",
        y_key="label",
        test_dataloader=dataloader,
    )

    eval_model = Model(
        name="food-category-classification",
        model=classifier,
    )

    eval_attack = ArtEvasionAttack(
        name="PGD",
        attack=track_init_params(art.attacks.evasion.ProjectedGradientDescent)(
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

    eval_metric = Metric(
        profiler=BasicProfiler(),
        perturbation={
            "linf_norm": PerturbationNormMetric(ord=torch.inf),
        },
        prediction={
            "accuracy": torchmetrics.classification.Accuracy(
                task="multiclass", num_classes=101
            ),
        },
    )

    evaluation = Evaluation(
        name="hf-food101-classification",
        description="Image classification of food-101 from HuggingFace",
        author="Kaludi",
        dataset=eval_dataset,
        model=eval_model,
        perturbations={
            "benign": [],
            "attack": [eval_attack],
        },
        metric=eval_metric,
    )

    ###
    # Engine
    ###

    task = ImageClassificationTask(
        evaluation, export_every_n_batches=args.export_every_n_batches
    )
    engine = EvaluationEngine(task, limit_test_batches=args.num_batches)
    results = engine.run()

    pprint(results)


if __name__ == "__main__":
    main(get_cli_args())
