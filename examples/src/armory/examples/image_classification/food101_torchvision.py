from pprint import pprint

import art.attacks.evasion
from art.estimators.classification import PyTorchClassifier
import torch
import torch.nn
import torchmetrics.classification
from torchvision import datasets
from torchvision import transforms as T
from transformers import AutoModelForImageClassification

from armory.data import ArmoryDataLoader, TupleDataset
from armory.engine import EvaluationEngine
from armory.evaluation import Dataset, Evaluation, Metric, Model, SysConfig
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


def load_model():
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

    eval_model = Model(
        name="ViT-finetuned-food101",
        model=classifier,
    )

    return eval_model, classifier


def load_dataset(batch_size: int, sysconfig: SysConfig):
    dataset = TupleDataset(
        datasets.Food101(
            root=str(sysconfig.dataset_cache),
            split="test",
            download=True,
            transform=T.Compose(
                [
                    T.Resize(size=(224, 224)),
                    T.ToTensor(),
                ]
            ),
        ),
        x_key="image",
        y_key="label",
    )

    dataloader = ArmoryDataLoader(dataset, batch_size=batch_size)

    eval_dataset = Dataset(
        name="food-101",
        x_key="image",
        y_key="label",
        test_dataloader=dataloader,
    )

    return eval_dataset


def create_attack(classifier: PyTorchClassifier):
    pgd = track_init_params(art.attacks.evasion.ProjectedGradientDescent)(
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

    eval_attack = ArtEvasionAttack(
        name="PGD",
        attack=pgd,
        use_label_for_untargeted=True,
    )

    return eval_attack


def create_metric():
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

    return eval_metric


def main(args):
    sysconfig = SysConfig()
    model, art_classifier = load_model()
    dataset = load_dataset(args.batch_size, sysconfig)
    attack = create_attack(art_classifier)
    metric = create_metric()

    ###
    # Evaluation
    ###

    evaluation = Evaluation(
        name="tv-food101-classification",
        description="Image classification of food-101 from TorchVision",
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
