from pprint import pprint

import art.attacks.evasion
from art.estimators.classification import PyTorchClassifier
import jatic_toolbox
import numpy as np
import torch.nn
from transformers.image_utils import infer_channel_dimension_format

from armory.examples.utils.args import create_parser
from armory.metrics.compute import BasicProfiler
from charmory.data import ArmoryDataLoader
from charmory.engine import EvaluationEngine
from charmory.evaluation import Attack, Dataset, Evaluation, Metric, Model
from charmory.model.image_classification import JaticImageClassificationModel
from charmory.tasks.image_classification import ImageClassificationTask
from charmory.track import track_init_params, track_params
from charmory.utils import create_jatic_dataset_transform


def get_cli_args():
    parser = create_parser(
        description="Run food classification example using models and datasets from the JATIC toolbox",
        batch_size=16,
        export_every_n_batches=5,
        num_batches=5,
    )
    return parser.parse_args()


def main(args):
    ###
    # Model
    ###
    model = track_params(jatic_toolbox.load_model)(
        provider="huggingface",
        model_name="Kaludi/food-category-classification-v2.0",
        task="image-classification",
    )

    classifier = track_init_params(PyTorchClassifier)(
        JaticImageClassificationModel(model),
        loss=torch.nn.CrossEntropyLoss(),
        optimizer=torch.optim.Adam(model.parameters(), lr=0.003),
        input_shape=(224, 224, 3),
        channels_first=False,
        nb_classes=12,
        clip_values=(0.0, 1.0),
    )

    ###
    # Dataset
    ###
    dataset = track_params(jatic_toolbox.load_dataset)(
        provider="huggingface",
        dataset_name="Kaludi/food-category-classification-v2.0",
        task="image-classification",
        split="validation",
    )

    def filter(sample):
        try:
            infer_channel_dimension_format(np.asarray(sample["image"]))
            return True
        except Exception:
            return False

    dataset._dataset = dataset._dataset.filter(filter)

    transform = create_jatic_dataset_transform(model.preprocessor)
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

    eval_attack = Attack(
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
    )

    evaluation = Evaluation(
        name="food-category-classification",
        description="Food category classification from HuggingFace",
        author="Kaludi",
        dataset=eval_dataset,
        model=eval_model,
        attack=eval_attack,
        metric=eval_metric,
    )

    ###
    # Engine
    ###

    task = ImageClassificationTask(
        evaluation, num_classes=12, export_every_n_batches=args.export_every_n_batches
    )
    engine = EvaluationEngine(task, limit_test_batches=args.num_batches)
    results = engine.run()

    pprint(results)

    print("JATIC Experiment Complete!")
    return 0


if __name__ == "__main__":
    main(get_cli_args())
