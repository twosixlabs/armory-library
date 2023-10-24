"""An mlflow experiment with varying parameters"""

from pprint import pprint

import art.attacks.evasion
from art.estimators.classification import PyTorchClassifier
import numpy as np
import torch
from transformers.image_utils import infer_channel_dimension_format

from armory.metrics.compute import BasicProfiler
from charmory.data import ArmoryDataLoader
from charmory.engine import EvaluationEngine
from charmory.evaluation import Attack, Dataset, Evaluation, Metric, Model
from charmory.model.image_classification import JaticImageClassificationModel
from charmory.tasks.image_classification import ImageClassificationTask
from charmory.track import track_evaluation, track_init_params, track_params
from charmory.utils import create_jatic_dataset_transform

NAME = "jatic-food-category-classification"
DESCRIPTION = "Food category classification from HuggingFace via JATIC-toolbox"


def make_evaluation_from_scratch(epsilon: float) -> Evaluation:
    """construct an evaluation with a variable epsilon."""

    import jatic_toolbox

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
        except Exception as err:
            print(err)
            return False

    print(f"Dataset length prior to filtering: {len(dataset)}")
    dataset._dataset = dataset._dataset.filter(filter)
    print(f"Dataset length after filtering: {len(dataset)}")

    transform = create_jatic_dataset_transform(model.preprocessor)
    dataset.set_transform(transform)

    generator = ArmoryDataLoader(
        dataset=dataset,
        batch_size=16,
    )

    eval_dataset = Dataset(
        name="food-category-classification",
        x_key="image",
        y_key="label",
        test_dataloader=generator,
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
            eps=epsilon,
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
        name=NAME,
        description=DESCRIPTION,
        author="Kaludi",
        dataset=eval_dataset,
        model=eval_model,
        attack=eval_attack,
        metric=eval_metric,
    )

    return evaluation


for epsilon in [x / 1000.0 for x in range(10, 40, 5)]:
    with track_evaluation("msw-food-3", "epsilon 0.010 to 0.040"):
        evaluation = make_evaluation_from_scratch(epsilon=epsilon)
        task = ImageClassificationTask(evaluation, num_classes=12)
        engine = EvaluationEngine(task)
        results = engine.run()
        print(f"Completed evaluation run with {epsilon=}")
        pprint(results)
