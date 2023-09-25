from pprint import pprint

import art.attacks.evasion
from art.estimators.classification import PyTorchClassifier
import jatic_toolbox
import numpy as np
import torch.nn
from transformers.image_utils import infer_channel_dimension_format

from armory.metrics.compute import BasicProfiler
from charmory.data import JaticVisionDatasetGenerator
from charmory.engine import LightningEngine
from charmory.evaluation import Attack, Dataset, Evaluation, Metric, Model, SysConfig
from charmory.tasks.image_classification import ImageClassificationTask
from charmory.track import track_init_params, track_params
from charmory.utils import (
    adapt_jatic_image_classification_model_for_art,
    create_jatic_image_classification_dataset_transform,
)


def main():
    ###
    # Model
    ###
    model = track_params(jatic_toolbox.load_model)(
        provider="huggingface",
        model_name="Kaludi/food-category-classification-v2.0",
        task="image-classification",
    )
    adapt_jatic_image_classification_model_for_art(model)

    classifier = track_init_params(PyTorchClassifier)(
        model,
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

    transform = create_jatic_image_classification_dataset_transform(model.preprocessor)
    dataset.set_transform(transform)

    generator = JaticVisionDatasetGenerator(
        dataset=dataset,
        batch_size=16,
        epochs=1,
    )

    ###
    # Evaluation
    ###
    eval_dataset = Dataset(
        name="food-category-classification",
        test_dataset=generator,
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

    eval_sysconfig = SysConfig(
        gpus=["all"],
        use_gpu=True,
    )

    evaluation = Evaluation(
        name="food-category-classification",
        description="Food category classification from HuggingFace",
        author="Kaludi",
        dataset=eval_dataset,
        model=eval_model,
        attack=eval_attack,
        metric=eval_metric,
        sysconfig=eval_sysconfig,
    )

    ###
    # Engine
    ###

    task = ImageClassificationTask(evaluation, num_classes=12, export_every_n_batches=5)
    engine = LightningEngine(task, limit_test_batches=5)
    results = engine.run()

    pprint(results)

    print("JATIC Experiment Complete!")
    return 0


if __name__ == "__main__":
    main()
