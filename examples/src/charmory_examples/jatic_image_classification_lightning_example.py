from pprint import pprint

import art.attacks.evasion
from art.estimators.classification import PyTorchClassifier
import jatic_toolbox
import lightning.pytorch as pl
import numpy as np
import torch.nn
from transformers.image_utils import infer_channel_dimension_format

# from charmory.engine import Engine
from armory.instrument.config import MetricsLogger
from armory.metrics.compute import BasicProfiler
from charmory.data import JaticVisionDatasetGenerator
from charmory.evaluation import (
    Attack,
    Dataset,
    Evaluation,
    Metric,
    Model,
    Scenario,
    SysConfig,
)
from charmory.scenarios.image_classification import (
    ImageClassificationModule,
    ImageClassificationTask,
)
from charmory.utils import (
    adapt_jatic_image_classification_model_for_art,
    create_jatic_image_classification_dataset_transform,
)


def main():
    ###
    # Model
    ###
    model = jatic_toolbox.load_model(
        provider="huggingface",
        model_name="Kaludi/food-category-classification-v2.0",
        task="image-classification",
    )
    adapt_jatic_image_classification_model_for_art(model)

    classifier = PyTorchClassifier(
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
    dataset = jatic_toolbox.load_dataset(
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
        attack=art.attacks.evasion.ProjectedGradientDescent(
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

    eval_scenario = Scenario(
        function=ImageClassificationTask,
        kwargs={},
    )

    eval_metric = Metric(
        profiler=BasicProfiler(),
        logger=MetricsLogger(
            supported_metrics=["accuracy"],
            perturbation=["linf"],
            task=["categorical_accuracy"],
            means=True,
            record_metric_per_sample=False,
        ),
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
        scenario=eval_scenario,
        metric=eval_metric,
        sysconfig=eval_sysconfig,
    )

    ###
    # Engine
    ###
    # engine = Engine(evaluation)
    # results = engine.run()
    # pprint(results)

    ###
    # Lightning
    ###
    module = ImageClassificationModule(evaluation)

    trainer = pl.Trainer(inference_mode=False)
    trainer.test(module)

    pprint(module.results)


if __name__ == "__main__":
    main()
