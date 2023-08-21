"""
Example programmatic entrypoint for scenario execution
"""
import json
from pprint import pprint
import sys

import art.attacks.evasion
from jatic_toolbox import __version__ as jatic_version
from jatic_toolbox import load_dataset as load_jatic_dataset
import torch


import armory.version

from charmory.data import JaticVisionDatasetGenerator
from charmory.engine import Engine
from charmory.evaluation import (
    Attack,
    Dataset,
    Evaluation,
    Metric,
    Model,
    Scenario,
    SysConfig,
)
import charmory.scenarios.image_classification
from charmory.utils import PILtoNumpy_HuggingFace_Variable_Length

BATCH_SIZE = 64
TRAINING_EPOCHS = 1
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


from typing import Optional
from art.estimators.classification import PyTorchClassifier
import torchvision.models as models
import torch.nn as nn

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Using the Resnet34 model and adding in the amount of classes for the pokemon dataset as a final layer
def make_image_net_model(**kwargs) -> models.resnet34(weights=True):
    model = models.resnet34(weights=True)
    num_ftrs = model.fc.in_features

    model.fc = nn.Linear(num_ftrs, 1000) #No. of classes = 1000 
    return model


def get_art_model(
    model_kwargs: dict, wrapper_kwargs: dict, weights_path: Optional[str] = None
) -> PyTorchClassifier:
    model = make_image_net_model(**model_kwargs)
    model.to(DEVICE)

    if weights_path:
        checkpoint = torch.load(weights_path, map_location=DEVICE)
        model.load_state_dict(checkpoint)

    wrapped_model = PyTorchClassifier(
        model,
        loss=nn.CrossEntropyLoss(),
        optimizer=torch.optim.Adam(model.parameters(), lr=0.01),
        input_shape=(3,200, 200),
        channels_first=False,
        nb_classes=1000,
        clip_values=(0.0, 1.0),
        **wrapper_kwargs,
    )
    return wrapped_model






# Loads Imagenet 1k Classification HuggingFace Example


def load_huggingface_dataset():

    transform = PILtoNumpy_HuggingFace_Variable_Length()
    train_dataset = load_jatic_dataset(
        provider="huggingface",
        dataset_name="imagenet-1k",
        task="image-classification",
        split="train",
        use_auth_token=True
    )

    train_dataset.set_transform(transform)

    train_dataset_generator = JaticVisionDatasetGenerator(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        epochs=TRAINING_EPOCHS,
        shuffle=True,
    )
    test_dataset = load_jatic_dataset(
        provider="huggingface",
        dataset_name="imagenet-1k",
        task="image-classification",
        split="test",
        use_auth_token=True
    )
    test_dataset.set_transform(transform)
    test_dataset_generator = JaticVisionDatasetGenerator(
        dataset=test_dataset,
        batch_size=BATCH_SIZE,
        epochs=1,
    )
    return train_dataset_generator, test_dataset_generator

def main(argv: list = sys.argv[1:]):
    if len(argv) > 0:
        if "--version" in argv:
            print(f"armory: {armory.version.__version__}")
            print(f"JATIC-toolbox: {jatic_version}")
            sys.exit(0)

    print("Armory: Example Programmatic Entrypoint for Scenario Execution")

    image_net_model = get_art_model(
        model_kwargs={},
        wrapper_kwargs={},
        weights_path=None,
    )

    model = Model(
        name="ImageNet1k",
        model=image_net_model,
    )

    train_dataset, test_dataset = load_huggingface_dataset()
    dataset = Dataset(
        name="POKEMON", train_dataset=train_dataset, test_dataset=test_dataset
    )
    
    train_dataset, test_dataset = load_huggingface_dataset()
    dataset = Dataset(
        name="imagenet", train_dataset=train_dataset, test_dataset=test_dataset
    )



    ###
    # The rest of this file was directly copied from the existing cifar example
    ###

    attack = Attack(
        function=art.attacks.evasion.ProjectedGradientDescent,
        kwargs={
            "batch_size": 32,
            "eps": 0.031,
            "eps_step": 0.007,
            "max_iter": 20,
            "num_random_init": 1,
            "random_eps": False,
            "targeted": False,
            "verbose": False,
        },
        knowledge="white",
        use_label=True,
        type=None,
    )

    scenario = Scenario(
        function=charmory.scenarios.image_classification.ImageClassificationTask,
        kwargs={},
    )

    metric = Metric(
        profiler_type="basic",
        supported_metrics=["accuracy"],
        perturbation=["linf"],
        task=["categorical_accuracy"],
        means=True,
        record_metric_per_sample=False,
    )

    sysconfig = SysConfig(gpus=["all"], use_gpu=True)

    baseline = Evaluation(
        name="pokemon",
        description="Baseline Pokemon image classification",
        author="msw@example.com",
        dataset=dataset,
        model=model,
        attack=attack,
        scenario=scenario,
        defense=None,
        metric=metric,
        sysconfig=sysconfig,
    )

    print(f"Starting Demo for {baseline.name}")

    image_net_engine = Engine(baseline)
    image_net_engine.train(nb_epochs=TRAINING_EPOCHS)
    results = image_net_engine.run()

    print("=" * 64)
    pprint(baseline)
    print("-" * 64)
    print(
        json.dumps(
            results, default=lambda o: "<not serializable>", indent=4, sort_keys=True
        )
    )

    print("=" * 64)
    print(dataset.train_dataset)
    print(dataset.test_dataset)
    print("-" * 64)
    print(model)

    print("=" * 64)
    print("Pokemon Experiment Complete!")
    return 0



if __name__ == "__main__":
    sys.exit(main())
