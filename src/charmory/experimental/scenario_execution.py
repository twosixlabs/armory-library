import json
from pprint import pprint

from jatic_toolbox import __version__ as jatic_version
from jatic_toolbox import load_dataset as load_jatic_dataset


from charmory.data import JaticVisionDatasetGenerator
from charmory.engine import Engine
from charmory.track import track_params


def load_huggingface_dataset(transform):
    print("Loading HuggingFace dataset from jatic_toolbox")

    train_dataset = track_params(load_jatic_dataset)(
        provider="huggingface",
        dataset_name="cifar10",
        task="image-classification",
        split="train",
    )
    train_dataset.set_transform(transform)
    train_dataset_generator = JaticVisionDatasetGenerator(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        epochs=TRAINING_EPOCHS,
        shuffle=True,
        size=512,  # Use a subset just for demo purposes
    )

    test_dataset = track_params(load_jatic_dataset)(
        provider="huggingface",
        dataset_name="cifar10",
        task="image-classification",
        split="test",
    )
    test_dataset.set_transform(transform)
    test_dataset_generator = JaticVisionDatasetGenerator(
        dataset=test_dataset,
        batch_size=BATCH_SIZE,
        epochs=1,
        shuffle=False,
        size=512,  # Use a subset just for demo purposes
    )

    return train_dataset_generator, test_dataset_generator


def execute_scenario(baseline, TRAINING_EPOCHS):
    #Runs basic scenario for armory that most example files run

    print(f"Starting Demo for {baseline.name}")

    pokemon_engine = Engine(baseline)
    pokemon_engine.train(nb_epochs=TRAINING_EPOCHS)
    results = pokemon_engine.run()

    print("=" * 64)
    pprint(baseline)
    print("-" * 64)
    print(
        json.dumps(
            results, default=lambda o: "<not serializable>", indent=4, sort_keys=True
        )
    )

