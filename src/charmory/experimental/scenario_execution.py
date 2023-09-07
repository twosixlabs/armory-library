import json
from pprint import pprint

from charmory.engine import Engine


def execute_scenario(baseline, TRAINING_EPOCHS):
    # Runs basic scenario for armory that most example files run

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
