from pprint import pprint

from charmory.engine import LightningEngine


def execute_lightning(task, limit_test_batches):
    # Runs basic Lightning Engine example

    engine = LightningEngine(task, limit_test_batches=limit_test_batches)
    results = engine.run()
    return results


def print_outputs(dataset, model, results):
    print("=" * 64)
    pprint(dataset.train_dataset)
    pprint(dataset.test_dataset)
    print("-" * 64)
    pprint(model)
    print("=" * 64)
    pprint(results)
