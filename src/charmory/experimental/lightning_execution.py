from charmory.engine import LightningEngine


def execute_lightning(task,limit_test_batches):
    # Runs basic Lightning Engine example 

    engine = LightningEngine(task, limit_test_batches)
    results = engine.run()
    return results

def print_outputs(dataset, model, results):
    print("=" * 64)
    print(dataset.train_dataset)
    print(dataset.test_dataset)
    print("-" * 64)
    print(model)
    print("=" * 64)
    print(results)