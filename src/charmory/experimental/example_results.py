from pprint import pprint


def print_outputs(dataset, model, results):
    print("=" * 64)
    pprint(dataset.train_dataloader)
    pprint(dataset.test_dataloader)
    print("-" * 64)
    pprint(model)
    print("=" * 64)
    pprint(results)
