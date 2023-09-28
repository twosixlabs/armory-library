from pprint import pprint

from charmory_examples.image_classification.resnet18_precomputed_pgd.evaluation import (
    create_pregenerated_evaluation_task,
    get_cli_args,
)

from charmory.engine import LightningEngine

if __name__ == "__main__":
    args = get_cli_args(with_attack=False)
    task = create_pregenerated_evaluation_task(batch_size=args.batch_size)
    engine = LightningEngine(task, limit_test_batches=args.num_batches)
    results = engine.run()
    pprint(results)
