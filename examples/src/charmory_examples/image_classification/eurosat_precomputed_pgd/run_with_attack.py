from pprint import pprint

from charmory_examples.image_classification.eurosat_precomputed_pgd.evaluation import (
    create_attack_evaluation_task,
    get_cli_args,
)

from charmory.engine import EvaluationEngine

if __name__ == "__main__":
    args = get_cli_args(with_attack=True)
    task = create_attack_evaluation_task(**vars(args))
    engine = EvaluationEngine(task, limit_test_batches=args.num_batches)
    results = engine.run()
    pprint(results)
