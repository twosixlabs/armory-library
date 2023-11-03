from pprint import pprint

from armory.examples.image_classification.eurosat_precomputed_pgd.evaluation import (
    create_evaluation_task,
    get_cli_args,
)
from charmory.engine import EvaluationEngine
from charmory.track import track_param

if __name__ == "__main__":
    args = get_cli_args(with_attack=False)
    task = create_evaluation_task(with_attack=False, **vars(args))
    track_param("main.num_batches", args.num_batches)
    engine = EvaluationEngine(task, limit_test_batches=args.num_batches)
    results = engine.run()
    pprint(results)
