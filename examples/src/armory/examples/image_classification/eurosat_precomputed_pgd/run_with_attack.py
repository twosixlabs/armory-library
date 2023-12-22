from pprint import pprint

from armory.examples.image_classification.eurosat_precomputed_pgd.evaluation import (
    create_evaluation,
    get_cli_args,
)
from charmory.engine import EvaluationEngine
from charmory.track import track_param

if __name__ == "__main__":
    args = get_cli_args(with_attack=True)
    evaluation = create_evaluation(with_attack=True, **vars(args))
    track_param("main.num_batches", args.num_batches)
    engine = EvaluationEngine(
        evaluation,
        export_every_n_batches=args.export_every_n_batches,
        limit_test_batches=args.num_batches,
    )
    results = engine.run()
    pprint(results)
