from pprint import pprint

from armory.engine import EvaluationEngine
from armory.examples.object_detection.coco_precomputed_patch.evaluation import (
    create_evaluation_task,
    get_cli_args,
)
from armory.track import track_param

if __name__ == "__main__":
    args = get_cli_args(with_attack=True)
    task = create_evaluation_task(with_attack=True, **vars(args))
    track_param("main.num_batches", args.num_batches)
    engine = EvaluationEngine(task, limit_test_batches=args.num_batches)
    results = engine.run()
    pprint(results)
