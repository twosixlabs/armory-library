from pprint import pprint

from armory.examples.image_classification.eurosat_precomputed_pgd.evaluation import (
    create_evaluation_task,
)
from armory.examples.utils.args import create_parser
from armory.matrix import matrix
from armory.matrix.range import frange
from charmory.engine import EvaluationEngine
from charmory.track import track_param


@matrix(
    model_name=("convnext", "swin", "vit"),
    attack_eps=frange(0.01, 0.03, 0.005),
    attack_max_iter=range(10, 26, 5),
)
def run_evaluation(num_batches, **kwargs):
    task = create_evaluation_task(
        with_attack=True,
        attack_batch_size=1,
        attack_eps_step=0.007,
        attack_num_random_init=1,
        attack_random_eps=False,
        attack_targeted=False,
        **kwargs,
    )
    track_param("main.num_batches", num_batches)
    engine = EvaluationEngine(task, limit_test_batches=num_batches)
    results = engine.run()
    pprint(results)


if __name__ == "__main__":
    parser = create_parser(
        description="Run EuroSAT image classification evaluation matrix",
        batch_size=4,
        export_every_n_batches=5,
    )
    args = parser.parse_args()

    run_evaluation(
        num_batches=args.num_batches,
        batch_size=args.batch_size,
        export_every_n_batches=args.export_every_n_batches,
    )
