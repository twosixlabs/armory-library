import albumentations as A

from armory.examples.image_classification.eurosat_precomputed_pgd.evaluation import (
    create_evaluation_task,
    get_cli_args,
)
from charmory.engine import AdversarialDatasetEngine
from charmory.track import track_param

if __name__ == "__main__":
    args = get_cli_args(with_attack=True)
    task = create_evaluation_task(with_attack=True, **vars(args))
    track_param("main.num_batches", args.num_batches)

    transform = A.FromFloat(dtype="uint8")

    def adapter(sample):
        # have to convert _back_ from CHW to HWC, as well as from
        # float [0.0,1.0] to int [0,255]
        sample["image"] = transform(image=sample["image"].transpose(1, 2, 0))["image"]
        # rename from JATIC-toolbox wrapper key back to original key
        sample["labels"] = sample["label"]
        del sample["label"]
        return sample

    engine = AdversarialDatasetEngine(
        task,
        output_dir=f"eurosat_with_{args.model_name}_pgd",
        adapter=adapter,
        features=task.evaluation.dataset.test_dataloader.dataset._dataset.features,
        num_batches=args.num_batches,
    )
    engine.generate()
