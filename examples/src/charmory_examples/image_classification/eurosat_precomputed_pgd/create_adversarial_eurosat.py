import albumentations as A
from charmory_examples.image_classification.eurosat_precomputed_pgd.evaluation import (
    create_evaluation,
    get_cli_args,
)

from charmory.engine import AdversarialDatasetEngine

if __name__ == "__main__":
    args = get_cli_args(with_attack=True)
    evaluation = create_evaluation(with_attack=True, **vars(args))

    transform = A.FromFloat(dtype="uint8")

    def adapter(sample):
        sample["image"] = transform(image=sample["image"].transpose(1, 2, 0))["image"]
        sample["labels"] = sample["label"]
        del sample["label"]
        return sample

    engine = AdversarialDatasetEngine(
        evaluation,
        output_dir=f"eurosat_with_{args.model_name}_pgd",
        adapter=adapter,
        features=evaluation.dataset.test_dataloader.dataset._dataset.features,
        num_batches=args.num_batches,
    )
    engine.generate()
