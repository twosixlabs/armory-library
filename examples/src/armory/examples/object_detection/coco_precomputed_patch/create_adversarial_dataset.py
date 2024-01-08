import albumentations as A
import torch
from torchvision.ops import box_convert

from armory.engine import AdversarialDatasetEngine
from armory.examples.object_detection.coco_precomputed_patch.evaluation import (
    create_evaluation_task,
    get_cli_args,
)
from armory.track import track_param

if __name__ == "__main__":
    args = get_cli_args(with_attack=True)
    task = create_evaluation_task(with_attack=True, **vars(args))
    track_param("main.num_batches", args.num_batches)

    transform = A.FromFloat(dtype="uint8")

    def adapter(sample):
        # have to convert _back_ from CHW to HWC, as well as from
        # float [0.0,1.0] to int [0,255]
        sample["image"] = transform(image=sample["image"].transpose(1, 2, 0))["image"]
        # convert `objects` from object of lists back to a list of objects
        objects = []
        for i in range(len(sample["objects"]["id"])):
            objects.append(
                dict(
                    area=sample["objects"]["area"][i],
                    bbox=box_convert(
                        torch.tensor(sample["objects"]["boxes"][i]), "xyxy", "xywh"
                    ),
                    id=sample["objects"]["id"][i],
                    iscrowd=sample["objects"]["iscrowd"][i],
                    label=sample["objects"]["labels"][i],
                )
            )
        sample["objects"] = objects

        return sample

    engine = AdversarialDatasetEngine(
        task,
        output_dir="coco_with_robustdpatch",
        adapter=adapter,
        features=task.evaluation.dataset.test_dataloader.dataset._dataset.features,
        num_batches=args.num_batches,
    )
    engine.generate()
