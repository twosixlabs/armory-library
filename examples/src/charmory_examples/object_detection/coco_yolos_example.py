import argparse
from pprint import pprint

from PIL import Image
import albumentations as A
import art.attacks.evasion
from art.estimators.object_detection import PyTorchObjectDetector
import jatic_toolbox
import numpy as np
import torch
from torchvision.ops import box_convert
from transformers import AutoImageProcessor

from armory.art_experimental.attacks.patch import AttackWrapper
from armory.metrics.compute import BasicProfiler
from charmory.data import ArmoryDataLoader, JaticObjectDetectionDataset
from charmory.engine import LightningEngine
from charmory.evaluation import Attack, Dataset, Evaluation, Metric, Model, SysConfig
from charmory.model import ArmoryModel
from charmory.tasks.object_detection import ObjectDetectionTask
from charmory.track import track_init_params, track_params
from charmory.utils import create_jatic_dataset_transform


def get_cli_args():
    parser = argparse.ArgumentParser(
        description="Run COCO object detection example using models and datasets from the JATIC toolbox",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--batch-size",
        default=4,
        type=int,
    )
    parser.add_argument(
        "--export-every-n-batches",
        default=5,
        type=int,
    )
    parser.add_argument(
        "--num-batches",
        default=20,
        type=int,
    )
    return parser.parse_args()


def main(args):
    ###
    # Model
    ###
    model = track_params(jatic_toolbox.load_model)(
        provider="huggingface",
        model_name="hustvl/yolos-tiny",
        task="object-detection",
    )

    # Bypass JATIC model wrapper to allow targeted adversarial attacks
    def hack(*args, **kwargs):
        return model.model(*args, **kwargs)

    model.forward = hack

    image_processor = AutoImageProcessor.from_pretrained("hustvl/yolos-tiny")

    def model_postadapter(output):
        result = image_processor.post_process_object_detection(
            output, target_sizes=[(512, 512) for _ in range(len(output.pred_boxes))]
        )
        return result

    detector = track_init_params(PyTorchObjectDetector)(
        ArmoryModel(model, postadapter=model_postadapter),
        channels_first=True,
        input_shape=(3, 400, 400),
        clip_values=(0.0, 1.0),
    )

    ###
    # Dataset
    ###
    dataset = track_params(jatic_toolbox.load_dataset)(
        provider="huggingface",
        dataset_name="rafaelpadilla/coco2017",
        task="object-detection",
        split="val",
        category_key="label",
    )

    # Have to filter out non-RGB images
    def filter(sample):
        shape = np.asarray(sample["image"]).shape
        return len(shape) == 3 and shape[2] == 3

    print(f"Dataset length prior to filtering: {len(dataset)}")
    dataset._dataset = dataset._dataset.filter(filter)
    print(f"Dataset length after filtering: {len(dataset)}")

    model_transform = create_jatic_dataset_transform(model.preprocessor)

    img_transforms = A.Compose(
        [
            A.LongestMaxSize(max_size=512),
            A.PadIfNeeded(
                min_height=512,
                min_width=512,
                border_mode=0,
                value=(0, 0, 0),
            ),
        ],
        bbox_params=A.BboxParams(
            format="coco",
            label_fields=["labels"],
        ),
    )

    def transform(sample):
        transformed = dict(image=[], objects=[])
        for i in range(len(sample["image"])):
            transformed_img = img_transforms(
                image=np.asarray(sample["image"][i]),
                bboxes=sample["objects"][i]["bbox"],
                labels=sample["objects"][i]["label"],
            )
            transformed["image"].append(Image.fromarray(transformed_img["image"]))
            transformed["objects"].append(
                dict(
                    bbox=transformed_img["bboxes"],
                    label=transformed_img["labels"],
                )
            )
        transformed = model_transform(transformed)
        for obj in transformed["objects"]:
            if len(obj.get("bbox", [])) > 1:
                obj["bbox"] = box_convert(
                    torch.tensor(obj["bbox"]), "xywh", "xyxy"
                ).numpy()
        return transformed

    dataset.set_transform(transform)

    dataloader = ArmoryDataLoader(
        JaticObjectDetectionDataset(dataset), batch_size=args.batch_size
    )

    ###
    # Evaluation
    ###
    eval_dataset = Dataset(
        name="coco",
        test_dataset=dataloader,
    )

    eval_model = Model(
        name="faster-rcnn-resnet50",
        model=detector,
    )

    patch = track_init_params(art.attacks.evasion.RobustDPatch)(
        detector,
        patch_shape=(3, 32, 32),
        batch_size=1,
        max_iter=20,
        targeted=False,
        verbose=False,
    )

    eval_attack = Attack(
        name="RobustDPatch",
        attack=AttackWrapper(patch),
        use_label_for_untargeted=False,
    )

    eval_metric = Metric(
        profiler=BasicProfiler(),
    )

    eval_sysconfig = SysConfig(
        gpus=["all"],
        use_gpu=True,
    )

    evaluation = Evaluation(
        name="coco-yolo-object-detection",
        description="COCO object detection using YOLO from HuggingFace",
        author="",
        dataset=eval_dataset,
        model=eval_model,
        attack=eval_attack,
        scenario=None,
        metric=eval_metric,
        sysconfig=eval_sysconfig,
    )

    ###
    # Engine
    ###

    task = ObjectDetectionTask(
        evaluation,
        skip_attack=True,
        export_every_n_batches=args.export_every_n_batches,
        class_metrics=False,
    )
    engine = LightningEngine(task, limit_test_batches=args.num_batches)
    results = engine.run()

    pprint(results)


if __name__ == "__main__":
    main(get_cli_args())
