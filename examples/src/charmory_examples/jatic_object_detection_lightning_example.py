from PIL import Image
import albumentations as A
import art.attacks.evasion
from art.estimators.object_detection import PyTorchFasterRCNN
import jatic_toolbox
import numpy as np

from armory.art_experimental.attacks.patch import AttackWrapper
from armory.metrics.compute import BasicProfiler
from charmory.data import JaticObjectDetectionDatasetGenerator
from charmory.engine import LightningEngine
from charmory.evaluation import Attack, Dataset, Evaluation, Metric, Model, SysConfig
from charmory.experimental.example_results import print_outputs
from charmory.tasks.object_detection import ObjectDetectionTask
from charmory.track import track_init_params, track_params
from charmory.utils import (
    adapt_jatic_object_detection_model_for_art,
    create_jatic_image_classification_dataset_transform,
)


def main():
    ###
    # Model
    ###
    model = track_params(jatic_toolbox.load_model)(
        provider="torchvision",
        model_name="fasterrcnn_resnet50_fpn",
        task="object-detection",
    )
    adapt_jatic_object_detection_model_for_art(model)

    detector = track_init_params(PyTorchFasterRCNN)(
        model,
        channels_first=True,
        clip_values=(0.0, 1.0),
    )

    ###
    # Dataset
    ###
    dataset = track_params(jatic_toolbox.load_dataset)(
        provider="huggingface",
        dataset_name="detection-datasets/coco",
        task="object-detection",
        split="val",
    )

    # Have to filter out non-RGB images
    def filter(sample):
        shape = np.asarray(sample["image"]).shape
        return len(shape) == 3 and shape[2] == 3

    print(f"Dataset length prior to filtering: {len(dataset)}")
    dataset._dataset = dataset._dataset.filter(filter)
    print(f"Dataset length after filtering: {len(dataset)}")

    model_transform = create_jatic_image_classification_dataset_transform(
        model.preprocessor
    )

    img_transforms = A.Compose(
        [
            A.LongestMaxSize(max_size=400),
            A.PadIfNeeded(
                min_height=400,
                min_width=400,
                border_mode=0,
                value=(0, 0, 0),
            ),
        ],
        bbox_params=A.BboxParams(
            format="pascal_voc",
            label_fields=["labels"],
        ),
    )

    def transform(sample):
        transformed = dict(image=[], objects=[])
        for i in range(len(sample["image"])):
            transformed_img = img_transforms(
                image=np.asarray(sample["image"][i]),
                bboxes=sample["objects"][i]["bbox"],
                labels=sample["objects"][i]["category"],
            )
            transformed["image"].append(Image.fromarray(transformed_img["image"]))
            transformed["objects"].append(
                dict(
                    bbox=transformed_img["bboxes"],
                    category=transformed_img["labels"],
                )
            )
        transformed = model_transform(transformed)
        return transformed

    dataset.set_transform(transform)

    generator = JaticObjectDetectionDatasetGenerator(
        dataset=dataset,
        batch_size=4,
        epochs=1,
    )

    ###
    # Evaluation
    ###
    eval_dataset = Dataset(
        name="coco",
        test_dataset=generator,
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
        name="coco-object-detection",
        description="COCO object detection from HuggingFace & TorchVision",
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
        export_every_n_batches=5,
        class_metrics=False,
    )

    engine = LightningEngine(task, limit_test_batches=10)
    results = engine.run()
    print_outputs(dataset, model, results)

    print("JATIC Experiment Complete!")
    return 0


if __name__ == "__main__":
    main()
