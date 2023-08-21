from pprint import pprint

import art.attacks.evasion
from art.estimators.object_detection import PyTorchFasterRCNN
import jatic_toolbox

from armory.metrics.compute import BasicProfiler
from charmory.data import JaticVisionDatasetGenerator
from charmory.engine import LightningEngine
from charmory.evaluation import Attack, Dataset, Evaluation, Metric, Model, SysConfig
from charmory.tasks.object_detection import ObjectDetectionTask
from charmory.track import track_evaluation, track_init_params, track_params
from charmory.utils import (
    adapt_jatic_object_detection_model_for_art,
    create_jatic_image_classification_dataset_transform,
)

NAME = "coco-object-detection"
DESCRIPTION = "COCO object detection from HuggingFace & TorchVision"


def main():
    with track_evaluation(NAME, description=DESCRIPTION):
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
            split="train",
        )

        transform = create_jatic_image_classification_dataset_transform(
            model.preprocessor
        )
        dataset.set_transform(transform)

        generator = JaticVisionDatasetGenerator(
            dataset=dataset,
            batch_size=1,  # have to use a batch size of 1 because of inhomogenous image sizes
            epochs=1,
            label_key="objects",
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

        eval_attack = Attack(
            name="PGD",
            attack=track_init_params(art.attacks.evasion.RobustDPatch)(
                detector,
                patch_shape=(3, 40, 40),
                batch_size=1,
                max_iter=20,
                targeted=False,
                verbose=False,
            ),
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
            name=NAME,
            description=DESCRIPTION,
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

        task = ObjectDetectionTask(evaluation, skip_attack=True)
        engine = LightningEngine(task, limit_test_batches=80)
        results = engine.run()

    pprint(results)


if __name__ == "__main__":
    main()
