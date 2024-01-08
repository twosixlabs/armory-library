from pprint import pprint

from PIL import Image
import art.attacks.evasion
from art.estimators.object_detection import PyTorchFasterRCNN
import jatic_toolbox
import numpy as np
import torchmetrics.detection

from armory.data import ArmoryDataLoader
from armory.engine import EvaluationEngine
from armory.evaluation import Dataset, Evaluation, Metric, Model
from armory.examples.utils.args import create_parser
from armory.experimental.patch import AttackWrapper
from armory.experimental.transforms import BboxFormat, create_object_detection_transform
from armory.metrics.compute import BasicProfiler
from armory.model.object_detection import JaticObjectDetectionModel
from armory.perturbation import ArtEvasionAttack
from armory.tasks.object_detection import ObjectDetectionTask
from armory.track import track_init_params, track_params
from armory.utils import create_jatic_dataset_transform


def get_cli_args():
    parser = create_parser(
        description="Run COCO object detection example using models and datasets from the JATIC toolbox",
        batch_size=4,
        export_every_n_batches=5,
        num_batches=20,
    )
    return parser.parse_args()


def main(args):
    ###
    # Model
    ###
    model = track_params(jatic_toolbox.load_model)(
        provider="torchvision",
        model_name="fasterrcnn_resnet50_fpn",
        task="object-detection",
    )

    # Bypass JATIC model wrapper to allow targeted adversarial attacks
    def hack(*args, **kwargs):
        return model._model(*args, **kwargs)

    model.forward = hack

    detector = track_init_params(PyTorchFasterRCNN)(
        JaticObjectDetectionModel(model),
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

    model_transform = create_jatic_dataset_transform(model.preprocessor)

    dataset.set_transform(
        create_object_detection_transform(
            image_from_np=Image.fromarray,
            max_size=400,
            format=BboxFormat.XYXY,
            label_fields=["category"],
            postprocessor=model_transform,
        )
    )

    dataloader = ArmoryDataLoader(dataset, batch_size=args.batch_size)

    ###
    # Evaluation
    ###
    eval_dataset = Dataset(
        name="coco",
        x_key="image",
        y_key="objects",
        test_dataloader=dataloader,
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

    eval_attack = ArtEvasionAttack(
        name="RobustDPatch",
        attack=AttackWrapper(patch),
        use_label_for_untargeted=False,
    )

    eval_metric = Metric(
        profiler=BasicProfiler(),
        prediction={
            "map": torchmetrics.detection.MeanAveragePrecision(class_metrics=False),
        },
    )

    evaluation = Evaluation(
        name="coco-object-detection",
        description="COCO object detection from HuggingFace & TorchVision",
        author="",
        dataset=eval_dataset,
        model=eval_model,
        perturbations={
            "benign": [],
            "attack": [eval_attack],
        },
        metric=eval_metric,
    )

    ###
    # Engine
    ###

    task = ObjectDetectionTask(
        evaluation,
        export_every_n_batches=args.export_every_n_batches,
        class_metrics=False,
    )
    engine = EvaluationEngine(task, limit_test_batches=args.num_batches)
    results = engine.run()
    pprint(results)

    print("JATIC Experiment Complete!")
    return 0


if __name__ == "__main__":
    main(get_cli_args())
