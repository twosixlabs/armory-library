from pprint import pprint

from art.attacks.evasion import RobustDPatch
from art.estimators.object_detection import PyTorchYolo
import datasets
import torch
import torchmetrics.detection
import yolov5
from yolov5.utils.loss import ComputeLoss

from armory.data import ArmoryDataLoader
from armory.engine import EvaluationEngine
import armory.evaluation as ev
from armory.examples.utils.args import create_parser
from armory.experimental.patch import AttackWrapper
from armory.experimental.transforms import BboxFormat, create_object_detection_transform
from armory.metrics.compute import BasicProfiler
from armory.perturbation import ArtEvasionAttack
from armory.tasks.object_detection import ObjectDetectionTask
from armory.track import track_init_params, track_params


def get_cli_args():
    parser = create_parser(
        description="Run YOLOv5m object detection for license plates",
        batch_size=4,
        export_every_n_batches=5,
        num_batches=20,
    )
    return parser.parse_args()


class Yolo(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.compute_loss = ComputeLoss(self.model.model.model)

    def forward(self, x, targets=None):
        if self.training:
            outputs = self.model.model.model(x)
            loss, _ = self.compute_loss(outputs, targets)
            return dict(loss_total=loss)
        return self.model(x)


@track_params
def load_model(
    model_path="keremberke/yolov5m-license-plate",
    conf=0.25,  # NMS confidence threshold
    iou=0.45,  # NMS IoU threshold
    agnostic=False,  # NMS class-agnostic
    multi_label=False,  # NMS multiple labels per box
    max_det=1000,  # maximum number of detections per image
):
    model = yolov5.load(model_path)
    # set model parameters
    model.conf = conf
    model.iou = iou
    model.agnostic = agnostic
    model.multi_label = multi_label
    model.max_det = max_det
    return Yolo(model)


@track_params
def main(batch_size, export_every_n_batches, num_batches):
    ###
    # Model
    ###
    model = load_model()
    detector = track_init_params(PyTorchYolo)(
        model,
        input_shape=(3, 512, 512),
        clip_values=(0, 1),
        attack_losses=("loss_total",),
    )

    ###
    # Dataset
    ###
    dataset = track_params(datasets.load_dataset)(
        "keremberke/license-plate-object-detection",
        name="full",
        split="test",
    )
    dataset.set_transform(
        create_object_detection_transform(
            max_size=512,
            float_max_value=255,
            format=BboxFormat.XYWH,
            label_fields=["category", "id"],
            rename_object_fields={"bbox": "boxes", "category": "labels"},
        )
    )
    dataloader = ArmoryDataLoader(dataset, batch_size=batch_size, shuffle=True)

    ###
    # Attack
    ###
    patch = track_init_params(RobustDPatch)(
        detector,
        patch_shape=(3, 50, 50),
        patch_location=(231, 231),  # middle of 512x512
        batch_size=batch_size,
        sample_size=10,
        learning_rate=0.01,
        max_iter=20,
        targeted=False,
        verbose=False,
    )
    attack = AttackWrapper(patch)

    ###
    # Evaluation
    ###
    evaluation = ev.Evaluation(
        name="yolo-license-plate",
        description="YOLOv5m object detection for license plates",
        author="TwoSix",
        dataset=ev.Dataset(
            name="Vehicle Registration Plates Dataset",
            x_key="image",
            y_key="objects",
            test_dataloader=dataloader,
        ),
        model=ev.Model(
            name="YOLOv5m",
            model=detector,
        ),
        perturbations=dict(
            benign=[],
            attack=[
                ArtEvasionAttack(
                    name="RobustDPatch",
                    attack=attack,
                    use_label_for_untargeted=False,
                )
            ],
        ),
        metric=ev.Metric(
            profiler=BasicProfiler(),
            prediction={
                "map": torchmetrics.detection.MeanAveragePrecision(class_metrics=False),
            },
        ),
    )

    ###
    # Engine
    ###
    task = ObjectDetectionTask(
        evaluation,
        export_every_n_batches=export_every_n_batches,
        iou_threshold=0.45,
        score_threshold=0.25,
    )
    engine = EvaluationEngine(task, limit_test_batches=num_batches)

    ###
    # Execute
    ###
    pprint(engine.run())


if __name__ == "__main__":
    main(**vars(get_cli_args()))
