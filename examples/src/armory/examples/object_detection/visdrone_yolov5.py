"""
Example Armory evaluation of VisDrone object detection with YOLOv5 against
a custom Robust DPatch attack
"""

from functools import partial
from pprint import pprint
import random
from typing import Callable, Literal, Optional, Sequence, Tuple, Union

import PIL.Image
import kornia
import torch
import torch.optim
import torchmetrics.detection
import yolov5

import armory.data
import armory.engine
import armory.evaluation
import armory.examples.object_detection.datasets.visdrone
import armory.export.criteria
import armory.export.object_detection
import armory.export.sink
import armory.metric
import armory.metrics.compute
import armory.metrics.detection
import armory.metrics.tide
import armory.model.object_detection
import armory.perturbation
import armory.track


def parse_cli_args():
    """Parse command-line arguments"""
    from armory.examples.utils.args import create_parser

    parser = create_parser(
        description="Perform VisDrone object detection",
        batch_size=4,
        export_every_n_batches=5,
        num_batches=20,
    )
    parser.add_argument(
        "--patch-batch-size",
        default=2,
        help="Batch size used to generate the patch",
        type=int,
    )
    parser.add_argument(
        "--patch-num-batches",
        default=10,
        help="Number of batches used to generate the patch",
        type=int,
    )
    parser.add_argument(
        "--patch-num-epochs",
        default=20,
        help="Number of epochs used to generate the patch",
        type=int,
    )
    return parser.parse_args()


def load_dataset(
    evaluation: armory.evaluation.Evaluation,
    batch_size: int,
    shuffle: bool,
    seed: Optional[int] = None,
    split: Union[Literal["validation"], Literal["train"]] = "validation",
):
    """Load VisDrone dataset"""
    with evaluation.autotrack():
        hf_dataset = armory.examples.object_detection.datasets.visdrone.load_dataset()
        dataloader = (
            armory.examples.object_detection.datasets.visdrone.create_dataloader(
                hf_dataset[split],
                max_size=640,
                batch_size=batch_size,
                shuffle=shuffle,
                seed=seed,
            )
        )
        dataset = armory.evaluation.Dataset(
            name="VisDrone2019",
            dataloader=dataloader,
        )
        return dataset


def load_model(evaluation: armory.evaluation.Evaluation):
    """Load YOLOv5 model from HuggingFace"""
    with evaluation.autotrack() as track_call:
        hf_model = track_call(yolov5.load, model_path="smidm/yolov5-visdrone")

        armory_model = armory.model.object_detection.YoloV5ObjectDetector(
            name="YOLOv5",
            model=hf_model,
        )

        return armory_model


def create_metrics():
    return {
        "map": armory.metric.PredictionMetric(
            torchmetrics.detection.MeanAveragePrecision(class_metrics=False),
            armory.data.TorchBoundingBoxSpec(format=armory.data.BBoxFormat.XYXY),
        ),
        "tide": armory.metrics.tide.TIDE.create(),
        "detection": armory.metrics.detection.ObjectDetectionRates.create(
            record_as_metrics=[
                "true_positive_rate_mean",
                "misclassification_rate_mean",
                "disappearance_rate_mean",
                "hallucinations_mean",
            ],
        ),
    }


def create_exporters(model, export_every_n_batches):
    """Create sample exporters"""
    return [
        armory.export.object_detection.ObjectDetectionExporter(
            criterion=armory.export.criteria.every_n_batches(export_every_n_batches)
        ),
    ]


class RobustDPatch(armory.evaluation.AttackProtocol):

    name = "RobustDPatch"

    def __init__(
        self,
        inputs_spec: armory.data.ImageSpec,
        optimizer: Optional[
            Callable[[Sequence[torch.Tensor]], torch.optim.Optimizer]
        ] = None,
        patch: Optional[torch.Tensor] = None,
        patch_shape: Optional[Tuple[int, int, int]] = None,
        patch_max: int = 255,
        patch_location: Optional[Tuple[int, int]] = None,
    ):
        if not patch and not patch_shape:
            raise ValueError("Either patch or patch_shape must be provided")
        elif not patch and patch_shape:
            self.patch = torch.rand(patch_shape) * patch_max
        elif patch:
            self.patch = patch
        self.patch.requires_grad = True

        self.inputs_spec = inputs_spec
        self.optimizer = optimizer if optimizer is not None else torch.optim.SGD
        self.patch_location = patch_location

    def optimizers(self):
        return self.optimizer([self.patch])

    def apply(self, batch: armory.data.Batch):
        inputs = batch.inputs.get(self.inputs_spec)
        assert isinstance(inputs, torch.Tensor)
        _, _, img_height, img_width = inputs.shape
        _, patch_height, patch_width = self.patch.shape

        if self.patch_location:
            x1, y1 = self.patch_location
        else:
            x1 = random.randint(0, img_width - patch_width)
            y1 = random.randint(0, img_height - patch_height)
        x2 = x1 + patch_width
        y2 = y1 + patch_height

        inputs_with_patch = inputs.clone()
        inputs_with_patch[:, :, x1:x2, y1:y2] = self.patch

        batch.inputs.set(inputs_with_patch, self.inputs_spec)

    def export(self, sink: armory.export.sink.Sink, epoch: int):
        if epoch % 5 == 0:
            patch_np = self.patch.detach().cpu().numpy().transpose(1, 2, 0) * 255
            patch = PIL.Image.fromarray(patch_np.astype("uint8"))
            sink.log_image(patch, f"patch_epoch_{epoch}.png")


def generate_patch(dataset, model, num_batches=10, num_epochs=20) -> torch.Tensor:

    attack = RobustDPatch(
        inputs_spec=model.inputs_spec,
        optimizer=partial(torch.optim.SGD, lr=0.1, momentum=0.9),
        patch_shape=(3, 50, 50),
        patch_location=(295, 295),  # middle of 640x640
    )

    augmentations = kornia.augmentation.container.ImageSequential(
        kornia.augmentation.RandomHorizontalFlip(p=0.5),
        kornia.augmentation.RandomBrightness(brightness=(0.75, 1.25), p=0.5),
        kornia.augmentation.RandomRotation(degrees=15, p=0.5),
        random_apply=True,
    )
    transform = armory.perturbation.CallablePerturbation(
        name="kornia",
        perturbation=augmentations,
        inputs_spec=armory.data.TorchSpec(),
    )

    optimization = armory.evaluation.Optimization(
        name="visdrone-yolov5-robustdpatch-generation",
        description="Optimization of a RobustDPatch attack against YOLOv5 using VisDrone2019",
        author="TwoSix",
        attack=attack,
        dataset=dataset,
        model=model,
        transforms=[transform],
    )

    engine = armory.engine.OptimizationEngine(
        optimization, limit_train_batches=num_batches, max_epochs=num_epochs
    )

    orig_loss = model.compute_loss

    def opt_loss(*args, **kwargs):
        # The Armory optimization engine works by optimizing the loss--that is,
        # the loss is minimized. Therefore in order to maximize the loss, we
        # need to negate it so that optimization results in a more effective
        # attack.
        return -orig_loss(*args, **kwargs)[0], None

    model.compute_loss = opt_loss
    engine.run()
    model.compute_loss = orig_loss

    return attack.patch


@armory.track.track_params
def main(
    batch_size,
    export_every_n_batches,
    num_batches,
    seed,
    shuffle,
    patch_batch_size,
    patch_num_batches,
    patch_num_epochs,
):
    """Perform the evaluation"""
    evaluation = armory.evaluation.Evaluation(
        name="visdrone-object-detection-yolov5",
        description="VisDrone object detection using YOLOv5",
        author="TwoSix",
    )

    dataset = load_dataset(evaluation, batch_size, shuffle, seed)
    model = load_model(evaluation)

    # Generate patch
    train_dataset = load_dataset(
        evaluation, patch_batch_size, shuffle, seed, split="train"
    )
    patch = generate_patch(train_dataset, model, patch_num_batches, patch_num_epochs)

    evaluation.use_dataset(dataset)
    evaluation.use_model(model)
    evaluation.use_metrics(create_metrics())
    evaluation.use_exporters(create_exporters(model, export_every_n_batches))

    with evaluation.add_chain("benign"):
        pass

    with evaluation.add_chain("patch") as chain:
        # x_1, y_1 = 295, 295  # middle of 640x640
        x_1, y_1 = 50, 50
        x_2 = x_1 + patch.shape[1]
        y_2 = y_1 + patch.shape[2]

        def apply_patch(inputs: torch.Tensor) -> torch.Tensor:
            with_patch = inputs.clone()
            with_patch[:, :, x_1:x_2, y_1:y_2] = patch
            return with_patch

        attack = armory.perturbation.CallablePerturbation(
            name="RobustDPatch",
            perturbation=apply_patch,
            inputs_spec=model.inputs_spec,
        )
        chain.add_perturbation(attack)

    eval_engine = armory.engine.EvaluationEngine(
        evaluation,
        profiler=armory.metrics.compute.BasicProfiler(),
        limit_test_batches=num_batches,
    )
    eval_results = eval_engine.run()

    pprint(eval_results)


if __name__ == "__main__":
    main(**vars(parse_cli_args()))
