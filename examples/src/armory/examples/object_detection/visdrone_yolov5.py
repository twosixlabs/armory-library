"""
Example Armory evaluation of VisDrone object detection with YOLOv5 against
a custom Robust DPatch attack
"""

from pprint import pprint
from typing import Optional

import torchmetrics.detection
import yolov5

import armory.data
import armory.engine
import armory.evaluation
import armory.examples.object_detection.datasets.visdrone
import armory.export.criteria
import armory.export.object_detection
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
    return parser.parse_args()


def load_dataset(
    evaluation: armory.evaluation.Evaluation,
    batch_size: int,
    shuffle: bool,
    seed: Optional[int] = None,
):
    """Load VisDrone dataset"""
    with evaluation.autotrack():
        hf_dataset = armory.examples.object_detection.datasets.visdrone.load_dataset()
        dataloader = (
            armory.examples.object_detection.datasets.visdrone.create_dataloader(
                hf_dataset["val"],
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


@armory.track.track_params
def main(batch_size, export_every_n_batches, num_batches, seed, shuffle):
    """Perform the evaluation"""
    evaluation = armory.evaluation.Evaluation(
        name="visdrone-object-detection-yolov5",
        description="VisDrone object detection using YOLOv5",
        author="TwoSix",
    )

    dataset = load_dataset(evaluation, batch_size, shuffle, seed)
    model = load_model(evaluation)

    evaluation.use_dataset(dataset)
    evaluation.use_model(model)
    evaluation.use_metrics(create_metrics())
    evaluation.use_exporters(create_exporters(model, export_every_n_batches))

    with evaluation.add_chain("benign"):
        pass

    eval_engine = armory.engine.EvaluationEngine(
        evaluation,
        profiler=armory.metrics.compute.BasicProfiler(),
        limit_test_batches=num_batches,
    )
    eval_results = eval_engine.run()

    pprint(eval_results)


if __name__ == "__main__":
    main(**vars(parse_cli_args()))
