"""
Armory evaluation of West Point object detection with YOLOv4 against an
adversarial patch attack
"""
from pprint import pprint

import art.attacks.evasion
import art.estimators.object_detection

import torch
import torchmetrics.detection
import torchvision.transforms.v2
import transformers

import armory.data  # type:ignore
import armory.dataset  # type:ignore
import armory.engine  # type:ignore
import armory.evaluation  # type:ignore
import armory.export.criteria  # type:ignore
import armory.export.object_detection  # type:ignore
import armory.metric  # type:ignore
import armory.metrics.compute  # type:ignore
import armory.metrics.detection  # type:ignore
import armory.metrics.perturbation  # type:ignore
import armory.metrics.tide  # type:ignore
import armory.model.object_detection  # type:ignore
import armory.perturbation  # type:ignore
import armory.track  # type:ignore

from torch.utils.data import Subset

from armory.experimental.hallucination_dataset import HallucinationDataset  # type:ignore
from pytorch_yolo import darknet2pytorch


############################################
######### LOAD MODEL #########
############################################

def load_model(
    cfg_path=None,
    is_weights_file=True,
    weights_path=None,
    torch_path=None
):
    model = darknet2pytorch.Darknet(
        f'{cfg_path}', inference=True, attack_mode=True)
    model_infer = darknet2pytorch.Darknet(f'{cfg_path}', inference=True)

    if is_weights_file:
        model.load_weights(f'{weights_path}')
        model_infer.load_weights(f'{weights_path}')
    else:
        model.load_state_dict(torch.load(f'{torch_path}'))
        model_infer.load_state_dict(torch.load(f'{torch_path}'))

    _ = model.eval()
    _ = model_infer.eval()

    art_detector = armory.track.track_init_params(
        art.estimators.object_detection.PyTorchObjectDetector
    )(
        model=model,
        channels_first=True,
        input_shape=(3, 608, 608),
        clip_values=(0.0, 1.0),
        attack_losses=(
            "cardinality_error",
            "loss_bbox",
            "loss_ce",
            "loss_giou",
        ),
    )

    return model, model_infer

############################################
######### DATA PROCESSING #########
############################################


def load_dataset(opts_data, opts_patch):

    # train_dataset = HallucinationDataset(
    #     "train_set",
    #     opts_data["data_dir"],
    #     opts_patch["x_patch_dim"],
    #     opts_patch["y_patch_dim"],
    #     opts_data["width"],
    #     opts_data["height"],
    #     eval_mode=False,
    #     input_shape=(608, 608),
    # )
    # train_dataloader = armory.dataset.ObjectDetectionDataLoader(
    #     train_dataset,
    #     batch_size=opts_data["batch_size"],
    #     shuffle=True,
    #     num_workers=2,
    #     pin_memory=False,
    #     collate_fn=HallucinationDataset.custom_collate_fn,
    # )

    test_dataset = HallucinationDataset(
        "test_set",
        opts_data["data_dir"],
        opts_patch["x_patch_dim"],
        opts_patch["y_patch_dim"],
        opts_data["width"],
        opts_data["height"],
        eval_mode=True,
        input_shape=(608, 608),
    )

    test_dataset = armory.dataset.TupleDatasetExpanded(
        test_dataset,
        "image",
        "boxes",
        "img_pixels",
        "patch_pixels_list",
        "weight_list",
        "mean_filter_list"
    )

    test_dataloader = armory.dataset.ObjectDetectionDataLoader(
        test_dataset,
        format=armory.data.BBoxFormat.XYXY,
        boxes_key="boxes",
        dim=armory.data.ImageDimensions.CHW,
        scale=armory.data.Scale(dtype=armory.data.DataType.FLOAT, max=1.0),
        image_key="image",
        labels_key="labels",
        objects_key="objects",
        batch_size=opts_data["batch_size"],
        shuffle=False,
        num_workers=2,
        pin_memory=False
    )

    evaluation_dataset = armory.evaluation.Dataset(
        name="Hallucination Dataset",
        dataloader=test_dataloader
    )

    return evaluation_dataset


############################################
######### CREATE ATTACK #########
############################################

def parse_cli_args():
    """Parse command-line arguments"""
    from armory.examples.utils.args import create_parser  # type:ignore

    parser = create_parser(
        description="Perform license plate object detection with YOLOS",
        batch_size=4,
        export_every_n_batches=5,
        num_batches=20,
    )
    return parser.parse_args()

# TODO: Revisit create_attack() to fit WP attacks; current attack serves as  placeholder


def create_attack(detector, batch_size: int = 1):
    dpatch = armory.track.track_init_params(art.attacks.evasion.RobustDPatch)(
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

    evaluation_attack = armory.perturbation.ArtPatchAttack(
        name="RobustDPatch",
        attack=dpatch,
        use_label_for_untargeted=False,
        generate_every_batch=True,
    )

    return evaluation_attack


def create_blur():
    blur = armory.track.track_init_params(torchvision.transforms.v2.GaussianBlur)(
        kernel_size=5,
    )

    evaluation_perturbation = armory.perturbation.CallablePerturbation(
        name="blur",
        perturbation=blur,
        inputs_accessor=armory.data.Images.as_torch(),
    )

    return evaluation_perturbation

############################################
######### CREATE METRICS #########
############################################


def create_metrics():
    return {
        "linf_norm": armory.metric.PerturbationMetric(
            armory.metrics.perturbation.PerturbationNormMetric(ord=torch.inf),
            armory.data.Images.as_torch(
                scale=armory.data.Scale(
                    dtype=armory.data.DataType.FLOAT, max=1.0)
            ),
        ),
        "map": armory.metric.PredictionMetric(
            torchmetrics.detection.MeanAveragePrecision(class_metrics=False),
            armory.data.BoundingBoxes.as_torch(
                format=armory.data.BBoxFormat.XYXY),
        ),
        # "tide": armory.metrics.tide.TIDE.create(),
        "detection": armory.metrics.detection.ObjectDetectionRates.create(),
    }


def create_exporters(export_every_n_batches):
    """Create sample exporters"""
    return [
        armory.export.object_detection.ObjectDetectionExporter(
            criterion=armory.export.criteria.every_n_batches(
                export_every_n_batches)
        ),
    ]


@armory.track.track_params(prefix="main")
def main(batch_size, export_every_n_batches, num_batches, seed, shuffle):
    """Perform evaluation"""
    if seed is not None:
        torch.manual_seed(seed)

    _, model_infer = load_model()

    dataset = load_dataset(batch_size, shuffle)
    # attack = create_attack(art_detector, batch_size)
    metrics = create_metrics()
    exporters = create_exporters(export_every_n_batches)

    evaluation = armory.evaluation.Evaluation(
        name="license-plate-detection-yolos",
        description="License plate object detection using yolos",
        author="TwoSix",
        dataset=dataset,
        model=model_infer,
        perturbations={
            "benign": [],
            "attack": [],
        },
        metrics=metrics,
        exporters=exporters,
        profiler=armory.metrics.compute.BasicProfiler(),
    )

    engine = armory.engine.EvaluationEngine(
        evaluation,
        limit_test_batches=num_batches,
    )
    results = engine.run()

    pprint(results)


if __name__ == "__main__":
    main(**vars(parse_cli_args()))
