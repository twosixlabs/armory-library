from typing import Iterable, Optional, Tuple

import torch
import torch.nn

from armory.data import (
    BBoxFormat,
    BoundingBoxes,
    BoundingBoxSpec,
    DataType,
    ImageDimensions,
    Images,
    ImageSpec,
    ObjectDetectionBatch,
    Scale,
    TorchBoundingBoxSpec,
    TorchImageSpec,
)
from armory.export.base import Exporter
from armory.export.drise.impl import (
    get_proposal_data,
    make_saliency_img,
    make_saliency_map,
)
from armory.model.object_detection import ObjectDetector


class DRiseSaliencyObjectDetectionExporter(Exporter):
    """An exporter for D-RISE object detection saliency maps."""

    spec = ImageSpec(
        dim=ImageDimensions.CHW,
        scale=Scale(dtype=DataType.FLOAT, max=1.0),
    )

    class ModelWrapper(torch.nn.Module):
        """
        A wrapper around an Armory object detector to make it work with the
        D-RISE implementation.
        """

        def __init__(self, model: ObjectDetector, num_classes: int, spec):
            super().__init__()
            self.model = model
            self.num_classes = num_classes
            self.spec = spec
            self.bbox_spec = TorchBoundingBoxSpec(format=BBoxFormat.XYXY)

        def forward(self, images_pt: torch.Tensor):
            """
            Invokes the wrapped model and returns the resulting boxes, class
            probabilities, and objectness as a tuple.
            """
            batch = ObjectDetectionBatch(
                inputs=Images(
                    images=images_pt,
                    spec=self.spec,
                ),
                targets=BoundingBoxes([], BoundingBoxSpec(format=BBoxFormat.XYXY)),
            )
            self.model.predict(batch)
            results = batch.predictions.get(self.bbox_spec)

            all_boxes = []
            all_cls_probs = []
            all_objs = []

            for result in results:
                assert result["scores"] is not None
                all_boxes.append(result["boxes"])
                all_cls_probs.append(
                    self.expand_scores(result["scores"], result["labels"])
                )
                all_objs.append(torch.tensor([1.0] * result["boxes"].shape[0]))

            return all_boxes, all_cls_probs, all_objs

        def expand_scores(
            self, scores: torch.Tensor, labels: torch.Tensor
        ) -> torch.Tensor:
            """
            Expands the given batch of scores and labels into probabilities for
            all classes. The score for the assigned label remains as specified,
            with all other classes receiving the evenly divided remaining
            probability.

            Args:
                scores: N-length tensor of predicted scores
                labels: N-length tensor of assigned labels

            Returns:
                (N, C) tensor of probabilities, where C is the total number of
                    classes
            """
            expanded_scores = torch.ones(scores.shape[0], self.num_classes)
            for i, (score, label) in enumerate(zip(scores, labels)):
                residual = (1.0 - score.item()) / self.num_classes
                expanded_scores[i, :] *= residual
                expanded_scores[i, int(label.item())] = score
            return expanded_scores

    def __init__(
        self,
        model: ObjectDetector,
        num_classes: int,
        batch_size: int = 1,
        name: Optional[str] = None,
        num_masks: int = 1000,
        score_threshold: float = 0.5,
        criterion: Optional[Exporter.Criterion] = None,
    ):
        """
        Initializes the exporter.

        Args:
            model: Armory object detector
            num_classes: Number of classes supported by the model
            batch_size: Number of images per batch when generating D-RISE
                saliency maps
            name: Description of the exporter
            num_masks: Number of masks to evaluate
            score_threshold: Minimum score for predicted objects to be included
                for D-RISE saliency map generation
            criterion: Criterion to determine when samples will be exported. If
                omitted, no samples will be exported.
        """
        super().__init__(name=name or "D-RISE", criterion=criterion)
        self.model = model
        self.batch_size = batch_size
        self.num_masks = num_masks
        self.score_threshold = score_threshold
        self.wrapper = self.ModelWrapper(model, num_classes, self.spec)
        self.image_spec = TorchImageSpec(dim=self.spec.dim, scale=self.spec.scale)
        self.bbox_spec = TorchBoundingBoxSpec(format=BBoxFormat.XYXY)

    def export_samples(
        self,
        batch_idx: int,
        batch: ObjectDetectionBatch,
        samples: Iterable[int],
    ) -> None:
        assert self.sink, "No sink has been set, unable to export"
        self.model.eval()

        images = batch.inputs.get(self.image_spec)
        batch_targets = batch.targets.get(self.bbox_spec)
        batch_preds = batch.predictions.get(self.bbox_spec)

        for sample_idx in samples:
            image = images[sample_idx]
            targets = batch_targets[sample_idx]
            preds = batch_preds[sample_idx]

            all_boxes, all_probs = self._get_all_boxes_and_preds(targets, preds)

            (masks, rand_offset_nums, boxes, class_probs, objectiveness) = (
                get_proposal_data(
                    image,
                    self.wrapper,
                    batch_size=self.batch_size,
                    device=self.model.device,
                    number_of_masks=self.num_masks,
                )
            )

            sal_maps = make_saliency_map(
                image,
                all_boxes,
                all_probs,
                masks,
                rand_offset_nums,
                boxes,
                class_probs,
                objectiveness,
                device=self.model.device,
            )

            num_targets = targets["boxes"].shape[0]
            for i in range(all_boxes.shape[0]):
                if i < num_targets:
                    color = "red"
                    name = "target"
                else:
                    color = "white"
                    name = "pred"
                img_contour_with_box = make_saliency_img(
                    image,
                    sal_maps[i],
                    all_boxes[i],
                    color=color,
                )
                filename = self.artifact_path(
                    batch_idx,
                    sample_idx,
                    f"drise_{i:02}_{name}.png",
                )
                self.sink.log_image(img_contour_with_box, filename)

    def _get_all_boxes_and_preds(
        self, targets: BoundingBoxes.BoxesTorch, preds: BoundingBoxes.BoxesTorch
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Produces single lists containing all target and predicted boxes and
        class probabilities.
        """
        assert preds["scores"] is not None

        num_targets = targets["boxes"].shape[0]
        above_threshold = preds["scores"] > self.score_threshold
        boxes_above_threshold = preds["boxes"][above_threshold]
        scores_above_threshold = preds["scores"][above_threshold]
        labels_above_threshold = preds["labels"][above_threshold]
        num_preds = boxes_above_threshold.shape[0]
        num_classes = self.wrapper.num_classes

        if num_targets and num_preds:
            all_boxes = torch.vstack(
                [targets["boxes"].cpu(), boxes_above_threshold.cpu()]
            )
        elif num_targets:  # but no preds
            all_boxes = targets["boxes"]
        elif num_preds:  # but no targets
            all_boxes = boxes_above_threshold
        else:  # no targets or preds
            all_boxes = torch.zeros((0, 4))

        all_probs = torch.zeros((num_targets + num_preds, num_classes))
        for i, label in enumerate(targets["labels"]):
            all_probs[i, label] = 1.0
        for j, prob in enumerate(
            self.wrapper.expand_scores(scores_above_threshold, labels_above_threshold)
        ):
            all_probs[num_targets + j] = prob

        return all_boxes, all_probs
