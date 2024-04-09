from typing import Iterable, Optional, Tuple

import torch
import torch.nn

from armory.data import (
    Batch,
    BBoxFormat,
    BoundingBoxes,
    DataType,
    ImageDimensions,
    Images,
    Scale,
)
from armory.export.base import Exporter
from armory.export.drise.impl import (
    get_proposal_data,
    make_saliency_img,
    make_saliency_map,
)
from armory.model.object_detection import ObjectDetector


class DRiseSaliencyObjectDetectionExporter(Exporter):
    """An exporter for D-Rise object detection saliency maps."""

    dim = ImageDimensions.CHW
    scale = Scale(dtype=DataType.FLOAT, max=1.0)

    class ModelWrapper(torch.nn.Module):

        def __init__(self, model: ObjectDetector, num_classes: int, dim, scale):
            super().__init__()
            self.model = model
            self.num_classes = num_classes
            self.dim = dim
            self.scale = scale
            self.bbox_accessor = BoundingBoxes.as_torch(format=BBoxFormat.XYXY)

        def forward(self, images_pt: torch.Tensor):
            images = Images(
                images=images_pt,
                dim=self.dim,
                scale=self.scale,
            )
            inputs = self.model.inputs_accessor.get(images)
            outputs = self.model(inputs)
            preds = BoundingBoxes([], format=BBoxFormat.XYXY)
            self.model.predictions_accessor.set(preds, outputs)
            results = self.bbox_accessor.get(preds)

            all_boxes = []
            all_cls_probs = []
            all_objs = []

            for result in results:
                all_boxes.append(result["boxes"])
                all_cls_probs.append(
                    self.expand_scores(result["scores"], result["labels"])
                )
                all_objs.append(torch.tensor([1.0] * result["boxes"].shape[0]))

            return all_boxes, all_cls_probs, all_objs

        def expand_scores(
            self, scores: torch.Tensor, labels: torch.Tensor
        ) -> torch.Tensor:
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
        criterion: Optional[Exporter.Criterion] = None,
    ):
        super().__init__(criterion=criterion)
        self.model = model
        self.batch_size = batch_size
        self.wrapper = self.ModelWrapper(model, num_classes, self.dim, self.scale)
        self.image_accessor = Images.as_torch(dim=self.dim, scale=self.scale)
        self.bbox_accessor = BoundingBoxes.as_torch(format=BBoxFormat.XYXY)

    def export_samples(
        self, chain_name: str, batch_idx: int, batch: Batch, samples: Iterable[int]
    ) -> None:
        assert self.sink, "No sink has been set, unable to export"
        self.model.eval()

        images = self.image_accessor.get(batch.inputs)
        batch_targets = self.bbox_accessor.get(batch.targets)
        batch_preds = self.bbox_accessor.get(batch.predictions)

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
                    number_of_masks=10,
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
                    color = "#50C878"
                    name = "target"
                else:
                    color = "#D22B2B"
                    name = "pred"
                img_contour_with_box = make_saliency_img(
                    image,
                    sal_maps[i],
                    all_boxes[i],
                    color=color,
                )
                filename = self.artifact_path(
                    chain_name,
                    batch_idx,
                    sample_idx,
                    f"drise_{i:02}_{name}.jpg",
                )
                self.sink.log_image(img_contour_with_box, filename)

    def _get_all_boxes_and_preds(
        self, targets: BoundingBoxes.BoxesTorch, preds: BoundingBoxes.BoxesTorch
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert preds["scores"] is not None

        num_targets = targets["boxes"].shape[0]
        num_preds = preds["boxes"].shape[0]
        num_classes = self.wrapper.num_classes

        if num_targets and num_preds:
            all_boxes = torch.vstack([targets["boxes"].cpu(), preds["boxes"].cpu()])
        elif num_targets:  # but no preds
            all_boxes = targets["boxes"]
        elif num_preds:  # but no targets
            all_boxes = preds["boxes"]
        else:  # no targets or preds
            all_boxes = torch.zeros((0, 4))

        all_probs = torch.zeros((num_targets + num_preds, num_classes))
        for i, label in enumerate(targets["labels"]):
            all_probs[i, label] = 1.0
        for j, prob in enumerate(preds["scores"]):
            all_probs[num_targets + j] = prob

        return all_boxes, all_probs
