from typing import TYPE_CHECKING, Tuple

import torchvision.transforms

if TYPE_CHECKING:
    from transformers.models.yolos import YolosImageProcessor, YolosModel

from charmory.model import ArmoryModel


class YolosTransformer(ArmoryModel):
    """
    Model wrapper with pre-applied input and output adapters for HuggingFace
    transformer YOLOS models
    """

    def __init__(
        self,
        model: "YolosModel",
        image_processor: "YolosImageProcessor",
        norm_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        norm_std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
        target_size: Tuple[int, int] = (512, 512),
    ):
        super().__init__(model, preadapter=self._preadapt, postadapter=self._postadapt)
        self.image_processor = image_processor
        self.normalize = torchvision.transforms.Normalize(norm_mean, norm_std)
        self.target_size = target_size

    def _preadapt(self, images, targets=None, **kwargs):
        # Prediction targets need a `class_labels` property rather than the
        # `labels` property that's being passed in from ART
        if targets is not None:
            for target in targets:
                target["class_labels"] = target["labels"]

        images = self.normalize(images)

        return (images, targets), kwargs

    def _postadapt(self, output):
        # The model is put in training mode during attack generation, and
        # we need to return the loss components instead of the predictions
        if output.loss_dict is not None:
            return output.loss_dict

        result = self.image_processor.post_process_object_detection(
            output,
            target_sizes=[self.target_size for _ in range(len(output.pred_boxes))],
        )
        return result
