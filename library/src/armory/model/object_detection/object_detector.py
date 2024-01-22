from typing import Optional

import torchvision.ops

from armory.data import Accessor, Batch, Images, TorchAccessor, to_torch
from armory.evaluation import ModelProtocol
from armory.model.base import ArmoryModel, ModelInputAdapter, ModelOutputAdapter


class ObjectDetector(ArmoryModel, ModelProtocol):
    """
    Wrapper around a model that produces object detection predictions.

    This wrapper applies score-threshold filtering and non-maximum suppression
    to the model's outputs if configured with a score threshold or an IOU
    threshold.

    Example::

        import armory.data
        from armory.model.object_detection import ObjectDetector

        # assuming `model` has been defined elsewhere
        detector = ObjectDetector(
            name="My model",
            model=model,
            inputs_accessor=armory.data.Images.as_torch(
                dim=armory.data.ImageDimensions.CHW
            ),
            predictions_accessor=armory.data.BoundingBoxes.as_torch(
                format=armory.data.BBoxFormat.XYXY
            ),
        )
    """

    def __init__(
        self,
        name: str,
        model,
        inputs_accessor: Images.Accessor,
        predictions_accessor: Accessor,
        preadapter: Optional[ModelInputAdapter] = None,
        postadapter: Optional[ModelOutputAdapter] = None,
        iou_threshold: Optional[float] = None,
        score_threshold: Optional[float] = None,
    ):
        """
        Initializes the model wrapper.

        Args:
            name: Name of the model.
            model: Object detection model being wrapped.
            inputs_accessor: Data accessor used to obtain low-level image data
                from the highly-structured image inputs contained in object
                detection batches.
            predictions_accessor: Data accessor used to update the object
                detection predictions in the batch.
            preadapter: Optional, model input adapter.
            postadapter: Optional, model output adapter.
            iou_threshold: Optional, IOU threshold for non-maximum suppression
            score_threshold: Optional, minimum score for predictions to be
                retained
        """
        super().__init__(
            name=name,
            model=model,
            preadapter=preadapter,
            postadapter=postadapter,
        )
        self.inputs_accessor = inputs_accessor
        self.predictions_accessor = predictions_accessor
        self.iou_threshold = iou_threshold
        self.score_threshold = score_threshold

    def _apply(self, *args, **kwargs):
        super()._apply(*args, **kwargs)
        if isinstance(self.inputs_accessor, TorchAccessor):
            self.inputs_accessor.to(device=self.device)

    def _filter_predictions(self, preds):
        for pred in preds:
            # Filter based on score shreshold, if configured
            if self.score_threshold is not None:
                print(f"started with {len(pred['boxes'])} boxes before score threshold")
                keep = pred["scores"] > self.score_threshold
                pred["boxes"] = pred["boxes"][keep]
                pred["scores"] = pred["scores"][keep]
                pred["labels"] = pred["labels"][keep]
                print(f"keeping {len(pred['boxes'])} boxes")
            # Perform non-maximum suppression, if configured
            if self.iou_threshold is not None:
                print(f"started with {len(pred['boxes'])} boxes before iou threshold")
                keep = torchvision.ops.nms(
                    boxes=to_torch(pred["boxes"]),
                    scores=to_torch(pred["scores"]),
                    iou_threshold=self.iou_threshold,
                )
                single = len(keep) == 1
                boxes = pred["boxes"][keep]
                scores = pred["scores"][keep]
                labels = pred["labels"][keep]
                pred["boxes"] = [boxes] if single else boxes
                pred["scores"] = [scores] if single else scores
                pred["labels"] = [labels] if single else labels
                print(f"keeping {len(pred['boxes'])} boxes")
        return preds

    def predict(self, batch: Batch):
        """
        Invokes the wrapped model using the image inputs in the given batch and
        updates the object detection predictions in the batch.

        If the wrapper has been configured with a score threshold or an IOU
        threshold, the predictions will be filtered before the batch predictions
        are updated.

        Args:
            batch: Object detection batch
        """
        self.eval()
        inputs = self.inputs_accessor.get(batch.inputs)
        outputs = self(inputs)
        outputs = self._filter_predictions(outputs)
        self.predictions_accessor.set(batch.predictions, outputs)
