from io import BytesIO
from typing import Iterable, Optional, Sequence

import PIL.Image
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import numpy as np
from smqtk_classifier import ClassifyImage
import torch
from xaitk_saliency import GenerateImageClassifierBlackboxSaliency
from xaitk_saliency.impls.gen_image_classifier_blackbox_sal.slidingwindow import (
    SlidingWindowStack,
)

from armory.data import (
    DataType,
    ImageClassificationBatch,
    ImageDimensions,
    Images,
    ImageSpec,
    NumpyImageSpec,
    Scale,
    to_numpy,
)
from armory.export.base import Exporter
from armory.model.image_classification import ImageClassifier


class XaitkSaliencyBlackboxImageClassificationExporter(Exporter):
    """An exporter for XAITK-Saliency."""

    spec = ImageSpec(
        dim=ImageDimensions.HWC,
        scale=Scale(dtype=DataType.FLOAT, max=1.0),
    )

    class ModelWrapper(ClassifyImage):
        def __init__(self, model: ImageClassifier, classes: Sequence[int], spec):
            super().__init__()
            self.model = model
            self.classes = classes
            self.labels = [str(c) for c in classes]
            self.spec = spec

        def get_labels(self):
            return self.labels

        @torch.no_grad()
        def classify_images(self, img_iter):
            for img in img_iter:
                images = Images(images=np.expand_dims(img, axis=0), spec=self.spec)
                inputs = images.get(self.model.inputs_spec)
                outputs = to_numpy(self.model(inputs)).squeeze()
                yield dict(zip(self.labels, outputs[self.classes]))

        def get_config(self):
            # Required by a parent class.
            return {}

    def __init__(
        self,
        name: str,
        model: ImageClassifier,
        classes: Sequence[int],
        algorithm: Optional[GenerateImageClassifierBlackboxSaliency] = None,
        criterion: Optional[Exporter.Criterion] = None,
    ):
        """
        Initializes the exporter.

        Args:
            model: Armory image classifier
            classes: List of classes for which to generate saliency maps
            algorithm: XAITK saliency algorithm. By default the sliding window
                stack algorithm is used.
            criterion: Criterion to determine when samples will be exported. If
                omitted, no samples will be exported.
        """
        super().__init__(
            name=name,
            criterion=criterion,
        )
        self.inputs_spec = NumpyImageSpec(dim=self.spec.dim, scale=self.spec.scale)
        self.blackbox = self.ModelWrapper(model, classes, self.spec)
        self.classes = classes

        if algorithm is None:
            algorithm = SlidingWindowStack((50, 50), (20, 20), threads=4)
        self.algorithm: GenerateImageClassifierBlackboxSaliency = algorithm

    def export_samples(
        self,
        batch_idx: int,
        batch: ImageClassificationBatch,
        samples: Iterable[int],
    ) -> None:
        assert self.sink, "No sink has been set, unable to export"

        images = batch.inputs.get(self.inputs_spec)

        for sample_idx in samples:
            ref_image: np.ndarray = images[sample_idx]
            sal_maps: np.ndarray = self.algorithm.generate(ref_image, self.blackbox)

            for i, class_sal_map in enumerate(sal_maps):
                label = self.classes[i]

                # Positive half saliency
                pos_sal_img = self._plot(
                    ref_image, class_sal_map, False, f"Class {label} Pos Saliency"
                )
                filename = self.artifact_path(
                    batch_idx,
                    sample_idx,
                    f"xaitk_saliency_{self.name}_{label}_pos.png",
                )
                self.sink.log_image(pos_sal_img, filename)

                # Negative half saliency
                neg_sal_img = self._plot(
                    ref_image, class_sal_map, True, f"Class {label} Neg Saliency"
                )
                filename = self.artifact_path(
                    batch_idx,
                    sample_idx,
                    f"xaitk_saliency_{self.name}_{label}_neg.png",
                )
                self.sink.log_image(neg_sal_img, filename)

    @staticmethod
    def _plot(
        ref_image: np.ndarray, sal_map: np.ndarray, neg: bool, title: str
    ) -> PIL.Image.Image:
        fig = Figure(figsize=(6, 6))
        axes = fig.subplots()
        assert isinstance(axes, Axes)

        if neg:
            cmap = "jet_r"
            vmin = -1
            vmax = 0
        else:
            cmap = "jet"
            vmin = 0
            vmax = 1

        axes.imshow(ref_image, alpha=0.7)
        im = axes.imshow(
            np.clip(sal_map, vmin, vmax), cmap=cmap, alpha=0.3, vmin=vmin, vmax=vmax
        )
        fig.colorbar(
            im,
            fraction=0.046 * (ref_image.shape[0] / ref_image.shape[1]),
            pad=0.04,
        )
        axes.set_title(title)
        axes.axis("off")

        buf = BytesIO()
        fig.savefig(buf)
        buf.seek(0)
        return PIL.Image.open(buf)
