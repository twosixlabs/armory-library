from io import BytesIO
from typing import Iterable, Optional, Sequence

import PIL.Image
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import numpy as np
from smqtk_classifier import ClassifyImage
import torch
from xaitk_saliency import GenerateImageClassifierBlackboxSaliency

from armory.data import (
    Accessor,
    Batch,
    DataType,
    DefaultTorchAccessor,
    ImageDimensions,
    Images,
    Scale,
    to_numpy,
)
from armory.export.base import Exporter
from armory.model.image_classification import ImageClassifier


class XaitkSaliencyBlackboxImageClassificationExporter(Exporter):
    """An exporter for XAITK-Saliency."""

    dim = ImageDimensions.HWC
    scale = Scale(dtype=DataType.FLOAT, max=1.0)

    class ModelWrapper(ClassifyImage):
        def __init__(self, model: ImageClassifier, classes: Sequence[int], dim, scale):
            super().__init__()
            self.model = model
            self.classes = classes
            self.labels = [str(c) for c in classes]
            self.dim = dim
            self.scale = scale

        def get_labels(self):
            return self.labels

        @torch.no_grad()
        def classify_images(self, img_iter):
            for img in img_iter:
                images = Images(
                    images=np.expand_dims(img, axis=0), dim=self.dim, scale=self.scale
                )
                inputs = self.model.accessor.get(images)
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
        algorithm: GenerateImageClassifierBlackboxSaliency,
        targets_accessor: Optional[Accessor] = None,
        criterion: Optional[Exporter.Criterion] = None,
    ):
        """
        Initializes the exporter.

        Args:
            targets_accessor: Optional, data exporter used to obtain low-level
                ground truth targets data from the high-ly structured targets
                contained in exported batches. By default, a generic NumPy
                accessor is used.
            criterion: Criterion dictating when samples will be exported. If
                omitted, no samples will be exported.
        """
        super().__init__(
            targets_accessor=targets_accessor or DefaultTorchAccessor(),
            criterion=criterion,
        )
        self.inputs_accessor = Images.as_numpy(dim=self.dim, scale=self.scale)
        self.name = name
        self.blackbox = self.ModelWrapper(model, classes, self.dim, self.scale)
        self.algorithm = algorithm
        self.classes = classes

    def export_samples(
        self, chain_name: str, batch_idx: int, batch: Batch, samples: Iterable[int]
    ) -> None:
        assert self.sink, "No sink has been set, unable to export"

        images = self.inputs_accessor.get(batch.inputs)

        for sample_idx in samples:
            ref_image: np.ndarray = images[sample_idx]
            sal_maps: np.ndarray = self.algorithm.generate(ref_image, self.blackbox)

            colorbar_kwargs = {
                "fraction": 0.046 * (ref_image.shape[0] / ref_image.shape[1]),
                "pad": 0.04,
            }

            for i, class_sal_map in enumerate(sal_maps):
                label = self.classes[i]

                # Positive half saliency
                fig = Figure(figsize=(6, 6))
                axes = fig.subplots()
                assert isinstance(axes, Axes)

                axes.imshow(ref_image, alpha=0.7)
                im = axes.imshow(
                    np.clip(class_sal_map, 0, 1), cmap="jet", alpha=0.3, vmin=0, vmax=1
                )
                fig.colorbar(im, **colorbar_kwargs)
                axes.set_title(f"Class {label} Pos Saliency")
                axes.axis("off")

                buf = BytesIO()
                fig.savefig(buf)
                buf.seek(0)
                img = PIL.Image.open(buf)

                filename = self.artifact_path(
                    chain_name,
                    batch_idx,
                    sample_idx,
                    f"xaitk_saliency_{self.name}_{label}_pos.png",
                )
                self.sink.log_image(img, filename)

                # Negative half saliency
                fig = Figure(figsize=(6, 6))
                axes = fig.subplots()
                assert isinstance(axes, Axes)

                axes.imshow(ref_image, alpha=0.7)
                im = axes.imshow(
                    np.clip(class_sal_map, -1, 0),
                    cmap="jet_r",
                    alpha=0.3,
                    vmin=-1,
                    vmax=0,
                )
                fig.colorbar(im, **colorbar_kwargs)
                axes.set_title(f"Class {label} Neg Saliency")
                axes.axis("off")

                buf = BytesIO()
                fig.savefig(buf)
                buf.seek(0)
                img = PIL.Image.open(buf)

                filename = self.artifact_path(
                    chain_name,
                    batch_idx,
                    sample_idx,
                    f"xaitk_saliency_{self.name}_{label}_neg.png",
                )
                self.sink.log_image(img, filename)
