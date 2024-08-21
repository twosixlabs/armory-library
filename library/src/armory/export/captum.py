from functools import partial
from io import BytesIO
from typing import TYPE_CHECKING, Callable, Iterable, Optional

import PIL.Image
from captum.attr import (
    DeepLift,
    IntegratedGradients,
    NoiseTunnel,
    Saliency,
    visualization,
)
import numpy as np

from armory.data import (
    DataSpecification,
    DataType,
    ImageClassificationBatch,
    ImageDimensions,
    Scale,
    TorchImageSpec,
)
from armory.export.base import Exporter

if TYPE_CHECKING:
    import matplotlib.figure
    import torch
    import torch.nn


class CaptumImageClassificationExporter(Exporter):
    """An exporter for Captum integrated gradients for samples."""

    def __init__(
        self,
        model: "torch.nn.Module",
        name: Optional[str] = None,
        do_saliency: bool = True,
        do_integrated_grads: bool = False,
        do_smoothgrad_squared: bool = False,
        do_deeplift: bool = False,
        internal_batch_size: int = 1,
        n_steps: int = 1,
        inputs_spec: Optional[TorchImageSpec] = None,
        targets_spec: Optional[DataSpecification] = None,
        criterion: Optional[Exporter.Criterion] = None,
    ):
        """
        Initializes the exporter.

        Args:
            model: Computer vision model
            name: Descriptive name of the exporter
            do_saliency: Whether to generate saliency maps
            do_integrated_grads: Whether to generate integrated gradient maps
            do_smoothgrad_squred: Whether to generated integrated gradient maps
                with SmoothGrad Squared
            do_deeplift: Whether to generate DeepLift maps
            internal_batch_size: Batch size to use with Captum
            n_steps: Number of steps to be used when generating maps
            inputs_spec: Optional, data specification used to obtain raw
                image data from the inputs contained in exported batches. By
                default, a Torch images specification is used.
            targets_spec: Optional, data specification used to obtain raw ground
                truth targets data from the exported batches. By default, a
                generic NumPy specification is used.
            criterion: Criterion to determine when samples will be exported. If
                omitted, no samples will be exported.
        """
        super().__init__(
            name=name or "CaptumImageClassification",
            targets_spec=targets_spec,
            criterion=criterion,
        )
        self.model = model
        self.saliency = Saliency(model) if do_saliency else None
        self.integrated_grads = (
            IntegratedGradients(model) if do_integrated_grads else None
        )
        self.smoothgrad_sq = (
            NoiseTunnel(IntegratedGradients(model)) if do_smoothgrad_squared else None
        )
        self.deeplift = DeepLift(model) if do_deeplift else None
        self.internal_batch_size = internal_batch_size
        self.n_steps = n_steps
        self.inputs_spec = inputs_spec or TorchImageSpec(
            dim=ImageDimensions.CHW, scale=Scale(dtype=DataType.FLOAT, max=1.0)
        )

    def export_samples(
        self, batch_idx: int, batch: ImageClassificationBatch, samples: Iterable[int]
    ) -> None:
        assert self.sink, "No sink has been set, unable to export"
        self.model.eval()
        images = batch.inputs.get(self.inputs_spec).to(self.model.device)
        targets = batch.targets.get(self.targets_spec)

        for sample_idx in samples:
            image = images[sample_idx].unsqueeze(0)
            image.requires_grad = True

            orig_image = np.transpose(
                images[sample_idx].cpu().detach().numpy(), (1, 2, 0)
            )

            target = targets[sample_idx].item()

            artifact_path = partial(self.artifact_path, batch_idx, sample_idx)

            self._export_saliency(artifact_path, orig_image, image, target)
            self._export_integrated_gradients(artifact_path, orig_image, image, target)
            self._export_smoothgrad_squared(artifact_path, orig_image, image, target)
            self._export_deeplift(artifact_path, orig_image, image, target)

    @staticmethod
    def _tensor2np(as_tensor: "torch.Tensor") -> np.ndarray:
        return np.transpose(as_tensor.squeeze().cpu().detach().numpy(), (1, 2, 0))

    @staticmethod
    def _fig2img(figure: "matplotlib.figure.Figure") -> PIL.Image.Image:
        buf = BytesIO()
        figure.savefig(buf)
        buf.seek(0)
        return PIL.Image.open(buf)

    def _export_saliency(
        self,
        artifact_path: Callable[[str], str],
        orig_image: np.ndarray,
        image: "torch.Tensor",
        target,
    ):
        if self.saliency is None:
            return

        self.model.zero_grad()
        grads = self.saliency.attribute(image, target=target)
        grads = self._tensor2np(grads)

        fig, _ = visualization.visualize_image_attr(
            grads,
            orig_image,
            method="blended_heat_map",
            sign="absolute_value",
            show_colorbar=True,
            title="Overlayed Gradient Magnitudes",
            use_pyplot=False,
        )
        filename = artifact_path("captum_saliency.png")
        self.sink.log_image(self._fig2img(fig), filename)

    def _export_integrated_gradients(
        self,
        artifact_path: Callable[[str], str],
        orig_image: np.ndarray,
        image: "torch.Tensor",
        target,
    ):
        if self.integrated_grads is None:
            return

        self.model.zero_grad()
        attr_ig, delta = self.integrated_grads.attribute(
            image,
            target=target,
            baselines=image * 0,
            return_convergence_delta=True,
            internal_batch_size=self.internal_batch_size,
        )
        attr_ig = self._tensor2np(attr_ig)

        fig, _ = visualization.visualize_image_attr(
            attr_ig,
            orig_image,
            method="blended_heat_map",
            sign="all",
            show_colorbar=True,
            title="Overlayed Integrated Gradients",
            use_pyplot=False,
        )
        filename = artifact_path("captum_integrated_gradients.png")
        self.sink.log_image(self._fig2img(fig), filename)

    def _export_smoothgrad_squared(
        self,
        artifact_path: Callable[[str], str],
        orig_image: np.ndarray,
        image: "torch.Tensor",
        target,
    ):
        if self.smoothgrad_sq is None:
            return

        self.model.zero_grad()
        attr_ig_nt = self.smoothgrad_sq.attribute(
            image,
            target=target,
            baselines=image * 0,
            nt_type="smoothgrad_sq",
            nt_samples=100,
            stdevs=0.2,
            n_steps=self.n_steps,
            internal_batch_size=self.internal_batch_size,
        )
        attr_ig_nt = self._tensor2np(attr_ig_nt)

        fig, _ = visualization.visualize_image_attr(
            attr_ig_nt,
            orig_image,
            method="blended_heat_map",
            sign="absolute_value",
            outlier_perc=10,
            show_colorbar=True,
            title="Overlayed Integrated Gradients \n with SmoothGrad Squared",
            use_pyplot=False,
        )
        filename = artifact_path("captum_integrated_gradients_smoothgrad_squared.png")
        self.sink.log_image(self._fig2img(fig), filename)

    def _export_deeplift(
        self,
        artifact_path: Callable[[str], str],
        orig_image: np.ndarray,
        image: "torch.Tensor",
        target,
    ):
        if self.deeplift is None:
            return

        self.model.zero_grad()
        attr_dl = self.deeplift.attribute(
            image,
            target=target,
            baselines=image * 0,
        )
        attr_dl = self._tensor2np(attr_dl)

        fig, _ = visualization.visualize_image_attr(
            attr_dl,
            orig_image,
            method="blended_heat_map",
            sign="all",
            show_colorbar=True,
            title="Overlayed DeepLift",
            use_pyplot=False,
        )
        filename = artifact_path("captum_deeplift.png")
        self.sink.log_image(self._fig2img(fig), filename)
