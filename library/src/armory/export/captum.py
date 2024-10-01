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

        :param model: Computer vision model
        :type model: torch.nn.Module
        :param name: Descriptive name of the exporter, defaults to None
        :type name: str, optional
        :param do_saliency: Whether to generate saliency maps, defaults to True
        :type do_saliency: bool, optional
        :param do_integrated_grads: Whether to generate integrated gradient maps, defaults to False
        :type do_integrated_grads: bool, optional
        :param do_smoothgrad_squared: Whether to generated integrated gradient maps
                with SmoothGrad Squared, defaults to False
        :type do_smoothgrad_squared: bool, optional
        :param do_deeplift: Whether to generate DeepLift maps, defaults to False
        :type do_deeplift: bool, optional
        :param internal_batch_size: Batch size to use with Captum, defaults to 1
        :type internal_batch_size: int, optional
        :param n_steps: Number of steps to be used when generating maps, defaults to 1
        :type n_steps: int, optional
        :param inputs_spec: Optional, data specification used to obtain raw
                image data from the inputs contained in exported batches. By
                default, a Torch images specification is used.
        :type inputs_spec: TorchImageSpec, optional
        :param targets_spec: Optional, data specification used to obtain raw ground
                truth targets data from the exported batches. By default, a
                generic NumPy specification is used.
        :type targets_spec: DataSpecification, optional
        :param criterion: Criterion to determine when samples will be exported. If
                omitted, no samples will be exported.
        :type criterion: Exporter.Criterion, optional
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
        """
        Exports samples from the given batch.

        :param batch_idx: The index/number of this batch.
        :type batch_idx: int
        :param batch: The batch to be exported.
        :type batch: Batch
        :param samples: The indices of samples in the batch to be exported.
        :type samples: Iterable[int]
        """
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
        """
        Tensor to numpy

        :param as_tensor: as_tensor
        :type as_tensor: torch.Tensor
        :return: np.ndarray
        :rtype: np.ndarray
        """
        return np.transpose(as_tensor.squeeze().cpu().detach().numpy(), (1, 2, 0))

    @staticmethod
    def _fig2img(figure: "matplotlib.figure.Figure") -> PIL.Image.Image:
        """
        Figure to image

        :param figure: Figure
        :type figure: matplotlib.figure.Figure
        :return: PIL.Image.Image
        :rtype: PIL.Image.Image
        """
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
        """
        Export saliency

        :param artifact_path: Artifact path
        :type artifact_path: Callable[[str], str]
        :param orig_image: Origin image
        :type orig_image: np.ndarray
        :param image: Image
        :type image: torch.Tensor
        :param target: Target
        :type target: _type_
        """
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
        """
        Export integrated gradients

        :param artifact_path: Artifact path
        :type artifact_path: Callable[[str], str]
        :param orig_image: Origin image
        :type orig_image: np.ndarray
        :param image: Image
        :type image: torch.Tensor
        :param target: Target
        :type target: _type_
        """
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
        """
        Export SmoothGrad Squared

        :param artifact_path: Artifact path
        :type artifact_path: Callable[[str], str]
        :param orig_image: Origin image
        :type orig_image: np.ndarray
        :param image: Image
        :type image: torch.Tensor
        :param target: Target
        :type target: _type_
        """
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
        """
        Export DeepLift

        :param artifact_path: Artifact path
        :type artifact_path: Callable[[str], str]
        :param orig_image: Origin image
        :type orig_image: np.ndarray
        :param image: Image
        :type image: torch.Tensor
        :param target: Target
        :type target: _type_
        """
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
