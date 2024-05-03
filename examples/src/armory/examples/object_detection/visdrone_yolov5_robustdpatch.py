import logging
from typing import Optional, Sequence

from lightning.pytorch import LightningModule, Trainer
import torch
import yolov5

import armory.data
import armory.examples.object_detection.datasets.visdrone
import armory.model.object_detection

logger = logging.getLogger(__name__)


class RobustDPatchModule(LightningModule):

    def __init__(self, model: armory.model.object_detection.YoloV5ObjectDetector):
        super().__init__()
        self.model = model
        self.target_spec = armory.data.TorchBoundingBoxSpec(
            format=armory.data.BBoxFormat.CXCYWH
        )
        self.patch_shape = (3, 50, 50)
        self.patch_location = (295, 295)
        self.patch = torch.zeros(self.patch_shape)
        self.targeted = False
        self.learning_rate = 0.01

    # def forward(self, inputs, target):
    #     # return self.model(inputs)
    #     return super().forward(inputs, target)

    def configure_optimizers(self):
        # return torch.optim.Adam(self.parameters(), lr=1e-3)
        return super().configure_optimizers()

    def on_train_epoch_start(self):
        self.patch_gradients = torch.zeros_like(self.patch, device=self.model.device)
        self.patch = self.patch.to(self.model.device)

    def on_train_epoch_end(self):
        self.patch = (
            self.patch
            + torch.sign(self.patch_gradients)
            * (1 - 2 * int(self.targeted))
            * self.learning_rate
        )
        # TODO handle normalized min/max
        self.patch = torch.clip(self.patch, 0, self.model.inputs_spec.scale.max)

    def training_step(self, batch: armory.data.Batch, batch_idx: int):
        # TODO transformations

        # Get inputs as Tensor with gradients required
        inputs = batch.inputs.get(self.model.inputs_spec)
        assert isinstance(inputs, torch.Tensor)
        if inputs.is_leaf:
            inputs.requires_grad = True
        else:
            inputs.retain_grad()

        # TODO apply patch to image

        # Get targets as Tensor
        _, _, height, width = inputs.shape
        targets = batch.targets.get(self.target_spec)
        yolo_targets = self._to_yolo_targets(targets, height, width, self.model.device)

        # Get loss from model outputs
        self.model.train()
        loss_components = self.model(inputs, yolo_targets)
        loss = loss_components["loss_total"]

        # Clean gradients
        self.model.zero_grad()

        # Compute gradients
        loss.backward(retain_graph=True)
        assert inputs.grad is not None
        grads = inputs.grad.clone()
        assert grads.shape == inputs.shape

        # Extract patch gradients
        x_1, y_1 = self.patch_location
        x_2 = x_1 + self.patch_shape[1]
        y_2 = y_1 + self.patch_shape[2]
        grads = grads[:, :, x_1:x_2, y_1:y_2]

        patch_gradients = self.patch_gradients + torch.sum(grads, dim=0)
        logger.debug(
            "Gradient percentage diff: %f)",
            torch.mean(
                torch.sign(patch_gradients) != torch.sign(self.patch_gradients),
                dtype=torch.float64,
            ),
        )
        self.patch_gradients = patch_gradients

    @staticmethod
    def _to_yolo_targets(
        targets: Sequence[armory.data.BoundingBoxes.BoxesTorch],
        height: int,
        width: int,
        device,
    ) -> torch.Tensor:
        targets_list = []

        for i, target in enumerate(targets):
            labels = torch.zeros(len(target["boxes"]), 6, device=device)
            labels[:, 0] = i
            labels[:, 1] = target["labels"]
            labels[:, 2:6] = target["boxes"]

            # normalize bounding boxes to [0, 1]}
            labels[:, 2:6:2] = labels[:, 2:6:2] / width
            labels[:, 3:6:2] = labels[:, 3:6:2] / height

            targets_list.append(labels)

        r = torch.vstack(targets_list)
        return r


def load_model():
    hf_model = yolov5.load(model_path="smidm/yolov5-visdrone")

    armory_model = armory.model.object_detection.YoloV5ObjectDetector(
        name="YOLOv5",
        model=hf_model,
    )

    return armory_model


def load_dataset(batch_size: int, shuffle: bool, seed: Optional[int] = None):
    hf_dataset = armory.examples.object_detection.datasets.visdrone.load_dataset()
    dataloader = armory.examples.object_detection.datasets.visdrone.create_dataloader(
        hf_dataset["val"],
        max_size=640,
        batch_size=batch_size,
        shuffle=shuffle,
        seed=seed,
    )
    return dataloader


if __name__ == "__main__":
    dataloader = load_dataset(batch_size=2, shuffle=False)
    model = load_model()

    module = RobustDPatchModule(model)
    trainer = Trainer(limit_train_batches=10, max_epochs=2)
    trainer.fit(module, dataloader)
