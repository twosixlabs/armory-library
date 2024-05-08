import logging
from typing import Optional, Sequence

import PIL.Image
import kornia
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
        self.patch_location = (295, 295)  # middle of 640x640
        # TODO non-zero min value
        self.patch = (
            torch.randint(0, 255, self.patch_shape)
            / 255
            * self.model.inputs_spec.scale.max
        )
        self.targeted = False
        self.learning_rate = 0.01
        self.sample_size = 10
        self.augmentation = kornia.augmentation.container.ImageSequential(
            kornia.augmentation.RandomHorizontalFlip(p=0.5),
            kornia.augmentation.RandomBrightness(brightness=(0.5, 2.0), p=0.5),
            kornia.augmentation.RandomRotation(degrees=15, p=0.5),
            random_apply=True,
        )

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
        for _ in range(self.sample_size):
            # Get inputs as Tensor
            inputs = batch.inputs.get(self.model.inputs_spec)
            assert isinstance(inputs, torch.Tensor)

            # Get patch as Tensor with gradients required
            patch = self.patch.clone()
            patch.requires_grad = True

            # Apply patch to image
            x_1, y_1 = self.patch_location
            x_2 = x_1 + self.patch_shape[1]
            y_2 = y_1 + self.patch_shape[2]
            inputs_with_patch = inputs.clone()
            inputs_with_patch[:, :, x_1:x_2, y_1:y_2] = patch

            # Apply random augmentations to images
            inputs_with_augmentations = self.augmentation(inputs_with_patch)

            # Get targets as Tensor
            _, _, height, width = inputs.shape
            targets = batch.targets.get(self.target_spec)
            yolo_targets = self._to_yolo_targets(
                targets, height, width, self.model.device
            )

            # Get loss from model outputs
            self.model.train()
            loss_components = self.model(inputs_with_augmentations, yolo_targets)
            loss = loss_components["loss_total"]

            # Clean gradients
            self.model.zero_grad()

            # Compute gradients
            loss.backward(retain_graph=True)
            assert patch.grad is not None
            grads = patch.grad.clone()
            assert grads.shape == self.patch.shape

            # Accumulate gradients
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
    trainer = Trainer(limit_train_batches=10, max_epochs=20)
    trainer.fit(module, dataloader)

    patch_np = (module.patch.cpu().numpy().transpose(1, 2, 0) * 255).astype("uint8")
    patch = PIL.Image.fromarray(patch_np)
    patch.save("patch.png")
