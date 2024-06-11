import logging
import random
from typing import Optional, Sequence

import PIL.Image
import kornia
from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.loggers import MLFlowLogger
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
        self.patch_min = 0
        self.patch_max = self.model.inputs_spec.scale.max
        self.patch = torch.randint(0, 256, self.patch_shape) / 255 * self.patch_max
        self.patch.requires_grad = True
        self.targeted = False
        self.learning_rate = 0.1
        self.augmentation = kornia.augmentation.container.ImageSequential(
            kornia.augmentation.RandomHorizontalFlip(p=0.5),
            kornia.augmentation.RandomBrightness(brightness=(0.75, 1.25), p=0.5),
            kornia.augmentation.RandomRotation(degrees=15, p=0.5),
            random_apply=True,
        )

    def configure_optimizers(self):
        return torch.optim.SGD([self.patch], lr=self.learning_rate, momentum=0.9)

    def on_train_epoch_end(self):
        if isinstance(self.logger, MLFlowLogger):
            if self.current_epoch % 5 == 0:
                patch_np = (
                    self.patch.detach().cpu().numpy().transpose(1, 2, 0) * 255
                ).astype("uint8")
                patch = PIL.Image.fromarray(patch_np)
                self.logger.experiment.log_image(
                    self.logger.run_id, patch, f"patch_epoch_{self.current_epoch}.png"
                )

    def training_step(self, batch: armory.data.Batch, batch_idx: int):
        # Get inputs as Tensor
        inputs = batch.inputs.get(self.model.inputs_spec)
        assert isinstance(inputs, torch.Tensor)

        # Apply patch to image
        x_1 = random.randint(0, inputs.shape[3] - self.patch_shape[2])
        y_1 = random.randint(0, inputs.shape[2] - self.patch_shape[1])
        # x_1, y_1 = self.patch_location
        x_2 = x_1 + self.patch_shape[1]
        y_2 = y_1 + self.patch_shape[2]
        inputs_with_patch = inputs.clone()
        inputs_with_patch[:, :, x_1:x_2, y_1:y_2] = self.patch

        # Apply random augmentations to images
        inputs_with_augmentations = self.augmentation(inputs_with_patch)

        # Get targets as Tensor
        _, _, height, width = inputs.shape
        targets = batch.targets.get(self.target_spec)
        yolo_targets = self._to_yolo_targets(targets, height, width, self.model.device)

        # Get loss from model outputs
        self.model.train()
        loss_components = self.model(inputs_with_augmentations, yolo_targets)
        # Negate the loss because the optimizer will be minimizing it
        loss = -loss_components["loss_total"]

        self.log("loss", loss)

        return loss

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

    patch_np = (module.patch.detach().cpu().numpy().transpose(1, 2, 0) * 255).astype(
        "uint8"
    )
    patch = PIL.Image.fromarray(patch_np)
    patch.save("initial_patch.png")

    trainer = Trainer(limit_train_batches=10, max_epochs=20)
    trainer.fit(module, dataloader)

    patch_np = (module.patch.detach().cpu().numpy().transpose(1, 2, 0) * 255).astype(
        "uint8"
    )
    patch = PIL.Image.fromarray(patch_np)
    patch.save("patch.png")
