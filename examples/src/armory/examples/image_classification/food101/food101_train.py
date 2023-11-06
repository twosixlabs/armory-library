"""
Example of PyTorch Lightning Data and ML pipeline on Food101 Dataset. Includes support for differing size of training datasets.
Give train dataset step and training log path as args
"""

import argparse
import os

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets
import torchvision.transforms

parser = argparse.ArgumentParser(
    description="Run the training and testing pipeline for the Food101 Dataset using Lightning",
    formatter_class=argparse.RawTextHelpFormatter,
)
parser.add_argument(
    "--step",
    type=int,
    default=1,
    help="The fraction of the training dataset you would like to use for training -> 2 meaning half of the dataset, 3 meaning a third of the dataset, etc.",
)
parser.add_argument(
    "--logdir",
    type=str,
    default=os.getcwd(),
    help="The directory that you would like the lightning training logs to be saved to. Default value is the current working directory",
)
args = parser.parse_args()
STEP_VALUE = args.step


class FoodNN(nn.Module):
    def __init__(self):
        super(FoodNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, stride=1, padding=1)
        self.fc1 = nn.Linear(16 * 32 * 256, 256)
        self.fc2 = nn.Linear(256, 101)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        output = F.log_softmax(x, dim=1)
        return output


class FoodClassifierTrainer(pl.LightningModule):
    def __init__(self, model):
        # Loosely adapted from the CIFAR10 Baseline model
        super(FoodClassifierTrainer, self).__init__()
        self.model = model
        self.correct_predictions = 0

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self.model(inputs)
        loss = F.cross_entropy(outputs, labels)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        return optim.SGD(self.parameters(), lr=0.001, momentum=0.9)

    def prepare_data(self):
        transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Resize(size=(512, 512)),
            ]
        )
        self.training_data = datasets.Food101(
            root="/home/rahul/cache", split="train", download=True, transform=transform
        )
        self.test_data = datasets.Food101(
            root="/home/rahul/cache", split="test", download=True, transform=transform
        )
        # Alter the download fields and the root to where the dataset is downloaded.

    def train_dataloader(self):
        """return a shuffled Dataloader for the training dataset using some subset of the training dataset"""
        mask = list(range(0, len(self.training_data), STEP_VALUE))
        masked_training_set = torch.utils.data.Subset(self.training_data, mask)
        return DataLoader(masked_training_set, shuffle=True)

    def test_dataloader(self):
        """return a shuffled Dataloader for the testing dataset"""
        return DataLoader(self.test_data, shuffle=False)

    def test_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self.model(inputs)
        _, predictions = torch.max(outputs, 1)
        correct_pred = torch.sum(predictions == labels)
        total_pred = labels.numel()
        self.correct_predictions += correct_pred.item()
        return correct_pred, total_pred

    def get_testing_accuracy(self) -> float:
        return float(self.correct_predictions / len(self.test_data))


trainer = pl.Trainer(
    max_epochs=5,
    accelerator="auto",
    devices="auto",
    strategy="auto",
    default_root_dir=args.logdir,
)

food_nn = FoodNN()
model = FoodClassifierTrainer(food_nn)
trainer.fit(model)
trainer.test(model)
print("Accuracy: " + str(model.get_testing_accuracy()))
