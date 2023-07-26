"""
CNN model for 512x512x3 image classification
"""
from typing import Optional

from art.estimators.classification import PyTorchClassifier
import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Net(nn.Module):
    """
    This is a simple CNN for food101 and does not achieve SotA performance
    """

    def __init__(self):
        # Model architecture loosely adapted from the cifar baseline_model
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, stride=1, padding=1)
        self.fc1 = nn.Linear(16 * 32 * 256, 256)
        self.fc2 = nn.Linear(256, 101)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
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


def make_food_model(**kwargs) -> Net:
    return Net()


def get_art_model(
    model_kwargs: dict, wrapper_kwargs: dict, weights_path: Optional[str] = None
) -> PyTorchClassifier:
    model = make_food_model(**model_kwargs)
    model.to(DEVICE)

    if weights_path:
        checkpoint = torch.load(weights_path, map_location=DEVICE)
        model.load_state_dict(checkpoint)

    wrapped_model = PyTorchClassifier(
        model,
        loss=nn.CrossEntropyLoss(),
        optimizer=torch.optim.Adam(model.parameters(), lr=0.003),
        input_shape=(512, 512, 3),
        channels_first=False,
        nb_classes=101,
        clip_values=(0.0, 1.0),
        **wrapper_kwargs,
    )
    return wrapped_model
