"""
CNN model for 244x244x3 pokemon image classification
"""
from typing import Optional

from art.estimators.classification import PyTorchClassifier
import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Net(nn.Module):
    """
    This is a simple CNN for Pokemon dataset and does not achieve SotA performance. It is a modified version of cifar.py.
    """

    def __init__(self) -> None:
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 4, 5, 1)
        self.conv2 = nn.Conv2d(4, 10, 5, 1)
        self.fc1 = nn.Linear(28090, 100)
        self.fc2 = nn.Linear(100, 150)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 3, 1, 2)  # from NHWC to NCHW
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
        output = F.log_softmax(x, dim=1)
        return output


def make_pokemon_model(**kwargs) -> Net:
    return Net()


def get_art_model(
    model_kwargs: dict, wrapper_kwargs: dict, weights_path: Optional[str] = None
) -> PyTorchClassifier:
    model = make_pokemon_model(**model_kwargs)
    model.to(DEVICE)

    if weights_path:
        checkpoint = torch.load(weights_path, map_location=DEVICE)
        model.load_state_dict(checkpoint)

    wrapped_model = PyTorchClassifier(
        model,
        loss=nn.CrossEntropyLoss(),
        optimizer=torch.optim.Adam(model.parameters(), lr=0.003),
        input_shape=(244, 244, 3),
        channels_first=False,
        nb_classes=150,
        clip_values=(0.0, 1.0),
        **wrapper_kwargs,
    )
    return wrapped_model
