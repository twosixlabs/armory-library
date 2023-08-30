"""
CNN model for variable length imagenet model
"""
from typing import Optional

from art.estimators.classification import PyTorchClassifier
import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import torchvision.models as models
from charmory.track import track_init_params,track_params


#Using the Resnet34 model and adding in the amount of classes for the pokemon dataset as a final layer
def make_image_net_model(**kwargs) -> models.resnet34(weights=True):
    model = models.resnet34(weights=True)
    num_ftrs = model.fc.in_features

    model.fc = nn.Linear(num_ftrs, 1000) #No. of classes = 1000 
    return model


def get_art_model(
    model_kwargs: dict, wrapper_kwargs: dict, weights_path: Optional[str] = None
) -> PyTorchClassifier:
    model = (make_image_net_model(**model_kwargs))
    

 


    wrapped_model = track_init_params(PyTorchClassifier)(
        model,
        loss=nn.CrossEntropyLoss(),
        optimizer=torch.optim.Adam(model.parameters(), lr=0.003),
        input_shape=(500, 500,3),
        channels_first=False,
        nb_classes=1000,
        clip_values=(0.0, 1.0),
    )

    return wrapped_model



