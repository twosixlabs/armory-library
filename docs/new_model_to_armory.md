# How to add a new model into armory
This file demonstrates three different examples of integrating a new model into armory-library: utilizing a model from HuggingFace, incorporating a model from GitHub, and integrating a model from another Python library.

The imports below are shared acrossed all examples. Each example creates the model and classifier objects to be used by armory-library.

```python
from armory.model.image_classification import ImageClassifier
from armory.track import track_init_params, track_params

from art.estimators.classification import PyTorchClassifier

import torch
from transformers import AutoModelForImageClassification
```

## Example 1: Using model from HuggingFace
This example uses built-in armory library capabilities since we support directly the use of models from Hugging Face.
```python 
model_name = "nateraw/food"

hf_model = track_params(
    transformers.AutoModelForObjectDetection.from_pretrained
)(pretrained_model_name_or_path=model_name)

armory_model = ImageClassifier(
    name="ViT-finetuned-food101",
    model=hf_model,
    inputs_spec=armory.data.TorchImageSpec(
        dim=armory.data.ImageDimensions.CHW, scale=normalized_scale
    ),
)

classifier = track_init_params(PyTorchClassifier)(
    model=armory_model,
    loss=torch.nn.CrossEntropyLoss(),
    optimizer=torch.optim.Adam(model.parameters(), lr=0.003),
    input_shape=(3, 224, 224),
    channels_first=True,
    nb_classes=101,
    clip_values=(-1, 1),
)
```
The section of code above defines the three variables `hf_model`, `armory_model` and `classifier`. We found a model on HuggingFace with model card ['nateraw/food'](https://huggingface.co/nateraw/food) that is trained on the Food 101 ([ethz/food101](https://huggingface.co/datasets/ethz/food101)) image classication dataset also available on Hugging Face. This model can be replaced with other models on Hugging Face that have been trained on the same dataset.
- The `hf_model` variable uses `AutoModelForImageClassification.from_pretrained` to load the model from Hugging Face specified by the model card name parameter.
    `track_params` is a function wrapper that stores the argument values as parameters in MLflow. Next, the `ImageClassifier` class wraps the Hugging Face model
     to make it compatible with Armory. This casts `armory-model` to have a standard output matching other armory-library image classification models.
- The `PyTorchClassifier` class wraps the armory-library model to be usable by the ART library. This ART class is specific to image classifier models written
     with the PyTorch framework and takes as arguments the armory-library model, loss function, and optimizer. The input image sizes are the shape of all the images
     in the dataset. The `channels_first` parameter is true because the images in the Pokemon dataset are in a (C, H, W) multi-dimensional array. The `nb_classes`
     describe the number of Pokemon classes predicted. Lastly the clip value specifies the min and max values of the input after scaling. We use `track_init_params`
     so that the constructor parameters for the ART wrapper are also tracked in MLflow.

## Example 2: Using model from PyTorch Image Models (timm)
In this example, a model downloaded from PyTorch Image Models is used by armory-library.

First, clone the `timm` Github repo and perform an editable install.
```bash
git clone https://github.com/rwightman/pytorch-image-models
cd pytorch-image-models && pip install -e .
```

Next, create a ResNet34 model pre-trained on the ImageNet-1K image classification dataset.
```python
import timm

timm_model = timm.create_model('resnet34', pretrained=True)
```

Lastly, run the same code from the first example for creating the armory-library model and classifier.
```python
armory_model = ImageClassifier(
    name="resnet34-imagenet",
    model=timm_model,
    inputs_spec=armory.data.TorchImageSpec(
        dim=armory.data.ImageDimensions.CHW, scale=normalized_scale
    ),
)

classifier = track_init_params(PyTorchClassifier)(
    model=armory_model,
    loss=torch.nn.CrossEntropyLoss(),
    optimizer=torch.optim.Adam(model.parameters(), lr=0.003),
    input_shape=(3, 224, 224),
    channels_first=True,
    nb_classes=1000,
    clip_values=(-1, 1),
)
```

## Example 3: Using model from PyPI
The third example demonstrates use of a model from a Python library on PyPI - the EfficientNet Lite PyTorch library.
First, install the `efficientnet_lite_pytorch` library into your environment.

```bash
pip install efficientnet_lite_pytorch
```

This code imports the Python library and creates a pre-trained ImageNet model from the specified weights path.
```python
from efficientnet_lite_pytorch import EfficientNet

from efficientnet_lite0_pytorch_model import EfficientnetLite0ModelFile
weights_path = EfficientnetLite0ModelFile.get_model_file_path()

lite0_model = EfficientNet.from_pretrained('efficientnet-lite0', weights_path = weights_path )
```

This is the same code from the last two examples that creates an armory-library model and classifier.
```python
armory_model = armory.model.image_classification.ImageClassifier(
    name="efficientnet-lite-imagenet",
    model=lite0_model,
    inputs_spec=armory.data.TorchImageSpec(
        dim=armory.data.ImageDimensions.CHW, scale=normalized_scale
    ),
)

classifier = track_init_params(PyTorchClassifier)(
    model=armory_model,
    loss=torch.nn.CrossEntropyLoss(),
    optimizer=torch.optim.Adam(model.parameters(), lr=0.003),
    input_shape=(3, 224, 224),
    channels_first=True,
    nb_classes=1000,
    clip_values=(-1, 1),
)
```
