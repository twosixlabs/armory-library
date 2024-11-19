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
This example uses built-in armory library capabilities since we support directly the use of models from HuggingFace.
```python 
model_name = "tianzhihui-isc/vit-base-patch16-224-in21k-finetuned-pokemon-classification"

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
    nb_classes=150,
    clip_values=(-1, 1),
)
```
This section of code defines the three variables `hf_model`, `armory_model` and `classifier`. We found a model on HuggingFace with the model card 'tianzhihui-isc/vit-base-patch16-224-in21k-finetuned-pokemon-classification' that is trained on the Pokemon image classiication dataset. This model can be replaced with other models on Hugging Face that have been trained on the same dataset.
- The `hf_model` variable uses `AutoModelForImageClassification.from_pretrained` to load the model from Hugging Face specified by the model card name parameter.
    `track_params` is a function wrapper that stores the argument values as parameters in MLflow. Next, the `ImageClassifier` class wraps the Hugging Face model
     to make it compatible with Armory. This casts `armory-model` to have a standard output matching other armory-library image classification models.
- The `PyTorchClassifier` class wraps the armory-library model to be usable by the ART library. This ART class is specific to image classifier models written
     with the PyTorch framework and takes as arguments the armory-library model, loss function, and optimizer. The input image sizes are the shape of all the images
     in the dataset. The `channels_first` parameter is true because the images in the Pokemon dataset are in a (C, H, W) multi-dimensional array. The `nb_classes`
     describe the number of Pokemon classes predicted. Lastly the clip value specifies the min and max values of the input after scaling. We use `track_init_params`
     so that the constructor parameters for the ART wrapper are also tracked in MLflow.

## Example 2: Using model from GitHub
In this example, a model downloaded from Github is used by armory-library.

First, clone the Github repo that contains the desired model.
```bash
git clone 'https://github.com/Lornatang/SRGAN-PyTorch'
```

Next, add the project folder of SRGAN-PyTorch to the system path, so that I can import the file into my example file. I also import SRGAN-PyTorch and create an instance of the model class.
```python
import sys
sys.path.insert(0,'/SRGAN-PyTorch')
from SRGAN-PyTorch import model as pytorch_new_model

SRRmodel = pytorch_new_model.SRResNet()
```

Lastly, I run the same code from the first example for creating the model and classifier variables.
```python
armory_model = ImageClassifier(
    name="ViT-finetuned-food101",
    model=SRRmodel,
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
    nb_classes=10,
    clip_values=(-1, 1),
)
```

## Example 3: Using model from PyPI
For the third example, I will be showing an example of using a python model from a python library on PyPI. I will be using the EfficientNet Lite PyTorch library.

This code will import the python library and create an example model from the case with a weights path added.
```python
from efficientnet_lite_pytorch import EfficientNet

from efficientnet_lite0_pytorch_model import EfficientnetLite0ModelFile
weights_path = EfficientnetLite0ModelFile.get_model_file_path()

lite0_model = EfficientNet.from_pretrained('efficientnet-lite0', weights_path = weights_path )
```

This is the same code from the last two examples to create a model and classifier variables.
```python
armory_model = armory.model.image_classification.ImageClassifier(
    name="ViT-finetuned-food101",
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
    nb_classes=10,
    clip_values=(-1, 1),
)

```
