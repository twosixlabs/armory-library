# How to add a new model into armory
In this file, I will show 3 different examples of how to add a model into armory-library. I will show how to use a model from HuggingFace, how to use a model from github, and how to use a model from a different python library.

Common imports shared acrossed all examples
```python
from charmory.model.image_classification import JaticImageClassificationModel
from charmory.track import track_init_params, track_params

from art.estimators.classification import PyTorchClassifier

import torch
from transformers import AutoModelForImageClassification
```
In each example the model and classifier objects are created to be used by armory-library

## Example 1: Using model from HuggingFace
This example using built in armory library capabilities since we support direct ability to use model from HuggingFace
```python 
model = JaticImageClassificationModel(
    track_params(AutoModelForImageClassification.from_pretrained)(
        "tianzhihui-isc/vit-base-patch16-224-in21k-finetuned-pokemon-classification"
    ),
)

classifier = track_init_params(PyTorchClassifier)(
    model,
    loss=torch.nn.CrossEntropyLoss(),
    optimizer=torch.optim.Adam(model.parameters(), lr=0.003),
    input_shape=(3, 224, 224),
    channels_first=True,
    nb_classes=150,
    clip_values=(-1, 1),
)
```
For this section of code we create two variables `model` and `classifier`. We found a model on HuggingFace that is trained on the same pokemon image dataset with the model card 'tianzhihui-isc/vit-base-patch16-224-in21k-finetuned-pokemon-classification'. This can be replaced with another model off of Huggingface.
- The model variable uses `AutoModelForImageClassification.from_pretrained` which takes in a HuggingFace model card name as a variable. This retrieves the model from
    HuggingFace that we will use for this example. `track_params` is a function wrapper that stores the argument values as parameters in MLflow. Lastly,
    the `JaticImageClassificationModel` is another wrapper to make the model compatible with Armory. This allows the model to have a standard output like other
    JATIC image classification models.
- The `PyTorchClassifier` class wraps the model to be usable by the ART library. It is specific to image classifier models written within the PyTorch framework. It takes in as arguments the model, loss function, and optimizer. The input image sizes are the shape of all the images inside the dataset. The `channels_first` variable is true because the images in the pokemon dataset are in a channels-first (C, H, W) multi-dimensional array. The `nb_classes` describe the number of classes model predicts on. Lastly the clip value  is the values that will be the min and max values of the input after scaling.We use `track_init_params` so that the constructor parameters for the ART wrapper are also tracked in MLflow.

## Example 2: Using model from GitHub
In this example case, a model will be downloaded from github and used by armory-library.

First I will clone the repo from github of the example model I will be using.
```bash
git clone 'https://github.com/Lornatang/SRGAN-PyTorch'
```

I first add the project folder of SRGAN-PyTorch to my system path, so that I can import the file into my example file. I also import SRGAN-PyTorch and create an instance of the model class.
```python
import sys
sys.path.insert(0,'/SRGAN-PyTorch')
from SRGAN-PyTorch import model as pytorch_new_model

SRRmodel = pytorch_new_model.SRResNet()
```

Lastly, I run the same code from the first example for creating the model and classifier variables.
```python
model = JaticImageClassificationModel(
    SRRmodel
)

classifier = track_init_params(PyTorchClassifier)(
    model,
    loss=torch.nn.CrossEntropyLoss(),
    optimizer=torch.optim.Adam(model.parameters(), lr=0.003),
    input_shape=(3, 224, 224),
    channels_first=True,
    nb_classes=10,
    clip_values=(-1, 1),
)
```

## Example 3: Using model from another python library
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
model = JaticImageClassificationModel(  
    SRRmodel
)

classifier = track_init_params(PyTorchClassifier)(
    model,
    loss=torch.nn.CrossEntropyLoss(),
    optimizer=torch.optim.Adam(model.parameters(), lr=0.003),
    input_shape=(3, 224, 224),
    channels_first=True,
    nb_classes=10,
    clip_values=(-1, 1),
)

```
