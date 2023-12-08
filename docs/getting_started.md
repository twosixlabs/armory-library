# Getting Started

## Lori's Journey

1. PyTorch self-tutorial introduction



## Prerequisite concepts in an Armory context

### Dataset

A dataset is a collection of images (samples) in a sequence-like structure such as a
tuple, map, or numpy array. Each can have a target (label) assigned. Datasets can be
imported from a variety of sources such as PyTorch, HuggingFace, or GitHub ???.

### Basic

The following is a basic example of loading tuple data:

```python
raw_dataset = [
        ([1, 2, 3], 4),
        ([5, 6, 7], 8),
    ]

keyed_dataset = TupleDataset(raw_dataset, x_key="data", y_key="target")
```

For this dataset we need a adapter because ????.

```python
from pprint import pprint
pprint(keyed_dataset)

def adapter(data):
    # ??? What goes here? Is armory expecting specific keys? Were the keys given
    # to TupleDataset arbitrary or necessary?
dataset = ArmoryDataset(keyed_dataset, adapter)
```

#### Huggingface

The following is an example of how to load a dataset from Huggingface:

```python
import datasets # hugging face dataset library
from transformers import AutoImageProcessor # hugging face image processor class
from armory.data import ArmoryLoader


dataset = datasets.load_dataset("mnist", split="test")
processor = AutoImageProcessor.from_pretrained(
        "farleyknight-org-username/vit-base-mnist"  # !!! this is a model card
    )

dataset.set_transform(functools.partial(transform, processor))
dataloader = ArmoryDataLoader(dataset, batch_size=batch_size, num_workers=5)
```

The `load_dataset` functions imports the [MNIST][mnist] (handwritten digit) dataset from
HuggingFace. The `split` parameters specifies which subset of the dataset to load,
is usually either `train` or `test` or possibly, depending on the dataset `validation`.

Here, `AutoImageProcessor.from_pretrained` expects a HuggingFace name for the dataset
card. Then `ArmoryDataLoader` generates the numpy arrays that are required by ART for
the evaluation.

[mnist]: https://huggingface.co/datasets/mnist

### Model

A model is the output of a machine learning algorithm run on a training set of data. It is used to identify patterns or make predictions on unseen datasets. Models can be imported from a variety of sources, including Huggingface, GitHub, PyPI, and jatic_toolbox.

#### Huggingface

The following is an example of how to import a model from Huggingface:
```python
model = JaticImageClassificationModel(
        track_params(AutoModelForImageClassification.from_pretrained)(
            "farleyknight-org-username/vit-base-mnist"
        ),
    )
```
Here, `farleyknight-org-username/vit-base-mnist` is the HUggingface model card name. `track_params` is a function wrapper that stores the argument values as parameters in MLflow and `JaticImageClassificationModel` is a wrapper to make the model compatible with Armory.

#### GitHub

The following is an example of how to import a model from GitHub, after having cloned the relevant repository:
```bash
git clone 'https://github.com/Lornatang/SRGAN-PyTorch'
```

```python
import sys
sys.path.insert(0,'/SRGAN-PyTorch')
from SRGAN-PyTorch import model as pytorch_new_model

SRRmodel = pytorch_new_model.SRResNet()

model = JaticImageClassificationModel(
    SRRmodel
)
```
Here, we add the GitHub folder to the system path and import the mode. Then as in the last example, we use the `JaticImageClassificationModel` to make the model compatible with Armory.

#### PyPI

The following is an example of loading a model from the EfficientNet Lite PyTorch library:
```python
from efficientnet_lite_pytorch import EfficientNet

from efficientnet_lite0_pytorch_model import EfficientnetLite0ModelFile
weights_path = EfficientnetLite0ModelFile.get_model_file_path()

lite0_model = EfficientNet.from_pretrained('efficientnet-lite0', weights_path = weights_path )

model = JaticImageClassificationModel(
    SRRmodel
)
```
Here, we import the library and create a model from the case with a weights path added.

#### jatic_toolbox

The following is an example of loading a Torchvision model from the jatic_toolbox:
```python
model = track_params(load_jatic_model)(
        provider="torchvision",
        model_name="resnet34",
        task="image-classification",
    )
```


In each case, we use the `PyTorchClassifier` wrapper to make the model compatible with the ART library. Note that this is specific to image classification models written within the PyTorch framework. The parameters can be adjusted as needed. `track_initial_params` is used so that these parameters are also tracked in MLflow.

```python
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

### Attack

An attack is a transformation of (each sample in) the dataset in order to disrupt the machine learning algorithm's results. For example, after an attack, the model may misclassify an image or fail to detect an object. In Armory, targeted attacks are designed to focus on only one class at a time.

Attacks can be loaded from [IBM's Adversarial Robustness Toolbox][art].

The following is an example of how to define an attack from ART's ProjectedGradientDescent:
```python
attack = Attack(
        name="PGD",
        attack=track_init_params(art.attacks.evasion.ProjectedGradientDescent)(
            classifier,
            batch_size=1,
            eps=epsilon,
            eps_step=0.007,
            max_iter=20,
            num_random_init=1,
            random_eps=False,
            targeted=False,
            verbose=False,
        ),
        use_label_for_untargeted=True,
    )
```
Here, ???

[art]: https://github.com/Trusted-AI/adversarial-robustness-toolbox
