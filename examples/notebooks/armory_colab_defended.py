from pprint import pprint

from art.attacks.evasion import ProjectedGradientDescent
from art.estimators.classification import PyTorchClassifier
import datasets
import torch
import torch.nn
import torchmetrics.classification
from transformers import AutoImageProcessor, AutoModelForImageClassification

from armory.metrics.compute import BasicProfiler
from charmory.data import ArmoryDataLoader
from charmory.engine import EvaluationEngine
import charmory.evaluation as ev
from charmory.metrics.perturbation import PerturbationNormMetric
from charmory.model.image_classification import JaticImageClassificationModel
from charmory.perturbation import ArtEvasionAttack
from charmory.tasks.image_classification import ImageClassificationTask
from charmory.track import track_init_params, track_params
from charmory.utils import Unnormalize
import mlflow
from PIL import Image
import os

# %% [markdown]
# ### We need to bring in the model and attach wrappers to run in this example

# %%
model = JaticImageClassificationModel(
    track_params(AutoModelForImageClassification.from_pretrained)(
        "farleyknight-org-username/vit-base-mnist"
    ),
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

# %% [markdown]
# For this section of code we create two variables `model` and `classifier`. We found a model on HuggingFace that is trained on the same mnist dataset with the model card `farleyknight-org-username/vit-base-mnist`. This can be replaced with another model off of Huggingface or you can use a custom local model.
# - The model variable uses `AutoModelForImageClassification.from_pretrained` which takes in a HuggingFace model card name as a variable. This retrieves the model from
#     HuggingFace that we will use for this example. `track_params` is a function wrapper that stores the argument values as parameters in MLflow. Lastly,
#     the `JaticImageClassificationModel` is another wrapper to make the model compatible with Armory. This allows the model to have a standard output like other
#     JATIC image classification models.
# - The `PyTorchClassifier` class wraps the model to be usable by the ART library. It is specific to image classifier models written within the PyTorch framework. It takes in as arguments the model, loss function, and optimizer. The input image sizes are the shape of all the images inside the dataset. The `channels_first` variable is true because the images in the MNIST dataset are in a channels-first (C, H, W) multi-dimensional array. The `nb_classes` describe the number of classes model predicts on. Lastly the clip value  is the values that will be the min and max values of the input after scaling.We use `track_init_params` so that the constructor parameters for the ART wrapper are also tracked in MLflow.

# %% [markdown]
# ### We need to get the dataset to use in this example.

# %%
dataset = datasets.load_dataset("mnist", split="test")
processor = AutoImageProcessor.from_pretrained(
    "farleyknight-org-username/vit-base-mnist"
)

# %% [markdown]
#
# For this section of code we create two variables `dataset` and `processor`.
# - The `dataset` variable uses the datasets module directly from the HuggingFace API. The path location "mnist" is the location that of the same `dataset` that was used
#   to train the model in this example. The parameter `split='test'` means that only the testing dataset is loading into the variable dataset. This is used because
#   armory-library only using the testing dataset to test the adversarial robustness of the model.
# - The `processor` variable obtains the model processor used by the model to use in our example. The is needed because the dataset needs to be in the correct form to be
#   inputted into the model. The path passed into this function is a HuggingFace model card location that is the same location of the model in this example.  This function
#   just obtains the `preprocessor`.
#
#

# %% [markdown]
# ### The dataset needs to be preprocessed in the correct form for both armory-library and to be inputted into the model. A pytorch dataloader is needed to use the ART API correcly

# %%
batch_size = 16


def transform(sample):
    # Use the HF image processor and convert from BW To RGB
    sample["image"] = processor([img.convert("RGB") for img in sample["image"]])[
        "pixel_values"
    ]
    return sample


dataset.set_transform(transform)
dataloader = ArmoryDataLoader(dataset, batch_size=batch_size)

# %% [markdown]
#
# For this section of code we transform the `dataset` and create a `dataloader`.
# - To transform the `dataset` we define a function `transform` to cycle through the `dataset` and convert each image into HuggingFace 'RGB' form. The transform function is
#   using the HuggingFace API where you first create a function that cycles through a `dataset` and preforms the operation on the `dataset` that is needed to get it in correct
#   form. Next the `set_transform` method is used by the HuggingFace dataset which applies the transform function to the entire HuggingFace dataset.
# - The `dataloader` variable is created by using the `ArmoryDataLoader` function. The `ArmoryDataloader` is a custom PyTorch DataLoader that produces numpy arrays instead of
#   Tensors which is required by ART. The only inputted used in this example is the `dataset` and batch size of images. All other variable can be passed in from the PyTorch
#   DataLoader API.
#

# %% [markdown]
# ## Next an attack is needed for armory-library that we will use to test the adversarial robustness of the machine learning model.

# %%
attack = track_init_params(ProjectedGradientDescent)(
    classifier,
    batch_size=batch_size,
    eps=0.031,
    eps_step=0.007,
    max_iter=20,
    num_random_init=1,
    random_eps=False,
    targeted=False,
    verbose=False,
)

# %% [markdown]
#
# For this section of code we create an `attack` variable.
# - The `attack` variable is create with the `ProjectedGradientDescent` class which comes from the ART library. Here is a link to the paper it was modeled after
#   https://arxiv.org/abs/1706.06083. The track_init_params is used to output the initial metrics to mlflow. It takes as an input the default specs for this `attack`, but
#   these can be changed to be optimized to other models and datasets.
#

# %% [markdown]
# ## A method is needed now to bring together the model, dataset, and attack that will execute all the code to test the adversarial robussness of the model.

# %%
evaluation = ev.Evaluation(
    name="mnist-vit-pgd",
    description="MNIST image classification using a ViT model and PGD attack",
    author="TwoSix",
    dataset=ev.Dataset(
        name="MNIST",
        x_key="image",
        y_key="label",
        test_dataloader=dataloader,
    ),
    model=ev.Model(
        name="ViT",
        model=classifier,
    ),
    perturbations={
        "benign": [],
        "attack": [
            ArtEvasionAttack(
                name="PGD",
                attack=attack,
                use_label_for_untargeted=False,
            ),
        ],
    },
    metric=ev.Metric(
        profiler=BasicProfiler(),
        perturbation={
            "linf_norm": PerturbationNormMetric(ord=torch.inf),
        },
        prediction={
            "accuracy": torchmetrics.classification.Accuracy(
                task="multiclass", num_classes=10
            ),
        },
    ),
)


# %% [markdown]
#
# For this section of code we create an `evalution` variable.
# - The `evalution` variable is created by `ev.Evalution` class. This variable holds all the essential elements to run an adversarial robustness attack. The elements are:
# ```python
#     name: 'user created name of the evalution (str)'
#     description: 'description of the evalution (str)'
#     model: ev.Model
#     dataset: ev.Dataset
#     author: 'string of the author of the evalution (str)'
#     perturbations: 'dictionary of names to lists of perturbations'
#     metric:  ev.Metric
# ```
#
# - The `model`, `dataset`, and `attack` were created in the earlier cells. The `metric` variable is using the basic standard metric function for image classification models in
#   armory-library. In the `dataset` variable, the `x_key` and `y_key` need to be properly listed for the `dataset` being used. The `x_key` needs to be the column name of the image
#   in the `dataset`. The `y_key` needs to be the column name of the label in the `dataset`.
#

# %% [markdown]
# ### Some final detials are needed to output metrics to mlflow and the an engine class is needed to run the evalution class.

# %%
export_every_n_batches = 10
num_batches = 10
task = ImageClassificationTask(
    evaluation,
    export_adapter=Unnormalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    export_every_n_batches=export_every_n_batches,
)
engine = EvaluationEngine(task, limit_test_batches=num_batches)

# %% [markdown]
#
# For this section of code we create a `task` and `engine`:
# - The task tells ART the type of model the attack is running on. Armory-library currently has two tasks image classification and object detection. It takes the `evalution`
#   variable, number of classes in the model, export adapter, and `export_every_n_batches`. The `export_every_n_batches` variable is the amount of batches that get exported to mlFlow, so if
#   10 is selected then every 10 batches are outputted on mlFlow. The adapter takes in the Unnormalize variable which preforms the inverse of the PyTorch Normalize function. This
#   operation allows the normalize dataset to be reversed to allow the unnormalized data to be displayed in mlflow.
# - The `engine` variable is created from the `Evaluation` Engine which takes the task and `limit_test_batches` as variables. The `limit_test_batches` is the amount of batches of testing
#   data that the engine will create metrics for. The `Evaluation` Engine only has one method `.run()` which runs the experiment.
#

# %% [markdown]
# #### Lastly, the engine variable executes .run() which runs the experiment. Function pprint is used to print the output of the run in a better format than regular python print.

# %%
pprint(engine.run())

# %%
# Simple script to retrieve 1 example of a benign and attacked image from the mlflow artifacts
mlflow_var = mlflow.search_experiments(filter_string="name = 'mnist-vit-pgd'")[0]
path = mlflow_var.artifact_location[7:] + "/" + engine.run_id + "/artifacts"
image_bin, image_adv = None, None
for f in os.listdir(path):
    if f.endswith("ex_0_x_adv.png"):
        image_adv = Image.open(os.path.join(path, f))
    elif f.endswith("ex_0_x.png"):
        image_bin = Image.open(os.path.join(path, f))
