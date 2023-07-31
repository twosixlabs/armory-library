"""
CNN model for 244x244x3 pokemon image classification
"""
"""
import art.attacks.evasion
from typing import Optional
from datasets import load_dataset
from art.estimators.classification import PyTorchClassifier
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import ToTensor
from jatic_toolbox import load_dataset as load_jatic_dataset
import armory.baseline_models.pytorch.pokemon
from armory.data.datasets import pokemon_context, pokemon_preprocessing
import armory.scenarios.image_classification
import armory.version
from charmory.engine import Engine
from charmory.data import JaticVisionDatasetGenerator
from charmory.evaluation import (
    Attack,
    Dataset,
    Evaluation,
    Metric,
    Model,
    Scenario,
    SysConfig,
)

def load_huggingface_dataset(
    split: str, epochs: int, batch_size: int, shuffle_files: bool, **kwargs
):
    print(
        "Loading HuggingFace dataset from jatic_toolbox, "
        f"{split=}, {batch_size=}, {epochs=}, {shuffle_files=}"
    )
    dataset = load_jatic_dataset(
        provider="huggingface",
        dataset_name="keremberke/pokemon-classification",
        task="image-classification",
        name='full',
        split=split,
    )
    return JaticVisionDatasetGenerator(
        dataset=dataset,
        batch_size=batch_size,
        epochs=epochs,
        shuffle=shuffle_files,
        preprocessing_fn=pokemon_preprocessing,
        context=pokemon_context,
    )



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


scenario = Scenario(
        function=armory.scenarios.image_classification.ImageClassificationTask,
        kwargs={},
    )
sysconfig = SysConfig(gpus=["all"], use_gpu=True)
def get_art_model_2(
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



def train_model():
    pokemon_model = get_art_model_2(
        model_kwargs={},
        wrapper_kwargs={},
    )

    model = Model(
        name="pokemon",
        model=pokemon_model,
        fit=True,
        fit_kwargs={"nb_epochs": 20},
    )



    criterion=nn.CrossEntropyLoss(),
    optimizer=torch.optim.Adam(model.model.model.parameters(), lr=0.003)
    input_shape=(244, 244, 3),
    channels_first=False,
    nb_classes=150,
    clip_values=(0.0, 1.0)
    #ds = load_dataset("keremberke/pokemon-classification", name="full")
    #train = ds['train']
    #validation = ds['validation']
    #test = ds['test']


    dataset = Dataset(
        name="POKEMON",
        train_dataset=load_huggingface_dataset(
            split="train",
            epochs=20,
            batch_size=64,
            shuffle_files=True,
        ),
        test_dataset=load_huggingface_dataset(
            split="test",
            epochs=1,
            batch_size=64,
            shuffle_files=False,
        ),
    )
    attack = Attack(
        function=art.attacks.evasion.ProjectedGradientDescent,
        kwargs={
            "batch_size": 1,
            "eps": 0.031,
            "eps_step": 0.007,
            "max_iter": 20,
            "num_random_init": 1,
            "random_eps": False,
            "targeted": False,
            "verbose": False,
        },
        knowledge="white",
        use_label=True,
        type=None,
    )

    metric = Metric(
        profiler_type="basic",
        supported_metrics=["accuracy"],
        perturbation=["linf"],
        task=["categorical_accuracy"],
        means=True,
        record_metric_per_sample=False,
    )
    #model.model.fit()
    baseline = Evaluation(
        name="pokemon",
        description="Baseline Pokemon image classification",
        author="msw@example.com",
        dataset=dataset,
        model=model,
        attack=None,
        scenario=scenario,
        defense=None,
        metric=metric,
        sysconfig=sysconfig,
    )
    pokemon_engine = Engine(baseline)
    results = pokemon_engine.run()
    print(results)
    
    '''
    for epoch in range(10):
        running_loss = 0.0
        for i, data in enumerate(dataset.train_dataset, 0):
            img_path = data['image_file_path'];labels = data['labels']
            image = Image.open(img_path)
            inputs = ToTensor()(image).unsqueeze(0)
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 100 == 99:    # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 100:.3f}')
                running_loss = 0.0

    #for batch, (X, y) in enumerate(ds):
    print('Finished Training') 

    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in test:
            images, labels = data
            # calculate outputs by running images through the network
            outputs = model(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')

    '''


if __name__ == '__main__':
    train_model()
    """