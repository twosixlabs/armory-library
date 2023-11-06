import time
from typing import List

import albumentations as A
from albumentations.pytorch import ToTensorV2
from datasets import load_from_disk
import numpy as np
import torch
from torch.utils.data import DataLoader
import torchvision
from torchvision.models.detection.faster_rcnn import (
    FasterRCNN_MobileNet_V3_Large_320_FPN_Weights,
    FastRCNNPredictor,
)
from tqdm.auto import tqdm

torch.set_float32_matmul_precision("high")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 8


def load_huggingface_dataset():
    train_dataset = load_from_disk(
        "s3://armory-library-data/datasets/xview_dataset/train/"
    )
    new_dataset = train_dataset.train_test_split(test_size=0.3, seed=1)
    train_dataset, test_dataset = new_dataset["train"], new_dataset["test"]

    return train_dataset, test_dataset


class Averager:
    def __init__(self):
        self.current_total = 0.0
        self.iterations = 0.0

    def send(self, value):
        self.current_total += value
        self.iterations += 1

    @property
    def value(self):
        if self.iterations == 0:
            return 0
        else:
            return 1.0 * self.current_total / self.iterations

    def reset(self):
        self.current_total = 0.0
        self.iterations = 0.0


def collate_fn(batch):
    """
    To handle the data loading as different images may have different number
    of objects and to handle varying size tensors as well.
    """
    return tuple(zip(*batch))


class xview(torch.utils.data.Dataset):
    def __init__(self, dataset, width, height):
        self.image_dataset = dataset
        self.height = height
        self.width = width

    def __getitem__(self, idx):
        img = self.image_dataset[idx]
        target = {}
        target["boxes"] = torch.as_tensor(img["objects"]["bbox"], dtype=torch.int64)
        target["labels"] = torch.as_tensor(
            img["objects"]["category"], dtype=torch.int64
        )
        target["image_id"] = torch.tensor([idx])
        image = img["image"].float()

        return image, target

    def __len__(self):
        return len(self.image_dataset)


def create_train_loader(train_dataset, num_workers=2):
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )
    return train_loader


def create_valid_loader(valid_dataset, num_workers=2):
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )
    return valid_loader


def create_model(num_classes):
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


train_dataset, test_dataset = load_huggingface_dataset()
model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(
    weights=FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT
)
img_transforms = A.Compose(
    [
        A.LongestMaxSize(max_size=500),
        A.PadIfNeeded(
            min_height=500,
            min_width=500,
            border_mode=0,
            value=(0, 0, 0),
        ),
        ToTensorV2(p=1.0),
    ],
    bbox_params=A.BboxParams(
        format="pascal_voc",
        label_fields=["labels"],
    ),
)


def transform(sample):
    transformed = dict(image=[], objects=[])
    for i in range(len(sample["image"])):
        transformed_img = img_transforms(
            image=np.asarray(sample["image"][i]),
            bboxes=sample["objects"][i]["bbox"],
            labels=sample["objects"][i]["category"],
        )
        transformed["image"].append(transformed_img["image"])
        transformed["objects"].append(
            dict(
                bbox=transformed_img["bboxes"],
                category=transformed_img["labels"],
            )
        )
    return transformed


train_dataset.set_transform(transform)
test_dataset.set_transform(transform)


train_dataset_1 = xview(train_dataset, 500, 500)
test_dataset_1 = xview(test_dataset, 500, 500)
train_loader = create_train_loader(train_dataset_1)
valid_loader = create_valid_loader(test_dataset_1)
print(f"Number of training samples: {len(train_dataset)}")
print(f"Number of validation samples: {len(test_dataset)}\n")
# initialize the model and move to the computation device
model = create_model(6)
model = model.to(DEVICE)


# function for running training iterations
def train(train_data_loader, model):
    print("Training")
    global train_itr

    # initialize tqdm progress bar
    prog_bar = tqdm(train_data_loader, total=len(train_data_loader))
    model.train()
    for i, data in enumerate(prog_bar):
        optimizer.zero_grad()

        images, targets = data

        images = list(image.to(DEVICE) for image in images)

        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        loss_value = losses.item()
        train_loss_list.append(loss_value)
        train_loss_hist.send(loss_value)
        losses.backward()
        optimizer.step()
        train_itr += 1

        # update the loss value beside the progress bar for each iteration
        prog_bar.set_description(desc=f"Loss: {loss_value:.4f}")
    return train_loss_list


def validate(valid_data_loader, model):
    print("Validating")
    global val_itr

    # initialize tqdm progress bar
    prog_bar = tqdm(valid_data_loader, total=len(valid_data_loader))
    # model.eval()
    for i, data in enumerate(prog_bar):
        images, targets = data

        images = list(image.to(DEVICE) for image in images)
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

        with torch.no_grad():
            loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        loss_value = losses.item()
        val_loss_list.append(loss_value)
        val_loss_hist.send(loss_value)
        val_itr += 1
        # update the loss value beside the progress bar for each iteration
        prog_bar.set_description(desc=f"Loss: {loss_value:.4f}")
    return val_loss_list


NUM_EPOCHS = 10
# get the model parameters
params = [p for p in model.parameters() if p.requires_grad]
# define the optimizer
optimizer = torch.optim.SGD(params, lr=0.00001, momentum=0.009, weight_decay=0.0005)
# initialize the Averager class
train_loss_hist = Averager()
val_loss_hist = Averager()
train_itr = 1
val_itr = 1
# train and validation loss lists to store loss values of all...

train_loss_list: List[float] = []
val_loss_list: List[float] = []
# name to save the trained model with
MODEL_NAME = "model"

if __name__ == "__main__":
    # start the training epochs
    for epoch in range(NUM_EPOCHS):
        print(f"\nEPOCH {epoch+1} of {NUM_EPOCHS}")
        # reset the training and validation loss histories for the current epoch
        train_loss_hist.reset()
        val_loss_hist.reset()
        # start timer and carry out training and validation
        start = time.time()
        train_loss = train(train_loader, model)
        val_loss = validate(valid_loader, model)
        print(f"Epoch #{epoch+1} train loss: {train_loss_hist.value:.3f}")
        print(f"Epoch #{epoch+1} validation loss: {val_loss_hist.value:.3f}")
        end = time.time()
        print(f"Took {((end - start) / 60):.3f} minutes for epoch {epoch}")

    torch.save(model, "fasterrcnn_mobilenet_v3")
