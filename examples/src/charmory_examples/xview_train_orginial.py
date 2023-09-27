import albumentations as A

import numpy as np
import torch


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#DEVICE = 'cpu'
BATCH_SIZE = 1
import torch

torch.set_float32_matmul_precision("high")
#import armory.data.datasets
import datasets

LABEL_MAP = {
    11: 1,  # Fixed-wing Aircraft
    12: 2,  # Small Aircraft
    13: 3,  # Cargo Plane
    15: 4,  # Helicopter
    17: 5,  # Passenger Vehicle
    18: 6,  # Small Car
    19: 7,  # Bus
    20: 8,  # Pickup Truck
    21: 9,  # Utility Truck
    23: 10,  # Truck
    24: 11,  # Cargo Truck
    25: 12,  # Truck w/Box
    26: 13,  # Truck Tractor
    27: 14,  # Trailer
    28: 15,  # Truck w/Flatbed
    29: 16,  # Truck w/Liquid
    32: 17,  # Crane Truck
    33: 18,  # Railway Vehicle
    34: 19,  # Passenger Car
    35: 20,  # Cargo Car
    36: 21,  # Flat Car
    37: 22,  # Tank Car
    38: 23,  # Locomotive
    40: 24,  # Maritime Vessel
    41: 25,  # Motorboat
    42: 26,  # Sailboat
    44: 27,  # Tugboat
    45: 28,  # Barge
    47: 29,  # Fishing Vessel
    49: 30,  # Ferry
    50: 31,  # Yacht
    51: 32,  # Container Ship
    52: 33,  # Oil Tanker
    53: 34,  # Engineering Vehicle
    54: 35,  # Tower Crane
    55: 36,  # Container Crane
    56: 37,  # Reach Stacker
    57: 38,  # Straddle Carrier
    59: 39,  # Mobile Crane
    60: 40,  # Dump Truck
    61: 41,  # Haul Truck
    62: 42,  # Scraper/Tractor
    63: 43,  # Front Loader/Bulldozer
    64: 44,  # Excavator
    65: 45,  # Cement Mixer
    66: 46,  # Ground Grader
    71: 47,  # Hut/Tent
    72: 48,  # Shed
    73: 49,  # Building
    74: 50,  # Aircraft Hanger
    75: 51,  # Unknown1
    76: 52,  # Damaged Building
    77: 53,  # Facility
    79: 54,  # Construction Site
    82: 55,  # Unknown2
    83: 56,  # Vehicle Lot
    84: 57,  # Helipad
    86: 58,  # Storage Tank
    89: 59,  # Shipping Container Lot
    91: 60,  # Shipping Container
    93: 61,  # Pylon
    94: 62,  # Tower
}

def load_huggingface_dataset():
    train_data = datasets.load_dataset("Honaker/xview_dataset", split="train")
    #train_data = train_data.train_test_split(test_size=0.5, seed=1)
    new_dataset = train_data.train_test_split(test_size=0.3, seed=1)
    train_dataset, test_dataset = new_dataset["train"], new_dataset["test"]

    #train_dataset, test_dataset = HuggingFaceObjectDetectionDataset(
    #    train_dataset
    #), HuggingFaceObjectDetectionDataset(test_dataset)

    return train_dataset, test_dataset
train_dataset, test_dataset = load_huggingface_dataset()

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
from albumentations.pytorch import ToTensorV2
def collate_fn(batch):
    """
    To handle the data loading as different images may have different number 
    of objects and to handle varying size tensors as well.
    """
    return tuple(zip(*batch))
def get_train_transform():
    return A.Compose([
        #A.Flip(0.5),
        #A.RandomRotate90(0.5),
        #A.MotionBlur(p=0.2),
        #A.MedianBlur(blur_limit=3, p=0.1),
        #A.Blur(blur_limit=3, p=0.1),
        ToTensorV2(p=1.0),
    ], bbox_params={
        'format': 'pascal_voc',
        'label_fields': ['labels']
    })
# define the validation transforms
def get_valid_transform():
    return A.Compose([
        ToTensorV2(p=1.0),
    ], bbox_params={
        'format': 'pascal_voc', 
        'label_fields': ['labels']
    })
import os
import torch

from torchvision.io import read_image
from torchvision.ops.boxes import masks_to_boxes
#from torchvision import tv_tensors
import PIL
from torchvision.transforms.v2 import functional as F
import cv2
from torch.nn import CrossEntropyLoss
from torch.nn import MSELoss
classLossFunc = CrossEntropyLoss()
bboxLossFunc = MSELoss()

class xview(torch.utils.data.Dataset):
    def __init__(self,dataset, width, height,  transforms=None):
        self.transforms = transforms
        self.image_dataset = dataset
        self.height = height
        self.width = width
  

    def __getitem__(self, idx):
        img = self.image_dataset[idx]


        image = np.asarray(img['image'])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image_resized = cv2.resize(image, (self.width, self.height))
        image_resized /= 255.0
        image_width = image.shape[1]
        image_height = image.shape[0]

        boxes = img['objects']['bbox']
        box_final = []
        for box in boxes:
            xmin_final = (box[0]/image_width)*self.width
            xmax_final = (box[2]/image_width)*self.width
            ymin_final = (box[1]/image_height)*self.height
            yamx_final = (box[3]/image_height)*self.height
            box_final.append([xmin_final, ymin_final, xmax_final, yamx_final])


        boxes = torch.as_tensor(box_final, dtype=torch.float32)
        area = img['objects']['area']
        area = torch.as_tensor(area, dtype=torch.float32)
        iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)
        labels = img['objects']['category']
        labels = list(
                map(lambda x: LABEL_MAP[x], labels)
            )  # map original classes to sequential classes
        labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["area"] = area
        target["iscrowd"] = iscrowd
        image_id = torch.tensor([idx])
        target["image_id"] = image_id


        if self.transforms:
            sample = self.transforms(image = image_resized,
                                     bboxes = target['boxes'],
                                     labels = labels)
            image_resized = sample['image']
            target['boxes'] = torch.Tensor(sample['bboxes'])
            
        return image_resized, target



    def __len__(self):
        return len(self.image_dataset)

from torch.utils.data import Dataset, DataLoader
def create_train_loader(train_dataset, num_workers=2):
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    return train_loader
def create_valid_loader(valid_dataset, num_workers=2):
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    return valid_loader
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor,FasterRCNN_MobileNet_V3_Large_320_FPN_Weights
#model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(weights=FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT)

def create_model(num_classes):
    num_classes = num_classes   # 63 classes
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

train_dataset_1 = xview(train_dataset, 3000, 3000,  transforms=get_train_transform())
test_dataset_1 = xview(test_dataset, 3000, 3000,  transforms=get_valid_transform())
train_loader = create_train_loader(train_dataset_1)
valid_loader = create_valid_loader(test_dataset_1)
print(f"Number of training samples: {len(train_dataset)}")
print(f"Number of validation samples: {len(test_dataset)}\n")
# initialize the model and move to the computation device
model = create_model(63)
model = model.to(DEVICE)


from tqdm.auto import tqdm

# function for running training iterations
def train(train_data_loader, model):
    print('Training')
    global train_itr
    global train_loss_list
    
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
    print('Validating')
    global val_itr
    global val_loss_list
    
    # initialize tqdm progress bar
    prog_bar = tqdm(valid_data_loader, total=len(valid_data_loader))
    #model.eval()
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


import time
NUM_EPOCHS= 5
# get the model parameters
params = [p for p in model.parameters() if p.requires_grad]
# define the optimizer
optimizer = torch.optim.SGD(params, lr=0.001, momentum=0.9, weight_decay=0.0005)
# initialize the Averager class
train_loss_hist = Averager()
val_loss_hist = Averager()
train_itr = 1
val_itr = 1
# train and validation loss lists to store loss values of all...
# ... iterations till ena and plot graphs for all iterations
train_loss_list = []
val_loss_list = []
# name to save the trained model with
MODEL_NAME = 'model'

# initialize SaveBestModel class
#save_best_model = SaveBestModel()
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
    # save the best model till now if we have the least loss in the...
    # ... current epoch
    #save_best_model(
    #    val_loss_hist.value, epoch, model, optimizer
    #)
    # save the current epoch model
    #save_model(epoch, model, optimizer)
    # save loss plot
    #save_loss_plot(OUT_DIR, train_loss, val_loss)
    
    # sleep for 5 seconds after each epoch
    time.sleep(5)
torch.save(model, 'fasterrcnn_mobilenet_v3')