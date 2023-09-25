"""
Example of PyTorch Lightning Data and ML pipeline on Food101 Dataset. Includes support for differing size of training datasets.
Give train dataset step and training log path as args
"""
import albumentations as A

import numpy as np
import torch


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#DEVICE = 'cpu'
BATCH_SIZE = 24
import torch
import datasets

torch.set_float32_matmul_precision("high")
#import armory.data.datasets

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
    new_dataset = train_data.train_test_split(test_size=0.3, seed=1)
    #new_dataset = train_data['train'].train_test_split(test_size=0.5, seed=1)
    train_dataset, test_dataset = new_dataset["train"], new_dataset["test"]

    #train_dataset, test_dataset = HuggingFaceObjectDetectionDataset(
    #    train_dataset
    #), HuggingFaceObjectDetectionDataset(test_dataset)

    return train_dataset, test_dataset
train_dataset, test_dataset = load_huggingface_dataset()
from albumentations.pytorch import ToTensorV2

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


class xview(torch.utils.data.Dataset):
    def __init__(self,dataset, width, height,  transforms=None):
        self.transforms = transforms
        self.image_dataset = dataset
        self.height = height
        self.width = width
  

    def __getitem__(self, idx):
        img = self.image_dataset[idx]


        image = np.asarray(img['image'])
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        #image_resized = cv2.resize(image, (self.width, self.height))
        #image_resized /= 255.0
        image_width = image.shape[1]
        image_height = image.shape[0]
        
        boxes = img['objects']['bbox']



        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = img['objects']['category']
        labels = list(
                map(lambda x: LABEL_MAP[x], labels)
            )  # map original classes to sequential classes
        labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {}
        target["bboxes"] = boxes
        target["labels"] = labels
        #target["area"] = area
        #target["iscrowd"] = iscrowd
        image_id = torch.tensor([idx])
        target["image_id"] = image_id
        target["img_size"] =  (self.height, self.width),
        target["img_scale"] = torch.tensor([1.0]),
        sample = self.transforms(image = image,
                                     bboxes = target['bboxes'],
                                     labels = labels)
        sample["bboxes"] = np.array(sample["bboxes"])
       

        


        image = sample["image"]
        labels = sample["labels"]
        _, new_h, new_w = image.shape
        box_final = []
        for box in sample["bboxes"]:
            xmin_final = (box[0]/image_width)*new_w
            xmax_final = (box[2]/image_width)*new_w
            ymin_final = (box[1]/image_height)*new_h
            yamx_final = (box[3]/image_height)*new_h
            box_final.append([xmin_final, ymin_final, xmax_final, yamx_final])




        target = {
            "boxes": torch.as_tensor(box_final, dtype=torch.float32),
            "labels": torch.as_tensor(labels),
            "image_id": torch.tensor([image_id]),
            "img_size": (new_h, new_w),
            "img_scale": torch.tensor([1.0]),
        }
            
        return image, target, image_id



    def __len__(self):
        return len(self.image_dataset)
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2


def get_train_transforms_1(target_img_size=512):
    return A.Compose(
        [
            #A.HorizontalFlip(p=0.5),
            A.Resize(height=target_img_size, width=target_img_size, p=1),
            A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ToTensorV2(p=1),
        ],
        p=1.0,
        bbox_params=A.BboxParams(
            format="pascal_voc", min_area=0, min_visibility=0, label_fields=["labels"]
        ),
    )


def get_valid_transforms_1(target_img_size=512):
    return A.Compose(
        [
            A.Resize(height=target_img_size, width=target_img_size, p=1),
            A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ToTensorV2(p=1),
        ],
        p=1.0,
        bbox_params=A.BboxParams(
            format="pascal_voc", min_area=0, min_visibility=0, label_fields=["labels"]
        ),
    )
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

class EfficientDetDataModule(LightningDataModule):
    
    def __init__(self,
                train_dataset_adaptor,
                validation_dataset_adaptor,
                train_transforms=get_train_transforms_1(512),
                valid_transforms=get_train_transforms_1(512),
                num_workers=4,
                height=100,
                width=100,
                batch_size=BATCH_SIZE):
        
        self.train_ds = train_dataset_adaptor
        self.valid_ds = validation_dataset_adaptor
        self.train_tfms = train_transforms
        self.valid_tfms = valid_transforms
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.height = height
        self.width = width
        super().__init__()

    def train_dataset(self) -> xview:
        return xview(dataset=self.train_ds, width=self.width,
                      height=self.height,  transforms=self.train_tfms   
        )

    def train_dataloader(self) -> DataLoader:
        train_dataset = self.train_dataset()
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True,
            drop_last=True,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )

        return train_loader

    def val_dataset(self) -> xview:
        return xview(dataset=self.valid_ds, width=self.width,
                      height=self.height,  transforms=self.valid_tfms   
        )

    def val_dataloader(self) -> DataLoader:
        valid_dataset = self.val_dataset()
        valid_loader = torch.utils.data.DataLoader(
            valid_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True,
            drop_last=True,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )

        return valid_loader
    
    @staticmethod
    def collate_fn(batch):
        
        images, targets, image_ids = tuple(zip(*batch))
        images = torch.stack(images)
        images = images.float()

        boxes = [target["boxes"].float() for target in targets]
        labels = [target["labels"].float() for target in targets]
        img_size = torch.tensor([target["img_size"] for target in targets]).float()
        img_scale = torch.tensor([target["img_scale"] for target in targets]).float()

        annotations = {
            "boxes": boxes,
            "cls": labels,
            "img_size": img_size,
            "img_scale": img_scale,
        }
        #return tuple(zip(*batch))

        return images, annotations, targets, image_ids
    

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")

def create_model(num_classes):
    num_classes = num_classes   # 63 classes
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model
import torch
from pytorch_lightning import LightningModule
#from pytorch_lightning.core.decorators import auto_move_data

class EfficientDetModel(LightningModule):
    def __init__(
        self,
        num_classes=63,
        img_size=512,
        prediction_confidence_threshold=0.2,
        learning_rate=0.0002,
        wbf_iou_threshold=0.44,
        inference_transforms=get_train_transforms_1(512), #get_valid_transforms(target_img_size=512),
        #model_architecture='tf_efficientnetv2_l',
    ):
        super().__init__()
        self.img_size = img_size
        self.model = create_model(num_classes)
        self.prediction_confidence_threshold = prediction_confidence_threshold
        self.lr = learning_rate
        self.wbf_iou_threshold = wbf_iou_threshold
        self.inference_tfms = inference_transforms


    #@auto_move_data
    def forward(self, images, targets):
        return self.model(images, targets)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.model.parameters(), lr=self.lr)


    def training_step(self, batch, batch_idx):
        images, annotations, targets, image_ids = batch

        losses = self.model(images, targets)

        '''      logging_losses = {
            "class_loss": losses["class_loss"].detach(),
            "box_loss": losses["box_loss"].detach(),
        }'''

        self.log("train_loss", losses["loss_objectness"], on_step=True, on_epoch=True, prog_bar=True,
                 logger=True)
        self.log(
            "train_class_loss", losses["loss_classifier"], on_step=True, on_epoch=True, prog_bar=True,
            logger=True
        )
        self.log("train_box_loss", losses["loss_box_reg"], on_step=True, on_epoch=True, prog_bar=True,
                 logger=True)

        return losses["loss_objectness"]


    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        images, annotations, targets, image_ids = batch
        #images = list(image.to(DEVICE) for image in images)
        #targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
        outputs = self.model(images, targets)

        detections = outputs["detections"]

        batch_predictions = {
            "predictions": detections,
            "targets": targets,
            "image_ids": image_ids,
        }

        logging_losses = {
            "class_loss": outputs["loss_classifier"].detach(),
            "box_loss": outputs["loss_box_reg"].detach(),
        }

        self.log("valid_loss", outputs["loss_objectness"], on_step=True, on_epoch=True, prog_bar=True,
                 logger=True, sync_dist=True)
        self.log(
            "valid_class_loss", logging_losses["class_loss"], on_step=True, on_epoch=True,
            prog_bar=True, logger=True, sync_dist=True
        )
        self.log("valid_box_loss", logging_losses["box_loss"], on_step=True, on_epoch=True,
                 prog_bar=True, logger=True, sync_dist=True)

        return {'loss': outputs["loss_objectness"], 'batch_predictions': batch_predictions}


from pytorch_lightning import Trainer
find_unused_parameters = True 
trainer = Trainer(max_epochs=1,
    accelerator="auto",
    devices="auto",
    strategy="auto",
    num_sanity_val_steps=0, 
    
    )

model_eff = EfficientDetModel()
dataload = EfficientDetDataModule(train_dataset, test_dataset)

trainer.fit(model_eff,train_dataloaders=dataload.train_dataloader(), 
                val_dataloaders=dataload.val_dataloader())

