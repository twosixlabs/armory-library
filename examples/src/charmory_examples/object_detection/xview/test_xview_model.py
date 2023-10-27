import albumentations as A
import numpy as np
import torch
from datasets import load_from_disk
from datasets.filesystems import S3FileSystem
import cv2

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 1

torch.set_float32_matmul_precision("high")

LABEL_MAP_1 = {
    0: 1,
    1: 2,  
    2: 3, 
    3: 4,  
    4: 5, 
    5: 6, 

}
CLASSES_1 = [
'None',
'Passenger Vehicle',
'Building',
'Other',
'Truck',
'Engineering Vehicle',
'Maritime Vessel',    
]
def load_huggingface_dataset():
    s3 = S3FileSystem(anon=False)
    train_dataset = load_from_disk("s3://armory-library-data/datasets/train/", fs=s3)

    new_dataset = train_dataset.train_test_split(test_size=0.3, seed=1)
    train_dataset, test_dataset = new_dataset["train"], new_dataset["test"]

    return train_dataset, test_dataset
train_dataset, test_dataset = load_huggingface_dataset()

detection_threshold = 0.2

from albumentations.pytorch import ToTensorV2
def collate_fn(batch):
    """
    To handle the data loading as different images may have different number 
    of objects and to handle varying size tensors as well.
    """
    return tuple(zip(*batch))
def get_train_transform():
    return A.Compose([

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
                map(lambda x: LABEL_MAP_1[x], labels)
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


model = torch.load('Path.cwd() / "fasterrcnn_mobilenet_v3_2"')

train_dataset_1 = xview(train_dataset, 500, 500,  transforms=get_train_transform())
test_dataset_1 = xview(test_dataset, 500, 500,  transforms=get_valid_transform())
train_loader = create_train_loader(train_dataset_1)
valid_loader = create_valid_loader(test_dataset_1)
print(f"Number of training samples: {len(train_dataset)}")
print(f"Number of validation samples: {len(test_dataset)}\n")
# initialize the model and move to the computation device


from torch.nn import CrossEntropyLoss
from torch.nn import MSELoss
classLossFunc = CrossEntropyLoss()
bboxLossFunc = MSELoss()

model.eval()
for image, target in valid_loader:
    orig_image = image[0].numpy().copy()
    orig_image = orig_image.transpose(1, 2, 0)
    orig_image = cv2.cvtColor(orig_image, cv2.COLOR_RGB2BGR)
    image= list(img.to(DEVICE) for img in image)
    with torch.no_grad():
        outputs = model(image)
    outputs = [{k: v.to('cpu') for k, v in t.items()} for t in outputs]
    i = 0
    if len(outputs[0]['boxes']) != 0:
        boxes = outputs[0]['boxes'].data.numpy()
        scores = outputs[0]['scores'].data.numpy()
        # filter out boxes according to `detection_threshold`

        boxes = boxes[scores >= detection_threshold].astype(np.int32)
        draw_boxes = boxes.copy()
        pred_classes = [CLASSES_1[i] for i in outputs[0]['labels'].cpu().numpy()]

        for j, box in enumerate(draw_boxes):
            cv2.rectangle(orig_image,
                        (int(box[0]), int(box[1])),
                        (int(box[2]), int(box[3])),
                        (255, 0, 0), 2)
            cv2.putText(orig_image, pred_classes[j], 
                        (int(box[0]), int(box[1]-5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 
                        2, lineType=cv2.LINE_AA)
        print(len(draw_boxes))
        cv2.imshow('Prediction', orig_image)
        res = cv2.waitKey(0)
       
        print(f"Image {i+1} done...")
        print('-'*50)
        i += 1