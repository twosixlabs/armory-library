from pathlib import Path

from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
from datasets import load_from_disk
import numpy as np
import torch
from torch.utils.data import DataLoader
from train_xview_model import xview

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 1

detection_threshold = 0.2

torch.set_float32_matmul_precision("high")
# label map to
LABEL_MAP_1 = {
    0: 1,
    1: 2,
    2: 3,
    3: 4,
    4: 5,
    5: 6,
}
CLASSES_1 = [
    "None",
    "Passenger Vehicle",
    "Building",
    "Other",
    "Truck",
    "Engineering Vehicle",
    "Maritime Vessel",
]


def load_huggingface_dataset():
    train_dataset = load_from_disk(
        "s3://armory-library-data/datasets/xview_dataset/train/"
    )

    new_dataset = train_dataset.train_test_split(test_size=0.3, seed=1)
    train_dataset, test_dataset = new_dataset["train"], new_dataset["test"]

    return train_dataset, test_dataset


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


def collate_fn(batch):
    """
    To handle the data loading as different images may have different number
    of objects and to handle varying size tensors as well.
    """
    return tuple(zip(*batch))


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


train_dataset, test_dataset = load_huggingface_dataset()
train_dataset.set_transform(transform)
test_dataset.set_transform(transform)
model = torch.load(Path.cwd() / "fasterrcnn_mobilenet_v3")

train_dataset_1 = xview(train_dataset, 500, 500)
test_dataset_1 = xview(test_dataset, 500, 500)
train_loader = create_train_loader(train_dataset_1)
valid_loader = create_valid_loader(test_dataset_1)
print(f"Number of training samples: {len(train_dataset)}")
print(f"Number of validation samples: {len(test_dataset)}\n")
# initialize the model and move to the computation device


model.eval()
for image, target in valid_loader:
    orig_image = image[0].numpy().copy()
    orig_image = np.array(
        Image.fromarray(image[0].numpy().transpose(1, 2, 0).astype(np.uint8))
    )

    image = list(img.to(DEVICE) for img in image)
    with torch.no_grad():
        outputs = model(image)
    outputs = [{k: v.to("cpu") for k, v in t.items()} for t in outputs]
    i = 0
    if len(outputs[0]["boxes"]) != 0:
        boxes = outputs[0]["boxes"].data.numpy()
        scores = outputs[0]["scores"].data.numpy()
        # filter out boxes according to `detection_threshold`

        boxes = boxes[scores >= detection_threshold].astype(np.int32)
        draw_boxes = boxes.copy()
        pred_classes = [CLASSES_1[i] for i in outputs[0]["labels"].cpu().numpy()]

        for j, box in enumerate(draw_boxes):
            cv2.rectangle(
                orig_image,
                (int(box[0]), int(box[1])),
                (int(box[2]), int(box[3])),
                (255, 0, 0),
                2,
            )
            cv2.putText(
                orig_image,
                pred_classes[j],
                (int(box[0]), int(box[1] - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
                lineType=cv2.LINE_AA,
            )
        print(len(draw_boxes))
        cv2.imshow("Prediction", orig_image)
        res = cv2.waitKey(0)

        print(f"Image {i+1} done...")
        print("-" * 50)
        i += 1
