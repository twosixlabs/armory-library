from torchvision.transforms.functional import to_pil_image
import torchvision.transforms as T
from typing import Optional, Tuple
from PIL import Image
import json
import torch
import torchvision.transforms.functional as F

import os


def process_directory(directory, img_type):
    entries = []
    json_file_path = os.path.join(directory, f"{img_type}_yolo_chest_v1.json")

    if os.path.exists(json_file_path):
        with open(json_file_path) as f:
            info_list = json.load(f)

            def update_item(item):
                item["image_name"] = os.path.join(
                    directory, item["image_name"]
                )
                return item

            updated_info_list = list(map(update_item, info_list))
            entries.extend(updated_info_list)

    return entries


def process_directories(directories, img_type):
    entries = []
    for directory in directories:
        entries.extend(process_directory(directory, img_type))
    return entries


def load_json(json_path):
    with open(json_path) as f:
        return json.load(f)


def load_image(
    image_pth: str,
    target_size: Optional[Tuple[int, int]] = (608, 608),
):
    try:
        image = Image.open(image_pth).convert("RGB")
        if target_size:
            image = image.resize(target_size)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {image_pth}")

    return image


def image_to_tensor(image: Image.Image) -> torch.Tensor:
    return T.ToTensor()(image)


def tensor_to_image(image_tensor: torch.Tensor) -> Image.Image:
    return to_pil_image(image_tensor)
