import os
import random
from typing import Dict, Tuple

import torch
from torch.utils.data import Dataset


from armory.experimental.image_transforms import (
    floor_odd,
    calculate_xy,
    calc_bounds,
    make_image_transforms,
)
from armory.experimental.yolo_utils import (
    load_json,
    load_image,
    image_to_tensor,
)
from armory.experimental.keypoint2pixel import Keypoint2Pixel


class HallucinationDataset(Dataset):
    def __init__(
        self,
        image_type: str,
        data_dir: str,
        x_patch_dim: int = 170,
        y_patch_dim: int = 150,
        width: int = 80,
        height: int = 80,
        eval_mode: bool = False,
        input_shape: Tuple[int, int] = (608, 608),
        small_ratio=0.5,
    ) -> None:
        self.image_type = image_type
        self.data_dir = data_dir
        self.x_patch_dim = x_patch_dim
        self.y_patch_dim = y_patch_dim
        self.width = width
        self.height = height
        self.eval_mode = eval_mode
        self.input_shape = input_shape

        self.image_transforms = make_image_transforms(image_type)
        self.kp2pix = Keypoint2Pixel(input_shape[0], bilinear=True)

        if self.image_type == "train_set":
            self.entries = load_json(
                f"{data_dir}/train/hallucination_train.json"
            )
            directory = f"{data_dir}/train"
        else:
            self.entries = load_json(
                f"{data_dir}/test/hallucination_test.json"
            )
            directory = f"{data_dir}/test"
        self.directory = directory
        self.keypoint_jitter = 0.02
        self.small_ratio = small_ratio

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        fixed_size = self.input_shape[0]
        x_patch_dim = self.x_patch_dim
        y_patch_dim = self.y_patch_dim

        if self.eval_mode:
            size_factor = 1
        else:
            if self.small_ratio > 0.99:
                size_factor = 1
            else:
                size_factor = (
                    random.random() ** 2 * (1 - self.small_ratio)
                    + self.small_ratio
                )

        w = self.width / self.input_shape[0] * size_factor
        h = self.height / self.input_shape[1] * size_factor

        if self.eval_mode:
            image_name, x, y = self.entries[idx]
        else:
            image_name = self.entries[idx]
            x = random.random() * (1 - w)
            y = random.random() * (1 - h)
        if self.image_type == "train_set":
            image_name = image_name[0]  # TODO: fix this
        image_path = os.path.join(self.directory, image_name)
        # print(image_path)
        x0 = x
        y0 = y
        x1 = min(x + w, 1)
        y1 = min(y + h, 1)

        bounds = calc_bounds(x0, y0, x1, y1)
        keypoints = self.kp2pix.keypoint_preprocess(
            [x0, y0, x1, y0, x1, y1, x0, y1]
        )
        if not self.eval_mode:
            keypoints = self.kp2pix.jitter_keypoints(
                keypoints, self.keypoint_jitter
            )

        image = load_image(image_path, target_size=(self.input_shape))
        if self.image_transforms is not None:
            image = self.image_transforms(image)
        image_pt = image_to_tensor(image)

        w1, h1, h2, w2 = self.kp2pix.calculate_lengths(keypoints)
        (
            image_pixels,
            patch_pixels_list,
            weights_list,
        ) = self.kp2pix.get_ind_map(keypoints, x_patch_dim, y_patch_dim, [])

        h_factor = max(
            floor_odd((y_patch_dim / ((h1 + h2) / 2 * fixed_size)).item()),
            1,
        )
        w_factor = max(
            floor_odd((x_patch_dim / ((w1 + w2) / 2 * fixed_size)).item()),
            1,
        )
        mean_filter = torch.ones(3, 1, h_factor, w_factor) / (
            h_factor * w_factor
        )
        return (
            image_pt,
            bounds,
            image_pixels,
            patch_pixels_list,
            weights_list,
            mean_filter,
        )

    @staticmethod
    def custom_collate_fn(batch):
        images = torch.stack([item[0] for item in batch])
        return (
            images,
            [item[1] for item in batch],
            [item[2] for item in batch],
            [item[3] for item in batch],
            [item[4] for item in batch],
            [item[5] for item in batch],
        )
