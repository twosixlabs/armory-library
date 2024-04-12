from math import floor, ceil
import random
from torchvision import transforms as T


def make_image_transforms(image_set):
    if image_set == "train_set" or image_set == "all":
        return T.Compose(
            [
                T.ColorJitter(
                    brightness=0.05, contrast=0.05, saturation=0.05, hue=0.05
                ),
            ]
        )

    if image_set == "test_set":
        return None

    raise ValueError(f"unknown: {image_set}")


def horizontal_flip(patch_pixels, x_patch_dim):
    patch_pixels[1] = x_patch_dim - 1 - patch_pixels[1]
    return patch_pixels


def floor_odd(x):
    return int(floor((x + 1) / 2) * 2 - 1)


def calculate_xy(
    height, width, input_shape, size_factor, eval_mode=True, x=None, y=None
):
    w = height / input_shape[0] * size_factor
    h = width / input_shape[1] * size_factor

    if eval_mode:
        return [
            x,
            y,
            min(x + w, 1),
            y,
            min(x + w, 1),
            min(y + h, 1),
            x,
            min(y + h, 1),
        ]
    else:
        x = random.random() * (1 - w)
        y = random.random() * (1 - h)
        return [
            x,
            y,
            min(x + w, 1),
            y,
            min(x + w, 1),
            min(y + h, 1),
            x,
            min(y + h, 1),
        ]


def calc_bounds(x0, y0, x1, y1):
    """
    spatial dimension of the output from the model for a given image,
    triplet that results from training + zmode=True: batch_size X 24 X spatial_dimension
    """
    spat_dims = [76, 38, 19]
    bounds = []
    for dims in spat_dims:
        bounds.append(
            [
                floor(x0 * dims),
                floor(y0 * dims),
                ceil(x1 * dims),
                ceil(y1 * dims),
            ]
        )
    return bounds
