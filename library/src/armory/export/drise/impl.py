from io import BytesIO
import os

from PIL import Image, ImageDraw
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import torch
from torchvision.ops import box_iou
import torchvision.transforms.functional as F
from tqdm import tqdm


def imgpath2tensor(directory, file_name):
    full_path = os.path.join(directory, file_name)
    no_ext = os.path.splitext(full_path)[0]
    mat = None
    txt_path = f"{no_ext}.txt"
    if os.path.exists(txt_path):
        with open(txt_path) as f:
            mat = f.readlines()
    image = Image.open(full_path).convert("RGB")
    resized_img = image.resize((608, 608))
    img_pt = F.to_tensor(resized_img)
    boxes = []
    labels = []
    if mat:
        for string in mat:
            label, cx, cy, w, h = string.split()
            labels.append(int(label))
            x0 = float(cx) - float(w) / 2
            y0 = float(cy) - float(h) / 2
            x1 = float(cx) + float(w) / 2
            y1 = float(cy) + float(h) / 2
            boxes.append([x0, y0, x1, y1])
    return img_pt, torch.tensor(boxes), torch.tensor(labels)


def generate_masks(w=16, h=16, p=0.5, number_of_masks=5000):
    masks = torch.randint(0, 256, (number_of_masks, h, w), dtype=torch.uint8) < int(
        p * 256
    )
    rand_offset_nums = torch.rand((number_of_masks, 2))
    return masks, rand_offset_nums


def masked_dataloader(img_pt, masks_all, rand_offset_nums, batch_size=4, device="cpu"):
    _, W, H = img_pt.shape
    number_of_masks, w, h = masks_all.shape
    imagenet_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
    upsample_h = (h + 1) * (H // h)
    upsample_w = (w + 1) * (W // w)
    W_list = torch.arange(W, device=device)
    H_list = torch.arange(H, device=device)
    X, Y = torch.meshgrid([W_list, H_list], indexing="xy")
    base_grid = torch.stack(
        [
            ((X + 0.5) / upsample_w - 0.5) * 2,
            ((Y + 0.5) / upsample_h - 0.5) * 2,
        ],
        axis=-1,
    ).unsqueeze(dim=0)
    xy_offset = torch.tensor(
        [2 - 2 * W / upsample_w, 2 - 2 * H / upsample_h], device=device
    )
    for ind in range(0, number_of_masks, batch_size):
        size = min(number_of_masks - ind, batch_size)
        grids = torch.tile(base_grid, (size, 1, 1, 1))
        offsets = rand_offset_nums[ind : (ind + size)].to(device) * xy_offset[None, :]
        grids = grids + offsets[:, None, None, :]
        masks = (masks_all[ind : (ind + size), None, :, :]).float().to(device)
        upscale_mask = torch.nn.functional.grid_sample(
            masks, grids, mode="bilinear", padding_mode="border"
        )
        masked_images = img_pt.to(device).unsqueeze(
            dim=0
        ) * upscale_mask + imagenet_mean[:, None, None] * (1 - upscale_mask)
        yield masked_images, upscale_mask


def get_proposal_data(
    img_pt,
    model,
    w=16,
    h=16,
    p=0.5,
    number_of_masks=5000,
    batch_size=16,
    object_thres=0.5,
    device="cpu",
):
    _, H, W = img_pt.shape
    masks, rand_offset_nums = generate_masks(
        w=w, h=h, p=p, number_of_masks=number_of_masks
    )
    mkdl = masked_dataloader(
        img_pt, masks, rand_offset_nums, batch_size=batch_size, device=device
    )
    objectiveness = []
    class_probs = []
    boxes_list = []
    with tqdm(total=number_of_masks) as progress_bar:
        for images_batch, masks_batch in mkdl:
            with torch.no_grad():
                boxes_batch, cls_probs_batch, objs_batch = model(images_batch)
                for boxes, cls_probs, objs in zip(
                    boxes_batch, cls_probs_batch, objs_batch
                ):
                    mask = objs > object_thres
                    objectiveness.append(objs[mask].cpu())
                    class_probs.append(cls_probs[mask].cpu())
                    boxes_list.append(boxes[mask].cpu())
            progress_bar.update(len(objs_batch))

    return masks, rand_offset_nums, boxes_list, class_probs, objectiveness


def make_saliency_map(
    img_pt,
    gt_box,
    gt_prob,
    masks,
    rand_offset_nums,
    boxes,
    cls_probs,
    objectiveness,
    device="cpu",
):
    _, H, W = img_pt.shape
    num_vecs = gt_box.shape[0]
    saliency_map = torch.zeros((num_vecs, H, W), device=device)
    mask_total = torch.zeros((1, H, W), device=device)
    mkdl = masked_dataloader(
        img_pt, masks, rand_offset_nums, batch_size=1, device=device
    )
    z = 0
    number_of_masks = masks.shape[0]
    gt_box = gt_box.to(device)
    gt_prob = gt_prob.to(device)
    with tqdm(total=number_of_masks) as progress_bar:
        for images_batch, masks_batch in mkdl:
            if torch.numel(objectiveness[z]) > 0:
                S_L = box_iou(gt_box, boxes[z].to(device))
                S_P = torch.nn.functional.cosine_similarity(
                    gt_prob[:, None], cls_probs[z].to(device)[None, :], dim=2
                )
                weight = torch.max(
                    S_L * S_P * objectiveness[z][None, :].to(device), dim=1
                ).values
                saliency_map += weight[:, None, None] * masks_batch[0].to(device)
            mask_total += masks_batch[0].to(device)
            z += 1
            progress_bar.update(1)
    saliency_map = saliency_map / mask_total
    return saliency_map.cpu()


def draw_contour(img_pt, cpu_map, thickness=1, levels=[0.95, 0.99], pixel_weight=True):
    """
    pixel_weight: if true, uses the pixels that contain <level> of the total value (after subtracting the min)
                  if false, <level> is quartiles.
    """
    quarter_thick = 1 * thickness
    half_thick = 2 * thickness
    three_thick = 3 * thickness
    full_thick = 4 * thickness
    img_pt_draw = img_pt.clone()
    partial_draw = img_pt_draw[:, half_thick:-half_thick, half_thick:-half_thick]

    if pixel_weight:
        min_value = cpu_map.min()
        new_map = cpu_map - min_value
        vals = torch.sort(new_map.flatten(), descending=True).values
        new_map_sum = new_map.sum()
        cum_sum_frac = torch.cumsum(vals, 0) / new_map_sum

    min_level = min(levels)
    max_level = max(levels)
    for level in levels:
        alpha = 2 ** ((level - min_level) / (max_level - min_level) * 4) / 2**4

        color = (
            torch.tensor([0.28, 0.51, 0.71]) * (1 - alpha)
            + torch.tensor([0.4, 0, 0.4]) * alpha
        )
        if pixel_weight:
            indices = torch.where(cum_sum_frac < (1 - level))[0]
            value = vals[indices[-1] if indices.numel() > 0 else 0] + min_value
        else:
            value = torch.quantile(cpu_map, level, dim=None)

        mask = (cpu_map >= value) * 1.0
        contour1 = torch.abs(
            mask[:, quarter_thick:-three_thick, quarter_thick:-three_thick]
            - mask[:, three_thick:-quarter_thick, three_thick:-quarter_thick]
        )
        contour2 = torch.abs(
            mask[:, quarter_thick:-three_thick, three_thick:-quarter_thick]
            - mask[:, three_thick:-quarter_thick, quarter_thick:-three_thick]
        )
        contour3 = torch.abs(
            mask[:, full_thick:, half_thick:-half_thick]
            - mask[:, :-full_thick, half_thick:-half_thick]
        )
        contour4 = torch.abs(
            mask[:, half_thick:-half_thick, full_thick:]
            - mask[:, half_thick:-half_thick, :-full_thick]
        )
        contour = ((contour1 + contour2 + contour3 + contour4) > 0.5) * 1.0
        partial_draw = partial_draw * (1 - contour) + color[:, None, None] * contour
    img_pt_draw[:, half_thick:-half_thick, half_thick:-half_thick] = partial_draw
    return img_pt_draw


def draw_heatmap(image, saliency_map: torch.Tensor, box):
    fig = Figure(figsize=(6, 6))
    axes = fig.subplots()
    assert isinstance(axes, Axes)

    sal_min = saliency_map.min()
    sal_max = saliency_map.max()
    saliency_map = (saliency_map - sal_min) / (sal_max - sal_min)

    axes.imshow(image.permute((1, 2, 0)), alpha=0.7, cmap="gray")
    heat_map = axes.imshow(
        saliency_map,
        cmap="jet",
        # vmin=0,
        # vmax=1,
        alpha=0.3,
    )
    axes.imshow(box)

    fig.colorbar(heat_map, fraction=0.046 * (image.shape[1] / image.shape[2]), pad=0.04)

    axes.axis("off")

    buf = BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    return Image.open(buf)


def draw_box(image, box, color="red"):
    img_c = image.copy()
    img = ImageDraw.Draw(img_c)
    img.rectangle(box, outline=color, width=2)
    return img_c


def make_saliency_img(
    img_pt,
    saliency_map,
    box,
    color="black",
    levels=[0.9, 0.95, 0.975, 0.99],
    contour=False,
):
    if contour:
        img_with_saliency = draw_contour(
            img_pt, saliency_map.unsqueeze(0), levels=levels
        )
        img = F.to_pil_image(img_with_saliency)
        return draw_box(
            img,
            box.tolist(),
            color,
        )
    else:
        _, height, width = img_pt.shape
        blank = Image.new("RGBA", (height, width), (255, 0, 0, 0))
        box_img = draw_box(blank, box.tolist(), color)
        return draw_heatmap(img_pt, saliency_map, box_img)
