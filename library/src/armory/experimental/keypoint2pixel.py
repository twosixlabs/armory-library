from typing import List, Tuple
import torch
from torch import Tensor

# https://math.stackexchange.com/questions/13404/mapping-irregular-quadrilateral-to-a-rectangle


class Keypoint2Pixel:
    def __init__(self, size: int = 608, bilinear: bool = True) -> None:
        """
        Calculates and maps all necessary labels for patch placement
        using pixel mapping

        Attributes:
            size : int
                The size of the image.
            bilinear : bool
                If True, output the pixel mapping for bilinear interpolation.
                If False, output the pixel mapping for nearest interpolation.

        Args:
            same as attributes

        Returns:
            None
        """
        self.size = size
        self.bilinear = bilinear

    @staticmethod
    def jitter_keypoints(keypoints: Tensor, factor: float = 0.1) -> Tensor:
        """
        Jitter the position of the keypoints.
        Args:
            keypoints : Tensor
                The keypoints.
            factor : maximum amount to jitter the points (less than .5).
            If negative only make the keypoints contract

        Returns:
            Tensor:
            New keypoints
        """

        if factor > 0:
            alpha = (torch.rand(4, 1) - 0.5) * 2 * factor
            beta = (torch.rand(4, 1) - 0.5) * 2 * factor
        else:
            alpha = -torch.rand(4, 1) * factor
            beta = -torch.rand(4, 1) * factor
        temp_f = torch.roll(keypoints, -1, 0)
        temp_b = torch.roll(keypoints, 1, 0)
        new_keypoints = (
            keypoints * ((1 - alpha) + (1 - beta)) / 2
            + temp_f * alpha / 2
            + temp_b * beta / 2
        )

        return new_keypoints.clamp(0, 1)

    @staticmethod
    def calculate_lengths(keypoints: Tensor) -> List[Tensor]:
        """
        Calculate distance between keypoints.
        The first two keypoints should be on the shoulders of the individual
        Args:
            keypoints : Tensor
                Tensor of shape (4, 2) containing the (x, y) coordinates
                of 4 keypoints.

        Returns:
            List[Tensor]:
            List of lengths.
        """
        p1p0 = keypoints[1] - keypoints[0]
        p2p1 = keypoints[2] - keypoints[1]
        p3p0 = keypoints[3] - keypoints[0]
        p2p3 = keypoints[2] - keypoints[3]
        side_lengths = [
            torch.linalg.vector_norm(p1p0),
            torch.linalg.vector_norm(p2p1),
            torch.linalg.vector_norm(p3p0),
            torch.linalg.vector_norm(p2p3),
        ]
        return side_lengths

    @staticmethod
    def _calculate_normals(keypoints: Tensor) -> List[Tensor]:
        """
        Calculate normal vectors based on the keypoints.
        The first two keypoints should be on the shoulders of the individual
        Args:
            keypoints : Tensor
                Tensor of shape (4, 2) containing the (x, y) coordinates
                of 4 keypoints.

        Returns:
            List[Tensor]:
            List of normal vectors, each represented as a 1D
            Tensor of shape (2,).
        """
        N0, N1, N2, N3 = [torch.zeros(2) for _ in range(4)]

        p1p0 = keypoints[1] - keypoints[0]
        p2p1 = keypoints[2] - keypoints[1]
        p3p0 = keypoints[3] - keypoints[0]
        p2p3 = keypoints[2] - keypoints[3]

        N0[0] = p3p0[1]
        N0[1] = -p3p0[0]
        N0 /= torch.linalg.vector_norm(N0)

        if torch.dot(N0, p1p0) < 0:
            N0 = -N0

        N1[0] = p1p0[1]
        N1[1] = -p1p0[0]
        N1 /= torch.linalg.vector_norm(N1)

        if torch.dot(N1, p3p0) < 0:
            N1 = -N1

        N2[0] = p2p1[1]
        N2[1] = -p2p1[0]
        N2 /= torch.linalg.vector_norm(N2)

        if torch.dot(N2, -p1p0) < 0:
            N2 = -N2

        N3[0] = p2p3[1]
        N3[1] = -p2p3[0]
        N3 /= torch.linalg.vector_norm(N3)

        if torch.dot(N3, -p3p0) < 0:
            N3 = -N3

        return [N0, N1, N2, N3]

    def get_box(self, keypoints):
        """
        - returns bounding box around the keypoints

        Args:
            keypoints Tensor:
                Array of four keypoints.

            box : Tuple[float, float, float, float]
                The bounding box as (x_min, y_min, x_max, y_max).
        """
        xs = keypoints[:, 0]
        ys = keypoints[:, 1]
        return [min(xs).item(), min(ys).item(), max(xs).item(), max(ys).item()]

    def _img_ind_2_uv(self, x_ind_img, y_ind_img, keypoints):
        """
        transforms image pixel space to uv space as defined by the keypoints
        see
        for explanation and formulas.


        Args:
            x_ind_img: Tensor
                indices in the x direction in image pixel space
            y_ind_img: Tensor
                indices in the y direction in image pixel space
            keypoints : Tensor
                Keypoints of the obscured region

        Returns:
            Tuple[Tensor, Tensor]: u, v coordinates of x_ind_img, y_ind_img.
        """
        N0, N1, N2, N3 = self._calculate_normals(keypoints)
        p = (torch.stack([x_ind_img, y_ind_img]).T + 0.5) / self.size
        u = torch.matmul(p - keypoints[0], N0) / (
            torch.matmul(p - keypoints[0], N0)
            + torch.matmul(p - keypoints[2], N2)
        )
        v = torch.matmul(p - keypoints[0], N1) / (
            torch.matmul(p - keypoints[0], N1)
            + torch.matmul(p - keypoints[3], N3)
        )
        return u, v

    def get_ind_map(
        self,
        keypoints: Tensor,
        patch_width: int,
        patch_height: int,
        obscured_pixels_list: List[Tensor],
    ):  # -> Tuple[List[List[Tensor]], List[List[Tensor]], List[Tensor]]:
        """
        - Calculates the mapping of image pixels to patch pixels given
        a list of pre-processed keypoints and bounding box. The result is
        a tuple containing lists of image and patch pixels.

        - The function also accepts a parameter `obscured_pixels_list` which
        is a list of regions (represented by quadrilaterals) that should not be considered (for example
        obscured or already labeled pixels).

        Args:
            keypoints : Tensor
                Tensor of four keypoints.
            patch_width : int
                Width of the patch.
            patch_height : int
                Height of the patch.
            obscured_pixels_list : List[Tensor]
                List of regions that should not be considered.

        Returns:
            Tuple[List[List[Tensor]], List[List[Tensor]],List[Tensor]]: The pixels in the image
            space, the corresponding pixels in the patch and the weights for bilinear from patch to image.
        """
        box = self.get_box(keypoints)
        x_ind_img_row = torch.arange(
            round(box[0] * self.size), round(box[2] * self.size)
        )
        y_ind_img_row = torch.arange(
            round(box[1] * self.size), round(box[3] * self.size)
        )
        x_ind_img, y_ind_img = torch.meshgrid(
            x_ind_img_row, y_ind_img_row, indexing="ij"
        )
        x_ind_img = torch.flatten(x_ind_img)
        y_ind_img = torch.flatten(y_ind_img)
        for obscured_keypoints in obscured_pixels_list:
            x_ind_img, y_ind_img = self.filter_obscured_region(
                x_ind_img, y_ind_img, obscured_keypoints
            )
        u, v = self._img_ind_2_uv(x_ind_img, y_ind_img, keypoints)
        tol = 1e-6
        mask = (u >= tol) & (u <= (1 - tol)) & (v >= tol) & (v <= (1 - tol))

        img_pixels = [y_ind_img[mask], x_ind_img[mask]]
        new_ys = v[mask] * patch_height - 0.5
        new_xs = u[mask] * patch_width - 0.5

        if self.bilinear:
            xu = torch.ceil(new_xs)
            xl = torch.ceil(new_xs - 1)
            yu = torch.ceil(new_ys)
            yl = torch.ceil(new_ys - 1)
            Q_11 = [
                torch.maximum(yl, torch.tensor(0)).long(),
                torch.maximum(xl, torch.tensor(0)).long(),
            ]
            w_11 = (xu - new_xs) * (yu - new_ys)
            Q_12 = [
                torch.minimum(yu, torch.tensor(patch_height - 1)).long(),
                torch.maximum(xl, torch.tensor(0)).long(),
            ]
            w_12 = (xu - new_xs) * (new_ys - yl)
            Q_21 = [
                torch.maximum(yl, torch.tensor(0)).long(),
                torch.minimum(xu, torch.tensor(patch_width - 1)).long(),
            ]
            w_21 = (new_xs - xl) * (yu - new_ys)
            Q_22 = [
                torch.minimum(yu, torch.tensor(patch_height - 1)).long(),
                torch.minimum(xu, torch.tensor(patch_width - 1)).long(),
            ]
            w_22 = (new_xs - xl) * (new_ys - yl)
            return (
                img_pixels,
                [Q_11, Q_12, Q_21, Q_22],
                [w_11, w_12, w_21, w_22],
            )
        else:
            return (
                img_pixels,
                [
                    [
                        torch.minimum(
                            torch.round(new_ys).long(),
                            torch.tensor(patch_height - 1),
                        ),
                        torch.minimum(
                            torch.round(new_xs).long(),
                            torch.tensor(patch_width - 1),
                        ),
                    ]
                ],
                [torch.ones_like(new_ys)],
            )

    def filter_obscured_region(
        self, x_ind_img: Tensor, y_ind_img: Tensor, keypoints: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """
        Find pixel indices inside the polygon that are obscured as defined
        by the keypoints and removes those indices.

        Args:
            x_ind_img: Tensor
                indices in the x direction in image pixel space
            y_ind_img: Tensor
                indices in the y direction in image pixel space
            keypoints : Tensor
                Keypoints of the obscured region

        Returns:
            Tuple[Tensor, Tensor]: x_ind_img and y_ind_img with the obscured indices removed.
        """
        u, v = self._img_ind_2_uv(x_ind_img, y_ind_img, keypoints)
        tol = 1e-6
        mask = ~((u >= tol) & (u <= (1 - tol)) & (v >= tol) & (v <= (1 - tol)))
        return x_ind_img[mask], y_ind_img[mask]

    @staticmethod
    def keypoint_preprocess(keypoints: List[float]) -> Tensor:
        """
        This function simply splits the keypoints into 4 parts

        Args:
            keypoints : List[float]
                The four keypoints in [x0,y0,x1,y1,x2,y2,x3,y3] form.

        Returns:
            Tensor: The four preprocessed keypoints
        """
        return torch.tensor(keypoints).reshape(4, 2)
