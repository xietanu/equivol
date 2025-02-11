import numpy as np

import masks
import borders

OFFSETS = [(0, 1), (1, 1), (0, -1), (1, -1)]
DIAG_OFFSETS = [(1, 1), (1, -1), (-1, 1), (-1, -1)]


def map_distance_inside_target(
    mask: np.ndarray,
    mask_borders: np.ndarray | None = None,
):
    """Map the distance from the outside of the target to the inside using the given mask and border set.

    Args:
        mask (np.ndarray): Histological mask as a 2D numpy array.
        mask_borders (np.ndarray, optional): A binary mask of the target borders. Defaults to None.
            If none, calculates the borders itself from the mask.
    Returns:
        np.ndarray: Distance map as a 2D numpy array.
    """

    assert masks.is_valid_mask(mask)

    if mask_borders is None:
        mask_borders = borders.find_borders(mask)

    assert np.any(
        mask_borders[:, :, 0]
    ), "No outer borders found. Please provide a valid mask."
    assert np.any(
        mask_borders[:, :, 1]
    ), "No inner borders found. Please provide a valid mask."

    distance = np.full(mask_borders[:, :, :2].shape, np.inf)
    distance[mask_borders[:, :, :2]] = 0

    cur_value = 0

    while np.inf in distance[:, :, :2]:
        next_distance = np.full(distance.shape, np.inf)
        next_distance[distance <= cur_value] = cur_value + 1

        for axis, offset in OFFSETS:
            offset_distance = np.roll(next_distance, offset, axis=axis)
            if axis == 0 and offset == 1:
                offset_distance[0, :, :] = np.inf
            elif axis == 0 and offset == -1:
                offset_distance[-1, :, :] = np.inf
            elif axis == 1 and offset == 1:
                offset_distance[:, 0, :] = np.inf
            elif axis == 1 and offset == -1:
                offset_distance[:, -1, :] = np.inf
            distance = np.minimum(distance, offset_distance)
        for offset in DIAG_OFFSETS:
            offset_distance = np.roll(next_distance, offset, axis=(0, 1)) * np.sqrt(2)
            if offset[0] == 1:
                offset_distance[0, :, :] = np.inf
            if offset[0] == -1:
                offset_distance[-1, :, :] = np.inf
            if offset[1] == 1:
                offset_distance[:, 0, :] = np.inf
            if offset[1] == -1:
                offset_distance[:, -1, :] = np.inf
            distance = np.minimum(distance, offset_distance)

        cur_value += 1

    distance = distance[:, :, 0] / (distance[:, :, 0] + distance[:, :, 1])

    return distance
