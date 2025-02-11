import numpy as np

import masks

OFFSETS = [(0, 1), (1, 1), (0, -1), (1, -1)]


def find_borders(
    mask: np.ndarray,
    allow_higher: bool = False,
):
    """Find the adjacent borders of a mask.
    An adjacent borders is defined the borders between two areas where the difference in the value of those areas
    is 1.

    Args:
        mask (np.ndarray): Histological mask as a 2D numpy array.
    Returns:
        np.ndarray: A boolean mask where True indicates an adjacent border.
        The first two dimensions of the border mask match the input mask, and the third dimension is the
        index of the border.
    """
    assert masks.is_valid_mask(mask)

    n_borders = len(np.unique(mask)) - 1
    borders = np.zeros((mask.shape[0], mask.shape[1], n_borders), dtype=bool)

    for axis, offset in OFFSETS:
        offset_mask = np.roll(mask, offset, axis=axis)
        if axis == 0 and offset == 1:
            offset_mask[0, :] = 0
        elif axis == 0 and offset == -1:
            offset_mask[-1, :] = 0
        elif axis == 1 and offset == 1:
            offset_mask[:, 0] = 0
        elif axis == 1 and offset == -1:
            offset_mask[:, -1] = 0

        for i in range(n_borders):
            val = np.unique(mask)[i]
            if allow_higher:
                borders[(mask == val) & (offset_mask >= val + 1), i] = True
            else:
                borders[(mask == val) & (offset_mask == val + 1), i] = True

    return borders
