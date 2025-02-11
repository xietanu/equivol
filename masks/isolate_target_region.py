import numpy as np
from skimage.measure import label

import masks


def isolate_target_region(mask: np.ndarray) -> np.ndarray:
    """Isolate the largest connected target of the masks.

    Args:
        mask: (np.ndarray): Histological mask as a 2D numpy array.
    Returns:
        isolated_mask: (np.ndarray): Isolated target region as a 2D numpy array.
    """
    assert masks.is_valid_mask(mask)

    isolation_mask = np.zeros_like(mask)
    isolation_mask[mask == 0] = 0
    isolation_mask[mask == 1] = 1
    isolation_mask[mask >= 2] = 0

    labelled = label(isolation_mask)

    biggest = 0
    biggest_size = 0

    for l in np.unique(labelled):
        size = np.sum(labelled == l)
        if l != 0 and size > biggest_size:
            biggest = l
            biggest_size = size

    isolated_mask = np.zeros_like(mask)
    isolated_mask[labelled == biggest] = 1
    isolated_mask[labelled != biggest] = 3
    isolated_mask[mask == 0] = 0
    isolated_mask[mask == 2] = 2

    return isolated_mask
