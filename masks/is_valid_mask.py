import numpy as np


def is_valid_mask(mask: np.ndarray) -> (bool, str):
    """Check if the given mask is valid for image segmentation.

    Must be a 2D uint8 numpy array with values only including 0,1,2,3.

    Args:
        mask (np.ndarray): A 2D numpy array representing the mask.
    Returns:
        bool: True if the mask is valid, False otherwise.
        msg: A message indicating the reason for invalidity if applicable.
    """
    if not isinstance(mask, np.ndarray) or mask.ndim != 2:
        return False, "Input must be a 2D numpy array."
    if mask.dtype != np.uint8:
        return False, "Mask must be of dtype uint8."
    if not np.isin(mask.flatten(), [0, 1, 2, 3]).all():
        return False, "Invalid value found in mask. Mask values must be 0, 1, 2, or 3."
    return True, ""
