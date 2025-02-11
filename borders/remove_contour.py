import numpy as np

import contours


def remove_contour(border: np.ndarray, contour: contours.Contour) -> np.ndarray:
    """
    Removes a given contour from the specified border set at a specific layer.

    Args:
        border (np.ndarray): The 2D boolean array representing the border set.
        contour (contours.Contour): The contour to be removed.

    Returns:
        np.ndarray: The updated border set with the specified contour removed.
    """
    border_set = border.copy()

    for vertex in contour:
        border_set[vertex.row, vertex.col] = False

    return border_set
