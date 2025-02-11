import numpy as np

import contours
import borders


def longest(border: np.ndarray) -> contours.Contour:
    all_contours = []
    trimmed_border = border
    while np.sum(trimmed_border) / np.sum(border) > 0.05:
        new_contour = contours.Contour.from_border(trimmed_border)
        trimmed_border = borders.remove_contour(trimmed_border, new_contour)
        all_contours.append(new_contour)

    return max(all_contours, key=lambda c: len(c))
