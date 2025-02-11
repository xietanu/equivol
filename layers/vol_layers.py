import itertools

import cv2
import numpy as np

import borders
import contours
import contours.find


class VolLayers:
    def __init__(self, border_contours: list[contours.Contour]):
        self.border_contours = border_contours

    @classmethod
    def from_depth_and_paired_contours(
        cls,
        mask: np.ndarray,
        depth_map: np.ndarray,
        layer_thicknesses: list[float] | np.ndarray,
        paired_contours: contours.PairedContours,
    ):
        if isinstance(layer_thicknesses, list):
            layer_thicknesses = np.array(layer_thicknesses)

        layer_quantiles = (
            np.array([0] + layer_thicknesses.cumsum().tolist())
            / layer_thicknesses.sum()
        )

        segment_masks = paired_contours.create_segment_mask(mask.shape)

        segment_quantiles = []

        for i in range(segment_masks.shape[2]):
            segment_distances = depth_map.copy()
            segment_distances[mask == 0] = 0
            segment_distances[mask == 2] = 1

            segment_quantiles.append(
                [
                    np.quantile(segment_distances[segment_masks[:, :, i]], ed)
                    for ed in layer_quantiles
                ]
            )

        segment_quantiles = (
            [segment_quantiles[0]] * 2 + segment_quantiles + [segment_quantiles[-1]] * 2
        )

        segment_quantiles = np.array(segment_quantiles)

        convolve_mat = [1 / 5] * 5

        smoothed_quantiles = []

        for i in range(len(layer_thicknesses) + 1):
            smoothed_quantiles.append(
                np.convolve(segment_quantiles[:, i], convolve_mat, mode="valid")
            )

        smoothed_quantiles = np.array(smoothed_quantiles).T

        new_mask = np.zeros_like(mask, dtype=np.int8)
        new_mask[mask == 3] = -1
        new_mask[mask == 1] = -1
        new_mask[mask == 2] = len(layer_thicknesses) + 1
        paired_contours.inner_contour.draw(new_mask, len(layer_thicknesses) + 1)

        for i in range(segment_masks.shape[2]):
            segment_distances = depth_map.copy()
            segment_distances[mask == 0] = 0
            segment_distances[mask == 2] = 1

            quantiles = smoothed_quantiles[i]
            for j, (q1, q2) in enumerate(itertools.pairwise(quantiles), start=1):
                new_mask[
                    np.logical_and(
                        segment_masks[:, :, i],
                        np.logical_and(segment_distances >= q1, segment_distances < q2),
                    )
                ] = j

        all_borders = borders.find_borders(new_mask, True)[:, :, 1:]

        layer_contours = [
            contours.find.longest(all_borders[:, :, i])
            for i in range(all_borders.shape[2])
        ]
        layer_contours[0] = paired_contours.outer_contour
        layer_contours[-1] = paired_contours.inner_contour

        return VolLayers(layer_contours)

    def smooth_inner_layers(self, degree: int):
        for contour in self.border_contours[1:-1]:
            contour.smooth(degree)

    def draw(self, canvas: np.ndarray):
        for i, (b1, b2) in enumerate(itertools.pairwise(self.border_contours), start=1):
            b1.draw(canvas, i)
            b2.draw(canvas, i)
            points = []
            for p in b1.vertices:
                points.append((p.col, p.row))

            if np.linalg.norm(
                np.array(b1.root.coord) - np.array(b2.tail.coord)
            ) > np.linalg.norm(np.array(b1.root.coord) - np.array(b2.root.coord)):
                b2.flip()

            for p in b2.vertices:
                points.append((p.col, p.row))

            cv2.fillPoly(canvas, np.array([points]), i)

        return canvas
