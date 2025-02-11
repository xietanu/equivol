import cv2
import numpy as np

import borders
import contours
import contours.find
import layers
import masks


def ev_segment(
    mask: np.ndarray,
    layer_thicknesses: list[float] | np.ndarray,
    scale: float = 1,
    smoothing_degree: int = 30,
    report=False,
):
    """Creates an equivolumetric segmentation of a histological masks.

    Args:
        mask (np.ndarray): Histological mask as a 2D numpy array.
            0 indicates area 'above' the area to be segmented.
            1 indicates area to be segmented.
            2 indicates area 'below' the area to be segmented.
            3 indicates area to ignore.
            E.g. for cortical layers, 0 would be background, 1 is the grey matter, 2 is the white matter.
        layer_thicknesses (list[float] | np.ndarray): Expected (volumetric) thickness of each layer.
            Unit is unimportant as long as they are consistent. Layers should be listed in descending order.
        scale (float, optional): Scaling factor calculat ing contours. Defaults to 1.
            A higher scale usually leads to better results but is also slower.
        smoothing_degree (int, optional): Smoothing degree for the final contours. Defaults to 30.
        report (bool, optional): If True, prints messages regarding progress on intermediate steps. Defaults to False.
    Returns:
        np.ndarray: Segmented equivolumetric masks.
            0 indicates area 'above' the segmented area.
            1-n indicates the equivolumetric layers.
            n+1 indicates area 'below' the area to be segmented.
            n+2 indicates area not segmented.
    """
    assert masks.is_valid_mask(mask)

    if report:
        print("Step 1: Rescaling and filtering the mask...")
    isolated = masks.isolate_target_region(mask)
    rescaled_mask = cv2.resize(
        isolated, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST
    )
    mask_borders = borders.find_borders(rescaled_mask)

    if report:
        print("Step 2: Calculating distances from inner and outer edges...")
    if scale <= 1:
        distances = borders.map_distance_inside_target(rescaled_mask, mask_borders)
    else:
        sml_mask_borders = borders.find_borders(isolated)
        sml_distances = borders.map_distance_inside_target(isolated, sml_mask_borders)
        distances = cv2.resize(
            sml_distances,
            (rescaled_mask.shape[1], rescaled_mask.shape[0]),
            interpolation=cv2.INTER_CUBIC,
        )

    distances[rescaled_mask != 1] = np.inf

    if report:
        print("Step 3: Calculating contours for inner and outer surfaces...")
    outer_contour = contours.find.longest(mask_borders[:, :, 0])
    inner_contour = contours.find.longest(mask_borders[:, :, 1])

    if report:
        print("Step 4: Pairing the contours...")
    paired = contours.PairedContours.from_contours(
        outer_contour,
        inner_contour,
        rescaled_mask,
        min_step_size=10,
        decross_links=True,
    )

    if report:
        print("Step 5: Calculating equivolumetric masks...")

    vol_layers = layers.VolLayers.from_depth_and_paired_contours(
        rescaled_mask, distances, layer_thicknesses, paired
    )
    vol_layers.smooth_inner_layers(smoothing_degree)

    output = vol_layers.draw(np.zeros_like(rescaled_mask))
    inner = len(layer_thicknesses) + 1
    padding = len(layer_thicknesses) + 2
    output[(rescaled_mask == 1) & (output == 0)] = padding
    output[rescaled_mask == 2] = inner
    output[rescaled_mask == 3] = padding

    output = cv2.resize(
        output, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_NEAREST
    )

    return output
