# equivol
A simple implementation of [equi-volumetric layering](https://layerfmri.com/2020/04/24/equivol/)
with minimal dependencies.

See example.ipynb for examples of usage on segmentations of grey matter from the
[BigBrain](https://bigbrainproject.org/).

## Usage

The package contains one function intended for use by users:

### ev_segment

```
ev_segment(
    mask: np.ndarray,
    layer_thicknesses: list[float] | np.ndarray,
    scale: float,
    smoothing_degree: int,
    report: bool
)
```


Creates an equivolumetric layering segmentation of a histological masks.

#### Args:
- mask (np.ndarray): Histological mask as a 2D numpy array. 
  - 0 indicates area 'above' the area to be segmented. 
  - 1 indicates area to be segmented. 
  - 2 indicates area 'below' the area to be segmented. 
  - 3 indicates area to ignore. 
  - E.g. for cortical layers, 0 would be background, 1 is the grey matter, 2 is the white matter. 
- layer_thicknesses (list[float] | np.ndarray): Expected (volumetric) thickness of each layer. 
  - Unit is unimportant as long as they are consistent.
  - Layers should be listed in descending order. 
- scale (float, optional): Scaling factor calculating contours. Defaults to 1.
  - A higher scale usually leads to better results but is also slower. 
- smoothing_degree (int, optional): Smoothing degree for the final contours. Defaults to 30. 
- report (bool, optional): If True, prints messages regarding progress on intermediate steps. Defaults to False.

#### Returns:
- np.ndarray: Segmented equivolumetric mask. 
  - 0 indicates area 'above' the segmented area. 
  - 1-n indicates the n equivolumetric layers. 
  - n+1 indicates area 'below' the area to be segmented. 
  - n+2 indicates area not segmented.

## Dependencies
- numpy
- opencv
- scikit-image
