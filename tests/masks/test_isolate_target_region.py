import cv2
import numpy as np
import pytest

import masks

test_data = [
    (
        cv2.imread(f"tests/example_imgs/{i}.png", cv2.IMREAD_UNCHANGED),
        cv2.imread(f"tests/example_imgs/{i}_isolated.png", cv2.IMREAD_UNCHANGED),
    )
    for i in range(1, 6)
]


@pytest.mark.parametrize("image, expected", test_data)
def test_isolate_target_region(image, expected):
    # Perform isolation
    isolated_image = masks.isolate_target_region(image)

    # Compare the isolated image with the expected result
    assert np.all(isolated_image == expected)
