import numpy as np

import masks


def test_is_valid_mask_true():
    inpt = np.array([[0, 1], [1, 2], [2, 3]], dtype=np.uint8)
    assert masks.is_valid_mask(inpt)[0] is True


def test_is_valid_mask_not_2d():
    inpt = np.array([0, 1, 1, 2, 2, 3], dtype=np.uint8)
    result, msg = masks.is_valid_mask(inpt)
    assert masks.is_valid_mask(inpt)[0] is False, "Returns valid"
    assert "2d numpy array" in msg.lower(), "Message doesn't mention 2D numpy array"


def test_is_valid_mask_not_numpy():
    inpt = [[0, 1], [1, 2], [2, 3]]
    result, msg = masks.is_valid_mask(inpt)
    assert masks.is_valid_mask(inpt)[0] is False, "Returns valid"
    assert "2d numpy array" in msg.lower(), "Message doesn't mention 2D numpy array"


def test_is_valid_mask_not_uint8():
    inpt = np.array([[0, 1], [1, 2], [2, 3]], dtype=np.int8)
    result, msg = masks.is_valid_mask(inpt)
    assert masks.is_valid_mask(inpt)[0] is False, "Returns valid"
    assert "dtype uint8" in msg.lower(), "Message doesn't mention uint8"


def test_is_valid_mask_invalid_numbers():
    inpt = np.array([[0, 1], [1, 4], [2, 3]], dtype=np.uint8)
    result, msg = masks.is_valid_mask(inpt)
    assert masks.is_valid_mask(inpt)[0] is False, "Returns valid"
    assert "invalid value" in msg.lower(), "Message doesn't mention invalid value"
