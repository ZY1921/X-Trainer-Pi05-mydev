import cv2
import numpy as np
import pytest

from examples.xtrainer_real import image_preprocessing


def test_preprocessing_matches_latest_collection_images():
    top = np.arange(480 * 640 * 3, dtype=np.uint32).reshape(480, 640, 3).astype(np.uint8)
    left_wrist = np.full((480, 640, 3), 17, dtype=np.uint8)
    right_wrist = np.arange(480 * 640 * 3, dtype=np.uint32).reshape(480, 640, 3).astype(np.uint8)
    observation = {
        "observation.images.top": top.copy(),
        "observation.images.left_wrist": left_wrist.copy(),
        "observation.images.right_wrist": right_wrist.copy(),
    }

    result = image_preprocessing.preprocess_inference_images(observation)

    expected_top = cv2.resize(top[:430, :], (640, 480), interpolation=cv2.INTER_LINEAR)
    np.testing.assert_array_equal(result["observation.images.top"], expected_top)
    np.testing.assert_array_equal(result["observation.images.left_wrist"], left_wrist)
    np.testing.assert_array_equal(result["observation.images.right_wrist"], right_wrist[::-1, ::-1])


def test_preprocessing_rejects_wrong_source_resolution():
    observation = {
        "observation.images.top": np.zeros((224, 224, 3), dtype=np.uint8),
        "observation.images.right_wrist": np.zeros((224, 224, 3), dtype=np.uint8),
    }

    with pytest.raises(ValueError, match=r"shape \(480, 640, 3\)"):
        image_preprocessing.preprocess_inference_images(observation)
