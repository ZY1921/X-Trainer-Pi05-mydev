"""Camera preprocessing shared by the X-Trainer inference clients."""

import cv2
import numpy as np

_TOP_SOURCE_SHAPE = (480, 640)
_TOP_CROP_ROWS = slice(150, 420)
_TOP_CROP_COLUMNS = slice(220, 480)


def match_legacy_dataset_images(observation: dict) -> dict:
    """Match the camera orientation and top-camera crop used during data collection."""
    for camera_name in ("top", "right_wrist"):
        image_key = f"observation.images.{camera_name}"
        observation[image_key] = np.ascontiguousarray(observation[image_key][::-1, ::-1])

    top_key = "observation.images.top"
    top = observation[top_key]
    if top.shape[:2] != _TOP_SOURCE_SHAPE:
        raise ValueError(
            "Legacy dataset image preprocessing requires top-camera observations "
            f"with shape (480, 640, 3), got {top.shape}. Start the client with "
            "--render-height 480 --render-width 640."
        )

    # The data-collection client cropped this 260x270 ROI and stretched it to
    # 640x480. Keep the RGB channel order unchanged for the inference server.
    top = top[_TOP_CROP_ROWS, _TOP_CROP_COLUMNS]
    observation[top_key] = np.ascontiguousarray(
        cv2.resize(top, (640, 480), interpolation=cv2.INTER_LINEAR)
    )
    return observation
