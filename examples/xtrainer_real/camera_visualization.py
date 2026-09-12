"""Optional OpenCV camera preview shared by X-Trainer inference clients."""

from collections.abc import Sequence

import cv2
import numpy as np


class CameraPreviewStopRequestedError(Exception):
    """Raised when the operator asks to stop through the preview window."""


class CameraVisualizer:
    """Display preprocessed RGB observations without modifying model inputs."""

    def __init__(
        self,
        *,
        enabled: bool,
        window_name: str,
        cameras: Sequence[tuple[str, str]],
    ) -> None:
        self._enabled = enabled
        self._window_name = window_name
        self._cameras = tuple(cameras)
        self._window_created = False

    def show(self, observation: dict) -> None:
        if not self._enabled:
            return

        preview_frames = []
        for camera_name, label in self._cameras:
            image_key = f"observation.images.{camera_name}"
            if image_key not in observation:
                raise KeyError(f"Missing camera image for preview: {image_key}")
            rgb = np.asarray(observation[image_key])
            if rgb.ndim != 3 or rgb.shape[-1] != 3:
                raise ValueError(f"Expected {image_key} with shape (height, width, 3), got {rgb.shape}.")

            # Work on a BGR copy so labels and OpenCV display never mutate the
            # RGB arrays that are sent to the policy server.
            bgr = np.ascontiguousarray(rgb[..., ::-1])
            cv2.putText(bgr, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3)
            cv2.putText(bgr, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
            preview_frames.append(bgr)

        preview = np.hstack(preview_frames)
        if not self._window_created:
            cv2.namedWindow(self._window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self._window_name, preview.shape[1], preview.shape[0])
            self._window_created = True

        cv2.imshow(self._window_name, preview)
        key = cv2.waitKey(1) & 0xFF
        try:
            window_visible = cv2.getWindowProperty(self._window_name, cv2.WND_PROP_VISIBLE)
        except cv2.error:
            window_visible = -1
        if key in (ord("q"), ord("Q"), 27) or window_visible < 1:
            raise CameraPreviewStopRequestedError

    def close(self) -> None:
        if not self._window_created:
            return
        try:
            cv2.destroyWindow(self._window_name)
            cv2.waitKey(1)
        except cv2.error:
            pass
        self._window_created = False
