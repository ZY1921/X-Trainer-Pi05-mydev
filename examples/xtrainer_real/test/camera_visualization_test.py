import cv2
import numpy as np
import pytest

from examples.xtrainer_real import camera_visualization


def _observation() -> dict:
    return {
        "observation.images.top": np.full((24, 32, 3), (10, 20, 30), dtype=np.uint8),
        "observation.images.right_wrist": np.full((24, 32, 3), (40, 50, 60), dtype=np.uint8),
    }


def test_disabled_preview_does_not_touch_observation_or_opencv(monkeypatch):
    observation = _observation()
    original = {key: value.copy() for key, value in observation.items()}
    monkeypatch.setattr(cv2, "imshow", lambda *_args: pytest.fail("disabled preview called imshow"))

    visualizer = camera_visualization.CameraVisualizer(
        enabled=False,
        window_name="test",
        cameras=(("top", "TOP"), ("right_wrist", "RIGHT WRIST")),
    )
    visualizer.show(observation)

    for key in observation:
        np.testing.assert_array_equal(observation[key], original[key])


def test_enabled_preview_displays_bgr_copies_and_preserves_model_images(monkeypatch):
    observation = _observation()
    original = {key: value.copy() for key, value in observation.items()}
    displayed = []

    monkeypatch.setattr(cv2, "putText", lambda image, *_args: image)
    monkeypatch.setattr(cv2, "namedWindow", lambda *_args: None)
    monkeypatch.setattr(cv2, "resizeWindow", lambda *_args: None)
    monkeypatch.setattr(cv2, "imshow", lambda _name, image: displayed.append(image.copy()))
    monkeypatch.setattr(cv2, "waitKey", lambda _delay: -1)
    monkeypatch.setattr(cv2, "getWindowProperty", lambda *_args: 1.0)

    visualizer = camera_visualization.CameraVisualizer(
        enabled=True,
        window_name="test",
        cameras=(("top", "TOP"), ("right_wrist", "RIGHT WRIST")),
    )
    visualizer.show(observation)

    assert displayed[0].shape == (24, 64, 3)
    np.testing.assert_array_equal(displayed[0][:, :32], original["observation.images.top"][..., ::-1])
    np.testing.assert_array_equal(displayed[0][:, 32:], original["observation.images.right_wrist"][..., ::-1])
    for key in observation:
        np.testing.assert_array_equal(observation[key], original[key])


def test_preview_stop_key_requests_client_shutdown(monkeypatch):
    observation = _observation()
    monkeypatch.setattr(cv2, "putText", lambda image, *_args: image)
    monkeypatch.setattr(cv2, "namedWindow", lambda *_args: None)
    monkeypatch.setattr(cv2, "resizeWindow", lambda *_args: None)
    monkeypatch.setattr(cv2, "imshow", lambda *_args: None)
    monkeypatch.setattr(cv2, "waitKey", lambda _delay: ord("q"))
    monkeypatch.setattr(cv2, "getWindowProperty", lambda *_args: 1.0)

    visualizer = camera_visualization.CameraVisualizer(
        enabled=True,
        window_name="test",
        cameras=(("top", "TOP"),),
    )
    with pytest.raises(camera_visualization.CameraPreviewStopRequestedError):
        visualizer.show(observation)
