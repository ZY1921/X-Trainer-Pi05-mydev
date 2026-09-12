"""Preprocessed real-robot environments shared by all inference clients."""

from typing_extensions import override

from examples.xtrainer_real import camera_visualization as _camera_visualization
from examples.xtrainer_real import env as _bimanual_env
from examples.xtrainer_real import image_preprocessing as _image_preprocessing
from examples.xtrainer_real import right_arm_env as _right_arm_env

BIMANUAL_ACTION_DIM = 14


class BimanualInferenceEnvironment(_bimanual_env.XTrainerRealEnvironment):
    """14D environment with the canonical three-camera preprocessing pipeline."""

    def __init__(self, *, show_camera_preview: bool = False, **kwargs) -> None:
        super().__init__(**kwargs)
        self._camera_visualizer = _camera_visualization.CameraVisualizer(
            enabled=show_camera_preview,
            window_name="X-Trainer bimanual: top | left wrist | right wrist",
            cameras=(("top", "TOP"), ("left_wrist", "LEFT WRIST"), ("right_wrist", "RIGHT WRIST")),
        )

    @override
    def get_observation(self) -> dict:
        observation = super().get_observation()
        _image_preprocessing.preprocess_inference_images(observation)
        self._camera_visualizer.show(observation)
        return observation

    @override
    def close(self) -> None:
        self._camera_visualizer.close()
        super().close()


class RightArmInferenceEnvironment(_right_arm_env.XTrainerRightArmRealEnvironment):
    """7D environment with the canonical two-camera preprocessing pipeline."""

    def __init__(self, *, show_camera_preview: bool = False, **kwargs) -> None:
        super().__init__(**kwargs)
        self._camera_visualizer = _camera_visualization.CameraVisualizer(
            enabled=show_camera_preview,
            window_name="X-Trainer right arm: top | right wrist",
            cameras=(("top", "TOP"), ("right_wrist", "RIGHT WRIST")),
        )

    @override
    def get_observation(self) -> dict:
        observation = super().get_observation()
        _image_preprocessing.preprocess_inference_images(observation)
        self._camera_visualizer.show(observation)
        return observation

    @override
    def close(self) -> None:
        self._camera_visualizer.close()
        super().close()
