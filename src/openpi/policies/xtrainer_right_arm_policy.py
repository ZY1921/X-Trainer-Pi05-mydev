"""Transforms for the right-arm-only X-Trainer policy interface."""

import dataclasses
from typing import ClassVar

import einops
import numpy as np

from openpi import transforms

PHYSICAL_ACTION_DIM = 7
MODEL_RIGHT_ARM_START = 7
MODEL_RIGHT_ARM_END = 14


def make_xtrainer_right_arm_example() -> dict:
    return {
        "observation.state": np.ones((PHYSICAL_ACTION_DIM,), dtype=np.float32),
        "observation.images.top": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation.images.right_wrist": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "prompt": "do something",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.ndim == 3 and image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"Expected an RGB image, got shape {image.shape}")
    return image


def _require_last_dim(values: np.ndarray, expected: int, name: str) -> np.ndarray:
    values = np.asarray(values)
    if values.ndim == 0 or values.shape[-1] != expected:
        raise ValueError(f"Expected {name} last dimension {expected}, got shape {values.shape}")
    return values


@dataclasses.dataclass(frozen=True)
class XTrainerRightArmInputs(transforms.DataTransformFn):
    """Convert physical 7D right-arm observations into the common policy schema."""

    EXPECTED_CAMERAS: ClassVar[tuple[str, ...]] = ("top", "right_wrist")

    def __call__(self, data: dict) -> dict:
        state = _require_last_dim(data["observation.state"], PHYSICAL_ACTION_DIM, "observation.state")
        images = self._extract_images(data)
        unexpected = set(images) - set(self.EXPECTED_CAMERAS)
        if unexpected:
            raise ValueError(f"Unexpected cameras for right-arm policy: {tuple(sorted(unexpected))}")
        missing = set(self.EXPECTED_CAMERAS) - set(images)
        if missing:
            raise ValueError(f"Missing required right-arm cameras: {tuple(sorted(missing))}")

        top_image = _parse_image(images["top"])
        right_wrist_image = _parse_image(images["right_wrist"])
        inputs = {
            "state": state,
            "image": {
                "base_0_rgb": top_image,
                "left_wrist_0_rgb": np.zeros_like(top_image),
                "right_wrist_0_rgb": right_wrist_image,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.False_,
                "right_wrist_0_rgb": np.True_,
            },
        }

        if "actions" in data:
            inputs["actions"] = _require_last_dim(data["actions"], PHYSICAL_ACTION_DIM, "actions")
        if "prompt" in data:
            prompt = data["prompt"]
            inputs["prompt"] = prompt.decode("utf-8") if isinstance(prompt, bytes) else prompt
        return inputs

    def _extract_images(self, data: dict) -> dict[str, np.ndarray]:
        images: dict[str, np.ndarray] = {}
        if "images" in data:
            images.update(data["images"])
        prefix = "observation.images."
        for key, value in data.items():
            if key.startswith(prefix):
                images[key.removeprefix(prefix)] = value
        return images


@dataclasses.dataclass(frozen=True)
class PackRightArmToModelSlots(transforms.DataTransformFn):
    """Place normalized physical right-arm values in model dimensions 7:14."""

    def __call__(self, data: dict) -> dict:
        data["state"] = self._pack(data["state"], "state")
        if "actions" in data:
            data["actions"] = self._pack(data["actions"], "actions")
        return data

    @staticmethod
    def _pack(values: np.ndarray, name: str) -> np.ndarray:
        values = _require_last_dim(values, PHYSICAL_ACTION_DIM, name)
        pad_width = [(0, 0)] * values.ndim
        pad_width[-1] = (MODEL_RIGHT_ARM_START, 0)
        packed = np.pad(values, pad_width, constant_values=0.0)
        if packed.shape[-1] != MODEL_RIGHT_ARM_END:
            raise AssertionError(f"Unexpected packed {name} shape: {packed.shape}")
        return packed


@dataclasses.dataclass(frozen=True)
class UnpackRightArmFromModelSlots(transforms.DataTransformFn):
    """Extract physical right-arm state/actions from model dimensions 7:14."""

    def __call__(self, data: dict) -> dict:
        if "state" in data:
            data["state"] = self._unpack(data["state"], "state")
        if "actions" in data:
            data["actions"] = self._unpack(data["actions"], "actions")
        return data

    @staticmethod
    def _unpack(values: np.ndarray, name: str) -> np.ndarray:
        values = np.asarray(values)
        if values.ndim == 0 or values.shape[-1] < MODEL_RIGHT_ARM_END:
            raise ValueError(
                f"Expected model {name} last dimension at least {MODEL_RIGHT_ARM_END}, got shape {values.shape}"
            )
        return np.ascontiguousarray(values[..., MODEL_RIGHT_ARM_START:MODEL_RIGHT_ARM_END])


@dataclasses.dataclass(frozen=True)
class XTrainerRightArmOutputs(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        actions = _require_last_dim(data["actions"], PHYSICAL_ACTION_DIM, "actions")
        return {"actions": actions}
