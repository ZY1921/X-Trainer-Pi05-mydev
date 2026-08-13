"""Image codecs for reducing websocket observation payloads."""

import io
import time
from typing import Any, Dict, Tuple

import numpy as np
from PIL import Image


_IMAGE_KEY_PREFIX = "observation.images."
_ENCODED_IMAGE_MARKER = "__openpi_encoded_image__"


def encode_observation_images(
    observation: Dict[str, Any],
    *,
    codec: str,
    jpeg_quality: int,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Encode observation images while leaving non-image fields unchanged."""
    _validate_codec(codec)
    _validate_jpeg_quality(jpeg_quality)

    start_time = time.monotonic()
    encoded_observation = dict(observation)
    raw_image_bytes = 0
    wire_image_bytes = 0
    image_count = 0

    for key, value in observation.items():
        if not key.startswith(_IMAGE_KEY_PREFIX):
            continue

        image = _validate_rgb_image(value, key)
        image_count += 1
        raw_image_bytes += image.nbytes

        if codec == "raw":
            wire_image_bytes += image.nbytes
            continue

        buffer = io.BytesIO()
        Image.fromarray(image).save(
            buffer,
            format="JPEG",
            quality=jpeg_quality,
        )
        payload = buffer.getvalue()
        wire_image_bytes += len(payload)
        encoded_observation[key] = {
            _ENCODED_IMAGE_MARKER: True,
            "codec": "jpeg",
            "shape": image.shape,
            "data": payload,
        }

    elapsed_ms = (time.monotonic() - start_time) * 1000
    return encoded_observation, {
        "codec": codec,
        "image_count": image_count,
        "raw_image_bytes": raw_image_bytes,
        "wire_image_bytes": wire_image_bytes,
        "encode_ms": elapsed_ms,
    }


def decode_observation_images(
    observation: Dict[str, Any],
    *,
    codec: str,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Decode wire-format images back to RGB uint8 arrays."""
    _validate_codec(codec)

    start_time = time.monotonic()
    decoded_observation = dict(observation)
    raw_image_bytes = 0
    wire_image_bytes = 0
    image_count = 0

    for key, value in observation.items():
        if not key.startswith(_IMAGE_KEY_PREFIX):
            continue
        image_count += 1

        if codec == "raw":
            image = _validate_rgb_image(value, key)
            raw_image_bytes += image.nbytes
            wire_image_bytes += image.nbytes
            continue

        if not isinstance(value, dict) or value.get(_ENCODED_IMAGE_MARKER) is not True:
            raise TypeError(f"Expected JPEG wire envelope for observation image {key!r}.")
        if value.get("codec") != "jpeg":
            raise ValueError(f"Unsupported encoded image codec for {key!r}: {value.get('codec')!r}.")
        payload = value.get("data")
        if not isinstance(payload, bytes):
            raise TypeError(f"Expected JPEG bytes for observation image {key!r}.")

        try:
            with Image.open(io.BytesIO(payload)) as encoded_image:
                image = np.ascontiguousarray(np.asarray(encoded_image.convert("RGB"), dtype=np.uint8))
        except Exception as error:
            raise ValueError(f"Failed to JPEG-decode observation image {key!r}.") from error

        declared_shape = value.get("shape")
        if not isinstance(declared_shape, (list, tuple)):
            raise TypeError(f"Missing JPEG image shape for observation image {key!r}.")
        expected_shape = tuple(int(dimension) for dimension in declared_shape)
        if image.shape != expected_shape:
            raise ValueError(
                f"Decoded observation image {key!r} has shape {image.shape}, expected {expected_shape}."
            )

        decoded_observation[key] = image
        raw_image_bytes += image.nbytes
        wire_image_bytes += len(payload)

    elapsed_ms = (time.monotonic() - start_time) * 1000
    return decoded_observation, {
        "codec": codec,
        "image_count": image_count,
        "raw_image_bytes": raw_image_bytes,
        "wire_image_bytes": wire_image_bytes,
        "decode_ms": elapsed_ms,
    }


def _validate_rgb_image(value: Any, key: str) -> np.ndarray:
    image = np.asarray(value)
    if image.dtype != np.uint8:
        raise TypeError(f"Observation image {key!r} must have dtype uint8, got {image.dtype}.")
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"Observation image {key!r} must have shape (height, width, 3), got {image.shape}.")
    return image


def _validate_codec(codec: str) -> None:
    if codec not in ("raw", "jpeg"):
        raise ValueError(f"Unsupported image transport codec: {codec!r}.")


def _validate_jpeg_quality(jpeg_quality: int) -> None:
    if not 1 <= jpeg_quality <= 100:
        raise ValueError(f"JPEG quality must be in [1, 100], got {jpeg_quality}.")
