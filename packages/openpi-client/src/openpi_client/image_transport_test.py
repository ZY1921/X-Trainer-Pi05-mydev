import numpy as np
import pytest

from openpi_client import image_transport
from openpi_client import msgpack_numpy


def _test_image() -> np.ndarray:
    y, x = np.mgrid[:480, :640]
    image = np.empty((480, 640, 3), dtype=np.uint8)
    image[..., 0] = x * 255 // 639
    image[..., 1] = y * 255 // 479
    image[..., 2] = 32
    image[160:320, 240:400] = (240, 20, 20)
    return image


def test_jpeg_round_trip_preserves_shape_and_reduces_wire_bytes():
    image = _test_image()
    observation = {
        "observation.state": np.ones(2, dtype=np.float32),
        "observation.images.top": image,
    }

    encoded, encode_stats = image_transport.encode_observation_images(
        observation,
        codec="jpeg",
        jpeg_quality=90,
    )
    unpacked = msgpack_numpy.unpackb(msgpack_numpy.packb(encoded))
    decoded, decode_stats = image_transport.decode_observation_images(unpacked, codec="jpeg")

    decoded_image = decoded["observation.images.top"]
    assert decoded_image.shape == image.shape
    assert decoded_image.dtype == np.uint8
    assert encode_stats["wire_image_bytes"] < encode_stats["raw_image_bytes"] / 4
    assert decode_stats["wire_image_bytes"] == encode_stats["wire_image_bytes"]
    assert np.mean(np.abs(decoded_image.astype(np.int16) - image.astype(np.int16))) < 5
    np.testing.assert_array_equal(decoded["observation.state"], observation["observation.state"])


def test_raw_transport_keeps_image_array_unchanged():
    image = _test_image()
    observation = {"observation.images.top": image}

    encoded, encode_stats = image_transport.encode_observation_images(
        observation,
        codec="raw",
        jpeg_quality=90,
    )
    decoded, decode_stats = image_transport.decode_observation_images(encoded, codec="raw")

    assert encoded["observation.images.top"] is image
    assert decoded["observation.images.top"] is image
    assert encode_stats["wire_image_bytes"] == image.nbytes
    assert decode_stats["raw_image_bytes"] == image.nbytes


def test_rejects_invalid_jpeg_quality():
    with pytest.raises(ValueError, match="JPEG quality"):
        image_transport.encode_observation_images({}, codec="jpeg", jpeg_quality=101)
