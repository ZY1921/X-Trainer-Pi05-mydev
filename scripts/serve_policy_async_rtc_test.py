import numpy as np
from openpi_client import image_transport
import pytest

from scripts import serve_policy_async_rtc


class _FakePolicy:
    def __init__(self) -> None:
        self.calls = []

    def infer(self, observation, **kwargs):
        self.calls.append((observation, kwargs))
        return {"actions": np.zeros((5, 2), dtype=np.float32)}


def _request(*, rtc=None):
    return {
        "protocol": serve_policy_async_rtc.PROTOCOL_NAME,
        "protocol_version": serve_policy_async_rtc.PROTOCOL_VERSION,
        "request_id": 7,
        "observation": {"state": np.ones(2, dtype=np.float32)},
        "image_transport": {"codec": "raw", "jpeg_quality": 90},
        "rtc": rtc,
    }


def test_baseline_request_calls_policy_without_rtc_kwargs():
    policy = _FakePolicy()
    response = serve_policy_async_rtc.run_inference_request(policy, _request())

    assert response["request_id"] == 7
    assert response["result"]["actions"].shape == (5, 2)
    assert response["result"]["server_timing"]["infer_ms"] >= 0
    assert policy.calls[0][1] == {}


def test_rtc_request_maps_protocol_fields_to_policy_kwargs():
    policy = _FakePolicy()
    previous = np.ones((4, 2), dtype=np.float32)
    rtc = {
        "prev_chunk_left_over": previous,
        "inference_delay": 2,
        "execution_horizon": 4,
        "prefix_attention_schedule": "exp",
        "max_guidance_weight": 10.0,
    }

    serve_policy_async_rtc.run_inference_request(policy, _request(rtc=rtc))

    kwargs = policy.calls[0][1]
    np.testing.assert_array_equal(kwargs["prev_chunk_left_over"], previous)
    assert kwargs["inference_delay"] == 2
    assert kwargs["execution_horizon"] == 4
    assert kwargs["rtc_prefix_attention_schedule"] == "exp"
    assert kwargs["rtc_max_guidance_weight"] == 10.0


def test_jpeg_request_is_decoded_before_policy_inference():
    policy = _FakePolicy()
    image = np.zeros((48, 64, 3), dtype=np.uint8)
    image[12:36, 16:48] = (220, 30, 20)
    wire_observation, _ = image_transport.encode_observation_images(
        {"observation.images.top": image},
        codec="jpeg",
        jpeg_quality=90,
    )
    request = _request()
    request["observation"] = wire_observation
    request["image_transport"] = {"codec": "jpeg", "jpeg_quality": 90}

    response = serve_policy_async_rtc.run_inference_request(policy, request)

    decoded = policy.calls[0][0]["observation.images.top"]
    assert decoded.shape == image.shape
    assert decoded.dtype == np.uint8
    assert response["result"]["server_timing"]["image_decode_ms"] >= 0
    assert response["result"]["image_transport"]["codec"] == "jpeg"


def test_rejects_incompatible_protocol():
    policy = _FakePolicy()
    request = _request()
    request["protocol_version"] = 999

    with pytest.raises(ValueError, match="Unsupported protocol"):
        serve_policy_async_rtc.run_inference_request(policy, request)
