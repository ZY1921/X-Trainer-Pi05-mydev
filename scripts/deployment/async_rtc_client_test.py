# ruff: noqa: SLF001
# Deployment protocol integration tests for the asynchronous RTC client.

import numpy as np
from openpi_client import image_transport
import pytest

from examples.xtrainer_real import async_rtc_main


class _FakeWorker:
    def __init__(self) -> None:
        self.metadata = {
            "async_rtc": {
                "protocol": async_rtc_main.PROTOCOL_NAME,
                "protocol_version": async_rtc_main.PROTOCOL_VERSION,
                "model_action_horizon": 10,
                "model_action_dim": 2,
                "rtc_supported": True,
            }
        }
        self.submitted = []
        self.ready_responses = []

    def submit(self, task):
        self.submitted.append(task)

    def wait_for_response(self, timeout_s):
        del timeout_s
        task = self.submitted.pop(0)
        return self.make_response(task, base=100.0)

    def get_response_nowait(self):
        return self.ready_responses.pop(0) if self.ready_responses else None

    @staticmethod
    def make_response(task, *, base):
        actions = np.arange(base, base + 10, dtype=np.float32)[:, None]
        return async_rtc_main.InferenceResponse(
            task=task,
            result={"actions": actions},
            round_trip_ms=25.0,
        )


def _make_agent(worker, *, rtc_enabled=True, max_delay=4):
    return async_rtc_main.AsyncRTCPolicyAgent(
        worker,
        request_interval=4,
        rtc_enabled=rtc_enabled,
        max_inference_delay_steps=max_delay,
        inference_timeout_s=1.0,
        warmup_timeout_s=2.0,
        warmup_rtc=False,
        debug_timing=False,
    )


def test_worker_encodes_jpeg_observations_in_request_envelope():
    worker = object.__new__(async_rtc_main.AsyncRTCInferenceWorker)
    worker._rtc_inference_delay = 2
    worker._rtc_execution_horizon = 4
    worker._rtc_prefix_attention_schedule = "exp"
    worker._rtc_max_guidance_weight = 10.0
    worker._image_transport_codec = "jpeg"
    worker._image_jpeg_quality = 90
    image = np.zeros((48, 64, 3), dtype=np.uint8)
    task = async_rtc_main.InferenceTask(
        request_id=3,
        episode_id=1,
        request_step=4,
        observation={"observation.images.top": image},
        prev_chunk_left_over=None,
        timeout_s=1.0,
    )

    request, stats = worker._make_request(task)
    decoded, _ = image_transport.decode_observation_images(request["observation"], codec="jpeg")

    assert request["image_transport"] == {"codec": "jpeg", "jpeg_quality": 90}
    assert isinstance(request["observation"]["observation.images.top"]["data"], bytes)
    assert stats["wire_image_bytes"] < stats["raw_image_bytes"]
    assert decoded["observation.images.top"].shape == image.shape


def test_rtc_request_runs_in_background_and_switches_using_actual_delay():
    worker = _FakeWorker()
    agent = _make_agent(worker)
    agent.reset()
    observation = {"state": np.zeros(2, dtype=np.float32)}

    assert agent.get_action(observation)["actions"].item() == 100
    assert agent.get_action(observation)["actions"].item() == 101
    assert agent.get_action(observation)["actions"].item() == 102
    assert agent.get_action(observation)["actions"].item() == 103
    assert agent.get_action(observation)["actions"].item() == 104

    online_task = worker.submitted[0]
    np.testing.assert_array_equal(online_task.prev_chunk_left_over[:, 0], np.arange(104, 110))
    worker.ready_responses.append(worker.make_response(online_task, base=200.0))

    # The response is accepted one control step after request_step=4, so index 1
    # is the first non-stale action from the new chunk.
    assert agent.get_action(observation)["actions"].item() == 201


def test_non_rtc_baseline_uses_the_same_async_switch_timing():
    worker = _FakeWorker()
    agent = _make_agent(worker, rtc_enabled=False)
    agent.reset()
    observation = {"state": np.zeros(2, dtype=np.float32)}

    for _ in range(5):
        agent.get_action(observation)
    online_task = worker.submitted[0]
    assert online_task.prev_chunk_left_over is None
    worker.ready_responses.append(worker.make_response(online_task, base=300.0))

    assert agent.get_action(observation)["actions"].item() == 301


def test_aborts_when_async_result_exceeds_step_deadline():
    worker = _FakeWorker()
    agent = _make_agent(worker, max_delay=1)
    agent.reset()
    observation = {"state": np.zeros(2, dtype=np.float32)}

    for _ in range(6):
        agent.get_action(observation)
    with pytest.raises(TimeoutError, match="max-inference-delay-steps"):
        agent.get_action(observation)
