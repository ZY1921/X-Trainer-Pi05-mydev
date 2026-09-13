# ruff: noqa: SLF001
import dataclasses
import queue
import threading
import time

import numpy as np
import pytest

from examples.xtrainer_real import bimanual_async_rtc_main as base_rtc
from examples.xtrainer_real.voice import rtc as voice_rtc


class FakeWorker:
    def __init__(self):
        self.metadata = {"async_rtc": {"model_action_horizon": 50, "rtc_supported": True}}
        self.submitted = []
        self.pending = []
        self.wait_error = None
        self.response_error = None
        self.wrong_session = False

    def submit(self, task):
        self.submitted.append(task)
        self.pending.append(task)

    def wait_for_response(self, _timeout):
        if self.wait_error:
            raise self.wait_error
        task = self.pending.pop(0)
        if self.wrong_session:
            task = dataclasses.replace(task, episode_id=-999)
        return base_rtc.InferenceResponse(task, {"actions": np.zeros((50, 7))}, round_trip_ms=1, error=self.response_error)

    def get_response_nowait(self):
        return None


def make_agent(worker):
    return voice_rtc.VoiceRightArmAsyncRTCPolicyAgent(
        worker,
        request_interval=1,
        rtc_enabled=True,
        max_inference_delay_steps=24,
        inference_timeout_s=3,
        warmup_timeout_s=600,
        warmup_rtc=True,
        debug_timing=False,
    )


def test_no_rtc_prefix_or_old_prompt_crosses_task_boundary():
    worker = FakeWorker()
    agent = make_agent(worker)
    agent.reset()
    agent.get_action({"prompt": "red"})
    agent.get_action({"prompt": "red"})
    old_request = worker.pending[0]
    assert old_request.prev_chunk_left_over is not None
    agent.finish_task(4)
    assert not worker.pending
    agent.set_action_recorder(None)
    agent.reset()
    agent.get_action({"prompt": "green"})
    new_initial = worker.submitted[-2]  # baseline, then RTC warmup
    assert new_initial.prev_chunk_left_over is None
    assert new_initial.observation["prompt"] == "green"
    assert new_initial.episode_id == old_request.episode_id + 1
    assert new_initial.request_id > old_request.request_id
    assert len({task.request_id for task in worker.submitted}) == len(worker.submitted)


@pytest.mark.parametrize("failure", ["timeout", "response_error", "wrong_session"])
def test_strict_task_finish_rejects_unusable_pending_result(failure):
    worker = FakeWorker()
    agent = make_agent(worker)
    agent.reset()
    agent.get_action({"prompt": "red"})
    agent.get_action({"prompt": "red"})
    if failure == "timeout":
        worker.wait_error = TimeoutError()
    elif failure == "response_error":
        worker.response_error = RuntimeError()
    else:
        worker.wrong_session = True
    with pytest.raises((RuntimeError, TimeoutError)):
        agent.finish_task(4)
    assert agent._active_actions is None


def test_worker_close_interrupts_connection_and_bounds_join():
    events = []

    class Connection:
        def close(self):
            events.append("close")

    class Thread:
        def join(self, timeout):
            events.append(timeout)

    worker = object.__new__(voice_rtc.VoiceAsyncRTCInferenceWorker)
    worker._voice_connection = Connection()
    worker._thread = Thread()
    worker._stop = threading.Event()
    worker._tasks = queue.Queue(maxsize=1)
    worker.close()
    assert worker._stop.is_set()
    assert events == ["close", 2.0]


@pytest.mark.parametrize("actions", [np.full((50, 7), float("nan")), np.empty((0, 7))])
def test_right_arm_rejects_bad_chunk_before_any_action(actions):
    agent = object.__new__(voice_rtc.VoiceRightArmAsyncRTCPolicyAgent)
    with pytest.raises(ValueError, match="finite actions"):
        agent._validated_result({"actions": actions})


def test_close_wakes_worker_blocked_on_long_warmup(monkeypatch):
    waiting = threading.Event()
    closed = threading.Event()

    class Connection:
        first = True

        def recv(self, timeout):
            if self.first:
                self.first = False
                return base_rtc.msgpack_numpy.packb(
                    {
                        "async_rtc": {
                            "protocol": base_rtc.PROTOCOL_NAME,
                            "protocol_version": base_rtc.PROTOCOL_VERSION,
                            "model_action_horizon": 50,
                            "image_transport_codecs": ["raw"],
                        }
                    }
                )
            waiting.set()
            if not closed.wait(timeout):
                raise TimeoutError("warmup")
            raise ConnectionError("closed")

        def send(self, _data):
            pass

        def close(self):
            closed.set()

    connection = Connection()
    monkeypatch.setattr(base_rtc.AsyncRTCInferenceWorker, "_connect", lambda _self: connection)
    worker = voice_rtc.VoiceAsyncRTCInferenceWorker(
        "fake",
        18080,
        connect_timeout_s=1,
        rtc_inference_delay=6,
        rtc_execution_horizon=15,
        rtc_prefix_attention_schedule="linear",
        rtc_max_guidance_weight=10,
        image_transport_codec="raw",
        image_jpeg_quality=90,
    )
    try:
        worker.submit(base_rtc.InferenceTask(0, 0, 0, {"prompt": "red"}, None, 600))
        assert waiting.wait(2)
        start = time.monotonic()
        worker.close()
        assert time.monotonic() - start < 2
        assert not worker._thread.is_alive()
    finally:
        worker.close()
