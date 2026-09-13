"""Voice-session lifecycle extensions for the normal right-arm async RTC client."""

import contextlib
import queue
from typing import Any

import numpy as np
from typing_extensions import override

from examples.xtrainer_real import bimanual_async_rtc_main as rtc
from examples.xtrainer_real.right_arm_single_async_rtc_main import RightArmAsyncRTCPolicyAgent


class VoiceAsyncRTCInferenceWorker(rtc.AsyncRTCInferenceWorker):
    """Allow Ctrl+C and startup failures to close a pending websocket receive."""

    def __init__(self, *args, **kwargs) -> None:
        self._voice_connection = None
        try:
            super().__init__(*args, **kwargs)
        except BaseException:
            self.close()
            raise

    @override
    def _connect(self):
        connection = super()._connect()
        self._voice_connection = connection
        return connection

    @override
    def close(self) -> None:
        if hasattr(self, "_stop"):
            self._stop.set()
        if hasattr(self, "_tasks"):
            with contextlib.suppress(queue.Full):
                self._tasks.put_nowait(None)
        if self._voice_connection is not None:
            with contextlib.suppress(Exception):
                self._voice_connection.close()
        if hasattr(self, "_thread"):
            self._thread.join(timeout=2.0)


class VoiceRightArmAsyncRTCPolicyAgent(RightArmAsyncRTCPolicyAgent):
    """Keep action chunks and outstanding requests isolated between spoken tasks."""

    @override
    def _validated_result(self, result: dict[str, Any] | None) -> tuple[dict[str, Any], np.ndarray]:
        result, actions = super()._validated_result(result)
        if len(actions) == 0 or not np.all(np.isfinite(actions)):
            raise ValueError("Right-arm action chunk must contain finite actions and be non-empty.")
        return result, actions

    def finish_task(self, timeout_s: float) -> None:
        self._active_actions = None
        self._active_result = None
        self._active_action_index = 0
        self._active_request_id = None
        task = self._inflight_task
        if task is not None:
            response = self._worker.wait_for_response(timeout_s)
            if response.task.request_id != task.request_id or response.task.episode_id != task.episode_id:
                raise RuntimeError("Unexpected response while ending task")
            if response.error is not None:
                raise RuntimeError("Pending inference failed while ending task") from response.error
            self._inflight_task = None
        if self._worker.get_response_nowait() is not None:
            raise RuntimeError("Unexpected leftover response at task boundary")

    def set_action_recorder(self, recorder) -> None:
        if self._inflight_task is not None or self._active_actions is not None:
            raise RuntimeError("Cannot switch recorder during an active task")
        self._action_recorder = recorder
