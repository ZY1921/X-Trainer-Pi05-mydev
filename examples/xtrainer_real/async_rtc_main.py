"""Run X-Trainer with non-blocking client/server inference and optional RTC.

The robot control loop stays on the main thread. A dedicated worker owns the
websocket connection and performs blocking network I/O while the robot keeps
executing the active action chunk.
"""

import contextlib
import copy
import dataclasses
import logging
import queue
import threading
import time
from typing import Any, Literal

import numpy as np
from openpi_client import msgpack_numpy
from openpi_client.runtime import agent as _agent
from openpi_client.runtime import runtime as _runtime
from typing_extensions import override
import tyro
import websockets.sync.client

from examples.xtrainer_real import diagnostics as _diagnostics
from examples.xtrainer_real import env as _env
from examples.xtrainer_real import image_preprocessing as _image_preprocessing
from examples.xtrainer_real import main as _sync_main

PROTOCOL_NAME = "openpi-async-rtc"
PROTOCOL_VERSION = 1

logger = logging.getLogger(__name__)


class RightArmOnlyEnvironment(_env.XTrainerRealEnvironment):
    """Discard model actions for the left arm after each episode reset."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._left_hold_action: np.ndarray | None = None

    @override
    def get_observation(self) -> dict:
        observation = super().get_observation()
        _image_preprocessing.match_legacy_dataset_images(observation)
        return observation

    @override
    def reset(self) -> None:
        # Let the base environment move both arms to the server-provided reset
        # pose, then hold the left arm and gripper at that pose for the episode.
        super().reset()
        if self._last_action is None:
            raise RuntimeError("X-Trainer reset did not initialize the current action.")
        self._left_hold_action = np.array(self._last_action[:7], copy=True)
        logger.info("Left arm frozen after reset at %s", self._left_hold_action.tolist())

    @override
    def apply_action(self, action: dict) -> None:
        if "actions" not in action:
            raise KeyError(f"Missing 'actions' in action dict: {tuple(action.keys())}")
        if self._left_hold_action is None:
            raise RuntimeError("Left-arm hold pose is unavailable; reset must run before applying actions.")

        target = np.asarray(action["actions"], dtype=np.float64).reshape(-1).copy()
        if target.shape[0] != 14:
            raise ValueError(f"Expected action length 14, got {target.shape[0]}")

        # [0:7] is left joints 1-6 plus the left gripper. Keep it fixed at
        # the post-reset pose; [7:14] (the right arm) remains model-controlled.
        target[:7] = self._left_hold_action

        filtered_action = dict(action)
        filtered_action["actions"] = target
        super().apply_action(filtered_action)


@dataclasses.dataclass
class Args(_sync_main.Args):
    """X-Trainer arguments with asynchronous inference and RTC controls."""

    rtc_enabled: bool = True
    """Enable RTC guidance. Disable this for an asynchronous non-RTC baseline."""

    rtc_inference_delay: int = 4
    """Expected inference delay, in control steps, used by RTC guidance."""

    rtc_execution_horizon: int = 10
    """Exclusive end of the RTC prefix transition region."""

    rtc_prefix_attention_schedule: Literal["exp", "linear", "ones", "zeros"] = "exp"
    """RTC prefix-attention schedule."""

    rtc_max_guidance_weight: float = 10.0
    """Maximum RTC denoising guidance weight."""

    max_inference_delay_steps: int = 15
    """Abort if an online inference request is this many control steps late."""

    inference_timeout_s: float = 3.0
    """Network timeout for online inference requests."""

    connect_timeout_s: float = 120.0
    """Maximum time to wait for the async RTC server at startup."""

    warmup_timeout_s: float = 180.0
    """Timeout for initial JAX compilation and RTC warmup requests."""

    warmup_rtc: bool = True
    """Compile the RTC path before robot motion starts."""

    debug_async_timing: bool = False
    """Log request latency, measured delay, and chunk switch indices."""


@dataclasses.dataclass(frozen=True)
class InferenceTask:
    request_id: int
    episode_id: int
    request_step: int
    observation: dict[str, Any]
    prev_chunk_left_over: np.ndarray | None
    timeout_s: float


@dataclasses.dataclass(frozen=True)
class InferenceResponse:
    task: InferenceTask
    result: dict[str, Any] | None
    round_trip_ms: float
    error: BaseException | None = None


class AsyncRTCInferenceWorker:
    """Own a websocket connection and execute one inference request at a time."""

    def __init__(
        self,
        host: str,
        port: int,
        *,
        connect_timeout_s: float,
        rtc_inference_delay: int,
        rtc_execution_horizon: int,
        rtc_prefix_attention_schedule: str,
        rtc_max_guidance_weight: float,
    ) -> None:
        self._uri = host if host.startswith("ws") else f"ws://{host}"
        self._uri = f"{self._uri}:{port}"
        self._connect_timeout_s = connect_timeout_s
        self._rtc_inference_delay = rtc_inference_delay
        self._rtc_execution_horizon = rtc_execution_horizon
        self._rtc_prefix_attention_schedule = rtc_prefix_attention_schedule
        self._rtc_max_guidance_weight = rtc_max_guidance_weight

        self._tasks: queue.Queue[InferenceTask | None] = queue.Queue(maxsize=1)
        self._responses: queue.Queue[InferenceResponse] = queue.Queue()
        self._ready = threading.Event()
        self._stop = threading.Event()
        self._metadata: dict[str, Any] | None = None
        self._startup_error: BaseException | None = None
        self._thread = threading.Thread(target=self._run, name="async-rtc-inference", daemon=True)
        self._thread.start()

        if not self._ready.wait(timeout=connect_timeout_s + 1.0):
            raise TimeoutError(f"Timed out waiting for inference worker startup at {self._uri}.")
        if self._startup_error is not None:
            raise RuntimeError(f"Failed to start inference worker at {self._uri}.") from self._startup_error

    @property
    def metadata(self) -> dict[str, Any]:
        if self._metadata is None:
            raise RuntimeError("Server metadata is not available.")
        return self._metadata

    def submit(self, task: InferenceTask) -> None:
        if not self._thread.is_alive():
            raise RuntimeError("Inference worker is not running.") from self._startup_error
        try:
            self._tasks.put_nowait(task)
        except queue.Full as error:
            raise RuntimeError("An inference request is already queued.") from error

    def get_response_nowait(self) -> InferenceResponse | None:
        try:
            return self._responses.get_nowait()
        except queue.Empty:
            return None

    def wait_for_response(self, timeout_s: float) -> InferenceResponse:
        try:
            return self._responses.get(timeout=timeout_s)
        except queue.Empty as error:
            raise TimeoutError(f"Inference did not finish within {timeout_s:.1f}s.") from error

    def close(self) -> None:
        self._stop.set()
        with contextlib.suppress(queue.Full):
            self._tasks.put_nowait(None)
        self._thread.join(timeout=max(self._connect_timeout_s, 1.0))

    def _run(self) -> None:
        connection = None
        try:
            connection = self._connect()
            metadata = msgpack_numpy.unpackb(connection.recv(timeout=self._connect_timeout_s))
            if not isinstance(metadata, dict):
                raise TypeError(f"Expected server metadata dictionary, got {type(metadata).__name__}.")
            self._validate_server_metadata(metadata)
            self._metadata = metadata
            self._ready.set()

            while not self._stop.is_set():
                task = self._tasks.get()
                if task is None:
                    break
                start_time = time.monotonic()
                try:
                    request = self._make_request(task)
                    connection.send(msgpack_numpy.packb(request))
                    response = msgpack_numpy.unpackb(connection.recv(timeout=task.timeout_s))
                    result = self._parse_response(task, response)
                    self._responses.put(
                        InferenceResponse(
                            task=task,
                            result=result,
                            round_trip_ms=(time.monotonic() - start_time) * 1000,
                        )
                    )
                except BaseException as error:
                    self._responses.put(
                        InferenceResponse(
                            task=task,
                            result=None,
                            round_trip_ms=(time.monotonic() - start_time) * 1000,
                            error=error,
                        )
                    )
                    break
        except BaseException as error:
            self._startup_error = error
            self._ready.set()
        finally:
            if connection is not None:
                try:
                    connection.close()
                except Exception:
                    logger.exception("Failed to close async RTC websocket cleanly.")

    def _connect(self) -> websockets.sync.client.ClientConnection:
        deadline = time.monotonic() + self._connect_timeout_s
        while not self._stop.is_set():
            try:
                logger.info("Connecting to async RTC server at %s", self._uri)
                return websockets.sync.client.connect(
                    self._uri,
                    compression=None,
                    max_size=None,
                    open_timeout=min(self._connect_timeout_s, 10.0),
                )
            except (ConnectionRefusedError, OSError) as error:
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        f"Could not connect to {self._uri} within {self._connect_timeout_s:.1f}s."
                    ) from error
                time.sleep(1.0)
        raise RuntimeError("Inference worker stopped before connecting.")

    def _validate_server_metadata(self, metadata: dict[str, Any]) -> None:
        protocol = metadata.get("async_rtc")
        if not isinstance(protocol, dict):
            raise ValueError("The connected server does not advertise async RTC support.")
        if protocol.get("protocol") != PROTOCOL_NAME or protocol.get("protocol_version") != PROTOCOL_VERSION:
            raise ValueError(f"Incompatible async RTC server metadata: {protocol}")
        model_horizon = protocol.get("model_action_horizon")
        if isinstance(model_horizon, int) and self._rtc_execution_horizon > model_horizon:
            raise ValueError(
                f"--rtc-execution-horizon={self._rtc_execution_horizon} exceeds the server model action horizon "
                f"({model_horizon})."
            )

    def _make_request(self, task: InferenceTask) -> dict[str, Any]:
        rtc = None
        if task.prev_chunk_left_over is not None:
            rtc = {
                "prev_chunk_left_over": task.prev_chunk_left_over,
                "inference_delay": self._rtc_inference_delay,
                "execution_horizon": self._rtc_execution_horizon,
                "prefix_attention_schedule": self._rtc_prefix_attention_schedule,
                "max_guidance_weight": self._rtc_max_guidance_weight,
            }
        return {
            "protocol": PROTOCOL_NAME,
            "protocol_version": PROTOCOL_VERSION,
            "request_id": task.request_id,
            "observation": task.observation,
            "rtc": rtc,
        }

    def _parse_response(self, task: InferenceTask, response: Any) -> dict[str, Any]:
        if not isinstance(response, dict):
            raise TypeError(f"Expected response dictionary, got {type(response).__name__}.")
        if response.get("request_id") != task.request_id:
            raise RuntimeError(
                f"Response request_id {response.get('request_id')!r} does not match request {task.request_id}."
            )
        if error := response.get("error"):
            raise RuntimeError(f"Inference server error:\n{error}")
        result = response.get("result")
        if not isinstance(result, dict):
            raise TypeError(f"Expected response result dictionary, got {type(result).__name__}.")
        return result


class AsyncRTCPolicyAgent(_agent.Agent):
    """Schedule chunk inference asynchronously while returning actions synchronously."""

    def __init__(
        self,
        worker: AsyncRTCInferenceWorker,
        *,
        request_interval: int,
        rtc_enabled: bool,
        max_inference_delay_steps: int,
        inference_timeout_s: float,
        warmup_timeout_s: float,
        warmup_rtc: bool,
        debug_timing: bool,
    ) -> None:
        self._worker = worker
        self._request_interval = request_interval
        self._rtc_enabled = rtc_enabled
        self._max_inference_delay_steps = max_inference_delay_steps
        self._inference_timeout_s = inference_timeout_s
        self._warmup_timeout_s = warmup_timeout_s
        self._warmup_rtc = warmup_rtc
        self._debug_timing = debug_timing

        self._episode_id = -1
        self._next_request_id = 0
        self._step = 0
        self._last_request_step = 0
        self._active_result: dict[str, Any] | None = None
        self._active_actions: np.ndarray | None = None
        self._active_action_index = 0
        self._inflight_task: InferenceTask | None = None

        self._validate_config()

    @override
    def reset(self) -> None:
        self._drain_inflight()
        self._episode_id += 1
        self._step = 0
        self._last_request_step = 0
        self._active_result = None
        self._active_actions = None
        self._active_action_index = 0
        self._inflight_task = None

    @override
    def get_action(self, observation: dict) -> dict:
        if self._active_actions is None:
            self._bootstrap(observation)
        else:
            self._accept_ready_response()
            self._check_inference_deadline()
            if self._inflight_task is None and self._step - self._last_request_step >= self._request_interval:
                self._submit_online_request(observation)

        assert self._active_actions is not None
        if self._active_action_index >= len(self._active_actions):
            raise RuntimeError(
                "The active action chunk was exhausted before a new inference result arrived. "
                "Reduce --action-horizon or --max-inference-delay-steps."
            )

        action = np.array(self._active_actions[self._active_action_index], copy=True)
        result = {"actions": action}
        if self._active_result is not None:
            for timing_key in ("policy_timing", "server_timing", "async_timing"):
                if timing_key in self._active_result:
                    result[timing_key] = self._active_result[timing_key]

        self._active_action_index += 1
        self._step += 1
        return result

    def _bootstrap(self, observation: dict) -> None:
        logger.info("Running blocking baseline warmup before robot motion")
        baseline_task = self._new_task(
            observation,
            request_step=0,
            prev_chunk_left_over=None,
            timeout_s=self._warmup_timeout_s,
        )
        baseline_response = self._submit_and_wait(baseline_task, self._warmup_timeout_s)
        self._install_initial_result(baseline_response.result)

        if self._rtc_enabled and self._warmup_rtc:
            assert self._active_actions is not None
            logger.info("Running blocking RTC warmup before robot motion")
            rtc_task = self._new_task(
                observation,
                request_step=0,
                prev_chunk_left_over=np.array(self._active_actions, copy=True),
                timeout_s=self._warmup_timeout_s,
            )
            self._submit_and_wait(rtc_task, self._warmup_timeout_s)
        logger.info("Async inference warmup complete; starting control actions")

    def _submit_online_request(self, observation: dict) -> None:
        assert self._active_actions is not None
        prev_chunk = None
        if self._rtc_enabled:
            prev_chunk = np.array(self._active_actions[self._active_action_index :], copy=True)
            if len(prev_chunk) == 0:
                raise RuntimeError("Cannot start RTC inference with an empty previous chunk.")
        task = self._new_task(
            observation,
            request_step=self._step,
            prev_chunk_left_over=prev_chunk,
            timeout_s=self._inference_timeout_s,
        )
        self._worker.submit(task)
        self._inflight_task = task
        self._last_request_step = self._step
        if self._debug_timing:
            logger.info(
                "ASYNC_RTC submit request=%d step=%d rtc=%s remaining=%d",
                task.request_id,
                self._step,
                self._rtc_enabled,
                len(self._active_actions) - self._active_action_index,
            )

    def _accept_ready_response(self) -> None:
        response = self._worker.get_response_nowait()
        if response is None:
            return
        if response.error is not None:
            raise RuntimeError(f"Inference request {response.task.request_id} failed.") from response.error
        if self._inflight_task is None or response.task.request_id != self._inflight_task.request_id:
            logger.warning("Discarding stale inference response %d", response.task.request_id)
            return
        if response.task.episode_id != self._episode_id:
            logger.warning("Discarding response %d from an earlier episode", response.task.request_id)
            self._inflight_task = None
            return

        actual_delay_steps = self._step - response.task.request_step
        result, actions = self._validated_result(response.result)
        if actual_delay_steps >= len(actions):
            raise RuntimeError(
                f"Inference result {response.task.request_id} arrived {actual_delay_steps} steps late, "
                f"but its chunk contains only {len(actions)} actions."
            )

        result["async_timing"] = {
            "round_trip_ms": response.round_trip_ms,
            "actual_delay_steps": actual_delay_steps,
            "chunk_start_index": actual_delay_steps,
        }
        self._active_result = result
        self._active_actions = actions
        self._active_action_index = actual_delay_steps
        self._inflight_task = None

        if self._debug_timing:
            server_ms = result.get("server_timing", {}).get("infer_ms", float("nan"))
            logger.info(
                "ASYNC_RTC switch request=%d step=%d delay_steps=%d chunk_index=%d rtt_ms=%.1f server_ms=%.1f",
                response.task.request_id,
                self._step,
                actual_delay_steps,
                self._active_action_index,
                response.round_trip_ms,
                server_ms,
            )

    def _check_inference_deadline(self) -> None:
        if self._inflight_task is None:
            return
        delay_steps = self._step - self._inflight_task.request_step
        if delay_steps > self._max_inference_delay_steps:
            raise TimeoutError(
                f"Inference request {self._inflight_task.request_id} exceeded "
                f"--max-inference-delay-steps={self._max_inference_delay_steps}."
            )

    def _new_task(
        self,
        observation: dict,
        *,
        request_step: int,
        prev_chunk_left_over: np.ndarray | None,
        timeout_s: float,
    ) -> InferenceTask:
        task = InferenceTask(
            request_id=self._next_request_id,
            episode_id=self._episode_id,
            request_step=request_step,
            observation=copy.deepcopy(observation),
            prev_chunk_left_over=prev_chunk_left_over,
            timeout_s=timeout_s,
        )
        self._next_request_id += 1
        return task

    def _submit_and_wait(self, task: InferenceTask, timeout_s: float) -> InferenceResponse:
        self._worker.submit(task)
        response = self._worker.wait_for_response(timeout_s + 1.0)
        if response.task.request_id != task.request_id:
            raise RuntimeError(f"Expected response {task.request_id}, got {response.task.request_id}.")
        if response.error is not None:
            raise RuntimeError(f"Inference request {task.request_id} failed.") from response.error
        return response

    def _install_initial_result(self, result: dict[str, Any] | None) -> None:
        result, actions = self._validated_result(result)
        self._active_result = result
        self._active_actions = actions
        self._active_action_index = 0

    def _validated_result(self, result: dict[str, Any] | None) -> tuple[dict[str, Any], np.ndarray]:
        if not isinstance(result, dict):
            raise TypeError(f"Expected inference result dictionary, got {type(result).__name__}.")
        if "actions" not in result:
            raise KeyError("Inference result is missing 'actions'.")
        actions = np.asarray(result["actions"])
        if actions.ndim != 2:
            raise ValueError(f"Expected actions with shape (horizon, action_dim), got {actions.shape}.")
        return result, actions

    def _drain_inflight(self) -> None:
        if self._inflight_task is None:
            return
        try:
            response = self._worker.wait_for_response(self._warmup_timeout_s)
            if response.error is not None:
                logger.warning("Previous episode inference failed while draining: %s", response.error)
        except TimeoutError:
            logger.warning("Timed out draining previous episode inference request")
        finally:
            self._inflight_task = None

    def _validate_config(self) -> None:
        if self._request_interval <= 0:
            raise ValueError("--action-horizon must be positive.")
        if self._max_inference_delay_steps <= 0:
            raise ValueError("--max-inference-delay-steps must be positive.")
        if self._inference_timeout_s <= 0 or self._warmup_timeout_s <= 0:
            raise ValueError("Inference timeouts must be positive.")

        protocol = self._worker.metadata["async_rtc"]
        model_horizon = protocol.get("model_action_horizon")
        if not isinstance(model_horizon, int):
            raise ValueError(f"Server did not provide a valid model action horizon: {model_horizon!r}.")
        if self._request_interval + self._max_inference_delay_steps > model_horizon:
            raise ValueError(
                "--action-horizon + --max-inference-delay-steps must not exceed the model action horizon "
                f"({self._request_interval} + {self._max_inference_delay_steps} > {model_horizon})."
            )
        if self._rtc_enabled and not protocol.get("rtc_supported", False):
            raise ValueError("The server policy does not support RTC; a JAX checkpoint is required.")


def _validate_rtc_args(args: Args) -> None:
    if args.rtc_inference_delay < 0:
        raise ValueError("--rtc-inference-delay must be non-negative.")
    if args.rtc_execution_horizon <= 0:
        raise ValueError("--rtc-execution-horizon must be positive.")
    if args.rtc_inference_delay > args.rtc_execution_horizon:
        raise ValueError("--rtc-inference-delay cannot exceed --rtc-execution-horizon.")
    if args.rtc_inference_delay >= args.action_horizon:
        raise ValueError("--rtc-inference-delay must be smaller than --action-horizon.")
    if args.rtc_max_guidance_weight <= 0:
        raise ValueError("--rtc-max-guidance-weight must be positive.")


def main(args: Args) -> None:
    if args.rtc_enabled:
        _validate_rtc_args(args)

    worker = AsyncRTCInferenceWorker(
        args.host,
        args.port,
        connect_timeout_s=args.connect_timeout_s,
        rtc_inference_delay=args.rtc_inference_delay,
        rtc_execution_horizon=args.rtc_execution_horizon,
        rtc_prefix_attention_schedule=args.rtc_prefix_attention_schedule,
        rtc_max_guidance_weight=args.rtc_max_guidance_weight,
    )
    try:
        _run_robot(args, worker)
    finally:
        worker.close()


def _run_robot(args: Args, worker: AsyncRTCInferenceWorker) -> None:
    metadata = worker.metadata
    logger.info("Server metadata: %s", metadata)

    environment = RightArmOnlyEnvironment(
        left_robot_ip=args.left_robot_ip,
        right_robot_ip=args.right_robot_ip,
        left_gripper_port=args.left_gripper_port,
        right_gripper_port=args.right_gripper_port,
        left_gripper_id=args.left_gripper_id,
        right_gripper_id=args.right_gripper_id,
        left_gripper_servo_pos=args.left_gripper_servo_pos,
        right_gripper_servo_pos=args.right_gripper_servo_pos,
        camera_top_serial=args.camera_top_serial,
        camera_left_wrist_serial=args.camera_left_wrist_serial,
        camera_right_wrist_serial=args.camera_right_wrist_serial,
        camera_fps=args.camera_fps,
        render_height=args.render_height,
        render_width=args.render_width,
        prompt=args.prompt,
        reset_pose=metadata.get("reset_pose"),
        max_joint_delta=args.max_joint_delta,
        ramp_step=args.ramp_step,
        ramp_max_steps=args.ramp_max_steps,
        gripper_update_threshold=args.gripper_update_threshold,
        servo_step_limit=args.servo_step_limit,
    )
    try:
        subscribers = []
        if args.debug_action_state_diagnostics:
            subscribers.append(
                _diagnostics.ActionStateDiagnosticsSubscriber(
                    interval=args.debug_action_state_interval,
                    max_steps=args.debug_action_state_max_steps,
                )
            )

        agent = AsyncRTCPolicyAgent(
            worker,
            request_interval=args.action_horizon,
            rtc_enabled=args.rtc_enabled,
            max_inference_delay_steps=args.max_inference_delay_steps,
            inference_timeout_s=args.inference_timeout_s,
            warmup_timeout_s=args.warmup_timeout_s,
            warmup_rtc=args.warmup_rtc,
            debug_timing=args.debug_async_timing,
        )
        runtime = _runtime.Runtime(
            environment=environment,
            agent=agent,
            subscribers=subscribers,
            max_hz=args.control_hz,
            num_episodes=args.num_episodes,
            max_episode_steps=args.max_episode_steps,
        )
        runtime.run()
    finally:
        environment.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
