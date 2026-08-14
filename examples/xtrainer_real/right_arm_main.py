"""Run the X-Trainer while executing model actions on the right arm only.

Both arms still move to the server-provided reset pose at the start of each
episode. After reset, the left arm and left gripper are held at that pose, and
only the right-arm portion of each model action is executed.
"""

import dataclasses
import logging
from pathlib import Path
import time
from typing import Any

import cv2
import numpy as np
from openpi_client import action_chunk_broker
from openpi_client import base_policy as _base_policy
from openpi_client import websocket_client_policy as _websocket_client_policy
from openpi_client.runtime import agent as _agent
from openpi_client.runtime import runtime as _runtime
from typing_extensions import override
import tyro

from examples.xtrainer_real import diagnostics as _diagnostics
from examples.xtrainer_real import env as _env
from examples.xtrainer_real import image_preprocessing as _image_preprocessing
from examples.xtrainer_real import inference_action_recorder as _action_recorder
from examples.xtrainer_real import main as _sync_main

logger = logging.getLogger(__name__)
action_logger = logging.getLogger(f"{__name__}.actions")
_PREVIEW_WINDOW_NAME = "X-Trainer observations: top | left wrist | right wrist"


class _PreviewStopRequestedError(Exception):
    """Raised when the operator closes the preview or presses Q/Esc."""


@dataclasses.dataclass
class Args(_sync_main.Args):
    """Right-arm synchronous inference arguments with action recording controls."""

    record_inference_actions: bool = True
    """Save received action chunks, selected actions, metadata, and a plot."""

    inference_action_output_dir: str = "output"
    """Root directory for timestamped inference action recording directories."""

    inference_action_plot_start_index: int = 7
    """First action dimension shown in the generated plot; 7 selects the right arm."""


def _configure_action_logger() -> Path:
    output_dir = Path.cwd() / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_path = output_dir / f"inference_actions_{timestamp}.log"

    action_logger.setLevel(logging.INFO)
    action_logger.propagate = False
    handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
    action_logger.addHandler(handler)
    return log_path


class _RecordingSynchronousPolicy(_base_policy.BasePolicy):
    """Record each complete action chunk returned by the blocking websocket policy."""

    def __init__(
        self,
        policy: _base_policy.BasePolicy,
        *,
        action_horizon: int,
        recorder: _action_recorder.InferenceActionRecorder,
    ) -> None:
        self._policy = policy
        self._action_horizon = action_horizon
        self._recorder = recorder
        self._episode_id = -1
        self._inference_index = 0
        self._next_request_id = 0
        self.active_request_id: int | None = None

    @override
    def infer(self, obs: dict[str, Any]) -> dict[str, Any]:
        start_time = time.monotonic()
        result = self._policy.infer(obs)
        round_trip_ms = (time.monotonic() - start_time) * 1000
        actions = np.asarray(result.get("actions"))
        if actions.ndim != 2:
            raise ValueError(f"Expected synchronous actions with shape (horizon, action_dim), got {actions.shape}.")

        request_id = self._next_request_id
        request_step = self._inference_index * self._action_horizon
        timing = result.get("server_timing", result.get("policy_timing", {}))
        server_infer_ms = float(timing.get("infer_ms", float("nan"))) if isinstance(timing, dict) else float("nan")
        self._recorder.record_received_chunk(
            actions,
            phase="sync_initial" if self._inference_index == 0 else "sync_online",
            request_id=request_id,
            episode_id=self._episode_id,
            request_step=request_step,
            arrival_step=request_step,
            actual_delay_steps=0,
            rtc_enabled=False,
            installed=True,
            round_trip_ms=round_trip_ms,
            server_infer_ms=server_infer_ms,
        )
        self.active_request_id = request_id
        self._next_request_id += 1
        self._inference_index += 1
        return result

    @override
    def reset(self) -> None:
        self._policy.reset()
        self._episode_id += 1
        self._inference_index = 0
        self.active_request_id = None


class _RecordingSynchronousAgent(_agent.Agent):
    """Record the per-step actions selected by ``ActionChunkBroker``."""

    def __init__(
        self,
        policy: _base_policy.BasePolicy,
        *,
        recording_policy: _RecordingSynchronousPolicy,
        recorder: _action_recorder.InferenceActionRecorder,
        action_horizon: int,
    ) -> None:
        self._policy = policy
        self._recording_policy = recording_policy
        self._recorder = recorder
        self._action_horizon = action_horizon
        self._episode_id = -1
        self._step = 0

    @override
    def get_action(self, observation: dict) -> dict:
        result = self._policy.infer(observation)
        request_id = self._recording_policy.active_request_id
        if request_id is None:
            raise RuntimeError("Synchronous action chunk does not have a request ID.")
        self._recorder.record_selected_action(
            np.asarray(result["actions"]),
            episode_id=self._episode_id,
            step=self._step,
            request_id=request_id,
            chunk_index=self._step % self._action_horizon,
        )
        self._step += 1
        return result

    @override
    def reset(self) -> None:
        self._policy.reset()
        self._episode_id += 1
        self._step = 0


class RightArmOnlyEnvironment(_env.XTrainerRealEnvironment):
    """X-Trainer environment that discards the model's left-arm actions."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._left_hold_action: np.ndarray | None = None
        self._action_log_step = 0
        self._preview_window_created = False

    @override
    def get_observation(self) -> dict:
        observation = super().get_observation()
        _image_preprocessing.match_legacy_dataset_images(observation)

        preview_frames = []
        for camera_name, label in (
            ("top", "TOP"),
            ("left_wrist", "LEFT WRIST"),
            ("right_wrist", "RIGHT WRIST"),
        ):
            # Observations are RGB. Convert a copy to BGR for OpenCV so the
            # arrays sent to the server remain completely unchanged.
            rgb = observation[f"observation.images.{camera_name}"]
            bgr = np.ascontiguousarray(rgb[..., ::-1])
            cv2.putText(bgr, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3)
            cv2.putText(bgr, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
            preview_frames.append(bgr)

        preview = np.hstack(preview_frames)
        if not self._preview_window_created:
            cv2.namedWindow(_PREVIEW_WINDOW_NAME, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(_PREVIEW_WINDOW_NAME, preview.shape[1], preview.shape[0])
            self._preview_window_created = True

        cv2.imshow(_PREVIEW_WINDOW_NAME, preview)
        key = cv2.waitKey(1) & 0xFF
        window_visible = cv2.getWindowProperty(_PREVIEW_WINDOW_NAME, cv2.WND_PROP_VISIBLE)
        if key in (ord("q"), ord("Q"), 27) or window_visible < 1:
            raise _PreviewStopRequestedError

        return observation

    @override
    def reset(self) -> None:
        # The base reset first moves both arms to the server-provided reset pose.
        super().reset()
        if self._last_action is None:
            raise RuntimeError("X-Trainer reset did not initialize the current action.")
        self._left_hold_action = np.array(self._last_action[:7], copy=True)
        self._action_log_step = 0
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

        action_logger.info("step=%d action=%s", self._action_log_step, target.tolist())
        self._action_log_step += 1

        # [0:7] is left joints 1-6 plus the left gripper. Keep it fixed at
        # the post-reset pose; [7:14] (the right arm) remains model-controlled.
        target[:7] = self._left_hold_action

        filtered_action = dict(action)
        filtered_action["actions"] = target
        super().apply_action(filtered_action)

    @override
    def close(self) -> None:
        if self._preview_window_created:
            try:
                cv2.destroyWindow(_PREVIEW_WINDOW_NAME)
                cv2.waitKey(1)
            except cv2.error:
                pass
            self._preview_window_created = False
        super().close()


def main(args: Args) -> None:
    if args.inference_action_plot_start_index < 0:
        raise ValueError("--inference-action-plot-start-index must be non-negative.")
    action_log_path = _configure_action_logger()
    logging.info("Writing inference action log to %s", action_log_path)

    action_recorder = None
    if args.record_inference_actions:
        action_recorder = _action_recorder.InferenceActionRecorder(
            args.inference_action_output_dir,
            mode="sync_baseline",
            plot_start_index=args.inference_action_plot_start_index,
            config=dataclasses.asdict(args),
        )
        logging.info("Recording synchronous inference actions in %s", action_recorder.output_dir)

    ws_client_policy = _websocket_client_policy.WebsocketClientPolicy(
        host=args.host,
        port=args.port,
    )
    metadata = ws_client_policy.get_server_metadata()
    logging.info("Server metadata: %s", metadata)

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

    subscribers = []
    if args.debug_action_state_diagnostics:
        subscribers.append(
            _diagnostics.ActionStateDiagnosticsSubscriber(
                interval=args.debug_action_state_interval,
                max_steps=args.debug_action_state_max_steps,
            )
        )

    if action_recorder is not None:
        recording_policy = _RecordingSynchronousPolicy(
            ws_client_policy,
            action_horizon=args.action_horizon,
            recorder=action_recorder,
        )
        chunk_policy = action_chunk_broker.ActionChunkBroker(
            policy=recording_policy,
            action_horizon=args.action_horizon,
        )
        agent: _agent.Agent = _RecordingSynchronousAgent(
            chunk_policy,
            recording_policy=recording_policy,
            recorder=action_recorder,
            action_horizon=args.action_horizon,
        )
    else:
        from openpi_client.runtime.agents import policy_agent as _policy_agent

        agent = _policy_agent.PolicyAgent(
            policy=action_chunk_broker.ActionChunkBroker(
                policy=ws_client_policy,
                action_horizon=args.action_horizon,
            )
        )

    runtime = _runtime.Runtime(
        environment=environment,
        agent=agent,
        subscribers=subscribers,
        max_hz=args.control_hz,
        num_episodes=args.num_episodes,
        max_episode_steps=args.max_episode_steps,
    )

    try:
        runtime.run()
    except _PreviewStopRequestedError:
        logging.info("Camera preview closed; stopping the client.")
    finally:
        try:
            environment.close()
        finally:
            if action_recorder is not None:
                action_recorder.save()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
