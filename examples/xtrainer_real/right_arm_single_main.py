"""Run synchronous inference on the physical right arm with two cameras."""

import dataclasses
import logging
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
from examples.xtrainer_real import image_preprocessing as _image_preprocessing
from examples.xtrainer_real import inference_action_recorder as _action_recorder
from examples.xtrainer_real import right_arm_env as _right_arm_env

logger = logging.getLogger(__name__)
_PREVIEW_WINDOW_NAME = "X-Trainer right arm: top | right wrist"
_EXPECTED_CAMERA_KEYS = ["observation.images.top", "observation.images.right_wrist"]


class _PreviewStopRequestedError(Exception):
    pass


@dataclasses.dataclass
class Args:
    host: str = "0.0.0.0"
    port: int = 8000
    prompt: str = "pick up the object"
    action_horizon: int = 25
    control_hz: float = 20.0
    num_episodes: int = 1
    max_episode_steps: int = 1000

    right_robot_ip: str = "192.168.5.2"
    right_gripper_port: str = "/dev/ttyUSB0"
    right_gripper_id: int = 22
    right_gripper_servo_pos: tuple[int, int] = (2048, 3052)
    camera_top_serial: str = ""
    camera_right_wrist_serial: str = ""
    camera_fps: float = 30.0
    render_height: int = 480
    render_width: int = 640

    max_joint_delta: float = 0.17
    ramp_step: float = 0.01
    ramp_max_steps: int = 100
    gripper_update_threshold: float = 0.02
    servo_step_limit: float = 0.9

    debug_action_state_diagnostics: bool = False
    debug_action_state_interval: int = 20
    debug_action_state_max_steps: int = 400
    record_inference_actions: bool = True
    inference_action_output_dir: str = "output"
    inference_action_plot_start_index: int = 0


def validate_right_arm_metadata(metadata: dict[str, Any]) -> None:
    expected = {
        "robot_side": "right",
        "physical_action_dim": _right_arm_env.PHYSICAL_ACTION_DIM,
        "model_action_start_index": 7,
        "model_action_end_index": 14,
    }
    mismatches = {key: (metadata.get(key), value) for key, value in expected.items() if metadata.get(key) != value}
    if list(metadata.get("camera_keys", ())) != _EXPECTED_CAMERA_KEYS:
        mismatches["camera_keys"] = (metadata.get("camera_keys"), _EXPECTED_CAMERA_KEYS)
    reset_pose = np.asarray(metadata.get("reset_pose", ())).reshape(-1)
    if reset_pose.shape != (_right_arm_env.PHYSICAL_ACTION_DIM,):
        mismatches["reset_pose"] = (metadata.get("reset_pose"), "7 values")
    if mismatches:
        raise ValueError(f"Server checkpoint is not compatible with the right-arm client: {mismatches}")


class PreviewRightArmEnvironment(_right_arm_env.XTrainerRightArmRealEnvironment):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._preview_window_created = False

    @override
    def get_observation(self) -> dict:
        observation = super().get_observation()
        _image_preprocessing.match_legacy_dataset_images(observation)
        preview_frames = []
        for camera_name, label in (("top", "TOP"), ("right_wrist", "RIGHT WRIST")):
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
        if key in (ord("q"), ord("Q"), 27) or cv2.getWindowProperty(_PREVIEW_WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
            raise _PreviewStopRequestedError
        return observation

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


class _ValidatedRecordingPolicy(_base_policy.BasePolicy):
    def __init__(
        self,
        policy: _base_policy.BasePolicy,
        *,
        action_horizon: int,
        recorder: _action_recorder.InferenceActionRecorder | None,
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
        if actions.ndim != 2 or actions.shape[-1] != _right_arm_env.PHYSICAL_ACTION_DIM:
            raise ValueError(f"Expected actions with shape (horizon, 7), got {actions.shape}")

        request_id = self._next_request_id
        self.active_request_id = request_id
        if self._recorder is not None:
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
        self._next_request_id += 1
        self._inference_index += 1
        return result

    @override
    def reset(self) -> None:
        self._policy.reset()
        self._episode_id += 1
        self._inference_index = 0
        self.active_request_id = None


class _RecordingAgent(_agent.Agent):
    def __init__(
        self,
        policy: _base_policy.BasePolicy,
        *,
        recording_policy: _ValidatedRecordingPolicy,
        recorder: _action_recorder.InferenceActionRecorder | None,
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
        if self._recorder is not None:
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


def main(args: Args) -> None:
    if (args.render_height, args.render_width) != (480, 640):
        raise ValueError("Right-arm image preprocessing requires --render-height 480 --render-width 640.")
    if not 0 <= args.inference_action_plot_start_index < _right_arm_env.PHYSICAL_ACTION_DIM:
        raise ValueError("--inference-action-plot-start-index must be in [0, 6].")

    ws_client_policy = _websocket_client_policy.WebsocketClientPolicy(host=args.host, port=args.port)
    metadata = ws_client_policy.get_server_metadata()
    validate_right_arm_metadata(metadata)
    logger.info("Validated right-arm server metadata: %s", metadata)

    recorder = None
    if args.record_inference_actions:
        recorder = _action_recorder.InferenceActionRecorder(
            args.inference_action_output_dir,
            mode="right_arm_sync",
            plot_start_index=args.inference_action_plot_start_index,
            config=dataclasses.asdict(args),
        )

    environment = PreviewRightArmEnvironment(
        right_robot_ip=args.right_robot_ip,
        right_gripper_port=args.right_gripper_port,
        right_gripper_id=args.right_gripper_id,
        right_gripper_servo_pos=args.right_gripper_servo_pos,
        camera_top_serial=args.camera_top_serial,
        camera_right_wrist_serial=args.camera_right_wrist_serial,
        camera_fps=args.camera_fps,
        render_height=args.render_height,
        render_width=args.render_width,
        prompt=args.prompt,
        reset_pose=metadata["reset_pose"],
        max_joint_delta=args.max_joint_delta,
        ramp_step=args.ramp_step,
        ramp_max_steps=args.ramp_max_steps,
        gripper_update_threshold=args.gripper_update_threshold,
        servo_step_limit=args.servo_step_limit,
    )
    validated_policy = _ValidatedRecordingPolicy(
        ws_client_policy,
        action_horizon=args.action_horizon,
        recorder=recorder,
    )
    chunk_policy = action_chunk_broker.ActionChunkBroker(validated_policy, action_horizon=args.action_horizon)
    agent = _RecordingAgent(
        chunk_policy,
        recording_policy=validated_policy,
        recorder=recorder,
        action_horizon=args.action_horizon,
    )

    subscribers = []
    if args.debug_action_state_diagnostics:
        subscribers.append(
            _diagnostics.ActionStateDiagnosticsSubscriber(
                interval=args.debug_action_state_interval,
                max_steps=args.debug_action_state_max_steps,
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
        logger.info("Camera preview closed; stopping the client.")
    finally:
        try:
            environment.close()
        finally:
            if recorder is not None:
                recorder.save()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
