"""Right-arm-only real-robot environment for X-Trainer inference."""

import contextlib
import logging
import time

import numpy as np
from openpi_client import image_tools
from openpi_client.runtime import environment as _environment
from typing_extensions import override

logger = logging.getLogger(__name__)
PHYSICAL_ACTION_DIM = 7


def _observation_to_action_array(observation: dict[str, float]) -> np.ndarray:
    values = np.zeros(PHYSICAL_ACTION_DIM, dtype=np.float64)
    for joint_index in range(6):
        values[joint_index] = float(observation[f"joint{joint_index + 1}.pos"])
    values[6] = float(observation.get("gripper.pos", 1.0))
    return values


def _create_follower(**kwargs):
    # Keep hardware-only dependencies out of policy-server and unit-test imports.
    from examples.xtrainer_real.hardware import DobotXTrainer

    return DobotXTrainer(**kwargs)


class XTrainerRightArmRealEnvironment(_environment.Environment):
    """Control only the right arm and expose top/right-wrist observations."""

    def __init__(
        self,
        *,
        right_robot_ip: str = "192.168.5.2",
        right_gripper_port: str = "/dev/ttyUSB0",
        right_gripper_id: int = 22,
        right_gripper_servo_pos: tuple[int, int] = (2048, 3052),
        camera_top_serial: str,
        camera_right_wrist_serial: str,
        camera_fps: float = 30.0,
        render_height: int = 480,
        render_width: int = 640,
        prompt: str | None = None,
        reset_pose: list[float] | None = None,
        max_joint_delta: float = 0.17,
        ramp_step: float = 0.01,
        ramp_max_steps: int = 100,
        gripper_update_threshold: float = 0.02,
        servo_step_limit: float = 0.9,
    ) -> None:
        if not camera_top_serial:
            raise ValueError("camera_top_serial is required for right-arm inference.")
        if not camera_right_wrist_serial:
            raise ValueError("camera_right_wrist_serial is required for right-arm inference.")

        self._follower = _create_follower(
            robot_ip=right_robot_ip,
            gripper_port=right_gripper_port,
            gripper_id=right_gripper_id,
            gripper_servo_pos=right_gripper_servo_pos,
            read_gripper_position=False,
            max_delta_per_step=servo_step_limit,
            camera_serials={
                "cam_top": camera_top_serial,
                "cam_right_wrist": camera_right_wrist_serial,
            },
            camera_fps=camera_fps,
        )
        self._render_height = render_height
        self._render_width = render_width
        self._prompt = prompt
        self._max_joint_delta = max_joint_delta
        self._ramp_step = ramp_step
        self._ramp_max_steps = max(ramp_max_steps, 1)
        self._gripper_update_threshold = max(gripper_update_threshold, 0.0)
        self._connected = False
        self._last_action: np.ndarray | None = None
        self._last_gripper_sent = 1.0

        self._reset_pose = None
        if reset_pose is not None:
            reset = np.asarray(reset_pose, dtype=np.float64).reshape(-1)
            if reset.shape[0] != PHYSICAL_ACTION_DIM:
                raise ValueError(f"Expected reset_pose length {PHYSICAL_ACTION_DIM}, got {reset.shape[0]}")
            self._reset_pose = reset

    @override
    def reset(self) -> None:
        self._ensure_connected()
        if self._reset_pose is not None:
            self._move_smooth(self._get_qpos(), self._reset_pose)
            time.sleep(0.2)
        self._last_action = self._get_qpos()
        self._last_gripper_sent = float(self._last_action[6])

    @override
    def is_episode_complete(self) -> bool:
        return False

    @override
    def get_observation(self) -> dict:
        self._ensure_connected()
        observation = {"observation.state": self._get_qpos().astype(np.float32)}
        for camera_name in ("top", "right_wrist"):
            frame = self._read_camera_frame(camera_name)
            frame = image_tools.convert_to_uint8(
                image_tools.resize_with_pad(frame, self._render_height, self._render_width)
            )
            observation[f"observation.images.{camera_name}"] = frame
        if self._prompt is not None:
            observation["prompt"] = self._prompt
        return observation

    @override
    def apply_action(self, action: dict) -> None:
        self._ensure_connected()
        if "actions" not in action:
            raise KeyError(f"Missing 'actions' in action dict: {tuple(action.keys())}")
        target = np.asarray(action["actions"], dtype=np.float64).reshape(-1).copy()
        if target.shape[0] != PHYSICAL_ACTION_DIM:
            raise ValueError(f"Expected action length {PHYSICAL_ACTION_DIM}, got {target.shape[0]}")
        target[6] = float(np.clip(target[6], 0.0, 1.0))

        if self._last_action is None:
            self._last_action = self._get_qpos()
        max_joint_delta = float(np.max(np.abs(target[:6] - self._last_action[:6])))
        if max_joint_delta > self._max_joint_delta:
            self._move_smooth(self._last_action, target)
        else:
            self._send_action(target)
        self._last_action = target.copy()

    def close(self) -> None:
        if not self._connected:
            return
        try:
            self._follower.disconnect()
        except Exception:
            logger.exception("Failed to disconnect right follower cleanly.")
        self._connected = False

    def _ensure_connected(self) -> None:
        if self._connected:
            return
        self._follower.connect()
        self._connected = True
        self._last_action = self._get_qpos()
        self._last_gripper_sent = float(self._last_action[6])

    def _read_camera_frame(self, camera_name: str, retries: int = 5) -> np.ndarray:
        camera_key = "cam_top" if camera_name == "top" else "cam_right_wrist"
        camera = self._follower.cameras[camera_key]
        for _ in range(max(retries, 1)):
            try:
                frame = camera.async_read(timeout_ms=50)
            except TypeError:
                frame = camera.async_read()
            except Exception:
                frame = None
            if isinstance(frame, np.ndarray) and frame.ndim == 3:
                return frame
            time.sleep(0.005)
        raise RuntimeError(f"Failed to read camera frame: {camera_name}")

    def _get_qpos(self) -> np.ndarray:
        return _observation_to_action_array(self._follower.get_low_latency_observation())

    def _send_action(self, action: np.ndarray) -> None:
        command = {f"joint{joint_index + 1}.pos": float(action[joint_index]) for joint_index in range(6)}
        gripper = float(action[6])
        if abs(gripper - self._last_gripper_sent) >= self._gripper_update_threshold:
            command["gripper.pos"] = gripper
            self._last_gripper_sent = gripper
        self._follower.send_action(command)

    def _move_smooth(self, start_action: np.ndarray, goal_action: np.ndarray) -> None:
        max_delta = float(np.max(np.abs(goal_action - start_action)))
        steps = min(int(np.ceil(max_delta / max(self._ramp_step, 1e-6))), self._ramp_max_steps)
        if steps <= 1:
            self._send_action(goal_action)
            return
        for action in np.linspace(start_action, goal_action, steps):
            self._send_action(action)

    def __del__(self) -> None:
        with contextlib.suppress(Exception):
            self.close()
