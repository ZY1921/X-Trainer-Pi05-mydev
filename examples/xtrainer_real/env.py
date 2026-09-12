import contextlib
import logging
import time

import numpy as np
from openpi_client import image_tools
from openpi_client.runtime import environment as _environment
from typing_extensions import override

logger = logging.getLogger(__name__)


def _create_follower(**kwargs):
    # Keep hardware-only dependencies out of policy-server and unit-test imports.
    from examples.xtrainer_real.hardware import DobotXTrainer

    return DobotXTrainer(**kwargs)


def _obs_dict_to_arm_array(obs: dict[str, float]) -> np.ndarray:
    arr = np.zeros(7, dtype=np.float64)
    for joint_index in range(6):
        arr[joint_index] = float(obs[f"joint{joint_index + 1}.pos"])
    arr[6] = float(obs.get("gripper.pos", 1.0))
    return arr


class XTrainerRealEnvironment(_environment.Environment):
    """XTrainer real-robot environment for openpi-client runtime."""

    def __init__(
        self,
        *,
        left_robot_ip: str = "192.168.5.1",
        right_robot_ip: str = "192.168.5.3",
        left_gripper_port: str = "/dev/ttyUSB1",
        right_gripper_port: str = "/dev/ttyUSB0",
        left_gripper_id: int = 21,
        right_gripper_id: int = 22,
        left_gripper_servo_pos: tuple[int, int] = (2048, 3052),
        right_gripper_servo_pos: tuple[int, int] = (2048, 3052),
        camera_top_serial: str = "",
        camera_left_wrist_serial: str = "",
        camera_right_wrist_serial: str = "",
        camera_fps: float = 30.0,
        render_height: int = 224,
        render_width: int = 224,
        prompt: str | None = None,
        reset_pose: list[float] | None = None,
        max_joint_delta: float = 0.17,
        ramp_step: float = 0.01,
        ramp_max_steps: int = 100,
        gripper_update_threshold: float = 0.02,
        servo_step_limit: float = 0.9,
    ) -> None:
        left_cameras: dict[str, str] = {}
        right_cameras: dict[str, str] = {}

        if camera_top_serial:
            left_cameras["cam_top"] = camera_top_serial
        if camera_left_wrist_serial:
            left_cameras["cam_left_wrist"] = camera_left_wrist_serial
        if camera_right_wrist_serial:
            right_cameras["cam_right_wrist"] = camera_right_wrist_serial

        self._follower_left = _create_follower(
            robot_ip=left_robot_ip,
            gripper_port=left_gripper_port,
            gripper_id=left_gripper_id,
            gripper_servo_pos=left_gripper_servo_pos,
            read_gripper_position=False,
            max_delta_per_step=servo_step_limit,
            camera_serials=left_cameras,
            camera_fps=camera_fps,
        )
        self._follower_right = _create_follower(
            robot_ip=right_robot_ip,
            gripper_port=right_gripper_port,
            gripper_id=right_gripper_id,
            gripper_servo_pos=right_gripper_servo_pos,
            read_gripper_position=False,
            max_delta_per_step=servo_step_limit,
            camera_serials=right_cameras,
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
        self._last_gripper_sent = np.array([1.0, 1.0], dtype=np.float64)

        self._reset_pose = None
        if reset_pose is not None:
            reset = np.asarray(reset_pose, dtype=np.float64).reshape(-1)
            if reset.shape[0] != 14:
                raise ValueError(f"Expected reset_pose length 14, got {reset.shape[0]}")
            self._reset_pose = reset

    @override
    def reset(self) -> None:
        self._ensure_connected()
        if self._reset_pose is not None:
            self._move_smooth(self._get_bimanual_qpos(), self._reset_pose)
            time.sleep(0.2)
        self._last_action = self._get_bimanual_qpos()
        self._last_gripper_sent = np.array([self._last_action[6], self._last_action[13]], dtype=np.float64)

    @override
    def is_episode_complete(self) -> bool:
        return False

    @override
    def get_observation(self) -> dict:
        self._ensure_connected()
        observation = {
            "observation.state": self._get_bimanual_qpos().astype(np.float32),
        }
        for camera_name in ("top", "left_wrist", "right_wrist"):
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
        if target.shape[0] != 14:
            raise ValueError(f"Expected action length 14, got {target.shape[0]}")

        target[6] = float(np.clip(target[6], 0.0, 1.0))
        target[13] = float(np.clip(target[13], 0.0, 1.0))

        if self._last_action is None:
            self._last_action = self._get_bimanual_qpos()

        max_joint_delta = max(
            float(np.max(np.abs(target[:6] - self._last_action[:6]))),
            float(np.max(np.abs(target[7:13] - self._last_action[7:13]))),
        )
        if max_joint_delta > self._max_joint_delta:
            self._move_smooth(self._last_action, target)
        else:
            self._send_bimanual_action(target)

        self._last_action = target.copy()

    def close(self) -> None:
        if not self._connected:
            return
        for follower in (self._follower_left, self._follower_right):
            try:
                follower.disconnect()
            except Exception:
                logger.exception("Failed to disconnect follower cleanly.")
        self._connected = False

    def _ensure_connected(self) -> None:
        if self._connected:
            return
        self._follower_left.connect()
        self._follower_right.connect()
        self._connected = True
        self._last_action = self._get_bimanual_qpos()
        self._last_gripper_sent = np.array([self._last_action[6], self._last_action[13]], dtype=np.float64)

    def _read_camera_frame(self, camera_name: str, retries: int = 5) -> np.ndarray:
        camera = self._get_camera(camera_name)
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

    def _get_camera(self, camera_name: str):
        if camera_name == "top":
            if "cam_top" not in self._follower_left.cameras:
                raise RuntimeError("Missing camera cam_top on left follower")
            return self._follower_left.cameras["cam_top"]
        if camera_name == "left_wrist":
            if "cam_left_wrist" not in self._follower_left.cameras:
                raise RuntimeError("Missing camera cam_left_wrist on left follower")
            return self._follower_left.cameras["cam_left_wrist"]
        if camera_name == "right_wrist":
            if "cam_right_wrist" not in self._follower_right.cameras:
                raise RuntimeError("Missing camera cam_right_wrist on right follower")
            return self._follower_right.cameras["cam_right_wrist"]
        raise ValueError(f"Unsupported camera name: {camera_name}")

    def _get_bimanual_qpos(self) -> np.ndarray:
        left_obs = _obs_dict_to_arm_array(self._follower_left.get_low_latency_observation())
        right_obs = _obs_dict_to_arm_array(self._follower_right.get_low_latency_observation())
        return np.concatenate([left_obs, right_obs])

    def _split_bimanual_action(self, action: np.ndarray) -> tuple[dict[str, float], dict[str, float]]:
        left = {f"joint{joint_index + 1}.pos": float(action[joint_index]) for joint_index in range(6)}
        left["gripper.pos"] = float(action[6])

        right = {f"joint{joint_index + 1}.pos": float(action[7 + joint_index]) for joint_index in range(6)}
        right["gripper.pos"] = float(action[13])
        return left, right

    def _send_bimanual_action(self, action: np.ndarray) -> None:
        left_action, right_action = self._split_bimanual_action(action)

        left_gripper = float(left_action["gripper.pos"])
        if abs(left_gripper - float(self._last_gripper_sent[0])) < self._gripper_update_threshold:
            left_action.pop("gripper.pos", None)
        else:
            self._last_gripper_sent[0] = left_gripper

        right_gripper = float(right_action["gripper.pos"])
        if abs(right_gripper - float(self._last_gripper_sent[1])) < self._gripper_update_threshold:
            right_action.pop("gripper.pos", None)
        else:
            self._last_gripper_sent[1] = right_gripper

        self._follower_left.send_action(left_action)
        self._follower_right.send_action(right_action)

    def _move_smooth(self, start_action: np.ndarray, goal_action: np.ndarray) -> None:
        max_delta = float(np.max(np.abs(goal_action - start_action)))
        steps = min(int(np.ceil(max_delta / max(self._ramp_step, 1e-6))), self._ramp_max_steps)
        if steps <= 1:
            self._send_bimanual_action(goal_action)
            return

        for joints in np.linspace(start_action, goal_action, steps):
            self._send_bimanual_action(joints)

    def __del__(self) -> None:
        with contextlib.suppress(Exception):
            self.close()
