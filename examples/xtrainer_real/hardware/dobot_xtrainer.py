import contextlib
import logging
import os
import re
import socket
import threading
import time

import numpy as np

from examples.xtrainer_real.hardware.realsense_camera import RealSenseCamera
from examples.xtrainer_real.scservo_sdk import COMM_SUCCESS
from examples.xtrainer_real.scservo_sdk import PortHandler
from examples.xtrainer_real.scservo_sdk import protocol_packet_handler
from examples.xtrainer_real.scservo_sdk import sms_sts

logger = logging.getLogger(__name__)

JOINT_NAMES = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]


class DobotConnection:
    """Strict V4 command connection to the controller's port 29999."""

    MAX_REPLY_BYTES = 65536

    def __init__(self, ip: str, port: int, timeout: float = 3.0):
        self.ip = ip
        self.port = int(port)
        self.timeout = float(timeout)
        self._socket: socket.socket | None = None
        self._lock = threading.RLock()

    def connect(self) -> None:
        connection = socket.socket()
        try:
            connection.settimeout(self.timeout)
            connection.connect((self.ip, self.port))
        except BaseException:
            connection.close()
            raise
        self._socket = connection

    def send_recv(self, cmd: str) -> str:
        with self._lock:
            if self._socket is None:
                raise RuntimeError("Dobot socket is not connected")
            try:
                deadline = time.monotonic() + self.timeout
                self._socket.settimeout(self.timeout)
                self._socket.sendall(cmd.encode("utf-8"))
                data = bytearray()
                while b";" not in data:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError(f"Timed out waiting for Dobot reply to {cmd}")
                    self._socket.settimeout(remaining)
                    fragment = self._socket.recv(min(1024, self.MAX_REPLY_BYTES - len(data)))
                    if not fragment:
                        raise ConnectionError(f"Incomplete Dobot reply to {cmd}")
                    data.extend(fragment)
                    if len(data) >= self.MAX_REPLY_BYTES:
                        raise ValueError("Dobot reply exceeds 64 KiB")
                reply, trailing = data.split(b";", 1)
                # Only one command is in flight. Extra replies cannot acknowledge a future command.
                if trailing.strip():
                    raise ValueError(f"Unexpected extra Dobot reply to {cmd}")
                response = (reply + b";").decode("ascii").strip()
                match = re.fullmatch(r"([+-]?\d+)\s*,\s*\{(.*)\}\s*,\s*(\w+)\s*\((.*)\)\s*;", response)
                if match is None:
                    raise ValueError(f"Malformed Dobot reply to {cmd}: {response}")
                if match[3] != cmd.split("(", 1)[0]:
                    raise ValueError(f"Unexpected Dobot reply to {cmd}: {response}")
                status = int(match[1])
            except BaseException:
                # After a timeout or broken frame, a late reply must never be used by the next command.
                self.close()
                raise
            if status != 0:
                raise RuntimeError(f"Dobot rejected {cmd}: {response}")
            return response

    def close(self) -> None:
        with self._lock:
            connection, self._socket = self._socket, None
            if connection is not None:
                with contextlib.suppress(OSError):
                    connection.close()


class DobotGripperWrapper:
    """Feetech SMS/STS servo wrapper."""

    def __init__(self, port: str, servo_id: int, servo_pos: tuple[int, int], torque_limit: int = 300):
        self.port = port
        self.servo_id = int(servo_id)
        self.servo_pos_min = int(servo_pos[0])
        self.servo_pos_max = int(servo_pos[1])
        self.torque_limit = int(torque_limit)
        self._port_handler: PortHandler | None = None
        self._servo: sms_sts | None = None
        self._pack_handler = None
        self._io_lock = threading.Lock()

    def connect(self) -> None:
        self._port_handler = PortHandler(self.port)
        self._pack_handler = protocol_packet_handler(portHandler=self._port_handler, protocol_end=0)
        self._servo = sms_sts(self._port_handler)

        if not self._port_handler.openPort():
            raise RuntimeError(f"Failed to open gripper port {self.port}")
        self._port_handler.setBaudRate(1000000)

        self._set_latency_timer()

        _, result, error = self._servo.ping(self.servo_id)
        if error != 0:
            raise ConnectionError(
                f"Failed to ping gripper servo {self.servo_id} on {self.port}, result={result}, error={error}"
            )

        comm_result, _ = self._pack_handler.write2ByteTxRx(self.servo_id, 48, self.torque_limit)
        if comm_result != COMM_SUCCESS:
            logger.warning("Failed to set gripper torque limit: %s", comm_result)

    def _set_latency_timer(self) -> None:
        port_name = self.port.split("/")[-1]
        latency_path = f"/sys/bus/usb-serial/devices/{port_name}/latency_timer"
        try:
            with open(latency_path, "w") as file_obj:
                file_obj.write("1")
        except PermissionError:
            try:
                os.system(f"echo 1 | sudo tee {latency_path} > /dev/null")
            except Exception:
                logger.debug("Could not set latency timer for %s", port_name)
        except FileNotFoundError:
            logger.debug("Latency timer path not found: %s", latency_path)

    def get_position(self, retries: int = 2) -> float | None:
        if self._servo is None:
            raise RuntimeError("Gripper servo is not connected")
        last_result = 0
        last_error = 0
        for _ in range(max(1, retries)):
            with self._io_lock:
                position, result, error = self._servo.ReadPos(self.servo_id)
            if result == 0:
                normalized = (position - self.servo_pos_min) / (self.servo_pos_max - self.servo_pos_min)
                return float(np.clip(normalized, 0.0, 1.0))
            last_result = result
            last_error = error
            time.sleep(0.001)

        logger.debug("Failed to read gripper position: result=%s, error=%s", last_result, last_error)
        return None

    def set_position(self, pos: float, speed: int = 100) -> None:
        if self._servo is None:
            raise RuntimeError("Gripper servo is not connected")
        pos = float(np.clip(pos, 0.0, 1.0))
        servo_pos = int(pos * (self.servo_pos_max - self.servo_pos_min) + self.servo_pos_min)
        servo_speed = int(speed / 100.0 * 4096)
        with self._io_lock:
            self._servo.WritePosEx(self.servo_id, servo_pos, servo_speed, 0)

    def disconnect(self) -> None:
        if self._port_handler is not None:
            self._port_handler.closePort()
            self._port_handler = None


class DobotXTrainer:
    """Minimal xtrainer follower arm for inference runtime."""

    def __init__(
        self,
        *,
        robot_ip: str,
        gripper_port: str,
        gripper_id: int,
        gripper_servo_pos: tuple[int, int],
        use_gripper: bool = True,
        read_gripper_position: bool = False,
        tool_offset: tuple[float, float, float, float, float, float] = (0, 0, 197, 0, 0, 0),
        speed_factor: int = 20,
        acc_j: int = 20,
        speed_j: int = 20,
        servo_j_time: float = 0.03,
        servo_j_aheadtime: float = 50.0,
        servo_j_gain: float = 500.0,
        disable_torque_on_disconnect: bool = True,
        max_delta_per_step: float = 0.9,
        camera_serials: dict[str, str] | None = None,
        camera_fps: float = 30.0,
    ):
        if not 0.004 <= servo_j_time <= 3600.0:
            raise ValueError("servo_j_time must be in [0.004, 3600.0]")
        if not 20.0 <= servo_j_aheadtime <= 100.0:
            raise ValueError("servo_j_aheadtime must be in [20.0, 100.0]")
        if not 200.0 <= servo_j_gain <= 1000.0:
            raise ValueError("servo_j_gain must be in [200.0, 1000.0]")
        self.robot_ip = robot_ip
        self.read_gripper_position = read_gripper_position
        self.tool_offset = tool_offset
        self.speed_factor = speed_factor
        self.acc_j = acc_j
        self.speed_j = speed_j
        self.servo_j_time = servo_j_time
        self.servo_j_aheadtime = servo_j_aheadtime
        self.servo_j_gain = servo_j_gain
        self.disable_torque_on_disconnect = disable_torque_on_disconnect
        self.max_delta_per_step = max_delta_per_step

        self._dashboard: DobotConnection | None = None
        self._connected = False
        self._enable_attempted = False
        self._faulted = False
        self._next_servo_time = 0.0
        self._last_action_rad: np.ndarray | None = None
        self._last_gripper_pos = 1.0

        self._gripper = None
        if use_gripper:
            self._gripper = DobotGripperWrapper(
                port=gripper_port,
                servo_id=gripper_id,
                servo_pos=gripper_servo_pos,
            )

        camera_serials = camera_serials or {}
        self.cameras: dict[str, RealSenseCamera] = {
            name: RealSenseCamera(serial, fps=round(camera_fps)) for name, serial in camera_serials.items() if serial
        }

    @property
    def is_connected(self) -> bool:
        return self._connected and all(camera.is_connected for camera in self.cameras.values())

    def _create_connection(self) -> DobotConnection:
        return DobotConnection(self.robot_ip, 29999)

    def connect(self) -> None:
        if self._connected:
            return
        if self._faulted:
            raise RuntimeError("Dobot connection faulted; recreate the follower before sending actions")
        try:
            self._dashboard = self._create_connection()
            self._dashboard.connect()
            self._dashboard.send_recv("RequestControl()")
            self._enable_attempted = True  # The controller may enable even if its reply is lost.
            self._dashboard.send_recv("EnableRobot()")
            time.sleep(0.5)
            self._dashboard.send_recv(f"SpeedFactor({self.speed_factor})")
            self._dashboard.send_recv(f"AccJ({self.acc_j})")
            self._dashboard.send_recv(f"VelJ({self.speed_j})")

            tx, ty, tz, trx, try_, trz = self.tool_offset
            self._dashboard.send_recv(f"SetTool(1,{{{tx:.3f},{ty:.3f},{tz:.3f},{trx:.3f},{try_:.3f},{trz:.3f}}})")
            self._dashboard.send_recv("Tool(1)")
            self._dashboard.send_recv("StopDrag()")
            self._last_action_rad = np.asarray(self._read_joint_positions(), dtype=np.float64)

            if self._gripper is not None:
                self._gripper.connect()
                current_gripper = self._gripper.get_position(retries=3)
                if current_gripper is not None:
                    self._last_gripper_pos = current_gripper
                self._gripper.set_position(0.0)
                time.sleep(0.3)
                self._gripper.set_position(1.0)
                self._last_gripper_pos = 1.0

            for camera in self.cameras.values():
                camera.connect()
            self._connected = True
        except BaseException:
            self._faulted = True
            self.disconnect()
            raise

    def set_do_status(self, channel: int, status: int) -> None:
        if not self._connected or self._dashboard is None:
            raise RuntimeError("Robot not connected")
        try:
            self._dashboard.send_recv(f"DOInstant({int(channel)},{int(status)})")
        except BaseException:
            self._faulted = True
            self.disconnect()
            raise

    def _read_joint_positions(self) -> list[float]:
        if self._dashboard is None:
            raise RuntimeError("Dashboard connection is not available")
        angle_str = self._dashboard.send_recv("GetAngle()")
        try:
            angles_deg = [float(value) for value in angle_str.split("{", 1)[1].split("}", 1)[0].split(",")]
        except (IndexError, ValueError) as error:
            raise RuntimeError(f"Invalid GetAngle response: {angle_str}") from error
        if len(angles_deg) != 6 or not np.all(np.isfinite(angles_deg)):
            raise RuntimeError(f"Invalid GetAngle response: {angle_str}")
        return [float(np.deg2rad(angle)) for angle in angles_deg]

    def get_low_latency_observation(self) -> dict[str, float]:
        if not self._connected:
            raise RuntimeError("Robot not connected")
        observation: dict[str, float] = {}
        try:
            angles_rad = self._read_joint_positions()
        except BaseException:
            self._faulted = True
            self.disconnect()
            raise
        for index, name in enumerate(JOINT_NAMES):
            observation[f"{name}.pos"] = float(angles_rad[index])
        if self._gripper is not None:
            observation["gripper.pos"] = float(self._last_gripper_pos)
        return observation

    def send_action(self, action: dict[str, float]) -> dict[str, float]:
        if not self._connected:
            raise RuntimeError("Robot not connected")
        if self._dashboard is None:
            raise RuntimeError("Dashboard connection is not available")
        try:
            return self._send_action(action)
        except BaseException:
            self._faulted = True
            self.disconnect()
            raise

    def _send_action(self, action: dict[str, float]) -> dict[str, float]:
        # Missing joints must not silently become a zero-angle motion target.
        goal_rad = np.array([action[f"{name}.pos"] for name in JOINT_NAMES], dtype=np.float64)
        if not np.all(np.isfinite(goal_rad)):
            raise ValueError("Joint action contains a non-finite value")
        if "gripper.pos" in action and not np.isfinite(action["gripper.pos"]):
            raise ValueError("Gripper action contains a non-finite value")

        max_delta = self.max_delta_per_step
        if max_delta is not None and max_delta > 0 and self._last_action_rad is not None:
            delta = goal_rad - self._last_action_rad
            abs_delta = np.abs(delta)
            if np.any(abs_delta > max_delta):
                for joint_index in range(6):
                    if abs_delta[joint_index] <= max_delta:
                        continue
                    pi_ratio = delta[joint_index] / np.pi
                    if 1.85 < abs(pi_ratio) < 2.15:
                        goal_rad[joint_index] = goal_rad[joint_index] - 2 * np.pi * np.sign(pi_ratio)
                    if abs(goal_rad[joint_index] - self._last_action_rad[joint_index]) > max_delta:
                        raise RuntimeError(
                            f"Action jump too large on joint {joint_index}: delta={abs_delta[joint_index]:.3f}rad"
                        )
        goal_deg = [float(np.rad2deg(value)) for value in goal_rad]
        command = (
            f"ServoJ({goal_deg[0]:.6f},{goal_deg[1]:.6f},{goal_deg[2]:.6f},"
            f"{goal_deg[3]:.6f},{goal_deg[4]:.6f},{goal_deg[5]:.6f},"
            f"t={self.servo_j_time:.3f},aheadtime={self.servo_j_aheadtime:.1f},gain={self.servo_j_gain:.1f})"
        )
        # V4 acknowledges receipt, not completion. Pace reset/interpolation bursts as well as inference.
        delay = self._next_servo_time - time.monotonic()
        if delay > 0:
            time.sleep(delay)
        self._next_servo_time = time.monotonic() + self.servo_j_time
        self._dashboard.send_recv(command)
        self._last_action_rad = goal_rad.copy()
        if self._gripper is not None and "gripper.pos" in action:
            gripper_pos = float(np.clip(action["gripper.pos"], 0.0, 1.0))
            self._gripper.set_position(gripper_pos)
            self._last_gripper_pos = gripper_pos

        sent = {f"{name}.pos": float(goal_rad[index]) for index, name in enumerate(JOINT_NAMES)}
        if self._gripper is not None:
            sent["gripper.pos"] = float(self._last_gripper_pos)
        return sent

    def disconnect(self) -> None:
        self._connected = False
        # Stop the motion queue before potentially slow camera/serial cleanup, even when keeping torque on.
        if self._dashboard is not None:
            if self._enable_attempted:
                commands = ["Stop()"]
                if self.disable_torque_on_disconnect:
                    commands.append("DisableRobot()")
                for command in commands:
                    try:
                        self._dashboard.send_recv(command)
                    except Exception:
                        logger.warning("Could not confirm Dobot %s during disconnect", command, exc_info=True)
            with contextlib.suppress(Exception):
                self._dashboard.close()
            self._dashboard = None
        self._enable_attempted = False
        self._last_action_rad = None
        self._next_servo_time = 0.0
        for camera in self.cameras.values():
            with contextlib.suppress(Exception):
                camera.disconnect()

        if self._gripper is not None:
            with contextlib.suppress(Exception):
                self._gripper.disconnect()
