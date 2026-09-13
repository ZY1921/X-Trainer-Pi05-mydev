"""Voice-only safety wrappers around the established V3 robot hardware driver."""

import contextlib
import time

import numpy as np

from examples.xtrainer_real.hardware.dobot_xtrainer import DobotConnection
from examples.xtrainer_real.hardware.dobot_xtrainer import DobotXTrainer
from examples.xtrainer_real.hardware.realsense_camera import RealSenseCamera


class StrictDobotConnection(DobotConnection):
    """Fail closed on missing, malformed, partial, or rejected Dobot replies."""

    def send_recv(self, cmd: str) -> str:
        if self._socket is None:
            raise RuntimeError("Dobot socket is not connected")
        with self._lock:
            self._socket.sendall(cmd.encode("utf-8"))
            data = self._socket.recv(1024)
            while data and not data.rstrip().endswith(b";"):
                if len(data) >= 65536:
                    raise ValueError("Dobot response exceeds 64 KiB without a terminator")
                fragment = self._socket.recv(1024)
                if not fragment:
                    raise ConnectionError(f"Incomplete Dobot reply to {cmd}")
                data += fragment
            if not data:
                raise ConnectionError(f"Empty Dobot reply to {cmd}")
            response = data.decode("utf-8")
            try:
                status = int(response.split(",", 1)[0])
            except ValueError as error:
                raise ValueError(f"Malformed Dobot reply to {cmd}: {response}") from error
            if status != 0:
                raise RuntimeError(f"Dobot rejected {cmd}: {response}")
            return response


class InterruptibleRealSenseCamera(RealSenseCamera):
    """Register the pipeline before warmup so an interrupted start can release it."""

    def connect(self) -> None:
        if self._connected:
            return

        import pyrealsense2 as rs

        pipeline = rs.pipeline()
        config = rs.config()
        if self.serial:
            config.enable_device(self.serial)
        config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
        pipeline.start(config)
        self._pipeline = pipeline
        self._connected = True
        try:
            for _ in range(10):
                pipeline.wait_for_frames(timeout_ms=5000)
        except BaseException:
            self.disconnect()
            raise


class VoiceDobotXTrainer(DobotXTrainer):
    """The normal V3 driver with strict replies and reliable partial-start cleanup."""

    def __init__(self, *, camera_serials: dict[str, str] | None = None, camera_fps: float = 30.0, **kwargs):
        super().__init__(camera_serials={}, camera_fps=camera_fps, **kwargs)
        camera_serials = camera_serials or {}
        self.cameras = {
            name: InterruptibleRealSenseCamera(serial, fps=round(camera_fps))
            for name, serial in camera_serials.items()
            if serial
        }

    def connect(self) -> None:
        if self._connected:
            return
        try:
            self._dashboard = StrictDobotConnection(self.robot_ip, 29999)
            self._dashboard.connect()
            self._move = StrictDobotConnection(self.robot_ip, 30003, timeout=1.0)
            self._move.connect()

            self._dashboard.send_recv("EnableRobot()")
            time.sleep(0.5)
            self._dashboard.send_recv(f"SpeedFactor({self.speed_factor})")
            self._dashboard.send_recv(f"AccJ({self.acc_j})")
            self._dashboard.send_recv(f"SpeedJ({self.speed_j})")

            tx, ty, tz, trx, try_, trz = self.tool_offset
            self._dashboard.send_recv(
                f"SetTool(1,{{{tx:.3f},{ty:.3f},{tz:.3f},{trx:.3f},{try_:.3f},{trz:.3f}}})"
            )
            self._dashboard.send_recv("Tool(1)")
            self._dashboard.send_recv("StopDrag()")

            if self._gripper is not None:
                self._gripper.connect()
                current_gripper = self._gripper.get_position(retries=3)
                if current_gripper is not None:
                    self._last_gripper_pos = current_gripper
                self._gripper.set_position(0.0)
                time.sleep(0.3)
                self._gripper.set_position(1.0)

            for camera in self.cameras.values():
                camera.connect()
            self._connected = True
        except BaseException:
            self.disconnect()
            raise

    def _read_joint_positions(self) -> list[float]:
        if self._dashboard is None:
            raise RuntimeError("Dashboard connection is not available")
        response = self._dashboard.send_recv("GetAngle()")
        try:
            angles = list(map(float, response.split("{", 1)[1].split("}", 1)[0].split(",")))
        except (IndexError, ValueError) as error:
            raise RuntimeError(f"Invalid GetAngle response: {response}") from error
        if len(angles) != 6 or not np.all(np.isfinite(angles)):
            raise RuntimeError(f"Invalid GetAngle response: {response}")
        return [float(np.deg2rad(angle)) for angle in angles]

    def disconnect(self) -> None:
        for resource in (*self.cameras.values(), self._gripper):
            if resource is not None:
                with contextlib.suppress(Exception):
                    resource.disconnect()
        if self._dashboard is not None and self.disable_torque_on_disconnect:
            with contextlib.suppress(Exception):
                self._dashboard.send_recv("DisableRobot()")
        for connection in (self._dashboard, self._move):
            if connection is not None:
                with contextlib.suppress(Exception):
                    connection.close()
        self._dashboard = None
        self._move = None
        self._connected = False
