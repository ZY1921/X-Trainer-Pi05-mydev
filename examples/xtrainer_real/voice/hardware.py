"""Voice camera cleanup and the shared V4 robot command transport."""

from examples.xtrainer_real.hardware.dobot_xtrainer import DobotConnection
from examples.xtrainer_real.hardware.dobot_xtrainer import DobotXTrainer
from examples.xtrainer_real.hardware.realsense_camera import RealSenseCamera


class StrictDobotConnection(DobotConnection):
    """Voice-compatible alias of the shared strict V4 command connection."""


class InterruptibleRealSenseCamera(RealSenseCamera):
    """Compatibility name for the shared camera's interruptible warmup."""


class VoiceDobotXTrainer(DobotXTrainer):
    """V4 follower with an interruptible camera warmup."""

    def __init__(self, *, camera_serials: dict[str, str] | None = None, camera_fps: float = 30.0, **kwargs):
        super().__init__(camera_serials={}, camera_fps=camera_fps, **kwargs)
        camera_serials = camera_serials or {}
        self.cameras = {
            name: InterruptibleRealSenseCamera(serial, fps=round(camera_fps))
            for name, serial in camera_serials.items()
            if serial
        }

    def _create_connection(self) -> StrictDobotConnection:
        return StrictDobotConnection(self.robot_ip, 29999)
