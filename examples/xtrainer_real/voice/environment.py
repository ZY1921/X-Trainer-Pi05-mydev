"""Voice-only extension points for the established right-arm inference environment."""

import numpy as np

from examples.xtrainer_real.inference_environment import RightArmInferenceEnvironment
from examples.xtrainer_real.voice.hardware import VoiceDobotXTrainer


class VoiceRightArmInferenceEnvironment(RightArmInferenceEnvironment):
    """Add task switching and strict hardware without changing normal inference."""

    def __init__(
        self,
        *,
        right_robot_ip: str,
        right_gripper_port: str,
        right_gripper_id: int,
        right_gripper_servo_pos: tuple[int, int],
        camera_top_serial: str,
        camera_right_wrist_serial: str,
        camera_fps: float = 30.0,
        **kwargs,
    ) -> None:
        super().__init__(
            right_robot_ip=right_robot_ip,
            right_gripper_port=right_gripper_port,
            right_gripper_id=right_gripper_id,
            right_gripper_servo_pos=right_gripper_servo_pos,
            camera_top_serial=camera_top_serial,
            camera_right_wrist_serial=camera_right_wrist_serial,
            camera_fps=camera_fps,
            **kwargs,
        )
        # The parent follower is not connected yet. Replace it with the voice-only
        # strict implementation while retaining the proven environment behavior.
        self._follower = VoiceDobotXTrainer(
            robot_ip=right_robot_ip,
            gripper_port=right_gripper_port,
            gripper_id=right_gripper_id,
            gripper_servo_pos=right_gripper_servo_pos,
            read_gripper_position=False,
            max_delta_per_step=kwargs.get("servo_step_limit", 0.9),
            camera_serials={
                "cam_top": camera_top_serial,
                "cam_right_wrist": camera_right_wrist_serial,
            },
            camera_fps=camera_fps,
        )

    def set_prompt(self, prompt: str | None) -> None:
        self._prompt = prompt

    def get_current_qpos(self) -> np.ndarray:
        self._ensure_connected()
        qpos = self._get_qpos()
        if qpos.shape != (7,) or not np.all(np.isfinite(qpos)):
            raise ValueError("Invalid right-arm joint feedback")
        return qpos.copy()
