"""Run asynchronous RTC inference on the physical right arm with two cameras."""

# The established async module exposes reusable worker/agent validation helpers.
# ruff: noqa: SLF001

import dataclasses
import logging
from typing import Any, Literal

import numpy as np
from openpi_client.runtime import runtime as _runtime
from typing_extensions import override
import tyro

from examples.xtrainer_real import async_rtc_main as _async_base
from examples.xtrainer_real import diagnostics as _diagnostics
from examples.xtrainer_real import image_preprocessing as _image_preprocessing
from examples.xtrainer_real import inference_action_recorder as _action_recorder
from examples.xtrainer_real import right_arm_env as _right_arm_env
from examples.xtrainer_real import right_arm_single_main as _sync_right_arm

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class Args(_sync_right_arm.Args):
    rtc_enabled: bool = True
    rtc_inference_delay: int = 4
    rtc_execution_horizon: int = 10
    rtc_prefix_attention_schedule: Literal["exp", "linear", "ones", "zeros"] = "exp"
    rtc_max_guidance_weight: float = 10.0
    max_inference_delay_steps: int = 15
    inference_timeout_s: float = 3.0
    connect_timeout_s: float = 120.0
    warmup_timeout_s: float = 180.0
    warmup_rtc: bool = True
    debug_async_timing: bool = False
    image_transport_codec: Literal["raw", "jpeg"] = "jpeg"
    image_jpeg_quality: int = 90


class PreprocessedRightArmEnvironment(_right_arm_env.XTrainerRightArmRealEnvironment):
    @override
    def get_observation(self) -> dict:
        observation = super().get_observation()
        return _image_preprocessing.match_legacy_dataset_images(observation)


class RightArmAsyncRTCPolicyAgent(_async_base.AsyncRTCPolicyAgent):
    def _validated_result(self, result: dict[str, Any] | None) -> tuple[dict[str, Any], np.ndarray]:
        result, actions = super()._validated_result(result)
        if actions.shape[-1] != _right_arm_env.PHYSICAL_ACTION_DIM:
            raise ValueError(f"Expected right-arm actions with shape (horizon, 7), got {actions.shape}.")
        return result, actions


def main(args: Args) -> None:
    if (args.render_height, args.render_width) != (480, 640):
        raise ValueError("Right-arm image preprocessing requires --render-height 480 --render-width 640.")
    if not 0 <= args.inference_action_plot_start_index < _right_arm_env.PHYSICAL_ACTION_DIM:
        raise ValueError("--inference-action-plot-start-index must be in [0, 6].")
    if args.rtc_enabled:
        _async_base._validate_rtc_args(args)
    _async_base._validate_image_transport_args(args)

    worker = _async_base.AsyncRTCInferenceWorker(
        args.host,
        args.port,
        connect_timeout_s=args.connect_timeout_s,
        rtc_inference_delay=args.rtc_inference_delay,
        rtc_execution_horizon=args.rtc_execution_horizon,
        rtc_prefix_attention_schedule=args.rtc_prefix_attention_schedule,
        rtc_max_guidance_weight=args.rtc_max_guidance_weight,
        image_transport_codec=args.image_transport_codec,
        image_jpeg_quality=args.image_jpeg_quality,
    )
    try:
        _run_robot(args, worker)
    finally:
        worker.close()


def _run_robot(args: Args, worker: _async_base.AsyncRTCInferenceWorker) -> None:
    metadata = worker.metadata
    _sync_right_arm.validate_right_arm_metadata(metadata)
    logger.info("Validated right-arm server metadata: %s", metadata)

    recorder = None
    if args.record_inference_actions:
        recorder = _action_recorder.InferenceActionRecorder(
            args.inference_action_output_dir,
            mode="right_arm_async_rtc" if args.rtc_enabled else "right_arm_async_baseline",
            plot_start_index=args.inference_action_plot_start_index,
            config=dataclasses.asdict(args),
        )

    environment = PreprocessedRightArmEnvironment(
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
    try:
        subscribers = []
        if args.debug_action_state_diagnostics:
            subscribers.append(
                _diagnostics.ActionStateDiagnosticsSubscriber(
                    interval=args.debug_action_state_interval,
                    max_steps=args.debug_action_state_max_steps,
                )
            )

        agent = RightArmAsyncRTCPolicyAgent(
            worker,
            request_interval=args.action_horizon,
            rtc_enabled=args.rtc_enabled,
            max_inference_delay_steps=args.max_inference_delay_steps,
            inference_timeout_s=args.inference_timeout_s,
            warmup_timeout_s=args.warmup_timeout_s,
            warmup_rtc=args.warmup_rtc,
            debug_timing=args.debug_async_timing,
            action_recorder=recorder,
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
        try:
            environment.close()
        finally:
            if recorder is not None:
                recorder.save()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
