"""Repeated local voice tasks using the September 9 right-arm RTC setup."""

import dataclasses
import json
import logging
import math
from pathlib import Path
from typing import Literal

import tyro

from examples.xtrainer_real.voice.command_parser import ParseResult
from examples.xtrainer_real.voice.command_parser import parse_command
from examples.xtrainer_real.voice.config import VoiceConfig
from examples.xtrainer_real.voice.task_completion import CompletionConfig
from examples.xtrainer_real.voice.task_registry import get_task

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class Args:
    dry_run: bool = False
    text: str | None = None
    audio_file: Path | None = None
    model_path: Path = VoiceConfig.model_path
    mic_device: int = 5
    require_wake_word: bool = True
    host: str = "100.100.41.112"
    port: int = 18080
    right_robot_ip: str = "192.168.5.2"
    right_gripper_port: str = "/dev/ttyUSB0"
    right_gripper_id: int = 22
    right_gripper_servo_pos: tuple[int, int] = (2048, 3052)
    camera_top_serial: str = "409122272170"
    camera_right_wrist_serial: str = "409122270904"
    control_hz: float = 22.0
    action_horizon: int = 25
    camera_fps: float = 30.0
    render_height: int = 480
    render_width: int = 640
    max_joint_delta: float = 0.17
    ramp_step: float = 0.01
    ramp_max_steps: int = 100
    gripper_update_threshold: float = 0.02
    servo_step_limit: float = 0.9
    rtc_enabled: bool = True
    rtc_inference_delay: int = 6
    rtc_execution_horizon: int = 15
    rtc_prefix_attention_schedule: Literal["exp", "linear", "ones", "zeros"] = "linear"
    rtc_max_guidance_weight: float = 10.0
    max_inference_delay_steps: int = 24
    inference_timeout_s: float = 3.0
    connect_timeout_s: float = 120.0
    warmup_timeout_s: float = 600.0
    warmup_rtc: bool = True
    debug_async_timing: bool = True
    image_transport_codec: Literal["raw", "jpeg"] = "jpeg"
    image_jpeg_quality: int = 90
    show_camera_preview: bool = False
    record_inference_actions: bool = True
    inference_action_output_dir: Path = Path("output/voice")
    motion_start_threshold: float = 0.02
    motion_start_confirm_frames: int = 3
    departure_threshold: float = 0.20
    completion_ignore_seconds: float = 6.0
    return_threshold: float = 0.15
    return_confirm_frames: int = 5
    max_episode_steps: int = 450
    motion_start_timeout_steps: int = 110


def print_command(command: ParseResult) -> None:
    output = dataclasses.asdict(command)
    output["prompt"] = get_task(command.task_id).prompt if command.task_id is not None else None
    print(json.dumps(output, ensure_ascii=False, indent=2))


def run_session(service, executor_factory, completion: CompletionConfig) -> None:
    """One recorded utterance authorizes at most one task; no input thread or queue."""
    executor = None
    while True:
        logger.info("LISTENING")
        command = service.listen_once()
        print_command(command)
        if command.task_id is None:
            continue
        if executor is None:
            executor = executor_factory()
        result = executor.execute(get_task(command.task_id, completion), command)
        print(json.dumps(dataclasses.asdict(result), ensure_ascii=False, indent=2))


def main(args: Args) -> None:
    if args.text is not None and args.audio_file is not None:
        raise ValueError("--text and --audio-file are mutually exclusive")
    if (args.text is not None or args.audio_file is not None) and not args.dry_run:
        raise ValueError("--text / --audio-file require --dry-run")
    completion = CompletionConfig(
        **{field.name: getattr(args, field.name) for field in dataclasses.fields(CompletionConfig)}
    )
    for name in ("control_hz", "inference_timeout_s", "connect_timeout_s", "warmup_timeout_s"):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if args.text is not None:
        print_command(parse_command(args.text, require_wake_word=args.require_wake_word))
        return

    from examples.xtrainer_real.voice.recognizer import SpeechRecognizer

    logger.info("Loading local Whisper: %s", args.model_path)
    recognizer = SpeechRecognizer(args.model_path)
    if args.audio_file is not None:
        print_command(parse_command(recognizer.transcribe(args.audio_file), require_wake_word=args.require_wake_word))
        return

    from examples.xtrainer_real.voice.recorder import AudioRecorder
    from examples.xtrainer_real.voice.service import VoiceService

    service = VoiceService(
        AudioRecorder(VoiceConfig(model_path=args.model_path, mic_device=args.mic_device)),
        recognizer,
        require_wake_word=args.require_wake_word,
    )
    if args.dry_run:
        print_command(service.listen_once())
        return
    _run_live(args, service, completion)


def _run_live(args: Args, service, completion: CompletionConfig) -> None:
    # Keep all robot/policy imports out of text and audio dry-run modes.
    from examples.xtrainer_real import bimanual_async_rtc_main as rtc
    from examples.xtrainer_real.right_arm_single_async_rtc_main import Args as RTCArgs
    from examples.xtrainer_real.right_arm_single_main import validate_right_arm_metadata
    from examples.xtrainer_real.voice.environment import VoiceRightArmInferenceEnvironment
    from examples.xtrainer_real.voice.rtc import VoiceAsyncRTCInferenceWorker
    from examples.xtrainer_real.voice.rtc import VoiceRightArmAsyncRTCPolicyAgent
    from examples.xtrainer_real.voice.task_executor import RightArmTaskExecutor

    runtime_args = RTCArgs(
        **{field.name: getattr(args, field.name) for field in dataclasses.fields(RTCArgs) if hasattr(args, field.name)}
    )
    if (args.render_height, args.render_width) != (480, 640):
        raise ValueError("Right-arm preprocessing requires height=480 and width=640")
    if args.rtc_enabled:
        rtc._validate_rtc_args(runtime_args)  # noqa: SLF001
    rtc._validate_image_transport_args(runtime_args)  # noqa: SLF001
    worker = VoiceAsyncRTCInferenceWorker(
        args.host,
        args.port,
        connect_timeout_s=runtime_args.connect_timeout_s,
        rtc_inference_delay=runtime_args.rtc_inference_delay,
        rtc_execution_horizon=runtime_args.rtc_execution_horizon,
        rtc_prefix_attention_schedule=runtime_args.rtc_prefix_attention_schedule,
        rtc_max_guidance_weight=runtime_args.rtc_max_guidance_weight,
        image_transport_codec=runtime_args.image_transport_codec,
        image_jpeg_quality=runtime_args.image_jpeg_quality,
    )
    environment = None
    try:
        validate_right_arm_metadata(worker.metadata)
        if not all(math.isfinite(value) for value in worker.metadata["reset_pose"]):
            raise ValueError("Server reset_pose contains non-finite values")
        agent = VoiceRightArmAsyncRTCPolicyAgent(
            worker,
            request_interval=args.action_horizon,
            rtc_enabled=args.rtc_enabled,
            max_inference_delay_steps=runtime_args.max_inference_delay_steps,
            inference_timeout_s=runtime_args.inference_timeout_s,
            warmup_timeout_s=runtime_args.warmup_timeout_s,
            warmup_rtc=runtime_args.warmup_rtc,
            debug_timing=runtime_args.debug_async_timing,
        )

        def make_executor():
            nonlocal environment
            environment = VoiceRightArmInferenceEnvironment(
                right_robot_ip=args.right_robot_ip,
                right_gripper_port=args.right_gripper_port,
                right_gripper_id=args.right_gripper_id,
                right_gripper_servo_pos=args.right_gripper_servo_pos,
                camera_top_serial=args.camera_top_serial,
                camera_right_wrist_serial=args.camera_right_wrist_serial,
                camera_fps=args.camera_fps,
                render_height=args.render_height,
                render_width=args.render_width,
                max_joint_delta=args.max_joint_delta,
                ramp_step=args.ramp_step,
                ramp_max_steps=args.ramp_max_steps,
                gripper_update_threshold=args.gripper_update_threshold,
                servo_step_limit=args.servo_step_limit,
                prompt=None,
                reset_pose=worker.metadata["reset_pose"],
                show_camera_preview=args.show_camera_preview,
            )
            return RightArmTaskExecutor(
                environment,
                agent,
                control_hz=args.control_hz,
                inference_timeout_s=runtime_args.inference_timeout_s,
                output_root=args.inference_action_output_dir,
                record_actions=args.record_inference_actions,
            )

        run_session(service, make_executor, completion)
    finally:
        try:
            if environment is not None:
                environment.close()
        finally:
            worker.close()


def cli() -> None:
    logging.basicConfig(level=logging.INFO, force=True)
    try:
        main(tyro.cli(Args))
    except (KeyboardInterrupt, EOFError):
        logger.info("STOPPING: input interrupted; exiting without an automatic reset")
    except Exception:
        logger.exception("ERROR: voice session stopped; no task will be retried")
        raise SystemExit(1) from None


if __name__ == "__main__":
    cli()
