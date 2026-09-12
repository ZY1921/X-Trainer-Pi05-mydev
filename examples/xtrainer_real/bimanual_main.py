"""Run synchronous inference on the physical bimanual X-Trainer."""

import dataclasses
import logging

from openpi_client import action_chunk_broker
from openpi_client import websocket_client_policy as _websocket_client_policy
from openpi_client.runtime import runtime as _runtime
from openpi_client.runtime.agents import policy_agent as _policy_agent
import tyro

from examples.xtrainer_real import camera_visualization as _camera_visualization
from examples.xtrainer_real import diagnostics as _diagnostics
from examples.xtrainer_real import inference_environment as _inference_environment


@dataclasses.dataclass
class Args:
    host: str = "0.0.0.0"
    port: int = 8000

    prompt: str = "pick up the object"
    action_horizon: int = 25
    control_hz: float = 20.0

    num_episodes: int = 1
    max_episode_steps: int = 1000

    left_robot_ip: str = "192.168.5.1"
    right_robot_ip: str = "192.168.5.2"

    left_gripper_port: str = "/dev/ttyUSB1"
    right_gripper_port: str = "/dev/ttyUSB0"
    left_gripper_id: int = 21
    right_gripper_id: int = 22
    left_gripper_servo_pos: tuple[int, int] = (2048, 3052)
    right_gripper_servo_pos: tuple[int, int] = (2048, 3052)

    camera_top_serial: str = ""
    camera_left_wrist_serial: str = ""
    camera_right_wrist_serial: str = ""
    camera_fps: float = 30.0

    render_height: int = 480
    render_width: int = 640
    show_camera_preview: bool = False
    """Show the preprocessed camera images. Q, Esc, or closing the window stops the client."""

    max_joint_delta: float = 0.17
    ramp_step: float = 0.01
    ramp_max_steps: int = 100
    gripper_update_threshold: float = 0.02
    servo_step_limit: float = 0.9
    debug_action_state_diagnostics: bool = False
    debug_action_state_interval: int = 20
    debug_action_state_max_steps: int = 400


def main(args: Args) -> None:
    if (args.render_height, args.render_width) != (480, 640):
        raise ValueError("Bimanual image preprocessing requires --render-height 480 --render-width 640.")

    ws_client_policy = _websocket_client_policy.WebsocketClientPolicy(
        host=args.host,
        port=args.port,
    )
    metadata = ws_client_policy.get_server_metadata()
    logging.info("Server metadata: %s", metadata)

    environment = _inference_environment.BimanualInferenceEnvironment(
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
        show_camera_preview=args.show_camera_preview,
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
        agent=_policy_agent.PolicyAgent(
            policy=action_chunk_broker.ActionChunkBroker(
                policy=ws_client_policy,
                action_horizon=args.action_horizon,
            )
        ),
        subscribers=subscribers,
        max_hz=args.control_hz,
        num_episodes=args.num_episodes,
        max_episode_steps=args.max_episode_steps,
    )

    try:
        runtime.run()
    except _camera_visualization.CameraPreviewStopRequestedError:
        logging.info("Camera preview closed; stopping the client.")
    finally:
        environment.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    args: Args = tyro.cli(Args)
    main(args)
