# ruff: noqa: SLF001

import numpy as np
import pytest

from examples.xtrainer_real.hardware.dobot_xtrainer import DobotConnection
from examples.xtrainer_real.hardware.dobot_xtrainer import DobotXTrainer
from examples.xtrainer_real.voice.hardware import VoiceDobotXTrainer


class FakeSocket:
    def __init__(self, fragments):
        self.fragments = iter(fragments)
        self.sent = []
        self.closed = False

    def settimeout(self, _timeout):
        pass

    def close(self):
        self.closed = True

    def sendall(self, data):
        self.sent.append(data.decode("utf-8"))

    def recv(self, _size):
        fragment = next(self.fragments)
        if isinstance(fragment, BaseException):
            raise fragment
        return fragment


def follower(arm_type=DobotXTrainer, **kwargs):
    return arm_type(
        robot_ip="test",
        gripper_port="unused",
        gripper_id=22,
        gripper_servo_pos=(0, 1),
        use_gripper=False,
        **kwargs,
    )


def test_v4_transport_reassembles_fragmented_replies():
    connection = DobotConnection("test", 29999)
    socket = FakeSocket([b"0,{},Request", b"Control();", b"0,{},EnableRobot();"])
    connection._socket = socket

    assert connection.send_recv("RequestControl()") == "0,{},RequestControl();"
    assert connection.send_recv("EnableRobot()") == "0,{},EnableRobot();"
    assert socket.sent == ["RequestControl()", "EnableRobot()"]


@pytest.mark.parametrize(
    ("reply", "expected"),
    [
        (b"-1,{},GetAngle();", RuntimeError),
        (b"garbage;", ValueError),
        (b"0,garbage;", ValueError),
        (b"0,{},ServoJ();", ValueError),
        (b"0,{},GetAngle(;", ValueError),
        (b"0,{},GetAngle();0,{},GetAngle();", ValueError),
        (b"", ConnectionError),
        (TimeoutError("recv"), TimeoutError),
    ],
)
def test_v4_transport_rejects_bad_replies(reply, expected):
    connection = DobotConnection("test", 29999)
    connection._socket = FakeSocket([reply])
    with pytest.raises(expected):
        connection.send_recv("GetAngle()")


@pytest.mark.parametrize("arm_type", [DobotXTrainer, VoiceDobotXTrainer])
def test_v4_startup_and_action_use_only_dashboard(monkeypatch, arm_type):
    commands = []

    class Connection:
        def connect(self):
            commands.append("connected")

        def send_recv(self, command):
            # V4 command names from the vendor protocol, not a blanket success stub.
            assert command.split("(", 1)[0] in {
                "RequestControl",
                "EnableRobot",
                "SpeedFactor",
                "AccJ",
                "VelJ",
                "SetTool",
                "Tool",
                "StopDrag",
                "GetAngle",
                "ServoJ",
                "DOInstant",
                "Stop",
                "DisableRobot",
            }
            commands.append(command)
            if command == "GetAngle()":
                return "0,{90,0,0,0,0,0},GetAngle();"
            return f"0,{{}},{command};"

        def close(self):
            commands.append("closed")

    arm = follower(arm_type)
    assert arm._create_connection().port == 29999
    monkeypatch.setattr(arm, "_create_connection", Connection)
    monkeypatch.setattr("examples.xtrainer_real.hardware.dobot_xtrainer.time.sleep", lambda _seconds: None)
    arm.connect()

    assert commands[:5] == [
        "connected",
        "RequestControl()",
        "EnableRobot()",
        "SpeedFactor(20)",
        "AccJ(20)",
    ]
    assert "VelJ(20)" in commands
    assert "SetTool(1,{0.000,0.000,197.000,0.000,0.000,0.000})" in commands
    assert "Tool(1)" in commands
    assert "StopDrag()" in commands
    assert arm._last_action_rad == pytest.approx([np.pi / 2, 0, 0, 0, 0, 0])
    assert arm.get_low_latency_observation()["joint1.pos"] == pytest.approx(np.pi / 2)

    action = {f"joint{index}.pos": 0.0 for index in range(1, 7)}
    action["joint1.pos"] = np.pi / 2
    sent = arm.send_action(action)
    assert sent["joint1.pos"] == pytest.approx(np.pi / 2)
    assert (
        "ServoJ(90.000000,0.000000,0.000000,0.000000,0.000000,0.000000,t=0.030,aheadtime=50.0,gain=500.0)"
    ) in commands
    arm.set_do_status(1, 1)
    assert "DOInstant(1,1)" in commands

    arm.disconnect()
    assert commands[-3:] == ["Stop()", "DisableRobot()", "closed"]


@pytest.mark.parametrize("reply", ["0,{1,2},GetAngle();", "0,{nan,0,0,0,0,0},GetAngle();"])
def test_invalid_v4_joint_feedback_faults_follower(reply):
    class Dashboard:
        def send_recv(self, _command):
            return reply

        def close(self):
            pass

    arm = follower()
    arm._dashboard = Dashboard()
    arm._connected = True
    with pytest.raises(RuntimeError, match="Invalid GetAngle"):
        arm.get_low_latency_observation()
    assert arm._faulted
    assert not arm.is_connected
    with pytest.raises(RuntimeError, match="faulted"):
        arm.connect()


def test_rejected_v4_servo_command_never_updates_last_action():
    class Dashboard:
        def send_recv(self, command):
            if command.startswith("ServoJ("):
                raise RuntimeError("rejected")
            return "0,{},DisableRobot();"

        def close(self):
            pass

    arm = follower()
    arm._dashboard = Dashboard()
    arm._connected = True
    arm._enable_attempted = True
    with pytest.raises(RuntimeError, match="rejected"):
        arm.send_action({f"joint{index}.pos": 0.0 for index in range(1, 7)})
    assert arm._faulted
    assert arm._last_action_rad is None
    assert not arm.is_connected


def test_v4_startup_failure_after_enable_disables_and_closes(monkeypatch):
    commands = []

    class Connection:
        def connect(self):
            commands.append("connected")

        def send_recv(self, command):
            commands.append(command)
            if command == "AccJ(20)":
                raise RuntimeError("rejected")
            return f"0,{{}},{command};"

        def close(self):
            commands.append("closed")

    arm = follower()
    monkeypatch.setattr(arm, "_create_connection", Connection)
    monkeypatch.setattr("examples.xtrainer_real.hardware.dobot_xtrainer.time.sleep", lambda _seconds: None)
    with pytest.raises(RuntimeError, match="rejected"):
        arm.connect()
    assert commands[-3:] == ["Stop()", "DisableRobot()", "closed"]
    assert arm._faulted
    assert not arm.is_connected


def test_joint_jump_limit_stops_v4_servo_before_transmission():
    class Dashboard:
        def send_recv(self, _command):
            raise AssertionError("Unexpected command transmission")

        def close(self):
            pass

    arm = follower()
    arm._dashboard = Dashboard()
    arm._connected = True
    arm._last_action_rad = np.zeros(6)
    with pytest.raises(RuntimeError, match="Action jump too large"):
        arm.send_action({f"joint{index}.pos": 1.0 for index in range(1, 7)})


def test_documented_reply_whitespace_is_accepted():
    connection = DobotConnection("test", 29999)
    connection._socket = FakeSocket([b" 0, {90,0,0,0,0,0}, GetAngle();\r\n"])
    assert "90" in connection.send_recv("GetAngle()")


def test_partial_timeout_invalidates_connection_before_another_command():
    connection = DobotConnection("test", 29999)
    socket = FakeSocket([b"0,{", TimeoutError("recv"), b"90,0,0,0,0,0},GetAngle();"])
    connection._socket = socket
    with pytest.raises(TimeoutError):
        connection.send_recv("GetAngle()")
    assert socket.closed
    with pytest.raises(RuntimeError, match="not connected"):
        connection.send_recv("GetAngle()")
    assert socket.sent == ["GetAngle()"]


def test_reply_deadline_applies_to_all_fragments(monkeypatch):
    elapsed = [0.0]

    class SlowSocket(FakeSocket):
        def recv(self, _size):
            elapsed[0] += 0.4
            return b"0"

    connection = DobotConnection("test", 29999, timeout=1.0)
    socket = SlowSocket([])
    connection._socket = socket
    monkeypatch.setattr("examples.xtrainer_real.hardware.dobot_xtrainer.time.monotonic", lambda: elapsed[0])
    with pytest.raises(TimeoutError):
        connection.send_recv("GetAngle()")
    assert elapsed[0] < 1.5
    assert socket.closed


def test_servo_burst_is_paced_without_catching_up_after_a_delay(monkeypatch):
    elapsed = [1.0]
    sent_at = []

    class Dashboard:
        def send_recv(self, _command):
            sent_at.append(elapsed[0])
            return "0,{1},ServoJ();"

    def sleep(seconds):
        elapsed[0] += seconds

    monkeypatch.setattr("examples.xtrainer_real.hardware.dobot_xtrainer.time.monotonic", lambda: elapsed[0])
    monkeypatch.setattr("examples.xtrainer_real.hardware.dobot_xtrainer.time.sleep", sleep)
    arm = follower()
    arm._connected = True
    arm._dashboard = Dashboard()
    action = {f"joint{index}.pos": 0.0 for index in range(1, 7)}
    arm.send_action(action)
    arm.send_action(action)
    elapsed[0] += 1.0  # A slow inference must not trigger a burst of catch-up commands.
    arm.send_action(action)
    arm.send_action(action)
    assert sent_at == pytest.approx([1.0, 1.03, 2.03, 2.06])


def test_disconnect_stops_before_camera_cleanup_even_when_keeping_torque():
    events = []

    class Dashboard:
        def send_recv(self, command):
            events.append(command)

        def close(self):
            events.append("socket closed")

    class Camera:
        def disconnect(self):
            events.append("camera closed")

    arm = follower(disable_torque_on_disconnect=False)
    arm._dashboard = Dashboard()
    arm._enable_attempted = True
    arm.cameras = {"top": Camera()}
    arm.disconnect()
    assert events == ["Stop()", "socket closed", "camera closed"]


def test_wraparound_does_not_bypass_a_small_step_limit():
    arm = follower(max_delta_per_step=0.05)
    arm._connected = True
    arm._dashboard = FakeSocket([])  # Validation must fail before any command is sent.
    arm._last_action_rad = np.zeros(6)
    action = {f"joint{index}.pos": 0.0 for index in range(1, 7)}
    action["joint1.pos"] = 2 * np.pi + 0.2
    with pytest.raises(RuntimeError, match="Action jump too large"):
        arm.send_action(action)


@pytest.mark.parametrize("invalid_action", [{}, {"joint1.pos": 0.0}])
def test_missing_joints_never_become_zero_angle_targets(invalid_action):
    arm = follower()
    arm._connected = True
    socket = FakeSocket([])
    arm._dashboard = socket  # No send_recv method: validation must reject before transmitting.
    with pytest.raises(KeyError):
        arm.send_action(invalid_action)
    assert not socket.sent
    assert socket.closed
    assert arm._faulted
