# ruff: noqa: SLF001
import sys
import types

import numpy as np
import pytest

from examples.xtrainer_real.voice.hardware import InterruptibleRealSenseCamera
from examples.xtrainer_real.voice.hardware import StrictDobotConnection
from examples.xtrainer_real.voice.hardware import VoiceDobotXTrainer


class FakeSocket:
    def __init__(self, reply):
        self.reply = reply

    def sendall(self, _data):
        pass

    def recv(self, _size):
        if isinstance(self.reply, Exception):
            raise self.reply
        return self.reply


@pytest.mark.parametrize("reply", [b"", b"-1,{},ServoJ();", b"garbage", TimeoutError("recv")])
def test_strict_robot_connection_does_not_hide_failures(reply):
    connection = StrictDobotConnection("test", 29999)
    connection._socket = FakeSocket(reply)
    with pytest.raises((ConnectionError, RuntimeError, ValueError, TimeoutError)):
        connection.send_recv("GetAngle()")


def robot():
    return VoiceDobotXTrainer(
        robot_ip="test",
        gripper_port="test",
        gripper_id=22,
        gripper_servo_pos=(0, 1),
        use_gripper=False,
    )


@pytest.mark.parametrize("reply", ["", "0,{1,2},GetAngle();", "0,{nan,0,0,0,0,0},GetAngle();"])
def test_invalid_joint_reply_never_becomes_zero_state(reply):
    class Dashboard:
        def send_recv(self, _command):
            return reply

    follower = robot()
    follower._dashboard = Dashboard()
    with pytest.raises(RuntimeError, match="Invalid GetAngle"):
        follower._read_joint_positions()


def test_valid_joint_feedback_converts_degrees():
    connection = StrictDobotConnection("test", 29999)
    connection._socket = FakeSocket(b"0,{90,0,0,0,0,0},GetAngle();")
    follower = robot()
    follower._dashboard = connection
    assert follower._read_joint_positions() == pytest.approx([np.pi / 2, 0, 0, 0, 0, 0])


def test_interrupted_partial_connect_closes_resources(monkeypatch):
    events = []

    class Connection:
        def __init__(self, _ip, _port, **_kwargs):
            pass

        def connect(self):
            raise KeyboardInterrupt

        def send_recv(self, command):
            events.append(command)

        def close(self):
            events.append("closed")

    follower = robot()
    monkeypatch.setattr("examples.xtrainer_real.voice.hardware.StrictDobotConnection", Connection)
    with pytest.raises(KeyboardInterrupt):
        follower.connect()
    assert events == ["DisableRobot()", "closed"]
    assert follower._dashboard is None


def test_fragmented_robot_reply_is_reassembled():
    class Socket(FakeSocket):
        def __init__(self):
            self.fragments = iter([b"0,{90,0,", b"0,0,0,0},GetAngle();"])

        def recv(self, _size):
            return next(self.fragments)

    connection = StrictDobotConnection("test", 29999)
    connection._socket = Socket()
    assert connection.send_recv("GetAngle()") == "0,{90,0,0,0,0,0},GetAngle();"


@pytest.mark.parametrize("failure", [KeyboardInterrupt(), RuntimeError("camera warmup")])
def test_camera_warmup_failure_releases_pipeline(monkeypatch, failure):
    events = []

    class Pipeline:
        def start(self, _config):
            events.append("start")

        def wait_for_frames(self, **_kwargs):
            raise failure

        def stop(self):
            events.append("stop")

    config = types.SimpleNamespace(enable_device=lambda _serial: None, enable_stream=lambda *_args: None)
    monkeypatch.setitem(
        sys.modules,
        "pyrealsense2",
        types.SimpleNamespace(
            pipeline=Pipeline,
            config=lambda: config,
            stream=types.SimpleNamespace(color=1),
            format=types.SimpleNamespace(bgr8=1),
        ),
    )
    camera = InterruptibleRealSenseCamera("fake")
    with pytest.raises(type(failure)):
        camera.connect()
    assert events == ["start", "stop"]
    assert not camera.is_connected
