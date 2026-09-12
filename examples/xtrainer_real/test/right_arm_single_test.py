# ruff: noqa: SLF001

import importlib
import sys
import types

import numpy as np
import pytest

from examples.xtrainer_real import right_arm_env
from examples.xtrainer_real import right_arm_single_main


def _metadata() -> dict:
    return {
        "robot_side": "right",
        "physical_action_dim": 7,
        "model_action_start_index": 7,
        "model_action_end_index": 14,
        "camera_keys": ["observation.images.top", "observation.images.right_wrist"],
        "reset_pose": [0.0] * 7,
    }


def test_metadata_validation_rejects_bimanual_checkpoint():
    right_arm_single_main.validate_right_arm_metadata(_metadata())
    bad_metadata = _metadata()
    bad_metadata["physical_action_dim"] = 14
    with pytest.raises(ValueError, match="not compatible"):
        right_arm_single_main.validate_right_arm_metadata(bad_metadata)


def test_async_agent_rejects_non_7d_physical_actions(monkeypatch):
    monkeypatch.setitem(sys.modules, "serial", types.ModuleType("serial"))
    async_main = importlib.import_module("examples.xtrainer_real.right_arm_single_async_rtc_main")
    agent = object.__new__(async_main.RightArmAsyncRTCPolicyAgent)
    with pytest.raises(ValueError, match="horizon, 7"):
        agent._validated_result({"actions": np.zeros((5, 14), dtype=np.float32)})


def test_async_main_rejects_negative_plot_start_before_connecting(monkeypatch):
    monkeypatch.setitem(sys.modules, "serial", types.ModuleType("serial"))
    async_main = importlib.import_module("examples.xtrainer_real.right_arm_single_async_rtc_main")
    args = async_main.Args(inference_action_plot_start_index=-1, rtc_enabled=False)
    with pytest.raises(ValueError, match=r"must be in \[0, 6\]"):
        async_main.main(args)


def test_right_arm_environment_connects_one_follower_and_sends_7d_action(monkeypatch):
    created = []

    class FakeFollower:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.cameras = {}
            self.sent = []
            created.append(self)

        def connect(self):
            pass

        def disconnect(self):
            pass

        def get_low_latency_observation(self):
            return {
                **{f"joint{index}.pos": float(index) for index in range(1, 7)},
                "gripper.pos": 0.5,
            }

        def send_action(self, action):
            self.sent.append(action)

    monkeypatch.setattr(right_arm_env, "_create_follower", lambda **kwargs: FakeFollower(**kwargs))
    environment = right_arm_env.XTrainerRightArmRealEnvironment(
        camera_top_serial="top",
        camera_right_wrist_serial="right",
        max_joint_delta=10.0,
        gripper_update_threshold=0.0,
    )
    environment.apply_action({"actions": np.arange(7, dtype=np.float32) / 10})

    assert len(created) == 1
    assert created[0].kwargs["camera_serials"] == {"cam_top": "top", "cam_right_wrist": "right"}
    assert created[0].sent[-1]["joint1.pos"] == pytest.approx(0.0)
    assert created[0].sent[-1]["joint6.pos"] == pytest.approx(0.5)
    assert created[0].sent[-1]["gripper.pos"] == pytest.approx(0.6)

    with pytest.raises(ValueError, match="length 7"):
        environment.apply_action({"actions": np.zeros(14)})
