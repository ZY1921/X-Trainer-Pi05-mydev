# ruff: noqa: SLF001

import numpy as np
import pytest

from examples.xtrainer_real import env as bimanual_env
from examples.xtrainer_real import inference_environment
from examples.xtrainer_real import right_arm_env


class _FakeVisualizer:
    def __init__(self) -> None:
        self.shown = []

    def show(self, observation: dict) -> None:
        self.shown.append(observation)


@pytest.mark.parametrize(
    ("environment_type", "base_type"),
    [
        (inference_environment.BimanualInferenceEnvironment, bimanual_env.XTrainerRealEnvironment),
        (inference_environment.RightArmInferenceEnvironment, right_arm_env.XTrainerRightArmRealEnvironment),
    ],
)
def test_inference_environments_preprocess_before_visualization(monkeypatch, environment_type, base_type):
    observation = {"processed": False}
    call_order = []

    monkeypatch.setattr(base_type, "get_observation", lambda _self: observation)

    def preprocess(value):
        assert value is observation
        value["processed"] = True
        call_order.append("preprocess")
        return value

    class OrderedVisualizer(_FakeVisualizer):
        def show(self, value: dict) -> None:
            assert value["processed"] is True
            call_order.append("visualize")
            super().show(value)

    monkeypatch.setattr(inference_environment._image_preprocessing, "preprocess_inference_images", preprocess)
    environment = object.__new__(environment_type)
    environment._camera_visualizer = OrderedVisualizer()

    result = environment.get_observation()

    assert result is observation
    assert call_order == ["preprocess", "visualize"]


def test_bimanual_environment_sends_both_halves_of_14d_action(monkeypatch):
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

    monkeypatch.setattr(bimanual_env, "_create_follower", lambda **kwargs: FakeFollower(**kwargs))
    environment = inference_environment.BimanualInferenceEnvironment(
        camera_top_serial="top",
        camera_left_wrist_serial="left",
        camera_right_wrist_serial="right",
        max_joint_delta=10.0,
        gripper_update_threshold=0.0,
    )
    target = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 0.4])

    environment.apply_action({"actions": target})

    assert len(created) == 2
    assert created[0].kwargs["camera_serials"] == {"cam_top": "top", "cam_left_wrist": "left"}
    assert created[1].kwargs["camera_serials"] == {"cam_right_wrist": "right"}
    assert created[0].sent[-1]["joint1.pos"] == pytest.approx(0.0)
    assert created[0].sent[-1]["joint6.pos"] == pytest.approx(0.5)
    assert created[0].sent[-1]["gripper.pos"] == pytest.approx(0.6)
    assert created[1].sent[-1]["joint1.pos"] == pytest.approx(0.7)
    assert created[1].sent[-1]["joint6.pos"] == pytest.approx(1.2)
    assert created[1].sent[-1]["gripper.pos"] == pytest.approx(0.4)

    with pytest.raises(ValueError, match="length 14"):
        environment.apply_action({"actions": np.zeros(7)})


@pytest.mark.parametrize("failure_stage", ["connect", "observe", "send"])
def test_bimanual_failure_closes_both_arms_and_blocks_reconnect(monkeypatch, failure_stage):
    created = []

    class Follower:
        def __init__(self, **_kwargs):
            self.right = bool(created)
            self.connected = False
            self.disconnects = 0
            self.cameras = {}
            created.append(self)

        def fail_at(self, stage):
            if self.right and stage == failure_stage:
                raise ConnectionError(f"right arm {stage} failed")

        def connect(self):
            self.connected = True
            self.fail_at("connect")

        def disconnect(self):
            self.connected = False
            self.disconnects += 1

        def get_low_latency_observation(self):
            self.fail_at("observe")
            return {f"joint{index}.pos": 0.0 for index in range(1, 7)}

        def send_action(self, _action):
            self.fail_at("send")

    monkeypatch.setattr(bimanual_env, "_create_follower", Follower)
    environment = bimanual_env.XTrainerRealEnvironment()
    with pytest.raises(ConnectionError, match=f"right arm {failure_stage} failed"):
        environment.apply_action({"actions": np.zeros(14)})

    assert all(not arm.connected and arm.disconnects >= 1 for arm in created)
    assert not environment._connected
    with pytest.raises(RuntimeError, match="faulted"):
        environment.apply_action({"actions": np.zeros(14)})


@pytest.mark.parametrize(
    ("environment_type", "dimension"),
    [(bimanual_env.XTrainerRealEnvironment, 14), (right_arm_env.XTrainerRightArmRealEnvironment, 7)],
)
@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_environment_rejects_nonfinite_action_before_sending(monkeypatch, environment_type, dimension, value):
    environment = object.__new__(environment_type)
    monkeypatch.setattr(environment, "_ensure_connected", lambda: None)
    target = np.zeros(dimension)
    # A bad gripper value must not disappear through clipping or change-threshold filtering.
    target[-1] = value
    with pytest.raises(ValueError, match="finite"):
        environment.apply_action({"actions": target})
