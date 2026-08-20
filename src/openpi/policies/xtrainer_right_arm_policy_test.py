# ruff: noqa: SLF001

import numpy as np
import pytest

from openpi import transforms
from openpi.policies import xtrainer_right_arm_policy
from openpi.shared import normalize
from openpi.training import config as training_config


def _sample() -> dict:
    state = np.linspace(-0.3, 0.3, 7, dtype=np.float32)
    actions = np.stack([state, state + 0.1], axis=0)
    actions[:, 6] = np.array([0.2, 0.8], dtype=np.float32)
    return {
        "observation.state": state,
        "observation.images.top": np.zeros((12, 16, 3), dtype=np.uint8),
        "observation.images.right_wrist": np.ones((3, 12, 16), dtype=np.uint8),
        "actions": actions,
        "prompt": "test",
    }


def test_right_arm_inputs_and_image_masks():
    result = xtrainer_right_arm_policy.XTrainerRightArmInputs()(_sample())

    assert result["state"].shape == (7,)
    assert result["actions"].shape == (2, 7)
    assert result["image"]["base_0_rgb"].shape == (12, 16, 3)
    assert result["image"]["right_wrist_0_rgb"].shape == (12, 16, 3)
    assert result["image_mask"] == {
        "base_0_rgb": np.True_,
        "left_wrist_0_rgb": np.False_,
        "right_wrist_0_rgb": np.True_,
    }


def test_right_arm_inputs_reject_missing_camera_and_wrong_dimension():
    sample = _sample()
    sample.pop("observation.images.right_wrist")
    with pytest.raises(ValueError, match="Missing required right-arm cameras"):
        xtrainer_right_arm_policy.XTrainerRightArmInputs()(sample)

    sample = _sample()
    sample["observation.state"] = np.zeros(14, dtype=np.float32)
    with pytest.raises(ValueError, match="last dimension 7"):
        xtrainer_right_arm_policy.XTrainerRightArmInputs()(sample)

    sample = _sample()
    sample["observation.images.left_wrist"] = np.zeros((12, 16, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="Unexpected cameras"):
        xtrainer_right_arm_policy.XTrainerRightArmInputs()(sample)


def test_pack_and_unpack_use_pretrained_right_arm_slots():
    state = np.arange(7, dtype=np.float32)
    actions = np.arange(14, dtype=np.float32).reshape(2, 7)
    packed = xtrainer_right_arm_policy.PackRightArmToModelSlots()({"state": state.copy(), "actions": actions.copy()})

    assert packed["state"].shape == (14,)
    assert packed["actions"].shape == (2, 14)
    np.testing.assert_array_equal(packed["state"][:7], 0.0)
    np.testing.assert_array_equal(packed["state"][7:14], state)
    np.testing.assert_array_equal(packed["actions"][:, :7], 0.0)
    np.testing.assert_array_equal(packed["actions"][:, 7:14], actions)

    padded = transforms.PadStatesAndActions(32)(packed)
    unpacked = xtrainer_right_arm_policy.UnpackRightArmFromModelSlots()(padded)
    np.testing.assert_array_equal(unpacked["state"], state)
    np.testing.assert_array_equal(unpacked["actions"], actions)


def test_delta_norm_pack_round_trip_returns_physical_7d_actions():
    sample = xtrainer_right_arm_policy.XTrainerRightArmInputs()(_sample())
    expected_actions = sample["actions"].copy()
    stats = {
        "state": normalize.NormStats(mean=np.zeros(7), std=np.ones(7)),
        "actions": normalize.NormStats(mean=np.zeros(7), std=np.ones(7)),
    }
    input_pipeline = transforms.compose(
        [
            transforms.DeltaActions(transforms.make_bool_mask(6, -1)),
            transforms.Normalize(stats),
            xtrainer_right_arm_policy.PackRightArmToModelSlots(),
            transforms.PadStatesAndActions(32),
        ]
    )
    model_data = input_pipeline(sample)
    assert model_data["actions"].shape == (2, 32)

    output_pipeline = transforms.compose(
        [
            xtrainer_right_arm_policy.UnpackRightArmFromModelSlots(),
            transforms.Unnormalize(stats),
            transforms.AbsoluteActions(transforms.make_bool_mask(6, -1)),
            xtrainer_right_arm_policy.XTrainerRightArmOutputs(),
        ]
    )
    physical = output_pipeline({"state": model_data["state"], "actions": model_data["actions"]})
    assert physical["actions"].shape == (2, 7)
    np.testing.assert_allclose(physical["actions"], expected_actions, atol=2e-6)


def test_right_arm_config_rejects_legacy_14d_norm_stats():
    stats = {
        "state": normalize.NormStats(mean=np.zeros(14), std=np.ones(14)),
        "actions": normalize.NormStats(mean=np.zeros(14), std=np.ones(14)),
    }
    with pytest.raises(ValueError, match="must have dimension 7"):
        training_config.LeRobotXTrainerRightArmDataConfig._validate_norm_stats(stats)
