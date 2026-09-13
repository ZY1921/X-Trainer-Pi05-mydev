import dataclasses

import pytest

from examples.xtrainer_real.voice.task_completion import CompletionConfig
from examples.xtrainer_real.voice.task_completion import TaskCompletionDetector


def q(value, gripper=1.0):
    return [value, 0, 0, 0, 0, 0, gripper]


def test_motion_timer_uses_first_confirmed_observation_and_real_seconds():
    detector = TaskCompletionDetector(q(0), CompletionConfig())
    assert detector.update(q(0), 1000, 0) is None  # warmup elapsed, no motion
    for step, now in enumerate([1001, 1001.1, 1001.2], start=1):
        assert detector.update(q(0.25), now, step) is None
    assert detector.motion_started_at == 1001
    for step in range(4, 20):
        assert detector.update(q(0.1), 1006.99, step) is None
    for step in range(20, 24):
        assert detector.update(q(0.1), 1007 + (step - 20) * 0.01, step) is None
    assert detector.update(q(0.1), 1007.1, 24) == "returned_to_start"


def test_no_motion_or_gripper_only_does_not_complete():
    detector = TaskCompletionDetector(q(0), CompletionConfig())
    for step in range(110):
        assert detector.update(q(0.005, step % 2), step, step) is None
    assert detector.update(q(0), 110, 110) == "failed_to_start"


def test_motion_without_departure_cannot_complete():
    detector = TaskCompletionDetector(q(0), CompletionConfig())
    for step in range(1, 450):
        assert detector.update(q(0.05), step, step) is None
    assert detector.update(q(0.05), 450, 450) == "max_steps"


def test_jitter_resets_motion_confirmation():
    detector = TaskCompletionDetector(q(0), CompletionConfig())
    for step, value in enumerate([0.03, 0.01, 0.03, 0.01, 0.03, 0.04, 0.05], start=1):
        detector.update(q(value), step, step)
    assert detector.motion_started_at == 5


def test_return_requires_consecutive_samples_and_ignores_duplicate_steps():
    detector = TaskCompletionDetector(q(0), CompletionConfig(return_confirm_frames=2))
    for step in range(1, 4):
        detector.update(q(0.3), step, step)
    assert detector.update(q(0.1), 10, 4) is None
    assert detector.update(q(0.1), 10, 4) is None
    assert detector.update(q(0.16), 11, 5) is None
    assert detector.update(q(0.1), 12, 6) is None
    assert detector.update(q(0.1), 13, 7) == "returned_to_start"


def test_nonzero_start_and_gripper_exclusion():
    detector = TaskCompletionDetector([1, 2, 3, 4, 5, 6, 0], CompletionConfig())
    detector.update([1, 2, 3, 4, 5, 6, 1], 0, 1)
    assert detector.distance == 0
    assert detector.motion_started_at is None


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_feedback_rejected(bad):
    detector = TaskCompletionDetector(q(0), CompletionConfig())
    with pytest.raises(ValueError, match="non-finite"):
        detector.update(q(bad), 0, 1)


@pytest.mark.parametrize(
    "override",
    [
        {"max_episode_steps": 0},
        {"completion_ignore_seconds": float("nan")},
        {"return_threshold": 0.3},
        {"motion_start_confirm_frames": 1.5},
        {"max_episode_steps": 100},
    ],
)
def test_invalid_configs(override):
    with pytest.raises(ValueError, match="must|Require"):
        dataclasses.replace(CompletionConfig(), **override)
