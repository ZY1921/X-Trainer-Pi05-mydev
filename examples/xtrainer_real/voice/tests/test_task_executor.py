import dataclasses
import json

import numpy as np
import pytest

from examples.xtrainer_real.voice.command_parser import ParseResult
from examples.xtrainer_real.voice.main import Args
from examples.xtrainer_real.voice.main import main
from examples.xtrainer_real.voice.main import run_session
from examples.xtrainer_real.voice.task_completion import CompletionConfig
from examples.xtrainer_real.voice.task_executor import RightArmTaskExecutor
from examples.xtrainer_real.voice.task_registry import get_task


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


class FakeEnvironment:
    def __init__(self):
        self.qpos = np.zeros(7)
        self.prompt = None
        self.events = []
        self.raise_on_apply = None
        self.raise_on_final_reset = False
        self.resets = 0

    def set_prompt(self, prompt):
        self.prompt = prompt

    def reset(self):
        self.events.append("reset")
        self.resets += 1
        if self.raise_on_final_reset and self.resets == 2:
            raise RuntimeError("reset failed")
        self.qpos = np.zeros(7)

    def get_current_qpos(self):
        return self.qpos.copy()

    def get_observation(self):
        return {
            "observation.state": self.qpos.copy(),
            "prompt": self.prompt,
            "observation.images.top": np.zeros((8, 8, 3), dtype=np.uint8),
            "observation.images.right_wrist": np.zeros((8, 8, 3), dtype=np.uint8),
        }

    def apply_action(self, action):
        if self.raise_on_apply is not None:
            raise self.raise_on_apply
        self.qpos = action["actions"].copy()
        self.events.append("apply")


class FakeAgent:
    def __init__(self, env, trajectory=None):
        self.env = env
        self.trajectory = trajectory if trajectory is not None else [0.03, 0.04, 0.25, 0.25, 0.1, 0.1]
        self.index = 0
        self.observed_prompts = []
        self.finish_error = None
        self.get_error = None

    def reset(self):
        self.index = 0

    def set_action_recorder(self, recorder):
        assert recorder is None

    def get_action(self, obs):
        if self.get_error is not None:
            raise self.get_error
        self.observed_prompts.append(obs["prompt"])
        value = self.trajectory[min(self.index, len(self.trajectory) - 1)]
        self.index += 1
        return {"actions": np.array([value, 0, 0, 0, 0, 0, 0], dtype=float)}

    def finish_task(self, timeout):
        assert timeout == 4
        self.env.events.append("drain")
        if self.finish_error is not None:
            raise self.finish_error


def setup_executor(tmp_path, trajectory=None):
    clock = FakeClock()
    env = FakeEnvironment()
    agent = FakeAgent(env, trajectory)
    executor = RightArmTaskExecutor(
        env, agent, output_root=tmp_path, record_actions=False, clock=clock, sleep=clock.sleep
    )
    return executor, env, agent


def command(task_id="press_red_button"):
    return ParseResult("example", "example", task_id, None)


def task(task_id="press_red_button"):
    config = CompletionConfig(completion_ignore_seconds=0.1, return_confirm_frames=2)
    return get_task(task_id, config)


def test_two_commands_have_separate_logs_and_only_reset_after_draining(tmp_path):
    executor, env, agent = setup_executor(tmp_path)
    for index, task_id in enumerate(["press_red_button", "press_green_button"], start=1):
        previous = len(agent.observed_prompts)
        result = executor.execute(task(task_id), command(task_id))
        assert result.reason == "returned_to_start"
        assert result.executed_steps == 6
        assert result.session_id == index
        assert agent.observed_prompts[previous:] == [get_task(task_id).prompt] * 6
        assert env.events[-2:] == ["drain", "reset"]
        assert env.prompt is None
    summaries = list(tmp_path.glob("*/task_summary.json"))
    assert len(summaries) == 2
    for path in summaries:
        data = json.loads(path.read_text())
        assert data["reset_completed"]
        assert data["motion_started_at"] is not None


@pytest.mark.parametrize(
    ("trajectory", "reason", "steps"),
    [
        ([0], "failed_to_start", 110),
        ([0.3], "max_steps", 450),
    ],
)
def test_exact_step_limits_without_extra_model_call(tmp_path, trajectory, reason, steps):
    executor, env, agent = setup_executor(tmp_path, trajectory)
    result = executor.execute(task(), command())
    assert result.reason == reason
    assert result.executed_steps == steps
    assert len(agent.observed_prompts) == steps
    assert env.events.count("apply") == steps
    assert env.resets == 2


@pytest.mark.parametrize("failure", [KeyboardInterrupt(), RuntimeError("network failure")])
def test_inference_failure_or_interrupt_does_not_reset_or_retry(tmp_path, failure):
    executor, env, agent = setup_executor(tmp_path)
    agent.get_error = failure
    with pytest.raises(type(failure)):
        executor.execute(task(), command())
    assert env.events == ["reset"]
    with pytest.raises(RuntimeError, match="failed"):
        executor.execute(task(), command())


def test_nonfinite_action_never_reaches_hardware(tmp_path):
    executor, env, _ = setup_executor(tmp_path, [float("nan")])
    with pytest.raises(ValueError, match="action"):
        executor.execute(task(), command())
    assert env.events == ["reset"]


def test_nonfinite_camera_observation_state_is_fatal(tmp_path, monkeypatch):
    executor, env, _ = setup_executor(tmp_path)
    observation = env.get_observation()
    observation["observation.state"][1] = float("inf")
    monkeypatch.setattr(env, "get_observation", lambda: observation)
    with pytest.raises(ValueError, match="observation"):
        executor.execute(task(), command())
    assert env.events == ["reset"]


def test_drain_timeout_never_resets_or_allows_another_task(tmp_path):
    executor, env, agent = setup_executor(tmp_path)
    agent.finish_error = TimeoutError("drain failed")
    with pytest.raises(TimeoutError):
        executor.execute(task(), command())
    assert env.resets == 1
    assert env.events[-1] == "drain"
    with pytest.raises(RuntimeError, match="failed"):
        executor.execute(task(), command())


def test_reset_failure_is_recorded_as_error(tmp_path):
    executor, env, _ = setup_executor(tmp_path)
    env.raise_on_final_reset = True
    with pytest.raises(RuntimeError, match="reset failed"):
        executor.execute(task(), command())
    data = json.loads(next(tmp_path.glob("*/task_summary.json")).read_text())
    assert data["reason"] == "error"
    assert not data["reset_completed"]


def test_reject_arbitrary_prompt_before_hardware(tmp_path):
    executor, env, _ = setup_executor(tmp_path)
    with pytest.raises(ValueError, match="whitelist"):
        executor.execute(dataclasses.replace(task(), prompt="other"), command())
    assert env.events == []


def test_session_constructs_hardware_only_for_first_accepted_command(tmp_path):
    executor, env, agent = setup_executor(tmp_path)
    events = []

    class Service:
        commands = iter([ParseResult("", "", None, "empty_text"), command(), command("press_green_button")])

        def listen_once(self):
            events.append("listen")
            try:
                return next(self.commands)
            except StopIteration:
                raise KeyboardInterrupt from None

    def factory():
        events.append("create")
        return executor

    with pytest.raises(KeyboardInterrupt):
        run_session(Service(), factory, task().completion)
    assert events == ["listen", "listen", "create", "listen", "listen"]
    assert env.resets == 4
    assert len(agent.observed_prompts) == 12


def test_file_or_text_input_requires_dry_run_before_any_import():
    with pytest.raises(ValueError, match="dry-run"):
        main(Args(text="anything"))
