import dataclasses
import json
import logging
import math
from pathlib import Path
import tempfile
import time

import numpy as np

from examples.xtrainer_real.voice.command_parser import ParseResult
from examples.xtrainer_real.voice.task_completion import TaskCompletionDetector
from examples.xtrainer_real.voice.task_registry import TASKS
from examples.xtrainer_real.voice.task_registry import TaskSpec

logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class TaskResult:
    session_id: int
    task_id: str
    reason: str
    executed_steps: int
    reset_completed: bool
    output_dir: str


class RightArmTaskExecutor:
    def __init__(
        self,
        environment,
        agent,
        *,
        control_hz: float = 22.0,
        inference_timeout_s: float = 3.0,
        output_root: Path = Path("output/voice"),
        record_actions: bool = True,
        clock=time.monotonic,
        sleep=time.sleep,
    ) -> None:
        if not math.isfinite(control_hz) or control_hz <= 0:
            raise ValueError("control_hz must be finite and positive")
        if not math.isfinite(inference_timeout_s) or inference_timeout_s <= 0:
            raise ValueError("inference_timeout_s must be finite and positive")
        self._environment = environment
        self._agent = agent
        self._period = 1.0 / control_hz
        self._drain_timeout = inference_timeout_s + 1.0
        self._output_root = Path(output_root)
        self._record_actions = record_actions
        self._clock = clock
        self._sleep = sleep
        self._session_id = 0
        self._busy = False
        self._failed = False

    def execute(self, task: TaskSpec, command: ParseResult) -> TaskResult:
        if self._busy or self._failed:
            raise RuntimeError("Task executor is busy or has failed; restart the session")
        if (
            command.reason is not None
            or command.task_id != task.task_id
            or task.task_id not in TASKS
            or TASKS[task.task_id].prompt != task.prompt
        ):
            raise ValueError("Only an accepted command with its exact whitelist prompt may execute")
        self._session_id += 1
        self._output_root.mkdir(parents=True, exist_ok=True)
        output_dir = Path(
            tempfile.mkdtemp(prefix=f"task_{self._session_id:04d}_{task.task_id}_", dir=self._output_root)
        )
        summary = {
            "session_id": self._session_id,
            "command": dataclasses.asdict(command),
            "task": dataclasses.asdict(task),
            "control_hz": 1.0 / self._period,
            "reason": "error",
            "executed_steps": 0,
            "initial_reset_completed": False,
            "reset_completed": False,
        }
        recorder = None
        detector = None
        self._busy = True
        try:
            if self._record_actions:
                from examples.xtrainer_real.inference_action_recorder import InferenceActionRecorder

                recorder = InferenceActionRecorder(
                    output_dir, mode="voice_async_rtc", plot_start_index=0, config=summary.copy()
                )
            self._agent.set_action_recorder(recorder)
            self._agent.reset()
            self._environment.set_prompt(None)
            logger.info("RESETTING before task %d: %s", self._session_id, task.task_id)
            self._environment.reset()
            summary["initial_reset_completed"] = True
            start_qpos = self._environment.get_current_qpos()
            detector = TaskCompletionDetector(start_qpos, task.completion)
            summary["start_qpos"] = np.asarray(start_qpos).tolist()
            self._environment.set_prompt(task.prompt)
            logger.info("EXECUTING task %d: %s", self._session_id, task.prompt)

            with (output_dir / "completion.jsonl").open("w", encoding="utf-8") as trace:
                while True:
                    step_start = self._clock()
                    qpos = self._environment.get_current_qpos()
                    reason = detector.update(qpos, self._clock(), summary["executed_steps"])
                    trace.write(
                        json.dumps(
                            {
                                "executed_steps": summary["executed_steps"],
                                "time_monotonic": self._clock(),
                                "qpos": np.asarray(qpos).tolist(),
                                "distance": detector.distance,
                                "motion_started_at": detector.motion_started_at,
                                "departed": detector.departed,
                                "reason": reason,
                            }
                        )
                        + "\n"
                    )
                    trace.flush()
                    if reason is not None:
                        summary["reason"] = reason
                        break
                    observation = self._environment.get_observation()
                    self._validate_observation(observation)
                    action = self._agent.get_action(observation)
                    values = np.asarray(action["actions"])
                    if values.shape != (7,) or not np.all(np.isfinite(values)):
                        raise ValueError("Invalid right-arm action; expected seven finite values")
                    self._environment.apply_action(action)
                    summary["executed_steps"] += 1
                    self._sleep(max(0.0, self._period - (self._clock() - step_start)))

            # No get_action/apply_action calls are allowed past this boundary.
            self._environment.set_prompt(None)
            self._agent.finish_task(self._drain_timeout)
            logger.info("RESETTING after task %d: %s", self._session_id, summary["reason"])
            self._environment.reset()
            summary["reset_completed"] = True
            return TaskResult(
                session_id=self._session_id,
                task_id=task.task_id,
                reason=summary["reason"],
                executed_steps=summary["executed_steps"],
                reset_completed=True,
                output_dir=str(output_dir),
            )
        except BaseException as error:
            self._failed = True
            summary["detected_reason"] = summary["reason"]
            summary["reason"] = "interrupted" if isinstance(error, KeyboardInterrupt) else "error"
            summary["error"] = f"{type(error).__name__}: {error}"
            raise
        finally:
            self._busy = False
            # No movement in finally: Ctrl+C and faults must not cause a reset.
            if detector is not None:
                summary["motion_started_at"] = detector.motion_started_at
            try:
                with (output_dir / "task_summary.json").open("w", encoding="utf-8") as output:
                    json.dump(summary, output, ensure_ascii=False, indent=2)
                if recorder is not None:
                    recorder.save()
            except Exception:
                logger.exception("Failed to save task logs to %s", output_dir)

    @staticmethod
    def _validate_observation(observation: dict) -> None:
        state = np.asarray(observation["observation.state"])
        if state.shape != (7,) or not np.all(np.isfinite(state)):
            raise ValueError("Invalid right-arm observation state")
        for key in ("observation.images.top", "observation.images.right_wrist"):
            image = np.asarray(observation[key])
            if image.ndim != 3 or image.shape[-1] != 3 or image.dtype != np.uint8 or image.size == 0:
                raise ValueError(f"Invalid RGB uint8 image: {key}")
