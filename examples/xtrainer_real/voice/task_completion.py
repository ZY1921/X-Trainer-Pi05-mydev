"""Voice-task completion detection based on right-arm joint motion."""

from collections.abc import Sequence
import dataclasses
import math
from typing import Literal

EndReason = Literal["returned_to_start", "failed_to_start", "max_steps"]


@dataclasses.dataclass(frozen=True)
class CompletionConfig:
    motion_start_threshold: float = 0.02
    motion_start_confirm_frames: int = 3
    departure_threshold: float = 0.20
    completion_ignore_seconds: float = 6.0
    return_threshold: float = 0.15
    return_confirm_frames: int = 5
    max_episode_steps: int = 450
    motion_start_timeout_steps: int = 110

    def __post_init__(self) -> None:
        for field in dataclasses.fields(self):
            value = getattr(self, field.name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{field.name} must be finite and positive")
            if field.name.endswith(("frames", "steps")) and (isinstance(value, bool) or int(value) != value):
                raise ValueError(f"{field.name} must be an integer")
        if not self.motion_start_threshold < self.return_threshold < self.departure_threshold:
            raise ValueError("Require motion_start_threshold < return_threshold < departure_threshold")
        if self.motion_start_timeout_steps > self.max_episode_steps:
            raise ValueError("motion_start_timeout_steps must not exceed max_episode_steps")


def joint_position(qpos: Sequence[float]) -> tuple[float, ...]:
    if len(qpos) not in (6, 7):
        raise ValueError("Expected six joints, optionally followed by a gripper")
    values = tuple(float(value) for value in qpos)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("Joint position contains non-finite values")
    return values[:6]


class TaskCompletionDetector:
    """Detect return to a region; this is not a button-success classifier."""

    def __init__(self, start_qpos: Sequence[float], config: CompletionConfig) -> None:
        self.start_qpos = joint_position(start_qpos)
        self.config = config
        self.distance = 0.0
        self.motion_started_at: float | None = None
        self.departed = False
        self.reason: EndReason | None = None
        self._motion_frames = 0
        self._candidate_time: float | None = None
        self._return_frames = 0
        self._last_time: float | None = None
        self._last_steps = -1

    def update(self, qpos: Sequence[float], now: float, executed_steps: int) -> EndReason | None:
        current = joint_position(qpos)
        if not math.isfinite(now) or (self._last_time is not None and now < self._last_time):
            raise ValueError("Expected a finite monotonic timestamp")
        if executed_steps < 0 or executed_steps < self._last_steps:
            raise ValueError("Executed steps must be non-negative and monotonic")
        if self.reason is not None:
            return self.reason
        self.distance = max(abs(a - b) for a, b in zip(current, self.start_qpos, strict=True))
        self._last_time = now
        # Count independent control observations, not repeated checks of the same step.
        if executed_steps == self._last_steps:
            return None
        self._last_steps = executed_steps
        if executed_steps >= self.config.max_episode_steps:
            self.reason = "max_steps"
            return self.reason
        if executed_steps == 0:
            return None
        if self.distance >= self.config.departure_threshold:
            self.departed = True
        if self.motion_started_at is None:
            if self.distance >= self.config.motion_start_threshold:
                if self._motion_frames == 0:
                    self._candidate_time = now
                self._motion_frames += 1
                if self._motion_frames >= self.config.motion_start_confirm_frames:
                    self.motion_started_at = self._candidate_time
            else:
                self._motion_frames = 0
                self._candidate_time = None
            if self.motion_started_at is None and executed_steps >= self.config.motion_start_timeout_steps:
                self.reason = "failed_to_start"
        if (
            self.motion_started_at is not None
            and now - self.motion_started_at >= self.config.completion_ignore_seconds
            and self.departed
        ):
            self._return_frames = self._return_frames + 1 if self.distance <= self.config.return_threshold else 0
            if self._return_frames >= self.config.return_confirm_frames:
                self.reason = "returned_to_start"
        else:
            self._return_frames = 0
        return self.reason
