import dataclasses
from types import MappingProxyType

from examples.xtrainer_real.voice.task_completion import CompletionConfig


@dataclasses.dataclass(frozen=True)
class TaskSpec:
    task_id: str
    prompt: str
    completion: CompletionConfig = dataclasses.field(default_factory=CompletionConfig)


TASKS = MappingProxyType(
    {
        "press_red_button": TaskSpec(
            "press_red_button", "Use the right arm to press the red trip button on the control panel."
        ),
        "press_green_button": TaskSpec(
            "press_green_button", "Use the right arm to press the green close button on the control panel."
        ),
    }
)


def get_task(task_id: str, completion: CompletionConfig | None = None) -> TaskSpec:
    task = TASKS[task_id]
    return dataclasses.replace(task, completion=completion) if completion is not None else task
