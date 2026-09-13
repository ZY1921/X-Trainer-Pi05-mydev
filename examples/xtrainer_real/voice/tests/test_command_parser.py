# ruff: noqa: RUF001
import dataclasses
from pathlib import Path
import subprocess
import sys

import pytest

from examples.xtrainer_real.voice.command_parser import parse_command
from examples.xtrainer_real.voice.normalizer import normalize_text
from examples.xtrainer_real.voice.service import VoiceService
from examples.xtrainer_real.voice.task_registry import TASKS


@pytest.mark.parametrize(
    ("text", "task"),
    [
        ("机器人机器人，按下红色按钮", "press_red_button"),
        ("機器人，機器人，幫我按一下綠色按鍵！", "press_green_button"),
        ("机器人 机器人，把面板上的红按钮摁下去", "press_red_button"),
        ("机器人。机器人！麻烦你帮我点击绿色开关", "press_green_button"),
        ("机器人机器人，请你按一下红色按扭", "press_red_button"),
        ("机器人机器人，把绿色摁钮压下", "press_green_button"),
    ],
)
def test_accepted_commands(text, task):
    result = parse_command(text)
    assert result.task_id == task
    assert result.reason is None
    assert result.transcript == text


@pytest.mark.parametrize(
    "text",
    [
        "机器人机器人，不要按红色按钮",
        "机器人机器人，别按绿色按钮",
        "机器人机器人，取消按红色按钮",
        "机器人机器人，停止",
        "机器人机器人，按下按钮",
        "机器人机器人，红色按钮在哪里",
        "机器人机器人，按红色或者绿色按钮",
        "机器人机器人，按红按钮然后打开柜门",
        "机器人机器人，按红按钮然后按红按钮",
        "机器人机器人，按红按钮两次",
        "机器人机器人，按红按钮吗",
        "机器人机器人，按红按钮？",
        "机器人机器人，如果准备好了就按红按钮",
        "机器人机器人，我刚刚按红按钮",
        "机器人机器人，按红按钮是一个任务",
        "机器人机器人，转动红色旋钮",
        "机器人机器人，打开柜门",
        "机器人机器人，按蓝色按钮",
        "机器人机器人，红按钮",
        "机器人机器人，按红按钮，打开门",
        "机器人机器人，按红按钮谢谢",
        "机器人机器人，先按红按钮",
        "按红按钮",
        "",
        "我说机器人机器人，按红按钮",
    ],
)
def test_rejected_commands(text):
    result = parse_command(text)
    assert result.task_id is None
    assert result.reason


def test_wake_word_optional_but_restrictions_still_apply():
    assert parse_command("按红按钮", require_wake_word=False).task_id == "press_red_button"
    assert parse_command("不要按红按钮", require_wake_word=False).task_id is None


def test_normalization_preserves_questions_negation_and_multitask():
    assert normalize_text("機器人，機器人　不要按紅色按扭？") == "机器人,机器人不要按红色按钮?"
    assert "然后" in normalize_text("按紅按鈕，然後按綠按鈕")


def test_prompts_are_exact_and_registry_is_immutable():
    assert {key: spec.prompt for key, spec in TASKS.items()} == {
        "press_red_button": "Use the right arm to press the red trip button on the control panel.",
        "press_green_button": "Use the right arm to press the green close button on the control panel.",
    }
    with pytest.raises(TypeError):
        TASKS["open_door"] = TASKS["press_red_button"]
    with pytest.raises(dataclasses.FrozenInstanceError):
        TASKS["press_red_button"].prompt = "other"


def test_service_does_not_transcribe_empty_recording():
    class Recorder:
        def record(self):
            return None

    class Recognizer:
        def transcribe(self, _path):
            pytest.fail("Should not transcribe without recorded audio")

    assert VoiceService(Recorder(), Recognizer()).listen_once().reason == "empty_text"


def test_service_passes_audio_and_keeps_original_transcript():
    class Recorder:
        def record(self):
            return Path("sample.wav")

    class Recognizer:
        def transcribe(self, path):
            assert path == Path("sample.wav")
            return "機器人機器人，按綠色按鈕"

    result = VoiceService(Recorder(), Recognizer()).listen_once()
    assert result.task_id == "press_green_button"
    assert "機器人" in result.transcript


def test_text_cli_does_not_import_robot_or_audio_dependencies():
    script = """
import importlib.abc
import sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"sounddevice", "faster_whisper", "pyrealsense2", "openpi_client", "torch", "jax"}:
            raise AssertionError("Unexpected import: " + fullname)
sys.meta_path.insert(0, Block())
from examples.xtrainer_real.voice.main import Args, main
main(Args(text="机器人机器人，按红按钮", dry_run=True))
"""
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=True)
    assert TASKS["press_red_button"].prompt in result.stdout
