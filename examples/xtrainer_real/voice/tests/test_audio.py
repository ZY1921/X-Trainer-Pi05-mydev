import dataclasses
import sys
import types
import wave

import pytest

from examples.xtrainer_real.voice.config import VoiceConfig
from examples.xtrainer_real.voice.recognizer import SpeechRecognizer
from examples.xtrainer_real.voice.recorder import AudioRecorder


def test_recognizer_loads_once_and_only_uses_local_chinese_model(tmp_path, monkeypatch):
    for name in ("config.json", "model.bin", "tokenizer.json", "vocabulary.txt"):
        (tmp_path / name).touch()
    audio = tmp_path / "audio.wav"
    audio.touch()
    loads = []
    calls = []

    class Model:
        def __init__(self, path, **kwargs):
            loads.append((path, kwargs))

        def transcribe(self, path, **kwargs):
            calls.append((path, kwargs))
            return iter([types.SimpleNamespace(text="机器人机器人"), types.SimpleNamespace(text="按红按钮")]), None

    monkeypatch.setitem(sys.modules, "faster_whisper", types.SimpleNamespace(WhisperModel=Model))
    recognizer = SpeechRecognizer(tmp_path)
    for _ in range(2):
        assert recognizer.transcribe(audio) == "机器人机器人按红按钮"
    assert loads == [(str(tmp_path), {"device": "cpu", "compute_type": "int8", "local_files_only": True})]
    assert calls == [(str(audio), {"language": "zh", "beam_size": 5})] * 2


def test_missing_model_reports_path_without_loading_whisper(tmp_path):
    with pytest.raises(FileNotFoundError, match="config.json"):
        SpeechRecognizer(tmp_path)


def install_audio_fake(monkeypatch, *, interrupted=False, overflow=False):
    events = []
    inputs = iter(["", ""])

    def get_input(_prompt):
        text = next(inputs)
        if interrupted and "opened" in events:
            raise KeyboardInterrupt
        return text

    class Stream:
        def __init__(self, **kwargs):
            assert kwargs["device"] == 5
            assert kwargs["channels"] == 2
            assert kwargs["samplerate"] == 48000
            self.callback = kwargs["callback"]

        def __enter__(self):
            events.append("opened")
            self.callback(b"\0" * 12, 3, None, "overflow" if overflow else None)

        def __exit__(self, *_args):
            events.append("closed")

    monkeypatch.setattr("builtins.input", get_input)
    monkeypatch.setitem(
        sys.modules,
        "sounddevice",
        types.SimpleNamespace(
            check_input_settings=lambda **_kwargs: None,
            RawInputStream=Stream,
        ),
    )
    return events


def test_recording_writes_stereo_pcm_wav(tmp_path, monkeypatch):
    events = install_audio_fake(monkeypatch)
    config = dataclasses.replace(VoiceConfig(), audio_path=tmp_path / "command.wav")
    path = AudioRecorder(config).record()
    with wave.open(str(path), "rb") as recorded:
        assert recorded.getnchannels() == 2
        assert recorded.getframerate() == 48000
        assert recorded.getsampwidth() == 2
        assert recorded.getnframes() == 3
    assert events == ["opened", "closed"]


@pytest.mark.parametrize("mode", ["interrupt", "overflow"])
def test_bad_recording_closes_stream_and_does_not_write_audio(tmp_path, monkeypatch, mode):
    events = install_audio_fake(monkeypatch, interrupted=mode == "interrupt", overflow=mode == "overflow")
    config = dataclasses.replace(VoiceConfig(), audio_path=tmp_path / "command.wav")
    with pytest.raises((KeyboardInterrupt, RuntimeError)):
        AudioRecorder(config).record()
    assert events == ["opened", "closed"]
    assert not config.audio_path.exists()
