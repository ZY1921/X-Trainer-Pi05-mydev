# ruff: noqa: RUF001
from pathlib import Path
import wave

from examples.xtrainer_real.voice.config import VoiceConfig


class AudioRecorder:
    def __init__(self, config: VoiceConfig) -> None:
        import sounddevice as sd

        self._sd = sd
        self._config = config
        sd.check_input_settings(
            device=config.mic_device, channels=config.channels, dtype="int16", samplerate=config.sample_rate
        )

    def record(self) -> Path | None:
        input("按 Enter 开始录音（Ctrl+C 退出）…")
        chunks = []
        errors = []

        def callback(indata, _frames, _time, status):
            if status:
                errors.append(str(status))
            chunks.append(bytes(indata))

        # Input stays on the main thread: Ctrl+C never leaves an input-reader thread behind.
        with self._sd.RawInputStream(
            device=self._config.mic_device,
            channels=self._config.channels,
            samplerate=self._config.sample_rate,
            dtype="int16",
            callback=callback,
        ):
            input("正在录音，说完后按 Enter 停止…")
        if errors:
            raise RuntimeError(f"录音失败: {errors[0]}")
        if not chunks:
            return None
        path = self._config.audio_path
        path.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(path), "wb") as output:
            output.setnchannels(self._config.channels)
            output.setsampwidth(2)
            output.setframerate(self._config.sample_rate)
            output.writeframes(b"".join(chunks))
        return path
