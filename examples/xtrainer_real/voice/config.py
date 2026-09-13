import dataclasses
from pathlib import Path

VOICE_ROOT = Path(__file__).resolve().parent


@dataclasses.dataclass(frozen=True)
class VoiceConfig:
    model_path: Path = VOICE_ROOT / "models/faster-whisper-base"
    audio_path: Path = VOICE_ROOT / "recordings/command.wav"
    mic_device: int = 5
    sample_rate: int = 48000
    channels: int = 2
    require_wake_word: bool = True
