from pathlib import Path


class SpeechRecognizer:
    def __init__(self, model_path: Path) -> None:
        for name in ("config.json", "model.bin", "tokenizer.json", "vocabulary.txt"):
            if not (model_path / name).is_file():
                raise FileNotFoundError(f"Whisper 本地模型缺少文件: {model_path / name}")
        from faster_whisper import WhisperModel

        self._model = WhisperModel(str(model_path), device="cpu", compute_type="int8", local_files_only=True)

    def transcribe(self, audio_path: Path) -> str:
        if not audio_path.is_file():
            raise FileNotFoundError(audio_path)
        segments, _ = self._model.transcribe(str(audio_path), language="zh", beam_size=5)
        return "".join(segment.text for segment in segments).strip()
