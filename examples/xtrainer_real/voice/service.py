from examples.xtrainer_real.voice.command_parser import ParseResult
from examples.xtrainer_real.voice.command_parser import parse_command


class VoiceService:
    def __init__(self, recorder, recognizer, *, require_wake_word: bool = True) -> None:
        self._recorder = recorder
        self._recognizer = recognizer
        self._require_wake_word = require_wake_word

    def listen_once(self) -> ParseResult:
        audio_path = self._recorder.record()
        text = self._recognizer.transcribe(audio_path) if audio_path is not None else ""
        return parse_command(text, require_wake_word=self._require_wake_word)
