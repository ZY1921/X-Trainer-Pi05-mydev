import dataclasses
import re

from examples.xtrainer_real.voice.normalizer import normalize_text

_WAKE_WORD = re.compile(r"^机器人[,。.!、]*机器人[,。.!、]*")
_NEGATIVE = re.compile(r"不|别|取消|停止|停下|禁止")
_QUESTION = re.compile(r"\?|吗|么|是否|能否|哪里|何处|如何|怎么|为什么")
_MULTIPLE = re.compile(r"或者|或是|还是|然后|接着|再|并且|同时|以及|和|及|两次|两下|重复|一直")
_PREFIX = r"(?:(?:请(?:你)?|麻烦(?:你)?)(?:帮我)?|帮我)?"
_LOCATION = r"(?:(?:控制)?面板上(?:的)?)?"
_TARGET = rf"{_LOCATION}(?P<color>红|绿)(?:色)?(?:按钮|按键|开关)"
_PRESS = r"(?:按下|按|点击|摁下|摁|压下)(?:一下|一次|下)?"
_COMMAND = re.compile(
    rf"{_PREFIX}(?:{_PRESS}{_TARGET}(?:一下|一次|下|下去)?|"
    rf"把{_LOCATION}(?P<ba_color>红|绿)(?:色)?(?:按钮|按键|开关){_PRESS}(?:去|下去)?)"
    r"[。.!]*"
)


@dataclasses.dataclass(frozen=True)
class ParseResult:
    transcript: str
    normalized_text: str
    task_id: str | None
    reason: str | None


def parse_command(text: str, *, require_wake_word: bool = True) -> ParseResult:
    normalized = normalize_text(text)

    def reject(reason: str) -> ParseResult:
        return ParseResult(text, normalized, None, reason)

    if not normalized:
        return reject("empty_text")
    if _NEGATIVE.search(normalized):
        return reject("negative_or_stop")
    if _QUESTION.search(normalized):
        return reject("question")
    wake = _WAKE_WORD.match(normalized)
    if require_wake_word and wake is None:
        return reject("missing_wake_word")
    body = normalized[wake.end() :] if wake else normalized
    if ("红" in body and "绿" in body) or _MULTIPLE.search(body):
        return reject("multiple_or_ambiguous_tasks")
    match = _COMMAND.fullmatch(body)
    if match is None:
        return reject("unsupported_command")
    color = match.group("color") or match.group("ba_color")
    task_id = "press_red_button" if color == "红" else "press_green_button"
    return ParseResult(text, normalized, task_id, None)
