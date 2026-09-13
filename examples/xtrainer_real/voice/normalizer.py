import functools
import re
import unicodedata


@functools.lru_cache(maxsize=1)
def _converter():
    try:
        from opencc import OpenCC
    except ImportError as error:
        raise RuntimeError("请先安装 examples/xtrainer_real/voice/requirements.txt 中的 OpenCC 依赖。") from error
    return OpenCC("t2s")


def normalize_text(text: str) -> str:
    """Preserve punctuation and sentence meaning for rejection checks."""
    text = unicodedata.normalize("NFKC", _converter().convert(text)).lower()
    text = re.sub(r"\s+", "", text)
    for original, replacement in (("按扭", "按钮"), ("摁钮", "按钮")):
        text = text.replace(original, replacement)
    return text
