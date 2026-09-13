# ruff: noqa
from faster_whisper import WhisperModel
from opencc import OpenCC

AUDIO_FILE = "mic_test.wav"

model = WhisperModel(
    "/home/hfy/voice_test/models/faster-whisper-base",
    device="cpu",
    compute_type="int8"
)

cc = OpenCC("t2s")

segments, info = model.transcribe(
    AUDIO_FILE,
    language="zh"
)

text = ""

for segment in segments:
    text += segment.text

text = text.strip()

print("Whisper 原始结果：", text)

# 繁体转简体
text = cc.convert(text)

print("转换后的结果：", text)

if "按钮" in text and "按" in text:
    print("识别到 press_button")
else:
    print("未识别到有效任务")
