# ruff: noqa
import threading
import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write

from faster_whisper import WhisperModel
from opencc import OpenCC


# =========================
# 基本配置
# =========================

MODEL_PATH = "/home/hfy/voice_test/models/faster-whisper-base"
AUDIO_FILE = "command.wav"

# 你的 USB 麦克风当前已验证可用：
# 48000 Hz
# 双声道
SAMPLE_RATE = 48000
CHANNELS = 2

# 如果系统默认录音设备就是 USB 麦克风，可以保持 None
MIC_DEVICE = 5


# =========================
# 加载 Whisper
# =========================

print("正在加载 Whisper 模型...")

model = WhisperModel(
    MODEL_PATH,
    device="cpu",
    compute_type="int8"
)

cc = OpenCC("t2s")

print("Whisper 模型加载完成")


# =========================
# 录音函数
# =========================

def record_audio():

    print("\n按 Enter 开始录音...")
    input()

    print("\n开始录音")
    print("请说：机器人机器人，按下红色按钮")
    print("说完以后，再按 Enter 停止录音")

    audio_chunks = []
    stop_recording = False

    def callback(indata, frames, time, status):

        if status:
            print("录音状态：", status)

        audio_chunks.append(indata.copy())

    def wait_for_enter():

        nonlocal stop_recording

        input()

        stop_recording = True

    stop_thread = threading.Thread(
        target=wait_for_enter,
        daemon=True
    )

    stop_thread.start()

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype="int16",
        device=MIC_DEVICE,
        callback=callback
    ):

        while not stop_recording:
            sd.sleep(100)

    if len(audio_chunks) == 0:

        print("没有录到音频")

        return None

    audio = np.concatenate(
        audio_chunks,
        axis=0
    )

    write(
        AUDIO_FILE,
        SAMPLE_RATE,
        audio
    )

    print("\n录音完成")
    print("录音文件保存为：", AUDIO_FILE)

    return AUDIO_FILE


# =========================
# Whisper 识别函数
# =========================

def speech_to_text(audio_file):

    print("\n开始语音识别...")

    segments, info = model.transcribe(
        audio_file,
        language="zh",
        beam_size=5
    )

    text = ""

    for segment in segments:

        print(
            f"[{segment.start:.2f}s -> "
            f"{segment.end:.2f}s] "
            f"{segment.text}"
        )

        text += segment.text

    text = text.strip()

    print("\nWhisper 原始结果：")
    print(text)

    # =========================
    # 繁体转简体
    # =========================

    text_simple = cc.convert(text)

    print("\n转换后的结果：")
    print(text_simple)

    return text_simple


# =========================
# 指令解析函数
# =========================

def parse_command(text):

    print("\n开始判断机器人指令...")

    # 双唤醒词
    wake_word = "机器人机器人"

    has_wake_word = wake_word in text
    has_press = "按" in text
    has_button = "按钮" in text

    if (
        has_wake_word
        and has_press
        and has_button
    ):

        task = "press_button"

        print("识别到有效机器人指令")
        print("task =", task)

        return task

    else:

        print("未识别到有效机器人指令")
        print("机器人不会执行动作")

        return None


# =========================
# π0.5 触发函数
# =========================

def trigger_pi05(task):

    if task == "press_button":

        # 当前先不真正控制机器人
        # 后续这里再接 X-Trainer π0.5 real client

        prompt = "Press the red button."

        print("\n准备触发 π0.5")
        print("task =", task)
        print("prompt =", prompt)

        print(
            "\n当前为测试模式，"
            "暂时不发送机器人动作"
        )


# =========================
# 主程序
# =========================

def main():

    print("\n==============================")
    print("X-Trainer 语音控制测试程序")
    print("==============================")

    # 1. 录音
    audio_file = record_audio()

    if audio_file is None:
        return

    # 2. Whisper 识别
    text = speech_to_text(audio_file)

    if not text:

        print("没有识别到语音内容")

        return

    # 3. 指令解析
    task = parse_command(text)

    # 4. 模拟触发 π0.5
    if task is not None:

        trigger_pi05(task)


# =========================
# 程序入口
# =========================

if __name__ == "__main__":

    main()
