# X-Trainer 右臂语音控制 v1

在 `feature-nova2-v4-protocol` 分支，语音入口与其他真机入口共用 Nova2 V4 `29999` 命令连接和严格回复处理。下文 9 月 9 日的运行记录属于 V3 环境；V4.6.5.0-Force-beta6 的现场验收步骤见 [实机部署](../../../docs/实机部署.md)。

语音入口支持按红色按钮、按绿色按钮。每次有效语音授权一次执行；模型服务、Whisper、相机和机器人连接在进程内复用。执行期间不录音，任务结束后 reset 并等待下一次输入。

## 安装与本地模型

在机器人控制机的 Python 3.11+ 环境中，从仓库根目录安装：

```bash
python -m pip install -e packages/openpi-client
python -m pip install -r examples/xtrainer_real/voice/requirements.txt
```

如果 Linux 提示找不到 PortAudio，安装系统包 `libportaudio2`。Whisper 默认使用本目录的 `models/faster-whisper-base`，目录内必须有 `model.bin`、`config.json`、`tokenizer.json`、`vocabulary.txt`。早期验证脚本保留在 `prototype/`。其他机器需自行复制模型，或使用 `--model-path /absolute/model/path`。不会自动联网下载模型。

固定录音配置：麦克风 5、48000 Hz、双声道、int16。Whisper：CPU、int8、中文、beam size 5。`--mic-device` 可覆盖设备编号。临时音频写入本目录 `recordings/command.wav`，后一次覆盖前一次。模型和录音目录均被 Git 忽略。

## 先验证文本和音频

```bash
python -m examples.xtrainer_real.voice.main \
  --text "机器人机器人，按红色按钮" --dry-run

python -m examples.xtrainer_real.voice.main \
  --text "机器人机器人，帮我按一下绿色按键" --dry-run

python -m examples.xtrainer_real.voice.main \
  --audio-file examples/xtrainer_real/voice/prototype/command.wav --dry-run

# 录音、识别一次，打印结果但不连接机器人或推理服务器
python -m examples.xtrainer_real.voice.main --dry-run
```

纯文本 dry-run 只需要 CLI 和 OpenCC 依赖，不加载 Whisper、sounddevice、机器人驱动或模型客户端。`--text` 和 `--audio-file` 互斥，且第一版必须与 `--dry-run` 一起使用。返回 JSON 包含原始转写、归一化文本、task ID、拒绝原因及固定 Prompt。

默认要求唤醒词“机器人机器人”，兼容“机器人，机器人”和空格。`--no-require-wake-word` 可关闭此要求。否定、停止、疑问、条件句、描述、多任务、重复动作、颜色不明确、柜门或旋钮命令均拒绝。第一版采用完整句式匹配，无法识别的口语也会拒绝，请重新按示例说出指令。

| task ID | 唯一 Prompt |
| --- | --- |
| `press_red_button` | `Use the right arm to press the red trip button on the control panel.` |
| `press_green_button` | `Use the right arm to press the green close button on the control panel.` |

语音入口没有自由 Prompt 参数，不能绕过白名单调用模型。

## 9 月 9 日服务端与语音客户端

在 GPU 服务器启动已验证的红绿混合模型：

```bash
uv run scripts/deployment/serve_policy_async_rtc.py \
  --port 18080 \
  --asset-id xtrainer_right_arm \
  policy:checkpoint \
  --policy.config=pi05_xtrainer_right_arm_custom \
  --policy.dir=/home/user/Dobot/checkpoints/pi05_right_finetune/13000
```

在机器人控制机启动：

```bash
python -m examples.xtrainer_real.voice.main
```

默认配置就是 9 月 9 日红绿混合任务的连接与 RTC 参数：

| 参数 | 默认值 |
| --- | --- |
| 服务器 | `100.100.41.112:18080` |
| 右臂 IP、夹爪 | `192.168.5.2`、`/dev/ttyUSB0`、ID 22 |
| top、right wrist 相机 | `409122272170`、`409122270904` |
| 相机 / 控制频率 | 30 / 22 Hz |
| action horizon | 25 |
| RTC delay / execution horizon | 6 / 15 |
| prefix schedule / guidance weight | linear / 10 |
| 最大推理延迟 | 24 步 |
| 连接 / 在线请求 / warmup 超时 | 120 / 3 / 600 秒 |
| 图像传输 | JPEG，质量 90 |

这些值可以通过 CLI 覆盖，完整列表见 `--help`。需要预览相机时传 `--show-camera-preview`。现有直接指定 Prompt 的同步和异步入口照常使用。

程序加载一次 Whisper 并连接推理服务。首次有效语音后才连接机器人、初始化夹爪并 reset；启动提示进入 LISTENING 后，按 Enter 开始录音，说完按 Enter 结束。有效指令立即开始执行，不要求第二次确认。

```text
LISTENING → 解析 → 任务前 RESETTING → EXECUTING
          → 结束检测 → RTC 清理 → RESETTING → LISTENING
```

每次任务使用独立的递增任务编号，推理请求编号在进程内持续递增。下一任务的首次推理从空 RTC 前缀开始。空闲期间不请求模型动作。

## 结束检测与参数调整

每次任务前 reset 后读取实际起点，以前六个关节与该起点的绝对偏差最大值作为距离，单位 rad；不比较夹爪，不要求精准落到固定 reset 位姿。

| CLI 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `--motion-start-threshold` | 0.02 | 开始运动距离 |
| `--motion-start-confirm-frames` | 3 | 连续观测确认开始运动 |
| `--departure-threshold` | 0.20 | 必须至少离开过这个范围 |
| `--completion-ignore-seconds` | 6.0 | 从首次确认序列的第一帧起屏蔽返回检测 |
| `--return-threshold` | 0.15 | 起点附近范围 |
| `--return-confirm-frames` | 5 | 屏蔽结束后连续进入范围才停止 |
| `--max-episode-steps` | 450 | 最多下发的模型动作次数 |
| `--motion-start-timeout-steps` | 110 | 此步数仍未确认运动则结束 |

6 秒使用实际单调时钟，与控制频率和推理耗时无关。模型编译、warmup、reset 和录音时间不计入。450 步在实际达到 22 Hz 时约 20.5 秒；它是步数上限，不是墙钟超时。环境内部为单次动作做的平滑插值不另计模型动作步数。

返回检测、未启动和最大步数触发后，均停止继续消费 action chunk，清空 Prompt，最多用在线请求超时加 1 秒等待并丢弃未完成响应，然后调用现有 reset。结束原因分别为 `returned_to_start`、`failed_to_start`、`max_steps`。

`returned_to_start` 仅表示控制器检测到返回邻域，不保证按钮成功按下。阈值来自已有训练轨迹的初步统计，需要实机记录校准。第一版不做视觉或接触成功判断。

Ctrl+C 在录音、warmup、任务执行及 reset 中均可退出，不再额外启动 reset。它停止新的软件动作下发，不是硬件急停；底层断连沿用原驱动的 DisableRobot 行为。相机预览关闭也结束整个会话。异常、无效状态/动作、严格通信检查失败或 RTC 清理失败时退出，不自动重试任务或故障后回位。

## 日志、代码与验证

任务日志位于 `output/voice/task_<编号>_<任务>_<随机后缀>/`：

- `task_summary.json`：转写、Prompt、配置、起点、步数、结束原因和 reset 是否完成。
- `completion.jsonl`：每个控制观测的时间、关节位置、起点距离、运动开始时间和离开标志。
- `inference_actions_*/`：复用现有动作记录器的 NPZ、CSV 和绘图。

`--inference-action-output-dir` 修改输出位置；`--no-record-inference-actions` 关闭动作 chunk 记录，仍保留任务和结束检测日志。

语音包只负责录音、识别、归一化和解析。`task_registry.py` 持有 Prompt 白名单，`task_completion.py` 是纯计算检测器，`task_executor.py` 管理任务运行，`main.py` 提供入口。严格硬件通信、语音环境和 RTC 生命周期适配也全部位于本目录；旧入口及其硬件驱动不受影响。

离线验证命令：

```bash
python -m pytest -q \
  examples/xtrainer_real/voice/tests \
  examples/xtrainer_real/test/right_arm_single_test.py \
  examples/xtrainer_real/test/inference_environment_test.py \
  scripts/deployment/test/async_rtc_client_test.py \
  scripts/deployment/test/inference_action_recorder_test.py
```

测试使用假机器人、假网络和模拟时钟，不连接真机。真机验收依次进行单次红色任务、单次绿色任务、红→绿连续任务，检查结束点、reset、旧动作清理和输入等待。真机测试不由离线测试自动启动。
