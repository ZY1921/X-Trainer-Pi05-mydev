# Dobot X-Trainer Pi0.5（JAX）

本仓库基于 OpenPI，为 Dobot X-Trainer 增加了双臂和单右臂的数据转换、Pi0.5 微调、离线评估、远程推理与真机控制能力。

当前代码同时保留 14 维双臂和 7 维单右臂链路。现阶段有实机运行记录的正式部署路线是：7 维单右臂、红/绿按钮任务、22 Hz 控制、远程异步 RTC 推理。

> RTX 5090 服务端安装、Dobot 客户端安装、FFmpeg/PyAV 排查、checkpoint 对应关系以及经过实机验证的启动命令，请直接阅读 **[实机部署文档](docs/实机部署.md)**。README 只提供项目总览和开发入口，不替代正式部署步骤。

## 功能概览

- 将 X-Trainer raw episode 转换为 LeRobot v2.1 数据集。
- 从双臂 raw 数据生成 14 维双臂数据集或 7 维单右臂数据集。
- 合并多个单右臂单任务数据集，并重写 episode、frame、task 和视频索引。
- 计算与训练 config 对应的 normalization statistics。
- 支持 Pi0.5 JAX 全参数、LoRA 和 LoRA r64 微调。
- 提供 7 维单右臂专用 policy、metadata 校验和开环评估。
- 提供同步 WebSocket 与异步 RTC 两套远程推理协议。
- 支持 RTC 前缀引导、JPEG 图像传输、延迟统计和动作记录。
- 提供双臂/单右臂、同步/异步四个真机客户端入口。
- 提供红/绿按钮白名单式语音控制入口。

## 目录结构

| 路径 | 用途 |
| --- | --- |
| `src/openpi/` | 模型、policy、数据变换、训练配置与公共库 |
| `src/openpi/training/config.py` | X-Trainer 双臂和单右臂 config 定义 |
| `src/openpi/policies/xtrainer_policy.py` | 14 维双臂输入输出变换 |
| `src/openpi/policies/xtrainer_right_arm_policy.py` | 7 维单右臂输入输出与模型槽位变换 |
| `examples/xtrainer_real/` | 真机环境、硬件封装、客户端和数据转换 |
| `examples/xtrainer_real/data_convert/` | raw → LeRobot v2.1 与多任务合并工具 |
| `examples/xtrainer_real/voice/` | 右臂红/绿按钮语音控制 |
| `scripts/training/` | norm stats、JAX/PyTorch 训练入口 |
| `scripts/evaluation/` | 通用和单右臂开环评估 |
| `scripts/deployment/` | 同步 policy server、异步 RTC server 与协议测试 |
| `packages/openpi-client/` | 机器人侧轻量客户端、运行时和图像传输协议 |
| `docs/` | 部署、norm stats、远程推理和 Docker 文档 |

数据采集与遥操作由独立的 `dobot_xtrainer` 项目负责；本仓库从 raw episode 转换开始，覆盖数据准备、训练、评估和推理。

## 端到端流程

```text
dobot_xtrainer 遥操作与采集
    → raw episode
    → LeRobot v2.1 转换
    → 数据集检查/多任务合并
    → 计算 norm stats
    → Pi0.5 全参数或 LoRA 微调
    → 单右臂开环评估
    → GPU policy server
    → Dobot 真机客户端
```

双臂和单右臂使用不同的数据维度、相机集合、norm stats 和 config，不能交叉混用。

## 环境安装

### GPU 训练与服务端环境

推荐 Ubuntu 22.04、Python 3.11、兼容 CUDA 12 的 NVIDIA 驱动，并使用 `uv.lock` 创建环境：

```bash
git submodule update --init --recursive
uv venv --python 3.11
source .venv/bin/activate
uv sync --frozen
```

检查 JAX：

```bash
uv run --no-sync python -c "import jax; print(jax.__version__); print(jax.devices())"
```

训练或推理服务器上的 `jax.devices()` 应显示 NVIDIA GPU。系统 FFmpeg 与 PyAV 的版本处理见[实机部署文档的 1.3 节](docs/实机部署.md#13-ffmpeg-与-pyav-依赖问题)。

当前关键版本由仓库固定：

| 依赖 | 当前约束 |
| --- | --- |
| Python | `>=3.11`，仓库默认 3.11 |
| JAX | `jax[cuda12]==0.5.3` |
| LeRobot | Git commit `0cf864870cf29f4738d3ade893e6fd13fbd7cdb5` |
| PyAV | `uv.lock` 当前锁定 `18.0.0` |

### 机器人客户端环境

机器人控制机只安装真机运行依赖和当前仓库中的客户端包：

```bash
uv venv --python 3.11
source .venv/bin/activate
uv pip install -r examples/xtrainer_real/requirements.txt
uv pip install -e packages/openpi-client
```

不要用外部同名 `openpi-client` 覆盖本仓库版本，否则异步 RTC 协议或 JPEG 传输可能不兼容。

## 数据契约

| 模式 | state/action | 相机 | asset id |
| --- | --- | --- | --- |
| 14 维双臂 | `[左臂 6 关节, 左夹爪, 右臂 6 关节, 右夹爪]` | Top、Left wrist、Right wrist | `xtrainer` |
| 7 维单右臂 | `[右臂 6 关节, 右夹爪]` | Top、Right wrist | `xtrainer_right_arm` |

公共字段：

- `observation.state`：机器人状态。
- `action`：训练目标动作。
- `observation.images.top`：顶部相机 RGB 图像。
- `observation.images.left_wrist`：左腕相机，仅双臂链路使用。
- `observation.images.right_wrist`：右腕相机。
- `task` / `prompt`：任务语言描述。

双臂数据默认将两组 6 关节转成 delta action，两个夹爪保持 absolute action。单右臂数据默认将 6 关节转成 delta action，右夹爪保持 absolute action。

单右臂物理数据始终是 7 维；`PackRightArmToModelSlots` 在进入 Pi0.5 前将其放入模型的 `[7:14]` 槽位，输出时再由 `UnpackRightArmFromModelSlots` 还原为 7 维。单右臂 norm stats 也必须是 7 维，不能使用旧的 14 维 stats。

## 数据转换

所有命令都从仓库根目录执行。完整参数可查看各模块的 `--help`。

### 14 维双臂数据集

```bash
python -m examples.xtrainer_real.data_convert.convert_raw_to_lerobot_2_1 \
  --raw_root /path/to/collect_data \
  --output_root /path/to/xtrainer_bimanual_lerobot \
  --repo_id local/xtrainer_bimanual \
  --fps 30 \
  --task "<training prompt>" \
  --use_videos
```

### 7 维单右臂数据集

该转换器从双臂 raw state/action 中选取源索引 `[7:14]`，并只保留 Top 和 Right wrist 图像：

```bash
python -m examples.xtrainer_real.data_convert.convert_raw_right_arm_to_lerobot_2_1 \
  --raw_root /path/to/collect_data \
  --output_root /path/to/xtrainer_right_arm_lerobot \
  --fps 22 \
  --task "Use the right arm to press the red trip button on the control panel." \
  --use_videos
```

转换器会生成 LeRobot v2.1 的 `meta/`、`data/` 和 `videos/` 结构，并按共同 frame ID 对齐 observation、action 和相机帧。默认会跳过无法组成完整样本的坏帧；需要严格失败时使用 `--fail_on_bad_frames`。

### 合并单右臂多任务数据集

输入必须是两个或更多完整的单任务 LeRobot v2.1 数据集。源数据不会被修改：

```bash
python -m examples.xtrainer_real.data_convert.merge_right_arm_lerobot_2_1 \
  --input_roots \
    /path/to/right_arm_red_lerobot \
    /path/to/right_arm_green_lerobot \
  --output_root /path/to/right_arm_red_green_lerobot \
  --media_mode auto
```

`auto` 会优先硬链接视频，无法硬链接时回退为复制。目标目录非空时默认拒绝覆盖；明确需要重建时再传 `--overwrite_output`。

## X-Trainer 配置

当前 `src/openpi/training/config.py` 中的 X-Trainer 配置如下：

| 模式 | config | 用途 | asset id |
| --- | --- | --- | --- |
| 双臂 | `pi05_xtrainer` | Pi0.5 base-compatible smoke inference | `trossen` |
| 双臂 | `pi05_xtrainer_custom` | 加载双臂全参数 checkpoint | `xtrainer` |
| 双臂 | `pi05_xtrainer_finetune` | 双臂全参数微调 | `xtrainer` |
| 双臂 | `pi05_xtrainer_lora_finetune` | 双臂 LoRA 微调/加载 | `xtrainer` |
| 双臂 | `pi05_xtrainer_lora_r64_finetune` | 双臂 LoRA r64 实验配置 | `xtrainer` |
| 单右臂 | `pi05_xtrainer_right_arm_custom` | 加载单右臂全参数 checkpoint | `xtrainer_right_arm` |
| 单右臂 | `pi05_xtrainer_right_arm_finetune` | 单右臂全参数微调 | `xtrainer_right_arm` |
| 单右臂 | `pi05_xtrainer_right_arm_lora_finetune` | 单右臂 LoRA 微调/加载 | `xtrainer_right_arm` |
| 单右臂 | `pi05_xtrainer_right_arm_lora_r64_finetune` | 单右臂 LoRA r64 实验配置 | `xtrainer_right_arm` |

全参数 checkpoint 用对应的 `*_custom` config 部署；LoRA checkpoint 必须继续使用训练时的 LoRA 结构 config，不能用全参数 config 加载。

当前 r64 variant 已在配置表中定义，但其字符串尚未加入 `Pi0Config` 对应字段的 `Literal` 类型范围，生成 Tyro CLI 时会出现类型注解警告。因此 r64 路线应视为实验性配置，正式训练前需单独验证 CLI、模型初始化和 checkpoint 恢复。

快速检查配置：

```bash
uv run --no-sync python -c "
from openpi.training import config
for name in (
    'pi05_xtrainer_finetune',
    'pi05_xtrainer_lora_finetune',
    'pi05_xtrainer_right_arm_finetune',
    'pi05_xtrainer_right_arm_lora_finetune',
):
    cfg = config.get_config(name)
    print(name, cfg.data.assets.asset_id, cfg.model.action_horizon)
"
```

## Norm stats 与训练

### 计算 norm stats

下面以 7 维单右臂全参数训练为例。`HF_LEROBOT_HOME` 应指向本地 LeRobot 数据集父目录，`repo-id` 应与其下的数据集目录匹配：

```bash
HF_LEROBOT_HOME=/path/to/lerobot_root \
uv run scripts/training/compute_norm_stats.py \
  pi05_xtrainer_right_arm_finetune \
  --data.repo-id local/xtrainer_right_arm_dataset \
  --data.assets.asset-id xtrainer_right_arm \
  --exp-name right_arm_norm_stats \
  --batch-size 32 \
  --num-workers 4
```

默认输出位置：

```text
assets/pi05_xtrainer_right_arm_finetune/xtrainer_right_arm/norm_stats.json
```

双臂训练时改用双臂 config 和 `xtrainer` asset id。更多背景和检查方法见 [docs/norm_stats.md](docs/norm_stats.md)。

### Smoke training

当前 config 中的 base checkpoint 路径是开发机路径，换机器时必须通过 `--weight-loader.params-path` 覆盖：

```bash
HF_LEROBOT_HOME=/path/to/lerobot_root \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.90 \
uv run scripts/training/train.py \
  pi05_xtrainer_right_arm_finetune \
  --data.repo-id local/xtrainer_right_arm_dataset \
  --data.assets.asset-id xtrainer_right_arm \
  --weight-loader.params-path /path/to/pi05_base/params \
  --exp-name right_arm_full_smoke \
  --batch-size 1 \
  --num-workers 0 \
  --num-train-steps 50 \
  --save-interval 50 \
  --no-wandb-enabled \
  --overwrite
```

先确认数据、tokenizer、base params、norm stats、显存和 checkpoint 保存均正常，再提高 batch size 和训练步数。

### 正式训练

- 单右臂全参数：`pi05_xtrainer_right_arm_finetune`，默认 20,000 steps、batch size 32。
- 单右臂 LoRA：`pi05_xtrainer_right_arm_lora_finetune`，默认 20,000 steps、batch size 8、无 EMA。
- 单右臂 LoRA r64：`pi05_xtrainer_right_arm_lora_r64_finetune`，默认 40,000 steps、batch size 8、无 EMA；当前属于实验性配置。
- 双臂使用对应的不含 `right_arm` 的配置。

训练命令与 smoke training 相同，只需选择 config、实验名、batch size、步数和保存间隔。恢复已有实验时使用 `--resume`，新建并覆盖同名实验时使用 `--overwrite`；两者不能同时使用。

checkpoint 默认输出到：

```text
checkpoints/<config_name>/<exp_name>/<step>/
```

部署前确认 step 目录至少包含 `params/`、`_CHECKPOINT_METADATA` 和对应 asset id 下的 `norm_stats.json`。

## 单右臂开环评估

`scripts/evaluation/xtrainer_right_arm_open_loop_eval.py` 会在加载模型前检查：

- LeRobot 数据版本为 v2.1；
- state/action 为 7 维；
- 相机键严格为 Top 和 Right wrist；
- 数据 FPS 与期望值一致；
- checkpoint 与 `xtrainer_right_arm` norm stats 完整；
- policy metadata 使用模型槽位 `[7:14]`。

示例：

```bash
uv run scripts/evaluation/xtrainer_right_arm_open_loop_eval.py \
  --dataset-path /path/to/right_arm_lerobot \
  --checkpoint-path /path/to/right_arm_lora_checkpoint \
  --traj-ids 0 50 100 \
  --expected-dataset-fps 22 \
  --metrics-output-dir output/xtrainer_right_arm_open_loop
```

该入口固定使用 `pi05_xtrainer_right_arm_lora_finetune` 和 `xtrainer_right_arm`，用于比较单右臂 baseline 与 RTC 开环结果。输出目录已存在且非空时，需明确传 `--overwrite-output`。

## 远程推理与真机入口

GPU 机器运行 policy server，机器人控制机运行客户端，两端通过 WebSocket 通信。同步和异步 RTC 是两套不同协议，入口必须配对：

| 控制范围 | 服务方式 | 服务端 | 客户端 | 动作维度 |
| --- | --- | --- | --- | ---: |
| 单右臂 | 同步 | `scripts/deployment/serve_policy.py` | `examples.xtrainer_real.right_arm_single_main` | 7 |
| 单右臂 | 异步 RTC | `scripts/deployment/serve_policy_async_rtc.py` | `examples.xtrainer_real.right_arm_single_async_rtc_main` | 7 |
| 双臂 | 同步 | `scripts/deployment/serve_policy.py` | `examples.xtrainer_real.bimanual_main` | 14 |
| 双臂 | 异步 RTC | `scripts/deployment/serve_policy_async_rtc.py` | `examples.xtrainer_real.bimanual_async_rtc_main` | 14 |

查看当前命令行参数：

```bash
uv run scripts/deployment/serve_policy_async_rtc.py --help
python -m examples.xtrainer_real.right_arm_single_async_rtc_main --help
```

当前正式实机路线使用 `openpi-async-rtc` version 2、JPEG 图像传输和 7 维单右臂 metadata 校验。完整服务器 IP、端口、相机序列号、夹爪参数、RTC 参数、红/绿 prompt、checkpoint 路径与启动顺序统一维护在 **[docs/实机部署.md](docs/实机部署.md)**，部署时以该文档为准。

四个客户端共享当前图像预处理：采集尺寸固定为 `640×480`，Top 裁剪到前 430 行后缩放回 `640×480`，Right wrist 旋转 180°，Left wrist 保持原方向。添加 `--show-camera-preview` 可查看模型实际接收的画面。

单右臂同步和两种异步客户端默认记录推理动作到 `output/`。异步客户端还可记录请求延迟、chunk 切换和 JPEG 传输统计。详细入口行为见 [examples/xtrainer_real/README.md](examples/xtrainer_real/README.md)。

> 双臂入口会实际控制左右两条机械臂，不会冻结左臂。当前正式部署文档只收录已实机运行的单右臂 RTC 命令；双臂和同步模式在完成对应真机验收前，不应直接照搬模板上线。

## 语音控制

语音入口只允许两个白名单任务：

| task ID | Prompt |
| --- | --- |
| `press_red_button` | `Use the right arm to press the red trip button on the control panel.` |
| `press_green_button` | `Use the right arm to press the green close button on the control panel.` |

安装可选依赖并先做纯文本 dry-run：

```bash
python -m pip install -r examples/xtrainer_real/voice/requirements.txt
python -m examples.xtrainer_real.voice.main \
  --text "机器人机器人，按红色按钮" \
  --dry-run
```

语音模式的本地 Whisper 模型、麦克风、结束检测、RTC 生命周期和真机启动说明见 [examples/xtrainer_real/voice/README.md](examples/xtrainer_real/voice/README.md)。

## 测试

测试 X-Trainer 客户端和语音模块前，先安装其可选运行依赖：

```bash
uv pip install -r examples/xtrainer_real/voice/requirements.txt
```

运行仓库自动测试（排除 manual 标记）：

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
python -m pytest --strict-markers -m "not manual"
```

只运行 X-Trainer 相关回归测试：

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
python -m pytest -q \
  examples/xtrainer_real/test \
  examples/xtrainer_real/voice/tests \
  scripts/deployment/test \
  scripts/evaluation/xtrainer_right_arm_open_loop_eval_test.py \
  src/openpi/policies/xtrainer_right_arm_policy_test.py
```

`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` 可避免已 source 的 ROS 环境把 Python 3.10 `launch_testing` 等外部 pytest 插件注入当前 Python 3.11 环境。仓库测试使用假机器人、假网络或离线数据，不会主动连接真机；真机验收必须人工执行并准备硬件急停。

## 文档索引

| 文档 | 内容 |
| --- | --- |
| [docs/实机部署.md](docs/实机部署.md) | 当前正式实机部署流程、PyAV/FFmpeg、已验证 client-server 命令与 RTC 参数 |
| [examples/xtrainer_real/README.md](examples/xtrainer_real/README.md) | 四个真机客户端入口、图像处理、预览、动作记录与安全说明 |
| [examples/xtrainer_real/voice/README.md](examples/xtrainer_real/voice/README.md) | 右臂语音控制安装、白名单、结束检测和验证 |
| [scripts/README.md](scripts/README.md) | training、deployment、evaluation 脚本分类 |
| [docs/norm_stats.md](docs/norm_stats.md) | normalization statistics 原理与使用 |
| [docs/remote_inference.md](docs/remote_inference.md) | OpenPI 通用远程推理接口 |
| [docs/docker.md](docs/docker.md) | Docker 相关说明 |

## 安全说明

- 首次连接新 checkpoint 前确认 config、动作维度、相机集合、reset pose 和 norm stats 全部匹配。
- 单右臂和双臂 checkpoint、asset id、客户端入口不得混用。
- 开始真机运动前清空工作区并确认急停有效。
- 先运行少量 episode 和较小最大步数，确认关节方向、夹爪方向与图像方向后再扩大测试。
- 软件停止不能替代硬件急停；发生方向异常、突然加速或碰撞风险时应立即触发急停。

## License

本仓库继承上游 OpenPI 的许可与模型许可要求，详见 [LICENSE](LICENSE) 和 [LICENSE_GEMMA.txt](LICENSE_GEMMA.txt)。
