# X-Trainer 部署 Pi0.5（JAX）手册

版本：V1.0  
日期：2026-05-14  

---

## 1. 文档目标

本文档用于指导在 Dobot X-Trainer 双臂平台上完成 Pi0.5（JAX）部署的完整流程，覆盖以下阶段：

1. 使用 `dobot_xtrainer` 项目完成硬件检查、遥操作和原始数据采集。
2. 将 X-Trainer 原始采集数据转换为 OpenPI 训练要求的 LeRobot 格式数据集。
3. 部署 OpenPI Pi0.5（JAX）环境，计算 normalization statistics，执行全参微调与 LoRA 微调。
4. 启动 OpenPI policy server，并通过 X-Trainer real client 进行真实机器人推理验证。


---

## 2. 总体架构

### 2.1 端到端流程

```text
X-Trainer 硬件配置
    -> leader / follower / gripper / RealSense 连接检查
    -> 遥操作 smoke test
    -> 原始 demonstration 采集
    -> raw X-Trainer data
    -> LeRobot dataset 转换
    -> OpenPI 读取 LeRobot dataset
    -> 计算 norm stats
    -> Pi0.5 全参微调或 LoRA 微调
    -> policy server
    -> X-Trainer real client
    -> 真实机器人闭环验证
```

### 2.2 项目分工

| 模块 | 当前目录 | 职责 |
|---|---|---|
| X-Trainer 控制与采集 | `dobot_xtrainer` | 连接 Dobot follower、Dynamixel leader、夹爪、RealSense；执行遥操作；保存 raw episode。 |
| Pi0.5 / OpenPI | `Xtrainer-PI05-feat-xtrainer-finetune` | Pi0.5 JAX 模型、训练配置、norm stats、policy server、X-Trainer 推理 client。 |
| 数据转换 | 建议放在 `dobot_xtrainer/scripts/` 或 `openpi/examples/xtrainer_real/` | 将 raw episode 转成 LeRobot dataset。 |

### 2.3 关键数据契约

| 字段 | 形状 | 含义 |
|---|---:|---|
| `observation.state` | `(14,)` | 双臂状态：左臂 6 关节 + 左夹爪 + 右臂 6 关节 + 右夹爪。 |
| `action` | `(14,)` | 双臂动作，维度顺序同 state。 |
| `observation.images.top` | image | 顶部相机 RGB 图像。 |
| `observation.images.left_wrist` | image | 左腕相机 RGB 图像。 |
| `observation.images.right_wrist` | image | 右腕相机 RGB 图像。 |
| `task` / `prompt` | string | 语言任务描述，训练和推理必须一致或语义一致。 |

OpenPI 适配代码中的 `LeRobotXTrainerDataConfig` 会把 LeRobot 样本映射成模型输入，并默认将 12 个关节维度转换为 delta action，两个夹爪维度保持 absolute action。

---

## 3. 硬件和软件前置条件

### 3.1 硬件组成

| 硬件 | 数量 | 用途 |
|---|---:|---|
| Dobot follower 机械臂 | 2 | 左右从臂执行动作。 |
| Leader 主手 | 2 | 人类遥操作输入。 |
| Feetech / X-Trainer 夹爪 | 2 | 左右夹爪开合控制。 |
| Intel RealSense 相机 | 3 | 顶部、左腕、右腕图像采集。 |
| GPU 服务器 | 1 | Pi0.5 训练和 policy server。 |
| 机器人控制机 | 1 | 连接 X-Trainer 硬件并运行采集或推理 client。 |

### 3.2 网络和设备约定

当前代码默认使用以下 Dobot IP：

```text
左臂 follower: 192.168.5.1
右臂 follower: 192.168.5.2
```

部署前应确认控制机网卡与 Dobot 控制盒处于同一网段，并能 ping 通：

```bash
ping 192.168.5.1
ping 192.168.5.2
```

串口设备通常包括：

```text
/dev/ttyACM*  leader 主手
/dev/ttyUSB*  夹爪
```

检查命令：

```bash
ls -l /dev/ttyACM* /dev/ttyUSB* 2>/dev/null || true
```

### 3.3 系统建议

OpenPI 官方 README 中说明该项目主要在 Ubuntu 22.04 上测试，JAX 训练需要 NVIDIA GPU。建议：

| 环境 | 建议 |
|---|---|
| OS | Ubuntu 22.04 |
| Python | OpenPI 使用 Python 3.11；`dobot_xtrainer` 可使用独立环境安装其 `requirements.txt`。 |
| GPU | LoRA 微调建议至少 RTX 4090 级别；全参微调建议 A100 80GB / H100 或多卡/FSDP。 |
| 依赖管理 | OpenPI 使用 `uv`；X-Trainer 控制侧可使用 `conda` 或 `venv`。 |

---

## 4. 环境部署

### 4.1 `dobot_xtrainer` 环境

用途：硬件连通性检查、遥操作、数据采集、原始数据检查。

```bash
cd /path/to/workspace/dobot_xtrainer
conda create -n xtrainer python=3.10 -y
conda activate xtrainer
pip install -r requirements.txt
```

如 `pyrealsense2` 安装失败，应优先使用与当前系统、Python 版本、RealSense SDK 匹配的安装方式。机器人控制机上还需要确保串口权限可用。当前项目的 `scripts/function_util.py` 会读取 `scripts/dobot_config/dobot_settings.ini` 中的 `COMPUTER.passcode`，并尝试对串口执行 `chmod 777`。

安装后做最小导入检查：

```bash
python -c "import cv2, numpy, tyro, h5py; print('xtrainer basic deps ok')"
```

### 4.2 OpenPI Pi0.5（JAX）环境

用途：LeRobot dataset 读取、norm stats 计算、Pi0.5 全参/LoRA 微调、policy server。

```bash
cd /path/to/workspace/Xtrainer-PI05-feat-xtrainer-finetune
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

本地 `pyproject.toml` 中固定了关键依赖：

```text
Python >= 3.11
jax[cuda12] == 0.5.3
flax == 0.10.2
orbax-checkpoint == 0.11.13
transformers == 4.53.2
lerobot git rev = 0cf864870cf29f4738d3ade893e6fd13fbd7cdb5
```

说明：

1. `GIT_LFS_SKIP_SMUDGE=1` 用于避免安装依赖时自动拉取较大的 LFS 内容。
2. `lerobot` 版本由 `pyproject.toml` 的 git rev 固定，不建议随意升级。OpenPI 的 LeRobot loader 与 dataset schema 对版本敏感。
3. 如果需要从 Google Cloud Storage 下载 OpenPI base checkpoint，训练机必须能访问 `gs://openpi-assets`，或提前下载到本地并在配置中指定本地路径。

验证 OpenPI 配置是否可读取：

```bash
uv run python -c "from openpi.training import config; [print(n, '->', config.get_config(n).name) for n in ['pi05_xtrainer', 'pi05_xtrainer_custom', 'pi05_xtrainer_finetune']]"
```

如果部署仓库包含 LoRA 配置，还应验证：

```bash
uv run python -c "from openpi.training import config; [print(n, '->', config.get_config(n).name) for n in ['pi05_xtrainer_lora_finetune', 'pi05_xtrainer_lora_r64_finetune']]"
```

如果第二段报错，说明当前分支尚未包含 LoRA 训练配置，需要从 LoRA 适配分支合入，或在 `src/openpi/training/config.py` 中补齐对应 `TrainConfig`。

---

## 5. X-Trainer 硬件配置

### 5.1 配置文件位置

`dobot_xtrainer` 采集侧使用：

```text
dobot_xtrainer/scripts/dobot_config/dobot_settings.ini
```

主要字段：

```ini
[COMPUTER]
passcode = "sudo所需的电脑密码"

[CAMERA]
top = <top_camera_serial>
left = <left_wrist_camera_serial>
right = <right_wrist_camera_serial>

[HAND_LEFT]
joint_ids = 1, 2, 4, 5, 6, 7
append_id = 3
joint_offsets = ...
start_joints = ...
joint_signs = ...
gripper_config = ...
port = /dev/ttyACM0
baud_rate = 2000000
using_sensor = 0

[HAND_RIGHT]
...
port = /dev/ttyACM1

[GRIPPER_LEFT]
id = 21
pos = 2048, 3052
port = /dev/ttyUSB1

[GRIPPER_RIGHT]
id = 22
pos = 2048, 3052
port = /dev/ttyUSB0
```

设备序列号、串口、密码等本地私有信息不应提交到公共仓库。

### 5.2 自动扫描串口

```bash
cd /path/to/workspace/dobot_xtrainer
python scripts/1_find_port.py
```

该脚本会扫描 `/dev/ttyACM*` 和 `/dev/ttyUSB*`，尝试识别左右 leader 与左右夹爪，并写回 `dobot_settings.ini`。

排查要点：

1. 至少应识别 4 个串口设备。
2. 如果设备重插，串口编号可能变化，应重新运行。
3. 如果脚本无权限访问串口，先检查 `COMPUTER.passcode`，或手动执行 `sudo chmod 777 /dev/ttyACM* /dev/ttyUSB*`。

### 5.3 标定 leader offset

```bash
python scripts/2_get_offset.py
```

该脚本会读取当前主手关节位置，根据 `start_joints`、`joint_signs` 计算 `joint_offsets`，并写回配置。标定前应确保 leader 处于期望初始位姿。

标定失败的常见原因：

1. 左右 leader 串口写反。
2. `joint_ids` 或 `append_id` 与硬件不一致。
3. 主手未处于标准初始姿态。
4. Dynamixel 波特率不匹配。

### 5.4 检查相机

```bash
python scripts/5_camera_read.py
```

预期结果：

1. 控制台打印找到的 RealSense device id。
2. 窗口中显示顶部、左腕、右腕三路画面拼接图。
3. 画面方向应与训练/推理约定一致。

如相机打不开，优先检查：

1. `dobot_settings.ini` 中序列号是否正确。
2. 是否有其他进程占用 RealSense。
3. USB 带宽是否不足。
4. `pyrealsense2` 与系统 RealSense SDK 是否匹配。

---

## 6. 遥操作与数据采集

### 6.1 启动 follower robot server

打开终端 1：

```bash
cd /path/to/workspace/dobot_xtrainer
conda activate xtrainer
python experiments/launch_nodes.py --hostname 127.0.0.1 --robot-port 6001
```

该脚本会创建左右 Dobot follower：

```text
left robot ip  = 192.168.5.1
right robot ip = 192.168.5.2
ZMQ server     = 127.0.0.1:6001
```

启动后如长时间无响应，检查：

1. 控制盒是否上电并完成连接。
2. 控制机是否能 ping 通左右臂 IP。
3. Dobot 控制器是否处于 TCP/IP 可控制状态。
4. 上一次安全保护是否导致控制盒红灯，需要复位或重启。

### 6.2 启动遥操作与采集程序

打开终端 2：

```bash
cd /path/to/workspace/dobot_xtrainer
conda activate xtrainer
python experiments/run_control.py \
  --hostname 127.0.0.1 \
  --robot-port 6001 \
  --show-img True
```

当前 `run_control.py` 默认数据根目录为：

```text
/path/to/workspace/datasets/<project_name>/collect_data/
```

其中 `project_name` 在 `experiments/run_control.py` 的 `Args` 中默认为：

```python
project_name = "task1_new"
```

正式采集前应根据任务修改 `project_name`，例如：

```python
project_name = "insert_test_tube_20260514"
```

如果不希望修改源码，建议后续将 `project_name` 改造成 tyro CLI 参数；当前代码中它不是 dataclass type-annotated 字段，命令行不一定能覆盖。

### 6.3 按钮语义

当前 `run_control.py` 中按钮状态机含义如下：

| 操作 | 作用 |
|---|---|
| Button A 短按 | leader lock / unlock 切换。 |
| Button A 长按超过 1 秒 | 对应侧 follower servo start / stop 切换。 |
| Button B 按下 | 在至少一侧 servo 启动后，开始 / 停止 recording。 |

状态数组含义：

```text
what_to_do[side] = [lock_state, servo_state, record_state]
lock_state:   0 lock, 1 unlock
servo_state:  0 stop servo, 1 servo
record_state: 0 stop recording, 1 recording
```

推荐操作顺序：

1. 启动 `launch_nodes.py`，确认 follower server 正常。
2. 启动 `run_control.py`，等待相机线程、agent、robot init 完成。
3. 短按 Button A 解锁 leader，将 leader 调整到合理初始位姿。
4. 长按 Button A 启动 servo，程序会执行 `dynamic_approach`，使 follower 平滑靠近 leader 对应位置。
5. 缓慢遥操作，确认左右臂跟随方向、夹爪方向、运动范围均正确。
6. 按 Button B 开始录制，完成一个 episode 后再次按 Button B 停止录制。
7. 停止 servo 前应先停止录制，结束后将 leader 锁定。

### 6.4 采集输出结构

`run_control.py` 采集的 raw episode 结构为：

```text
datasets/<project_name>/collect_data/
  <episode_timestamp>/
    topImg/
      1.jpg
      2.jpg
      ...
    leftImg/
      1.jpg
      2.jpg
      ...
    rightImg/
      1.jpg
      2.jpg
      ...
    observation/
      1.pkl
      2.pkl
      ...
```

每个 `observation/*.pkl` 中至少包含：

```text
joint_positions    当前机器人状态，期望 14 维
joint_velocities   当前机器人速度，期望 14 维
ee_pos_quat        末端位姿相关信息
gripper_position   夹爪状态
control            当前发送的 action，期望 14 维
```

### 6.5 数据采集质量要求

建议采集规范：

1. 每个任务至少先采集 5 到 10 条短 episode 做 pipeline smoke test。
2. 正式训练前根据任务复杂度采集足够数量的成功 demonstrations。
3. 每条 episode 应从稳定初始场景开始，到任务完成后结束，避免过长空闲片段。
4. 相机视野必须覆盖关键物体、末端执行器和目标区域。
5. 操作动作应平滑，不要快速大幅拖动 leader。
6. 任务语言描述应固定，例如 `Insert the test tube on the desktop into the rack.`，后续转换和推理都使用同一语义。

### 6.6 采集后检查

检查 episode 数量：

```bash
find ../datasets/<project_name>/collect_data -maxdepth 1 -mindepth 1 -type d | wc -l
```

检查每条 episode 的帧数：

```bash
find ../datasets/<project_name>/collect_data/<episode_timestamp>/observation -name "*.pkl" | wc -l
find ../datasets/<project_name>/collect_data/<episode_timestamp>/topImg -name "*.jpg" | wc -l
find ../datasets/<project_name>/collect_data/<episode_timestamp>/leftImg -name "*.jpg" | wc -l
find ../datasets/<project_name>/collect_data/<episode_timestamp>/rightImg -name "*.jpg" | wc -l
```

使用项目内脚本统计最大帧数：

```bash
python scripts/6_dataset_count.py --dataset-name <project_name>
```

如需转为项目原有 HDF5 训练数据，可运行：

```bash
python scripts/4_collect2train_data.py
```

但 Pi0.5 / OpenPI 训练不直接使用该 HDF5 输出，仍需转换为 LeRobot dataset。

---

## 7. Raw 数据转换为 LeRobot 格式

### 7.1 为什么必须转换

OpenPI 的训练数据入口基于 LeRobot dataset loader。`dobot_xtrainer` 的 raw 文件夹结构便于采集和排查，但不能直接被 OpenPI Pi0.5 训练脚本稳定读取。因此必须完成以下转换：

```text
raw X-Trainer episode
    -> LeRobot dataset
    -> OpenPI DataConfig / transforms
    -> Pi0.5 training batch
```

### 7.2 LeRobot 版本要求

当前 `Xtrainer-PI05-feat-xtrainer-finetune/pyproject.toml` 中固定 LeRobot 依赖为：

```toml
lerobot = { git = "https://github.com/huggingface/lerobot", rev = "0cf864870cf29f4738d3ade893e6fd13fbd7cdb5" }
```

因此转换脚本必须与该 LeRobot 版本的数据结构兼容。handoff 中推荐使用 LeRobot v2.1 风格数据集，并通过 `--use_videos` 输出视频形式图像数据。不要用当前最新版 LeRobot API 盲目重写转换脚本，否则可能出现 OpenPI loader 读不到 meta、features 或 video index 的问题。

### 7.3 目标 LeRobot dataset 结构

转换后的数据集建议放在 Hugging Face LeRobot cache 结构下：

```text
$HF_LEROBOT_HOME/<repo_id>/
  data/
    chunk-000/
      episode_000000.parquet
      ...
  videos/
    chunk-000/
      observation.images.top/
        episode_000000.mp4
      observation.images.left_wrist/
        episode_000000.mp4
      observation.images.right_wrist/
        episode_000000.mp4
  meta/
    info.json
    stats.json
    tasks.jsonl
    episodes.jsonl
    episodes_stats.jsonl
```

必要 features：

```python
features = {
    "observation.state": {
        "dtype": "float32",
        "shape": (14,),
        "names": [[
            "left_j1", "left_j2", "left_j3", "left_j4", "left_j5", "left_j6", "left_gripper",
            "right_j1", "right_j2", "right_j3", "right_j4", "right_j5", "right_j6", "right_gripper",
        ]],
    },
    "action": {
        "dtype": "float32",
        "shape": (14,),
        "names": [[
            "left_j1", "left_j2", "left_j3", "left_j4", "left_j5", "left_j6", "left_gripper",
            "right_j1", "right_j2", "right_j3", "right_j4", "right_j5", "right_j6", "right_gripper",
        ]],
    },
    "observation.images.top": {
        "dtype": "video",
        "shape": (3, 480, 640),
        "names": ["channels", "height", "width"],
    },
    "observation.images.left_wrist": {
        "dtype": "video",
        "shape": (3, 480, 640),
        "names": ["channels", "height", "width"],
    },
    "observation.images.right_wrist": {
        "dtype": "video",
        "shape": (3, 480, 640),
        "names": ["channels", "height", "width"],
    },
}
```

OpenPI 的 `XTrainerInputs` 期望 camera key 为 `top`、`left_wrist`、`right_wrist`，不要写成 `cam_top`、`left`、`right` 或其他命名。

### 7.4 推荐转换命令模板

建议使用：

```bash
python examples/xtrainer_real/convert_raw_to_lerobot_2_1.py \
  --raw_root "$RAW_ROOT" \
  --output_root "$OUTPUT_ROOT" \
  --task "$TASK" \
  --fps 30 \
  --use_videos \
  --overwrite_output
```

如果当前没有该脚本，需要按第 7.5 节的数据映射实现，并建议将脚本长期维护在以下位置之一：

```text
Xtrainer-PI05-feat-xtrainer-finetune/examples/xtrainer_real/convert_xtrainer_raw_to_lerobot.py
```

### 7.5 Raw 到 LeRobot 的字段映射

| raw 来源 | LeRobot 字段 | 处理 |
|---|---|---|
| `observation/*.pkl -> joint_positions` | `observation.state` | 转成 `float32`，必须为 14 维。 |
| `observation/*.pkl -> control` | `action` | 转成 `float32`，必须为 14 维。 |
| `topImg/*.jpg` | `observation.images.top` | 读取 RGB 图像，按 LeRobot 要求写入 image/video。 |
| `leftImg/*.jpg` | `observation.images.left_wrist` | 同上。 |
| `rightImg/*.jpg` | `observation.images.right_wrist` | 同上。 |
| 命令行 `--task` | episode task / prompt | 每个 episode 调用 `dataset.save_episode(task=task)` 时写入。 |

转换脚本必须保证同一帧的 state、action、三路图像严格对齐。如果某一帧图像损坏或缺失，应整帧跳过，而不是只删除单路图像。



---

## 8. OpenPI 数据配置

### 8.1 XTrainer DataConfig

当前 OpenPI 适配项目中，关键入口为：

```text
Xtrainer-PI05-feat-xtrainer-finetune/src/openpi/training/config.py
Xtrainer-PI05-feat-xtrainer-finetune/src/openpi/policies/xtrainer_policy.py
```

`LeRobotXTrainerDataConfig` 的核心映射：

```python
{
    "observation.state": "observation.state",
    "observation.images.top": "observation.images.top",
    "observation.images.left_wrist": "observation.images.left_wrist",
    "observation.images.right_wrist": "observation.images.right_wrist",
    "actions": "action",
    "prompt": "prompt",
}
```

`XTrainerInputs` 期望：

```text
observation.state: [14]
observation.images.top
observation.images.left_wrist
observation.images.right_wrist
actions: [action_horizon, 14]
prompt: string
```

`XTrainerOutputs` 返回：

```text
actions: 前 14 维双臂 action
```

### 8.2 常用配置名称

| 配置 | 用途 | 说明 |
|---|---|---|
| `pi05_xtrainer` | base / smoke inference | 使用 Pi0.5 base checkpoint 与 base-compatible stats，适合快速测试接口。 |
| `pi05_xtrainer_custom` | fine-tuned checkpoint inference | 期望 checkpoint 中有 `assets/xtrainer`。 |
| `pi05_xtrainer_finetune` | 全参微调 | 使用 Pi0.5 base params 初始化，默认更新全量参数。 |
| `pi05_xtrainer_lora_finetune` | LoRA 微调 | 需要 LoRA 适配配置，冻结非 LoRA 参数。 |
| `pi05_xtrainer_lora_r64_finetune` | LoRA r64 微调 | LoRA rank 更高，显存和训练参数量更大。 |

当前 `Xtrainer-PI05-feat-xtrainer-finetune` 分支至少包含 `pi05_xtrainer`、`pi05_xtrainer_custom`、`pi05_xtrainer_finetune`。如需运行 LoRA，必须确认 LoRA config 与 `gemma_*_lora` variant 已合入。

### 8.3 必须统一的变量

| 变量 | 示例 | 说明 |
|---|---|---|
| `REPO_ID` | `dobot/insert_test_tube_xtrainer` | LeRobot dataset id。 |
| `TASK` / `prompt` | `Insert the test tube on the desktop into the rack.` | 训练和推理使用的语言任务描述。 |
| `asset_id` | `xtrainer` | OpenPI norm stats assets 子目录。 |
| `checkpoint config` | `pi05_xtrainer_custom` | 推理加载 fine-tuned checkpoint 时使用。 |
| `action_dim` | `14` | 双臂动作维度。 |
| `camera keys` | `top`, `left_wrist`, `right_wrist` | 训练与推理必须一致。 |

---

## 9. 计算 Norm Stats

### 9.1 作用

Pi0.5 训练和推理都会对 proprioceptive state 与 action 做 normalization。norm stats 是 checkpoint 的核心资产之一，必须与训练数据、训练配置、推理 checkpoint 配套。

如果缺失或不匹配，常见问题包括：

1. 训练脚本报找不到 norm stats。
2. loss / grad 出现 NaN 或 Inf。
3. 推理动作幅度异常。
4. 夹爪输出范围不稳定。

### 9.2 推荐命令

```bash
cd /path/to/workspace/Xtrainer-PI05-feat-xtrainer-finetune

HF_LEROBOT_HOME=/path/to/lerobot_cache \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
uv run scripts/training/compute_norm_stats.py \
  --exp-name test1
  --config-name pi05_xtrainer_finetune \
  --data.repo-id <your_hf_username>/<your_xtrainer_dataset> \
  --data.assets.asset-id xtrainer-custom
```

如果项目使用本地 YAML 或其他本地配置覆盖方式，也可使用 handoff 中的形式：

```bash
uv run --no-sync scripts/training/compute_norm_stats.py \
  --local-config-path examples/xtrainer_real/dobot_settings.yaml
```

以当前项目实际 CLI 为准。核心要求是：最终生成的 norm stats 应位于训练配置的 assets 目录下，并能通过 `asset_id='xtrainer'` 被加载。

### 9.3 输出位置

OpenPI `TrainConfig` 默认：

```text
assets_base_dir = ./assets
assets_dirs = ./assets/<config_name>
```

因此常见输出为：

```text
assets/pi05_xtrainer_finetune/xtrainer/
assets/pi05_xtrainer_lora_finetune/xtrainer/
assets/pi05_xtrainer_lora_r64_finetune/xtrainer/
```

### 9.4 检查 norm stats

```bash
find assets -path "*xtrainer*" -type f -maxdepth 5 -print
```

建议训练前做 batch finite check。如果项目包含 `scripts/check_batch_finite.py`，执行：

```bash
uv run scripts/check_batch_finite.py \
  pi05_xtrainer_finetune \
  --data.repo-id <your_hf_username>/<your_xtrainer_dataset> \
  --data.assets.asset-id xtrainer \
  --num-workers=0 \
  --batch-size=4
```

增强检查：

```bash
uv run scripts/check_batch_finite.py \
  pi05_xtrainer_finetune \
  --data.repo-id <your_hf_username>/<your_xtrainer_dataset> \
  --data.assets.asset-id xtrainer \
  --num-workers=0 \
  --batch-size=4 \
  --check-model-loss \
  --check-grad
```

如果当前分支没有该脚本，可先用 `scripts/training/train.py --num-train-steps=10` 做 smoke training，但不如专用检查脚本安全。

---

## 10. Pi0.5 全参微调

### 10.1 适用场景

全参微调会更新 Pi0.5 模型的主要参数，适合以下情况：

1. 有足够 GPU 显存或多卡资源。
2. 数据量较大，任务与 base distribution 差异明显。
3. 需要尽可能提高任务性能，且可以接受更高训练成本。

OpenPI README 中给出的资源估计是：全参微调单卡通常需要 70GB 以上显存。实际部署时，如果使用 RTX 4090，应优先做 LoRA；全参微调建议使用 A100 80GB / H100 或 FSDP 多卡。

### 10.2 关键配置

全参微调配置：

```text
pi05_xtrainer_finetune
```

当前本地配置特征：

```python
model = pi0_config.Pi0Config(pi05=True)
data = LeRobotXTrainerDataConfig(
    repo_id="your_hf_username/my_xtrainer_dataset",
    assets=AssetsConfig(asset_id="xtrainer"),
    base_config=DataConfig(prompt_from_task=True),
)
weight_loader = CheckpointWeightLoader("<pi05_base>/params")
num_train_steps = 20_000
batch_size = 32
```

注意：本地 `config.py` 中可能存在历史个人路径，可修改为用户真实绝对路径：

```text
/media/xxx/4t_12/xxx/openpi/openpi-assets/checkpoints/pi05_base/params
```

正式部署时不要依赖个人绝对路径。应改为以下二选一：

```text
gs://openpi-assets/checkpoints/pi05_base/params
/path/to/local/openpi-assets/checkpoints/pi05_base/params
```

### 10.3 Smoke training

先跑 10 到 100 step，确认数据、stats、checkpoint、显存链路正确：

```bash
cd /path/to/workspace/Xtrainer-PI05-feat-xtrainer-finetune

HF_LEROBOT_HOME=/path/to/lerobot_cache \
HF_HUB_OFFLINE=1 \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
uv run scripts/training/train.py \
  pi05_xtrainer_finetune \
  --data.repo-id <your_hf_username>/<your_xtrainer_dataset> \
  --data.assets.asset-id xtrainer-custom \
  --exp-name xtrainer_full_smoke \
  --batch-size 1 \
  --num-train-steps 50 \
  --num-workers 0 \
  --overwrite
```

说明：

1. 如果数据集已经在本地 cache，且不希望访问 Hugging Face Hub，设置 `HF_HUB_OFFLINE=1`。
2. 全参微调 smoke test 建议从 `batch-size 1` 开始。
3. 确认无 OOM 后，再逐步增加 batch size。

### 10.4 正式训练

```bash
HF_LEROBOT_HOME=/path/to/lerobot_cache \
HF_HUB_OFFLINE=1 \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 \
uv run scripts/training/train.py \
  pi05_xtrainer_finetune \
  --data.repo-id <your_hf_username>/<your_xtrainer_dataset> \
  --data.assets.asset-id xtrainer-custom \
  --exp-name xtrainer_full_<task>_<date> \
  --batch-size 32 \
  --num-train-steps 20000 \
  --save-interval 1000 \
  --keep-period 5000 \
  --overwrite
```

如出现 OOM，按顺序处理：

1. 降低 `--batch-size`。
2. 降低 `--num-workers`。
3. 设置 `XLA_PYTHON_CLIENT_MEM_FRACTION=0.9` 或 `0.95`。
4. 使用多卡并配置 `--fsdp-devices`。
5. 改用 LoRA 微调。

### 10.5 输出 checkpoint

默认 checkpoint 目录：

```text
checkpoints/<config_name>/<exp_name>/<step>/
```

例如：

```text
checkpoints/pi05_xtrainer_finetune/xtrainer_full_insert_tube_20260514/20000/
```

训练完成后应确认 checkpoint 中包含或能找到配套 assets：

```text
checkpoint_dir/
  params/
  assets/
    xtrainer/
      norm_stats...
```

---

## 11. Pi0.5 LoRA 微调

### 11.1 适用场景

LoRA 微调适合单卡资源有限但需要适配真实 X-Trainer 数据的场景。其核心策略是：

1. 使用 Pi0.5 base checkpoint 初始化。
2. 在 Gemma / action expert 相关层插入 LoRA adapter。
3. 冻结非 LoRA 参数，仅训练 adapter 参数。
4. 降低显存占用和训练成本。

OpenPI README 中给出的 LoRA 微调显存估计约为 22.5GB 以上，RTX 4090 级别 GPU 通常可作为起点。

### 11.2 LoRA 配置要求

LoRA 配置通常应包含：

```python
model = pi0_config.Pi0Config(
    pi05=True,
    paligemma_variant="gemma_2b_lora",
    action_expert_variant="gemma_300m_lora",
)
freeze_filter = pi0_config.Pi0Config(
    pi05=True,
    paligemma_variant="gemma_2b_lora",
    action_expert_variant="gemma_300m_lora",
).get_freeze_filter()
ema_decay = None
```

r64 版本通常将 LoRA rank 提升到 64，例如：

```text
gemma_2b_lora_r64
gemma_300m_lora_r64
```

如果当前分支没有这些 variant，需要在 `src/openpi/models/gemma.py` 和 `src/openpi/training/config.py` 中合入 LoRA r64 适配后再运行。

### 11.3 推荐配置名称

```text
pi05_xtrainer_lora_finetune
pi05_xtrainer_lora_r64_finetune
```

两者区别：

| 配置 | 特点 | 建议 |
|---|---|---|
| `pi05_xtrainer_lora_finetune` | rank 较低，显存压力较小 | 先用于 baseline。 |
| `pi05_xtrainer_lora_r64_finetune` | rank 更高，容量更大 | baseline 跑通后再尝试。 |

### 11.4 LoRA smoke training

```bash
cd /path/to/workspace/Xtrainer-PI05-feat-xtrainer-finetune

HF_LEROBOT_HOME=/path/to/lerobot_cache \
HF_HUB_OFFLINE=1 \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
uv run scripts/training/train.py \
  pi05_xtrainer_lora_finetune \
  --data.repo-id <your_hf_username>/<your_xtrainer_dataset> \
  --data.assets.asset-id xtrainer-custom \
  --exp-name xtrainer_lora_smoke \
  --batch-size 4 \
  --num-train-steps 100 \
  --num-workers 0 \
  --overwrite
```

### 11.5 LoRA 正式训练

```bash
HF_LEROBOT_HOME=/path/to/lerobot_cache \
HF_HUB_OFFLINE=1 \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
uv run scripts/training/train.py \
  pi05_xtrainer_lora_finetune \
  --data.repo-id <your_hf_username>/<your_xtrainer_dataset> \
  --data.assets.asset-id xtrainer-custom \
  --exp-name xtrainer_lora_<task>_<date> \
  --batch-size 8 \
  --num-train-steps 20000 \
  --save-interval 1000 \
  --keep-period 5000 \
  --overwrite
```

r64 训练示例：

```bash
HF_LEROBOT_HOME=/path/to/lerobot_cache \
HF_HUB_OFFLINE=1 \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
uv run scripts/training/train.py \
  pi05_xtrainer_lora_r64_finetune \
  --data.repo-id <your_hf_username>/<your_xtrainer_dataset> \
  --data.assets.asset-id xtrainer-custom \
  --exp-name xtrainer_lora_r64_<task>_<date> \
  --batch-size 4 \
  --num-train-steps 20000 \
  --save-interval 1000 \
  --keep-period 5000 \
  --overwrite
```

### 11.6 LoRA 训练检查

训练前建议确认只有 LoRA 参数可训练：

```bash
uv run python -c "from openpi.training import config; cfg=config.get_config('pi05_xtrainer_lora_finetune'); print(cfg.name); print(cfg.model); print(cfg.freeze_filter); print(cfg.ema_decay)"
```

必须满足：

1. `model` 中使用 LoRA variant。
2. `freeze_filter` 冻结非 LoRA 参数。
3. `ema_decay=None`，避免 LoRA 训练维护不必要的 EMA 状态。
4. `data.assets.asset_id='xtrainer-custom'(你计算的norm_stats)`。
5. `repo_id` 指向转换后的 X-Trainer LeRobot dataset。

---

## 12. 推理验证

### 12.1 推理模式

推荐使用 OpenPI 的远程推理模式：

```text
GPU 服务器运行 policy server
机器人控制机运行 X-Trainer real client
二者通过 websocket 通信
```

优点：

1. 机器人控制环境和大模型推理环境解耦。
2. GPU 服务器可以使用更强显卡。
3. 控制机只需安装轻量 runtime 依赖。

### 12.2 启动 policy server

base smoke test：

```bash
cd /path/to/workspace/Xtrainer-PI05-feat-xtrainer-finetune
uv run scripts/deployment/serve_policy.py --env XTRAINER
```

等价形式：

```bash
uv run scripts/deployment/serve_policy.py policy:checkpoint \
  --policy.config=pi05_xtrainer \
  --policy.dir=gs://openpi-assets/checkpoints/pi05_base
```

使用训练好的 X-Trainer checkpoint：

```bash
uv run scripts/deployment/serve_policy.py policy:checkpoint \
  --policy.config=pi05_xtrainer_custom \
  --policy.dir=/path/to/checkpoints/<config_name>/<exp_name>/<step>
```

关键要求：

1. `--policy.config` 与 checkpoint 的训练配置兼容。
2. checkpoint 中包含 `assets/xtrainer`，或 `policy.config` 能加载到相同 norm stats。
3. server 默认端口为 `8000`，机器人控制机应能访问该端口。

### 12.3 安装 robot client 依赖

在机器人控制机：

```bash
cd /path/to/workspace/Xtrainer-PI05-feat-xtrainer-finetune
uv pip install -r examples/xtrainer_real/requirements.txt
cd packages/openpi-client
pip install -e .
```

`examples/xtrainer_real/requirements.txt` 当前包含：

```text
openpi-client
numpy
tyro
pyrealsense2
pyserial
```

### 12.4 启动 X-Trainer real client

```bash
cd /path/to/workspace/Xtrainer-PI05-feat-xtrainer-finetune

python -m examples.xtrainer_real.main \
  --host <policy_server_ip> \
  --port 8000 \
  --prompt "Insert the test tube on the desktop into the rack." \
  --camera-top-serial <top_camera_serial> \
  --camera-left-wrist-serial <left_wrist_camera_serial> \
  --camera-right-wrist-serial <right_wrist_camera_serial> \
  --left-robot-ip 192.168.5.1 \
  --right-robot-ip 192.168.5.2 \
  --left-gripper-port /dev/ttyUSB1 \
  --right-gripper-port /dev/ttyUSB0 \
  --control-hz 20 \
  --action-horizon 25
```

调试动作与状态差异：

```bash
python -m examples.xtrainer_real.main \
  --host <policy_server_ip> \
  --port 8000 \
  --prompt "Insert the test tube on the desktop into the rack." \
  --camera-top-serial <top_camera_serial> \
  --camera-left-wrist-serial <left_wrist_camera_serial> \
  --camera-right-wrist-serial <right_wrist_camera_serial> \
  --debug-action-state-diagnostics True
```

### 12.5 推理端安全逻辑

`examples/xtrainer_real/env.py` 中包含以下保护：

| 参数 | 默认值 | 作用 |
|---|---:|---|
| `max_joint_delta` | `0.17` | 如果目标动作与上一动作差异过大，执行平滑过渡。 |
| `ramp_step` | `0.01` | 平滑过渡时每步最大变化参考。 |
| `ramp_max_steps` | `100` | 平滑过渡最多步数。 |
| `gripper_update_threshold` | `0.02` | 夹爪变化小于阈值时不重复发送。 |
| `servo_step_limit` | `0.9` | follower 单步动作限制。 |

真实机器人第一次验证时建议：

1. 降低 `--control-hz`，例如先用 5 到 10Hz。
2. 缩短 `--max-episode-steps`。
3. 保持手在急停附近。
4. 先空场景、低风险动作测试，再加入任务物体。
5. 观察 action 是否有明显跳变，再执行完整任务。

### 12.6 推理成功标准

最低成功标准：

1. policy server 正常启动并输出 metadata。
2. client 能收到 metadata 和 action chunk。
3. observation 包含 14 维 state 和三路 224x224 uint8 图像。
4. action 为 14 维，无 NaN/Inf。
5. follower 动作平滑，无大幅跳变。
6. 夹爪开合方向正确。
7. prompt 与训练任务一致。

任务成功标准：

1. 在固定初始分布下，机器人能完成目标操作。
2. 连续多次运行成功率达到项目要求。
3. 失败可归类为场景分布、数据质量、动作延迟、相机遮挡、模型能力或硬件异常。

---

## 13. 常见问题排查

### 13.1 串口识别失败

现象：`1_find_port.py` 检测不到 4 个串口，或 leader / gripper 初始化失败。

```bash
ls -l /dev/ttyACM* /dev/ttyUSB* 2>/dev/null || true
sudo chmod 777 /dev/ttyACM* /dev/ttyUSB*
python scripts/1_find_port.py
```

检查：

1. USB 线是否松动。
2. 设备是否被其他进程占用。
3. 左右 leader 是否接反。
4. `dobot_settings.ini` 中 baud rate 是否正确。

### 13.2 follower 连接失败

现象：`launch_nodes.py` 或 `run_control.py` 卡在机器人连接阶段。

```bash
ping 192.168.5.1
ping 192.168.5.2
```

检查：

1. 控制盒是否上电并连接完成。
2. 控制机网段是否正确。
3. Dobot 是否处于 TCP/IP 控制模式。
4. 上次是否触发安全限制导致红灯。
5. 固件版本是否满足代码检查要求，当前代码要求 V3 且版本号大于等于 `3.5.8.1`、小于 `4.0.0.0`。

### 13.3 动作方向不对

处理：

1. 重新运行 `scripts/2_get_offset.py`。
2. 检查 `joint_signs`。
3. 检查左右 leader 是否接反。
4. 检查 `start_joints` 是否与当前机械位姿一致。
5. 只启动单侧 servo 做小幅测试。

### 13.4 采集帧数不齐

处理：

1. 删除明显损坏或极短 episode。
2. 检查相机线程是否报错。
3. 降低采集频率或关闭 `--show-img`。
4. 转换脚本中必须按共同 frame id 对齐，不允许单路图像错位。

### 13.5 LeRobot dataset 加载失败

常见原因：

1. `repo_id` 与本地目录不一致。
2. `HF_LEROBOT_HOME` 未指向正确 cache。
3. LeRobot 版本与转换格式不兼容。
4. `meta/info.json` 或 parquet 缺失。
5. camera key 与 OpenPI `RepackTransform` 不一致。

排查：

```bash
find ${HF_LEROBOT_HOME:-$HOME/.cache/huggingface/lerobot} -maxdepth 3 -type d | grep <dataset_name>
```

### 13.6 Norm stats 缺失或异常

处理：

1. 重新计算 norm stats。
2. 确认 `asset_id='xtrainer'`。
3. 确认训练和推理使用同一套 assets。
4. 检查 raw 数据中 state/action 是否有 NaN/Inf 或极端跳变。
5. 如有 `check_batch_finite.py`，先用 `--skip-norm-stats` 定位是否是 stats 问题。

### 13.7 训练 OOM

处理顺序：

1. 降低 batch size。
2. 降低 num workers。
3. 使用 LoRA 替代全参微调。
4. 设置 `XLA_PYTHON_CLIENT_MEM_FRACTION=0.9` 或 `0.95`。
5. 关闭不必要的后台程序。
6. 使用多卡/FSDP。

### 13.8 推理动作异常

检查：

1. `prompt` 是否与训练任务一致。
2. checkpoint 是否加载正确。
3. checkpoint assets 是否为训练时的 `xtrainer` stats。
4. 三路相机是否对应训练时视角。
5. action 是否 14 维。
6. gripper 是否在 `[0, 1]`。
7. `max_joint_delta` 是否过大或过小。
8. control hz 是否高于实际网络与推理能力。

---

---

## 14. 附录：关键文件索引

### 14.1 `dobot_xtrainer`

| 文件 | 作用 |
|---|---|
| `README.md` | X-Trainer 项目基础说明。 |
| `requirements.txt` | 控制和采集依赖。 |
| `scripts/dobot_config/dobot_settings.ini` | 本地硬件配置。 |
| `scripts/1_find_port.py` | 自动识别 leader 和 gripper 串口。 |
| `scripts/2_get_offset.py` | 计算并写入 leader joint offsets。 |
| `scripts/5_camera_read.py` | RealSense 三路相机检查。 |
| `experiments/launch_nodes.py` | 启动 Dobot follower ZMQ server。 |
| `experiments/run_control.py` | 遥操作、按钮状态机、数据采集主入口。 |
| `scripts/6_dataset_count.py` | 统计采集 episode 帧数。 |
| `scripts/script_collect2train.py` | 将 raw 数据转为项目原 HDF5 格式，非 OpenPI 必需。 |

### 14.2 `Xtrainer-PI05-feat-xtrainer-finetune`

| 文件 | 作用 |
|---|---|
| `README.md` | OpenPI 官方项目说明和 Pi0.5 训练/推理概述。 |
| `pyproject.toml` | Python、JAX、LeRobot、OpenPI 依赖版本。 |
| `src/openpi/training/config.py` | Pi0.5 / XTrainer 数据和训练配置。 |
| `src/openpi/policies/xtrainer_policy.py` | XTrainer observation/action transform。 |
| `scripts/training/compute_norm_stats.py` | 计算 norm stats。 |
| `scripts/training/train.py` | JAX 训练入口。 |
| `scripts/deployment/serve_policy.py` | policy server 启动入口。 |
| `docs/remote_inference.md` | OpenPI 远程推理说明。 |
| `docs/norm_stats.md` | norm stats 说明。 |
| `examples/xtrainer_real/README.md` | XTrainer real robot 推理说明。 |
| `examples/xtrainer_real/main.py` | XTrainer 推理 client 主入口。 |
| `examples/xtrainer_real/env.py` | XTrainer real environment 与安全执行逻辑。 |
| `examples/xtrainer_real/hardware/dobot_xtrainer.py` | 推理侧 Dobot follower / gripper / camera 封装。 |

---

## 15. 最小验收清单

部署完成前，逐项确认：

1. `dobot_xtrainer` 环境可以导入依赖。
2. `1_find_port.py` 能识别左右 leader 和左右 gripper。
3. `2_get_offset.py` 已在标准初始姿态下运行并写回配置。
4. `5_camera_read.py` 能显示三路相机。
5. `launch_nodes.py` 能连接左右 follower。
6. `run_control.py` 能完成遥操作。
7. 至少采集 2 条 raw episode，并确认四类文件数量对齐。
8. raw 数据成功转换为 LeRobot dataset。
9. LeRobotDataset 能加载，state/action 都是 14 维。
10. OpenPI 能读取 XTrainer config。
11. norm stats 已计算并可加载。
12. 全参或 LoRA smoke training 能保存 checkpoint。
13. policy server 能加载 checkpoint 并启动。
14. XTrainer real client 能连接 policy server。
15. 推理 action 维度正确、无 NaN/Inf、动作平滑。
16. 真实任务验证结果已记录。
