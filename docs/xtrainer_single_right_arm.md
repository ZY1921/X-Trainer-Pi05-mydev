# X-Trainer 单右臂数据、训练与推理流程

- 修改日期：2026-09-01
- 适用模型：Pi0.5
- 物理设备：Dobot X-Trainer 右臂、top 相机、right wrist 相机

## 1. 修改目标与最终结果

原始采集数据仍是双臂格式，但实际任务只有右臂运动。本次修改新增了一套独立的单右臂流程：

- 从原始 14 维双臂数据中提取右臂 `7:14`，生成 7 维状态和动作。
- 保留 top 和 right wrist 两路图像，完全移除左臂相机数据。
- norm stats 按真实的 7 维数据计算，不让静止左臂影响归一化结果。
- 训练时仍把右臂放入预训练模型原有的 `[7:14]` 槽位，保持 Pi0.5 双臂预训练语义。
- 部署时只连接和控制右臂，只初始化 top、right wrist 两个相机。
- 新增同步推理和异步 RTC 推理入口，原有双臂入口保持不变。
- 支持把多个单任务 LeRobot v2.1 目录安全合并为一个语言条件多任务数据集。

新数据集接口如下：

| 字段 | 维度/内容 |
|---|---|
| `observation.state` | 7：右臂 6 个关节 + 右夹爪 |
| `action` | 7：右臂 6 个关节 + 右夹爪 |
| `observation.images.top` | top 相机 RGB 图像 |
| `observation.images.right_wrist` | 右腕相机 RGB 图像 |

## 2. 代码修改内容

### 2.1 新增单右臂转换脚本

新增 `examples/xtrainer_real/convert_raw_right_arm_to_lerobot_2_1.py`：

- 从每帧 `joint_positions[7:14]` 提取右臂状态。
- 从每帧 `control[7:14]` 提取右臂动作。
- 只读取 `topImg` 和 `rightImg`，不要求原始 episode 中存在 `leftImg`。
- 只对 observation、top 和 right 三者共有的帧编号进行转换。
- 支持原转换脚本已有的图片嵌入、视频编码、坏帧跳过、编码重试和统计信息生成。
- 输出 LeRobot v2.1 数据集，状态和动作的 feature shape 均为 `(7,)`。

原文件 `convert_raw_to_lerobot_2_1.py` 没有修改，双臂数据仍可使用原入口转换。

### 2.2 新增单右臂 Policy Transform

新增 `src/openpi/policies/xtrainer_right_arm_policy.py`，负责数据集、模型和物理设备之间的映射：

1. 输入必须是 7 维状态/动作，并且必须包含 top、right wrist 图像。
2. top 图像映射到模型 `base_0_rgb`。
3. right wrist 图像映射到模型 `right_wrist_0_rgb`。
4. 模型仍保留 `left_wrist_0_rgb` 输入槽，但填入全零图像并将 mask 设置为 `False`，不会作为有效视觉输入。
5. 7 维状态和动作完成归一化后，被放入模型空间 `[7:14]`；其余模型维度由 padding 补零。
6. 推理输出只取模型动作 `[7:14]`，再执行反归一化和 delta-to-absolute，最终交给机器人的是 7 维动作。

该映射的重点是：数据集和 norm stats 都是真实的 7 维，但模型仍沿用预训练时的右臂位置。

### 2.3 新增训练与 norm stats 配置

在 `src/openpi/training/config.py` 中增加：

- `XTrainerRightArmModelTransformFactory`
- `LeRobotXTrainerRightArmDataConfig`
- 单右臂 checkpoint metadata
- 四个单右臂配置

| 配置名 | 用途 |
|---|---|
| `pi05_xtrainer_right_arm_finetune` | Pi0.5 全参数微调 |
| `pi05_xtrainer_right_arm_lora_finetune` | Pi0.5 LoRA 微调 |
| `pi05_xtrainer_right_arm_lora_r64_finetune` | Pi0.5 LoRA R64 微调 |
| `pi05_xtrainer_right_arm_custom` | 加载全参数单右臂 checkpoint 推理 |

所有配置使用独立 asset id `xtrainer_right_arm`。加载 norm stats 时会检查 `state` 和 `actions` 的最后一维必须为 7，避免误用原双臂 14 维统计文件。

动作转换中，前 6 维关节使用 delta action，最后 1 维夹爪保持绝对值。当前流程只支持 Pi0.5。

### 2.4 新增右臂实机环境

新增 `examples/xtrainer_real/right_arm_env.py`：

- 只创建一个右臂 `DobotXTrainer` follower。
- 只连接右臂机器人和右夹爪串口。
- 只创建 `cam_top` 和 `cam_right_wrist`。
- 状态、动作和 reset pose 都严格检查为 7 维。
- 保留关节动作限幅、超限平滑插值、夹爪阈值更新和退出断连处理。

### 2.5 新增同步推理客户端

新增 `examples/xtrainer_real/right_arm_single_main.py`：

- 使用普通 WebSocket 同步请求 action chunk。
- 显示 `top | right wrist` 双画面预览，可按 `q`、`Esc` 或关闭窗口退出。
- 将实机图像旋转、裁剪到与历史训练数据一致；因此采集分辨率固定为 `480x640`。
- 启动机器人前校验服务端 metadata，包括右臂标识、7 维动作、模型槽位、相机键和 reset pose。
- 检查每次服务端返回动作必须为 `(horizon, 7)`。
- 默认记录收到的 action chunk、实际执行动作、延迟统计、CSV、NPZ 和动作曲线。

### 2.6 新增异步 RTC 推理客户端

新增 `examples/xtrainer_real/right_arm_single_async_rtc_main.py`：

- 复用已有异步 RTC worker 和 RTC action merge 逻辑。
- observation 只发送 7 维右臂状态和两路有效图像。
- 支持 RTC guidance、异步请求、推理超时、JPEG 图像传输和 warmup。
- 对异步返回结果额外执行 7 维动作检查。
- 使用独立的右臂环境，不会连接左臂。

### 2.7 其他修改和测试

- `examples/xtrainer_real/inference_action_recorder.py` 增加 7 维右臂动作名称，绘图会显示 `right_joint_1` 到 `right_gripper`。
- 增加转换、Policy Transform 和客户端校验测试。
- 已验证无 `leftImg` 的原始 episode 可以完成转换。
- 静态检查、格式检查和相关回归测试均通过，测试结果为 `34 passed`。

## 3. 使用方法

### 3.1 原始数据要求

每个原始 episode 至少包含：

```text
<raw_root>/<episode>/
├── observation/<frame_id>.pkl
├── topImg/<frame_id>.jpg
└── rightImg/<frame_id>.jpg
```

每个 pkl 中必须有：

- `joint_positions`：长度 14。
- `control`：长度 14。

`leftImg` 可以存在，也可以不存在；新转换脚本不会读取它。

### 3.2 转换数据集

默认生成视频格式：

```bash
python examples/xtrainer_real/convert_raw_right_arm_to_lerobot_2_1.py \
  --raw_root <raw_collect_data> \
  --output_root <right_arm_dataset_root> \
  --task "<task description>" \
  --fps 30 \
  --use_videos
```

如果不希望编码视频：

```bash
python examples/xtrainer_real/convert_raw_right_arm_to_lerobot_2_1.py \
  --raw_root <raw_collect_data> \
  --output_root <right_arm_dataset_root> \
  --task "<task description>" \
  --fps 30 \
  --no_videos
```

常用可选参数：

- `--overwrite_output`：覆盖已有的非空输出目录。
- `--skip_first_frames N`：跳过每个 episode 开头 N 帧。
- `--min_frames N`：episode 最少有效帧数，默认 10。
- `--fail_on_bad_frames`：遇到坏帧立即失败；默认是跳过坏帧。
- `--keep_images_for_video`：视频编码完成后保留临时图片。

转换后的目录需要放到当前 LeRobot 数据缓存可识别的位置，或上传为一个 dataset repo。后续命令中的 `<dataset_repo_id>` 必须指向这个新单右臂数据集，不能指向原 14 维双臂数据集。

### 3.3 合并多个右臂任务

每个任务先单独完成转换，再按命令中的顺序合并。例如先放红按钮、再放绿按钮，合并后它们的 `task_index` 分别为 0 和 1：

```bash
python examples/xtrainer_real/merge_right_arm_lerobot_2_1.py \
  --input_roots \
    <red_button_lerobot_root> \
    <green_button_lerobot_root> \
  --output_root <right_arm_multitask_lerobot_root> \
  --media_mode auto
```

合并脚本会：

- 检查每个输入均为完整的 LeRobot v2.1 视频数据集，且只包含一个任务。
- 检查 FPS、7 维状态/动作、top 和 right wrist 相机及 feature schema 完全一致。
- 重写全局 `index`、`episode_index`、`task_index`、episode metadata 和 dataset stats。
- 默认优先硬链接视频；无法硬链接时自动复制。输入数据不会被修改。
- 在输出目录已存在时拒绝运行；确认需要替换时显式增加 `--overwrite_output`。

不同任务的 prompt 必须不同。合并完成后，后续 norm stats 和训练命令中的 `<dataset_repo_id>` 都使用合并输出目录。当前红、绿按钮数据的平均 episode 长度接近，直接使用普通 shuffle 即可，不需要额外的任务均衡采样器。

### 3.4 检查基础 checkpoint 路径

三个训练配置当前沿用了仓库内原 X-Trainer 配置的 Pi0.5 base 权重路径：

```text
/home/dobot/gbw/openpi-assets/checkpoints/pi05_base/params
```

如果本机路径不同，训练前需要在 `src/openpi/training/config.py` 中将对应 `CheckpointWeightLoader` 改成实际的 Pi0.5 base params 路径。

### 3.5 计算 7 维 norm stats

norm stats 必须使用准备训练的同一个配置计算。以 LoRA 为例：

```bash
uv run scripts/training/compute_norm_stats.py \
  pi05_xtrainer_right_arm_lora_finetune \
  --data.repo-id <dataset_repo_id> \
  --data.assets.asset-id xtrainer_right_arm \
  --exp-name norm-stats
```

如果进行全参数训练，将配置名替换成：

```text
pi05_xtrainer_right_arm_finetune
```

生成的 stats 中 `state` 和 `actions` 都应为 7 维。不要复用原 `xtrainer` 的 14 维 norm stats。

### 3.6 启动训练

LoRA 示例：

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
uv run scripts/training/train.py \
  pi05_xtrainer_right_arm_lora_finetune \
  --data.repo-id <dataset_repo_id> \
  --data.assets.asset-id xtrainer_right_arm \
  --exp-name <experiment_name> \
  --overwrite
```

全参数训练使用 `pi05_xtrainer_right_arm_finetune`；R64 LoRA 使用 `pi05_xtrainer_right_arm_lora_r64_finetune`。计算 stats 和训练时的配置名、dataset repo id、asset id 必须保持一致。

### 3.7 同步推理

先在 GPU 机器启动服务。LoRA checkpoint 必须使用对应的 LoRA 配置加载：

```bash
uv run scripts/deployment/serve_policy.py policy:checkpoint \
  --policy.config=pi05_xtrainer_right_arm_lora_finetune \
  --policy.dir=<checkpoint_step_dir>
```

全参数 checkpoint 可以使用：

```bash
uv run scripts/deployment/serve_policy.py policy:checkpoint \
  --policy.config=pi05_xtrainer_right_arm_custom \
  --policy.dir=<checkpoint_step_dir>
```

在机器人控制机器安装运行依赖：

```bash
uv pip install -r examples/xtrainer_real/requirements.txt
uv pip install -e packages/openpi-client
```

第二条命令用于确保机器人端使用当前仓库内带异步 RTC 和图像传输功能的 `openpi-client`，避免误用环境里较旧的同名包。

然后启动新的同步右臂客户端：

```bash
python -m examples.xtrainer_real.right_arm_single_main \
  --host <server_ip> \
  --port 8000 \
  --prompt "<task description>" \
  --right-robot-ip 192.168.5.2 \
  --right-gripper-port /dev/ttyUSB0 \
  --camera-top-serial <top_serial> \
  --camera-right-wrist-serial <right_wrist_serial>
```

### 3.8 异步 RTC 推理

在 GPU 机器启动异步 RTC 服务：

```bash
uv run scripts/deployment/serve_policy_async_rtc.py policy:checkpoint \
  --policy.config=pi05_xtrainer_right_arm_lora_finetune \
  --policy.dir=<checkpoint_step_dir>
```

在机器人控制机器启动新客户端：

```bash
python -m examples.xtrainer_real.right_arm_single_async_rtc_main \
  --host <server_ip> \
  --port 8000 \
  --prompt "<task description>" \
  --right-robot-ip 192.168.5.2 \
  --right-gripper-port /dev/ttyUSB0 \
  --camera-top-serial <top_serial> \
  --camera-right-wrist-serial <right_wrist_serial> \
  --rtc-inference-delay 4 \
  --rtc-execution-horizon 10
```

默认通过 JPEG 传输图像，质量为 90；可通过 `--image-jpeg-quality` 调整。同步和异步客户端默认把推理动作记录到 `output/`，可用 `--inference-action-output-dir <dir>` 修改目录，或用 `--no-record-inference-actions` 关闭。

## 4. 兼容性和实机注意事项

- 原双臂转换脚本、双臂环境、原同步推理入口和原异步 RTC 推理入口均保留，可继续运行旧流程。
- 新流程要求服务端 checkpoint metadata 明确声明右臂、7 维动作、模型槽位 `[7:14]` 和两路相机；误连双臂 checkpoint 时客户端会在连接机器人前报错。
- top 和 right wrist 都是必需相机，缺少任意一个 serial 都不会启动。
- 图像预处理要求客户端输出 `480x640`，不要修改 `--render-height` 和 `--render-width`。
- 当前配置的右臂 reset pose 是 `[1.57, 0.0, 1.57, 0.0, -1.57, -1.01, 1.0]`。首次实机运行前必须确认它与设备坐标系、夹爪方向和现场安全初始位姿一致。
- 建议首次运行降低 `--control-hz`、缩短 episode，并在可随时急停的状态下验证关节和夹爪方向。
