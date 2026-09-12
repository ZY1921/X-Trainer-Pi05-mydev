# X-Trainer 真机推理客户端

本目录是自包含的 X-Trainer 真机运行环境，只保留四个正式推理入口。四个入口共用相同的图像预处理和可选实时预览模块。

## 推理入口

| 控制范围 | 推理方式 | 客户端入口 | 动作维度 | 相机 |
| --- | --- | --- | ---: | --- |
| 单右臂 | 同步 | `examples.xtrainer_real.right_arm_single_main` | 7 | Top、Right wrist |
| 单右臂 | 异步 RTC | `examples.xtrainer_real.right_arm_single_async_rtc_main` | 7 | Top、Right wrist |
| 双臂 | 同步 | `examples.xtrainer_real.bimanual_main` | 14 | Top、Left wrist、Right wrist |
| 双臂 | 异步 RTC | `examples.xtrainer_real.bimanual_async_rtc_main` | 14 | Top、Left wrist、Right wrist |

单右臂动作顺序为右臂 6 个关节加右夹爪。双臂动作顺序为：

```text
[左臂 6 个关节, 左夹爪, 右臂 6 个关节, 右夹爪]
```

## 环境安装与服务端

在机器人控制机安装运行依赖：

```bash
uv pip install -r examples/xtrainer_real/requirements.txt
cd packages/openpi-client
pip install -e .
pip install pyrealsense2 pyserial
```

同步客户端连接 `scripts/deployment/serve_policy.py`。例如：

```bash
uv run scripts/deployment/serve_policy.py --env XTRAINER
```

使用 X-Trainer 专用 checkpoint 和 norm stats 时可运行：

```bash
uv run scripts/deployment/serve_policy.py \
  policy:checkpoint \
  --policy.config=pi05_xtrainer_custom \
  --policy.dir=/path/to/your/checkpoint
```

异步客户端连接 `scripts/deployment/serve_policy_async_rtc.py`，客户端与服务端必须使用相同的异步协议版本。

## 统一图像处理

四个入口均通过 `inference_environment.py` 调用 `image_preprocessing.py` 中的统一处理：

- 客户端采集和基础缩放尺寸固定为 `640×480`。
- Top 图像保留 `top[:430, :]`，再拉伸回 `640×480`。
- Right wrist 图像旋转 180°。
- Left wrist 图像保持原方向。
- 模型输入保持 RGB、`uint8`、`640×480`。

不要把 `--render-height` 和 `--render-width` 改成其他值，否则客户端会在连接机械臂前拒绝启动。

## 可选实时预览

四个入口的实时预览默认关闭。需要显示模型实际接收的预处理后图像时添加：

```bash
--show-camera-preview
```

预览模块只操作图像副本，不会修改发送给模型的 RGB 数组。按 `Q`、`Esc` 或关闭窗口会结束客户端。无图形界面的机器人控制机不要开启此参数。

## 同步客户端示例

单右臂：

```bash
python -m examples.xtrainer_real.right_arm_single_main \
  --host <server_ip> \
  --port 8000 \
  --prompt "press the red button" \
  --camera-top-serial <top_serial> \
  --camera-right-wrist-serial <right_wrist_serial>
```

双臂：

```bash
python -m examples.xtrainer_real.bimanual_main \
  --host <server_ip> \
  --port 8000 \
  --prompt "perform the bimanual task" \
  --camera-top-serial <top_serial> \
  --camera-left-wrist-serial <left_wrist_serial> \
  --camera-right-wrist-serial <right_wrist_serial>
```

## 异步 RTC 客户端示例

异步客户端默认启用 RTC 和 JPEG quality 90 传输。

单右臂：

```bash
python -m examples.xtrainer_real.right_arm_single_async_rtc_main \
  --host <server_ip> \
  --port 18080 \
  --prompt "press the red button" \
  --camera-top-serial <top_serial> \
  --camera-right-wrist-serial <right_wrist_serial>
```

双臂：

```bash
python -m examples.xtrainer_real.bimanual_async_rtc_main \
  --host <server_ip> \
  --port 18080 \
  --prompt "perform the bimanual task" \
  --camera-top-serial <top_serial> \
  --camera-left-wrist-serial <left_wrist_serial> \
  --camera-right-wrist-serial <right_wrist_serial>
```

使用 `--no-rtc-enabled` 可运行异步非 RTC 对照；使用 `--image-transport-codec raw` 可关闭 JPEG 传输。

## 动作记录

单右臂同步以及两种异步客户端默认记录推理动作，输出到 `output/`。可使用以下参数控制：

```bash
--inference-action-output-dir <directory>
--no-record-inference-actions
```

每次运行会保存完整动作块、逐步选择动作、请求和时序信息、摘要及动作图。双臂异步和单右臂的记录图默认从动作维度 0 开始。同步双臂入口沿用基础同步运行逻辑，不额外保存动作文件。

## 安全说明

- `bimanual_main.py` 和 `bimanual_async_rtc_main.py` 会实际控制左右机械臂，不会冻结左臂。
- 启动双臂客户端前必须分别确认左右臂 IP、关节方向、初始姿态、夹爪端口和急停状态。
- 单臂任务应使用 7 维单右臂入口及对应的 7 维 checkpoint 和 norm stats。
- 7 维与 14 维 checkpoint、reset pose 和 norm stats 不得交叉使用。
- 现有环境继续使用动作插值、关节变化限制和夹爪更新阈值，首次实机验证仍应低速运行并准备急停。
