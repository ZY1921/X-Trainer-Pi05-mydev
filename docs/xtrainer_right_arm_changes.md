# X-Trainer 修改日志

## 第一次日志记录：右臂单臂执行与图像预处理

对应提交：`e9c9ed1`（完善 X-Trainer 右臂单臂执行与图像预处理）

### 修改范围

本次记录并提交的是 JPEG 图片压缩传输功能实现之前的 X-Trainer 客户端修改。主要目标是让模型继续接收双臂状态和三路相机图像，但在机器人执行阶段屏蔽模型的左臂输出，仅由模型控制右臂。

本次提交不包含 JPEG 图片压缩、图片传输协议升级、服务端 JPEG 解码及其测试代码。上述内容保留在工作区，供后续独立验证和提交。

### 具体修改

#### 1. 新增右臂专用同步客户端

文件：`examples/xtrainer_real/right_arm_main.py`

- 新增 `RightArmOnlyEnvironment`，继承现有 `XTrainerRealEnvironment`。
- 每个 episode reset 时，仍由基础环境将双臂移动到服务端提供的 reset pose。
- reset 完成后保存左臂六个关节和左夹爪的七维状态。
- 执行每个模型动作前，将动作向量 `[0:7]` 覆盖为保存的左臂状态。
- 动作向量 `[7:14]` 保持模型输出，因此右臂六个关节和右夹爪继续由模型控制。
- 对模型原始 14 维动作进行逐步日志记录，写入 `output/inference_actions_<timestamp>.log`。
- 增加三路相机预览窗口，显示 top、left wrist 和 right wrist 图像。
- 操作者按 `Q`、`Esc` 或关闭预览窗口时，可以安全停止客户端并关闭环境。

#### 2. 异步 RTC 客户端屏蔽左臂模型输出

文件：`examples/xtrainer_real/async_rtc_main.py`

- 在异步客户端中加入同语义的 `RightArmOnlyEnvironment`。
- reset 后固定左臂六个关节和左夹爪状态。
- 首次阻塞推理、后续异步动作块和 RTC 动作块切换产生的动作，全部在环境执行层统一覆盖左侧七维输出。
- 左臂屏蔽不改变模型输入：服务端仍能接收双臂状态和三路相机图像。
- 左臂屏蔽不改变 reset 行为：每个 episode 开始时双臂仍会回到服务端 reset pose。
- 保留原有异步请求、实际延迟步数统计、RTC 引导和超时保护逻辑。

#### 3. 统一推理图像与旧数据集预处理

文件：`examples/xtrainer_real/image_preprocessing.py`

- 新增共享的 `match_legacy_dataset_images()` 预处理函数。
- 对 top 和 right wrist 图像进行上下、左右翻转，以匹配旧数据采集时的相机方向。
- 对 top 图像裁取行 `150:420`、列 `220:480` 的区域，并拉伸回 `640×480`。
- 保持 RGB 通道顺序不变。
- 对 top 图像输入尺寸进行严格校验；必须以 `--render-height 480 --render-width 640` 启动。
- 同步右臂客户端和异步 RTC 客户端复用同一预处理函数，避免两条运行路径出现图像差异。

#### 4. 调整 RealSense 启动预热超时

文件：`examples/xtrainer_real/hardware/realsense_camera.py`

- 将相机启动预热阶段单帧等待超时从 1000 ms 调整为 5000 ms。
- 降低多相机或相机启动较慢时在预热阶段误报超时的概率。
- 不改变相机分辨率、帧率、颜色格式或正常运行阶段的读取逻辑。

### 动作索引约定

14 维动作向量的含义如下：

- `[0:6]`：左臂关节 1～6。
- `[6]`：左夹爪。
- `[7:13]`：右臂关节 1～6。
- `[13]`：右夹爪。

右臂专用环境只覆盖 `[0:7]`，不会修改 `[7:14]`。

### 运行注意事项

- 图像预处理要求 `--render-height 480 --render-width 640`。
- 左臂不是断电或解除使能，而是持续接收 reset 后的固定目标状态。
- 模型仍然输出完整 14 维动作；屏蔽发生在机器人环境的 `apply_action()` 执行入口。
- 同步右臂客户端会保存完整的模型原始动作日志，日志记录发生在左臂覆盖之前。

### 验证

- 对相关 Python 文件执行语法编译检查。
- 对相关文件执行 Ruff 静态检查。
- 对提交内容执行 `git diff --check`，检查空白符和补丁格式。

---

## 第二次日志记录：异步 RTC 图像 JPEG 压缩传输

对应提交：本次提交（实现异步 RTC 图像 JPEG 压缩传输）

### 修改目标

在不改变三路相机图像空间分辨率和视野范围的前提下，减少异步 RTC 客户端每次推理请求通过 WebSocket 上传的数据量，降低大尺寸 RGB 图像造成的网络传输延迟和延迟抖动。

压缩仅用于客户端与服务端之间的网络传输。客户端仍从环境取得 `640×480×3` 的 RGB 图像；服务端在模型推理前将 JPEG 数据解码回相同尺寸的 `uint8 RGB` 数组。

### 1. 新增共享图像传输编解码模块

文件：`packages/openpi-client/src/openpi_client/image_transport.py`

- 新增 `raw` 和 `jpeg` 两种图片传输模式。
- `jpeg` 模式使用 Pillow 将 RGB 图像编码为 JPEG 字节，并将编码数据、原始形状和编码类型封装到消息中。
- 服务端收到消息后，将 JPEG 字节解码为 RGB `uint8` 数组。
- 解码时校验图片声明尺寸和实际尺寸是否一致；不一致时立即报错，避免错误图像进入模型。
- 编解码只处理名称以 `observation.images.` 开头的字段，不修改机器人状态、prompt、RTC 动作前缀等其他数据。
- 统计图片数量、原始图像字节数、传输图像字节数以及编解码耗时。
- `raw` 模式保留原始 NumPy 数组传输，可用于效果对比和问题回退。

### 2. 异步 RTC 客户端加入 JPEG 传输

文件：`examples/xtrainer_real/async_rtc_main.py`

- 默认图片传输模式设置为 JPEG，默认质量为 90：

  ```text
  --image-transport-codec jpeg
  --image-jpeg-quality 90
  ```

- JPEG 质量参数允许范围为 1～100，超出范围时在启动阶段报错。
- 图片编码在异步推理工作线程中完成，不在机器人主控制线程中执行网络发送。
- 编码完成后再进行 msgpack 打包，并记录完整请求大小。
- 客户端连接服务端时检查服务端公布的图片传输能力；服务端不支持所选编码时拒绝启动。
- 保留以下回退参数，可恢复原来的无图片压缩传输：

  ```text
  --image-transport-codec raw
  ```

- 当启用 `--debug-async-timing` 时，增加如下传输诊断信息：

  ```text
  ASYNC_RTC transport phase=online request=... codec=jpeg
  request_kib=... images_raw_kib=... images_wire_kib=...
  ratio=... encode_ms=... decode_ms=...
  ```

- baseline warmup、RTC warmup 和在线推理请求均可记录传输指标。

### 3. 异步 RTC 服务端加入 JPEG 解码

文件：`scripts/deployment/serve_policy_async_rtc.py`

- 服务端声明支持 `raw` 和 `jpeg` 两种图片传输编码。
- 在执行 `policy.infer()` 前解码图片，使模型仍接收标准 NumPy RGB 图像。
- 拒绝未知图片编码以及格式不完整的图片消息。
- 在响应的 `server_timing` 中增加 `image_decode_ms`。
- 在推理响应中返回服务端图片传输统计，便于分析解码开销与实际压缩效果。

### 4. 异步 RTC 协议升级

客户端文件：`examples/xtrainer_real/async_rtc_main.py`

服务端文件：`scripts/deployment/serve_policy_async_rtc.py`

- 协议版本由 v1 升级为 v2。
- v2 请求新增 `image_transport` 描述，并允许图像字段使用编码后的线缆格式。
- 客户端和服务端必须同时更新并重启。
- v2 客户端连接旧版 v1 服务端时会在启动阶段明确报告协议不兼容，不会把 JPEG 字节误传给旧服务端执行推理。

### 5. 新增和扩展测试

文件：

- `packages/openpi-client/src/openpi_client/image_transport_test.py`
- `scripts/deployment/async_rtc_client_test.py`
- `scripts/deployment/serve_policy_async_rtc_test.py`

覆盖内容：

- JPEG 编码、msgpack 打包/解包和 JPEG 解码的完整往返。
- 解码后的图片尺寸、数据类型和 RGB 格式检查。
- JPEG 传输数据量小于原始 RGB 数据量。
- `raw` 模式保持原始数组传输。
- 非法 JPEG 质量参数被拒绝。
- 客户端请求包含正确的图片传输描述和 JPEG 字节。
- 服务端在调用策略前恢复图片数组。
- 原有 RTC 参数仍正确映射到策略推理接口。

### 图像信息变化说明

保持不变：

- 图像空间尺寸仍为 `640×480`。
- 相机视野、原有旋转和裁剪范围不变。
- 服务端交给模型的数据仍为三通道 RGB。
- 数据类型仍为 `uint8`，数值范围仍为 0～255。

存在变化：

- JPEG 质量 90 属于轻微有损压缩，解码后的单个像素值可能与压缩前略有不同。
- 细小纹理、锐利边缘和颜色边界可能出现轻微 JPEG 压缩痕迹。
- 如果要求逐像素完全一致，应使用 `--image-transport-codec raw`。

### 部署与运行注意事项

- 客户端与推理服务器必须部署同一个包含协议 v2 的版本，启动前应核对两端 Git 提交号完全一致。
- 修改服务端代码后必须重启异步 RTC 服务端。
- 首次实机运行建议保留 `--debug-async-timing`，同时观察 `delay_steps`、`rtt_ms`、`encode_ms`、`decode_ms` 和压缩比。
- 启用 JPEG 后需要重新测量实际推理延迟步数，再调整 `rtc_inference_delay`；不能直接沿用原始 RGB 传输下测得的延迟参数。
- `.bak` 备份文件不属于功能代码，也未包含在该次提交中。

### 验证结果

- Ruff 静态检查通过。
- 相关 Python 文件语法编译检查通过。
- `git diff --check` 通过。
- JPEG/msgpack 编解码往返测试共 3 项通过。
- 本地三路合成 `640×480` 图像测试中，解码结果保持 `(480, 640, 3) uint8`；实际压缩比和耗时取决于真实相机画面内容及运行电脑性能。
