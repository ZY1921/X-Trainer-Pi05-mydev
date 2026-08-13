# Scripts 目录

脚本按用途分类，命令应从仓库根目录运行。

| 目录 | 用途 | 主要入口 |
| --- | --- | --- |
| `training/` | 数据统计、JAX/PyTorch 训练与训练测试 | `train.py`、`train_pytorch.py`、`compute_norm_stats.py` |
| `deployment/` | Policy 服务、异步 RTC 协议测试与 Docker 部署 | `serve_policy.py`、`serve_policy_async_rtc.py` |
| `evaluation/` | 离线评估和结果分析 | `open_loop_eval.py` |

代码同步工具和机器凭据属于本机配置，已通过 `.gitignore` 排除，不会提交到仓库。
其他脚本的详细使用方式见各脚本的 `--help` 和项目文档。
