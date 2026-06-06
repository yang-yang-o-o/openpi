# 03 · 推理快速上手

> 前置：已完成 [02 · 依赖安装](./02-installation.md)，`uv run` 可用。
>
> 目标：用 simple_client 跑通端到端推理，确认整个链路无误。

---

## 3.1 两终端方案

### 终端 1 — 启动策略服务

```bash
cd /home/featurize/work/openpi
uv run scripts/serve_policy.py --env DROID
```

首次运行会自动从 `gs://openpi-assets` 下载 checkpoint 到 `~/.cache/openpi`（DROID checkpoint 约 **11.6 GB**）。建议指定缓存到本地盘：

```bash
export OPENPI_DATA_HOME=/home/featurize/openpi_cache
```

### 终端 2 — 发送模拟观测

```bash
cd /home/featurize/work/openpi
uv run examples/simple_client/main.py --env DROID --host 127.0.0.1
```

---

## 3.2 ⚠️ 必须显式加 `--host 127.0.0.1`

客户端默认 `host=0.0.0.0`，在 Featurize / 容器化网络环境下**连不上**本机 server，会报：

```
websockets.exceptions.InvalidMessage: did not receive a valid HTTP response
EOFError: stream ends after 0 bytes, before end of line
```

这**不是** server 故障（server 监听在 `0.0.0.0:8000` 是正确的），而是 `0.0.0.0` 不是合法的**客户端目标地址**。改成 `127.0.0.1` 或 `localhost` 即可。

---

## 3.3 Server 启动成功的标志

正常日志关键行：

```
INFO:absl:Finished restoring checkpoint in 5.35 seconds from .../pi05_droid/params.
INFO:root:Loaded norm stats from .../pi05_droid/assets/droid
INFO:root:Creating server (host: featurize, ip: 127.0.0.1)
INFO:websockets.server:server listening on 0.0.0.0:8000
```

以下日志**属于正常**：

| 日志 | 含义 |
|------|------|
| `Unable to initialize backend 'rocm'/'tpu'` | JAX 尝试 ROCm/TPU 失败回退 CUDA，不影响 |
| `Norm stats not found in .../assets/<cfg>/<repo>, skipping` | 会回退到 checkpoint 自带的 norm stats |
| `gsutil not found, falling back to gcsfs` | 用 Python 客户端下 GCS，速度可接受 |

---

## 3.4 客户端成功输出

```
INFO:root:Waiting for server at ws://127.0.0.1:8000...
INFO:root:Server metadata: {...}
Running policy: 100%|##########| 20/20 [00:XX<00:00, X.XX it/s]
+----------------- Timing Statistics -----------------+
| Metric              | Mean | Std | P25 | ... |
| client_infer_ms     | ...  |     |     |     |
| server_infer_ms     | ...  |     |     |     |
+-----------------------------------------------------+
```

---

## 3.5 Python 代码直接调用（不走 server）

```python
from openpi.training import config as _config
from openpi.policies import policy_config
from openpi.shared import download

config = _config.get_config("pi05_droid")
checkpoint_dir = download.maybe_download("gs://openpi-assets/checkpoints/pi05_droid")
policy = policy_config.create_trained_policy(config, checkpoint_dir)

example = {
    "observation/exterior_image_1_left": ...,
    "observation/wrist_image_left": ...,
    "observation/joint_position": ...,
    "observation/gripper_position": ...,
    "prompt": "pick up the fork",
}
action_chunk = policy.infer(example)["actions"]
```

也可在 `examples/inference.ipynb` 中交互式测试。

---

## 3.6 想换其他 env / 模型？

`--env` 支持 `ALOHA / ALOHA_SIM / DROID / LIBERO`，各 env 的观测格式不同。详见 [04 · 模型与环境选择](./04-models-and-envs.md)。

---

## 下一步

- 想了解每个 env 的观测格式和模型差异 → [04 · 模型与环境选择](./04-models-and-envs.md)
- 想用自己的数据微调 → [06 · 微调流程](./06-finetuning.md)
- 推理报错没解决 → [99 · 常见问题](./99-troubleshooting.md)
