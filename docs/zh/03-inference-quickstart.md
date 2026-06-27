# 03 · 推理快速上手

> 前置：已完成 [02 · 依赖安装](./02-installation.md)，`uv run` 可用。
>
> 目标：用 simple_client 跑通端到端推理，确认整个链路无误。

---

## 3.1 两终端方案

### 终端 1 — 启动策略服务

**方式 A — 指定本地权重（推荐，跳过自动下载）**

权重已在 repo 内时（见 [3.2](#32-可选项把权重缓存放到-repo-内手动管理)），直接传路径，不会触发 `gs://` 下载：

```bash
cd /home/featurize/work/openpi
uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config=pi05_droid \
  --policy.dir=/home/featurize/work/openpi/openpi_cache/openpi-assets/checkpoints/pi05_droid
```

**不需要**加 `--env`：server 只靠 `--policy.config`（训练配置名）和 `--policy.dir`（权重目录）加载模型；`--env` 仅在方式 B 里用来选默认 checkpoint。

**方式 B — 默认自动下载**（须指定 `--env`，用来选默认 checkpoint 并触发下载）

```bash
cd /home/featurize/work/openpi
uv run scripts/serve_policy.py --env DROID
```

首次运行会自动从 `gs://openpi-assets` 下载 checkpoint 到缓存目录（DROID 的 `pi05_droid` 约 **11.6 GB**）。默认缓存位置：`~/.cache/openpi`（实例销毁后可能丢失）。

### 终端 2 — 发送模拟观测

无论终端 1 用方式 A 还是 B，client **都要**加 `--env DROID`（决定发什么格式的模拟观测，与 server 是否写了 `--env` 无关）：

```bash
cd /home/featurize/work/openpi
uv run examples/simple_client/main.py --env DROID --host 127.0.0.1
```

---

## 3.2 可选项：把权重缓存放到 repo 内（手动管理）

> 适合：只跑 OpenPI、会换多个示例/env、希望**换实例也不重下**、work 配额够用（50 GB 级）。
>
> 代价：`~/work` 是云同步盘，**读写比本地盘慢**；大文件不要 git commit。

权重放在 repo 内后，启动 server 时用 `policy:checkpoint` **直接传本地路径**，完全跳过 `gs://...` 自动下载逻辑（不依赖 `OPENPI_DATA_HOME` 是否在当次 shell 里生效）。

`OPENPI_DATA_HOME` 仍可用于**首次下载**或把文件落到统一目录，但日常推理推荐下面这种显式指定方式。

### 目录结构

```
openpi/
├── openpi_cache/                          ← 权重缓存（自行管理，勿 commit）
│   ├── openpi-assets/checkpoints/
│   │   ├── pi05_droid/                    (~11.6 GB)
│   │   ├── pi05_libero/
│   │   └── ...
│   └── big_vision/
│       └── paligemma_tokenizer.model      (~4 MB)
├── src/
└── ...
```

URL 与本地路径的对应关系（`src/openpi/shared/download.py`）：

| 远程 URL | 本地路径 |
|---------|---------|
| `gs://openpi-assets/checkpoints/pi05_droid` | `$OPENPI_DATA_HOME/openpi-assets/checkpoints/pi05_droid` |
| `gs://big_vision/paligemma_tokenizer.model` | `$OPENPI_DATA_HOME/big_vision/paligemma_tokenizer.model` |

### 一次性准备目录

```bash
cd /home/featurize/work/openpi
mkdir -p openpi_cache

# 若已在 ~/.cache/openpi 下过，迁过来（避免重下）
if [ -d ~/.cache/openpi ] && [ "$(ls -A ~/.cache/openpi 2>/dev/null)" ]; then
  mv ~/.cache/openpi/* openpi_cache/
  rmdir ~/.cache/openpi 2>/dev/null
fi
```

权重文件自行放进 `openpi_cache/`，或首次用 `OPENPI_DATA_HOME` 下载到该目录后再改用手动路径启动（见文末「首次下载」）。

### 启动 server（指定本地权重，跳过自动下载）

**终端 1 — DROID 示例**（`--policy.config` 须与 checkpoint 匹配，`--policy.dir` 指向本地目录）：

```bash
cd /home/featurize/work/openpi

uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config=pi05_droid \
  --policy.dir=/home/featurize/work/openpi/openpi_cache/openpi-assets/checkpoints/pi05_droid
```

**终端 2 — client 不变**（`--env` 仍须与观测格式一致）：

```bash
cd /home/featurize/work/openpi
uv run examples/simple_client/main.py --env DROID --host 127.0.0.1
```

换其他 env 时，改 `config` + `dir`，client 的 `--env` 同步修改：

| env（client `--env`） | `--policy.config` | `--policy.dir`（本地路径） |
|----------------------|-------------------|---------------------------|
| `DROID` | `pi05_droid` | `.../openpi_cache/openpi-assets/checkpoints/pi05_droid` |
| `LIBERO` | `pi05_libero` | `.../openpi_cache/openpi-assets/checkpoints/pi05_libero` |
| `ALOHA_SIM` | `pi0_aloha_sim` | `.../openpi_cache/openpi-assets/checkpoints/pi0_aloha_sim` |
| `ALOHA` | `pi05_aloha` | `.../openpi_cache/openpi-assets/checkpoints/pi05_base` |

正常应直接出现 `Restoring checkpoint from .../openpi_cache/...`，**不应出现** `Downloading gs://openpi-assets/...`。

### 首次下载到 repo 内（可选）

还没有权重、需要自动拉取时，可临时设下载目录，**仍建议之后改用上节的手动 `--policy.dir` 启动**：

```bash
export OPENPI_DATA_HOME=/home/featurize/work/openpi/openpi_cache
uv run scripts/serve_policy.py --env DROID   # 仅首次下载用；下完改用手动路径
```

### 配额与 git 注意

```bash
# 查看 work 用量（openpi_cache 会计入配额）
du -sh ~/work/
du -sh ~/work/openpi/openpi_cache/
```

- `openpi_cache/` 已在 `.gitignore` 中，**不要** `git add` 权重文件
- 多个 checkpoint 叠加可能很快到 30–50 GB，删旧模型时直接 `rm -rf openpi_cache/openpi-assets/checkpoints/<不需要的>`

### 与「本地盘缓存」的取舍

| 方案 | 路径示例 | 跨实例保留 | 速度 |
|------|---------|----------|------|
| **repo 内（本节）** | `~/work/openpi/openpi_cache` | ✅ work 云盘持久 | 慢 |
| 本地盘 | `/home/featurize/openpi_cache` | ❌ 实例销毁即没 | 快 |

只做推理、频繁换 env → 用本节 repo 内方案更省事。训练时大量读数据 → 仍建议数据集放本地盘。

---

## 3.3 ⚠️ 必须显式加 `--host 127.0.0.1`

客户端默认 `host=0.0.0.0`，在 Featurize / 容器化网络环境下**连不上**本机 server，会报：

```
websockets.exceptions.InvalidMessage: did not receive a valid HTTP response
EOFError: stream ends after 0 bytes, before end of line
```

这**不是** server 故障（server 监听在 `0.0.0.0:8000` 是正确的），而是 `0.0.0.0` 不是合法的**客户端目标地址**。改成 `127.0.0.1` 或 `localhost` 即可。

---

## 3.4 Server 启动成功的标志

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

## 3.5 客户端成功输出

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

## 3.6 Python 代码直接调用（不走 server）

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

## 3.7 想换其他 env / 模型？

`--env` 支持 `ALOHA / ALOHA_SIM / DROID / LIBERO`，各 env 的观测格式不同。详见 [04 · 模型与环境选择](./04-models-and-envs.md)。

---

## 下一步

- 想跑 LIBERO benchmark、对论文、看仿真录像 → [08 · LIBERO 推理实战](./08-libero-inference-walkthrough.md)（Featurize Smoke 约 9 分钟，需双终端 + 单独 3.8 venv）
- 想了解每个 env 的观测格式和模型差异 → [04 · 模型与环境选择](./04-models-and-envs.md)
- 想用自己的数据微调 → [06 · 微调流程](./06-finetuning.md)
- 推理报错没解决 → [99 · 常见问题](./99-troubleshooting.md)
