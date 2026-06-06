# 99 · 常见问题速查

按报错关键字快速定位。

---

## 安装相关

| 问题 | 解决方案 |
|------|---------|
| `git submodule update` 卡住 | 加 `--depth 1 --progress`；GitHub 慢用 ghproxy（详见 [01 · 环境准备](./01-environment-setup.md#15-拉取子模块)） |
| `uv sync` 报 `invalid Content-Range header: bytes 18446744...` | 清华 HTTPS 源 + uv 已知 bug。换腾讯云/阿里云，并 `uv cache clean` |
| `uv sync` 很慢 | 用腾讯云镜像 `https://mirrors.cloud.tencent.com/pypi/simple/` |
| `uv sync` 中断后状态混乱 | `rm -rf .venv && uv cache clean` 重新来 |
| `Failed to hardlink files; falling back to full copy` | 警告，不影响。可 `export UV_LINK_MODE=copy` 消音 |
| CUDA 报错 | 不要装系统级 CUDA，让 uv 管理；必要时卸载系统 CUDA |
| GitHub 拉取（含 lerobot/dlimp）很慢 | 配置 `git config --global url."https://ghproxy.com/https://github.com/".insteadOf "https://github.com/"` |

---

## 推理相关

| 问题 | 解决方案 |
|------|---------|
| Client 报 `did not receive a valid HTTP response` / `EOFError: stream ends after 0 bytes` | Client 默认 `host=0.0.0.0` 在 Featurize 上连不通。**加 `--host 127.0.0.1`** |
| Server 日志 `Unable to initialize backend 'rocm'/'tpu'` | **正常**，JAX 自动回退到 CUDA |
| Server 日志 `Norm stats not found in .../assets/<cfg>/<repo>, skipping` | **正常**，会回退到 checkpoint 自带的 norm stats |
| Server 日志 `gsutil not found, falling back to gcsfs` | 正常，用 Python 客户端下 GCS |
| Checkpoint 下载慢 | 设置 `OPENPI_DATA_HOME` 到本地盘（不要放 `~/work`） |
| `connection refused` 连不上 server | 确认 server 真的在跑：`ss -tlnp \| grep 8000` 应能看到 python 进程监听 |
| HF 数据集下载失败 | `huggingface-cli login`；或 `export HF_ENDPOINT=https://hf-mirror.com` |

---

## 训练相关

| 问题 | 解决方案 |
|------|---------|
| 训练 OOM（JAX） | `export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9`；改用 `_low_mem`/`_low_mem_finetune` config；禁用 EMA |
| 用 `pi0_aloha_sim` 报 "48 GiB > 22 GiB" OOM | 该 config 是全量微调 + EMA，4090 装不下。改用 `pi0_aloha_sim_low_mem`（LoRA 版） |
| 训练 OOM（PyTorch） | PyTorch 无 LoRA/FSDP，4090 基本只能跑 `_low_mem_finetune` |
| `Missing norm stats` | 训练前先 `uv run scripts/compute_norm_stats.py --config-name <cfg>` |
| 训练 loss 发散 | 检查 `norm_stats.json` 的 `q01/q99/std`，某些维度若极小会导致归一化后值爆炸，可手动调整 |
| 训练启动崩 `api_key not configured (no-tty)` | wandb 没登录 + 后台跑没 tty。先 `wandb login <key>`；或加 `WANDB_MODE=disabled` |
| `wandb login` 报 `API key must be 40 characters long, yours was 86` | 粘错内容了，可能带了 URL / 邮箱 / PAT。重去 https://wandb.ai/authorize 只复制纯 40 位 key |
| Dataset download fails | HF 数据：`huggingface-cli login`；网络：用 `HF_ENDPOINT` 国内镜像 |
| Action dimensions mismatch | 检查 policy class 的 `Inputs/Outputs` 维度是否对应你的机器人 |

---

## 平台相关（Featurize）

| 问题 | 解决方案 |
|------|---------|
| `~/work` 配额报警 | `du -sh ~/work/` 查用量；把 checkpoint / 数据迁到本地盘 |
| HF 数据集偷偷占 100GB+ | Xet 协议会缓存双份。设 `export HF_HUB_DISABLE_XET=1`，并清理 `rm -rf $HF_HOME/xet` |
| `.venv` 撑爆 work 配额 | 软链到本地盘：`ln -s /home/featurize/openpi_venv ~/work/openpi/.venv` |
| LIBERO 数据下载到不知道哪 | 默认 `~/.cache/huggingface/lerobot/`，约 60GB。先设 `HF_HOME=/home/featurize/hf_cache` |
| 实例销毁后本地盘数据丢失 | 重要 checkpoint 拷回 `~/work` 或下载到本地 |
| 训练巨慢 | 检查数据是否在 `~/work`（云同步盘很慢）。数据放 `/home/featurize/data` 或其他本地目录 |
| MUJOCO_GL 报错 | 无显示器服务器用 `export MUJOCO_GL=egl`；安装 `libegl1-mesa-dev libgles2-mesa-dev` |

---

## 上报前自查清单

提 issue 前，先确认：

1. ✅ uv 版本 ≥ 0.11（`uv --version`）
2. ✅ 镜像源不是清华 HTTPS（`echo $UV_INDEX_URL`）
3. ✅ `uv cache clean && rm -rf .venv` 后重装一次
4. ✅ GPU 可用：`uv run python -c "import torch; print(torch.cuda.is_available())"`
5. ✅ Server 真的在监听 8000：`ss -tlnp | grep 8000`
6. ✅ Client 用的 `--host 127.0.0.1` 不是 `0.0.0.0`

---

## 还没解决？

回到 [文档首页](./README.md)，按场景重新走流程；或 提 issue 时附上：

- `uv --version`、`nvidia-smi`、`python --version`
- 完整 server 日志 + 完整 client 日志
- 用的 env / config 名 / checkpoint 路径
