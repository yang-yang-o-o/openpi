# setup.md · 极简安装速查

> 想快速跑通？4 步走。完整文档见 `[docs/zh/](./docs/zh/README.md)`。

---

## 一、4 步装好

```bash
git clone https://github.com/yang-yang-o-o/openpi.git
git checkout dev

# 1. 装 uv 并配腾讯云镜像（清华 HTTPS 源有 bug，不要用）
curl -LsSf https://astral.sh/uv/install.sh | sh
export UV_INDEX_URL=https://mirrors.cloud.tencent.com/pypi/simple/
export GIT_LFS_SKIP_SMUDGE=1

# 2. 拉子模块（仅 LIBERO/ALOHA 需要，DROID 可跳过）
cd /home/featurize/work/openpi
GIT_PROGRESS=1 git submodule update --init --recursive --depth 1 --progress

# 3. 装依赖（3GB+，5–15 分钟，不要中断）
uv cache clean && rm -rf .venv
uv sync
uv pip install -e .

# 4. 验证
uv run python -c "import torch, jax; \
  print('torch:', torch.__version__, 'cuda:', torch.cuda.is_available()); \
  print('jax devices:', jax.devices())"
```

期望输出 `cuda: True` 和 `[CudaDevice(id=0)]`。

---

## 二、跑通推理（两个终端）

终端 1：

```bash
uv run scripts/serve_policy.py --env DROID
```

终端 2：

```bash
uv run examples/simple_client/main.py --env DROID --host 127.0.0.1
```

⚠️ Client **必须** `--host 127.0.0.1`，默认的 `0.0.0.0` 在 Featurize 上连不通。

---

## 三、跑通微调（LIBERO 为例）

> 前置：子模块已拉好（步骤一中第 2 步）。4090 只能跑 **LoRA / 部分冻结**，跑不了全量微调。
>
> ⚠️ `pi05_libero` 是 **全量微调** config（`batch_size=256` + EMA），4090 必 OOM。
> 4090 上要用 **`pi0_libero_low_mem_finetune`**（LoRA + 无 EMA + `batch_size=32`）。

### 1. 配缓存路径（必做，否则会偷偷占爆磁盘）

LIBERO 数据集 \~60 GB，加 Xet 双倍缓存可到 \~120 GB。务必把 HF / OpenPI 缓存指向**本地盘**（`/dev/vda1` 有 600+ GB），不要让它进 `~/work`（仅 30 GB 配额）：

```bash
cat >> ~/.zshrc << 'EOF'
export HF_HOME=/home/featurize/hf_cache
export HF_DATASETS_CACHE=/home/featurize/hf_cache/datasets
export HF_HUB_DISABLE_XET=1                # 关 Xet，省一半磁盘
export HF_ENDPOINT=https://hf-mirror.com   # 国内镜像加速
export OPENPI_DATA_HOME=/home/featurize/openpi_cache
EOF
source ~/.zshrc
mkdir -p /home/featurize/hf_cache /home/featurize/openpi_cache

# 若 ~/.cache/huggingface 已有内容，迁移过去（避免重下）
if [ -d /home/featurize/.cache/huggingface ] && [ ! -L /home/featurize/.cache/huggingface ]; then
  mv /home/featurize/.cache/huggingface/* /home/featurize/hf_cache/ 2>/dev/null
  rmdir /home/featurize/.cache/huggingface
  ln -s /home/featurize/hf_cache /home/featurize/.cache/huggingface
fi
```

### 2. 跑微调三步

```bash
cd /home/featurize/work/openpi

# (1) 计算归一化统计（一次性，必做；会下载 ~60GB LIBERO 数据）
uv run scripts/compute_norm_stats.py --config-name pi0_libero_low_mem_finetune

# (2) 启动 LoRA 训练（JAX 路径，4090 友好）
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
  uv run scripts/train.py pi0_libero_low_mem_finetune \
    --exp-name=my_experiment \
    --overwrite

# (3) 训练完启 server（按实际 checkpoint 步数改）
uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config=pi0_libero_low_mem_finetune \
  --policy.dir=checkpoints/pi0_libero_low_mem_finetune/my_experiment/30000
```

### 3. 4090 上要训多久？

`pi0_libero_low_mem_finetune` 默认 `num_train_steps=30000`、`batch_size=32`：

| 阶段 | 4090 单卡耗时（估） |
|------|-------------------|
| 首次下数据（60 GB） | 30–90 分钟（视网速） |
| `compute_norm_stats.py` | 10–30 分钟（一次性） |
| **训练 30k steps** | **~20–40 小时**（约 1–2 天） |
| 单 step | ~2–5 秒 |

**想更快出第一个能看的 checkpoint**：

- 改 `num_train_steps=5000` 或 `10000`（在 config.py 里改，或加 CLI flag），**5–15 小时**就能看到效果
- 训练默认每 1000 step 存 checkpoint（`save_interval=1000`），随时可中断用任意中间 checkpoint 推理
- 想再降显存/提速：把 `batch_size` 调到 16 或 8

### 4. 要点

- `XLA_PYTHON_CLIENT_MEM_FRACTION=0.9` **必加**，让 JAX 用满 4090 显存
- **千万别用 `pi05_libero`**（全量微调 config），4090 必 OOM
- `--exp-name` 决定 checkpoint 目录 `checkpoints/<config>/<exp-name>/<step>/`
- 自己的数据需先转 LeRobot 格式 + 定义 config，详见 [06 · 微调流程](./docs/zh/06-finetuning.md)

---

那我感觉还是“2000
~1.5–2 小时
5–25%
看得出在伸手抓方块，但抓不稳/掉/递不出去”吧，数据下载比较简单，而且训练2小时以内，而且我能在仿真里看看推理效果，后续我找个能长时间运行的环境，跑一晚，还能和论文里的结果对比一下，仿真也能有90%左右的成功率，你觉得呢？有没有更好的建议

## 四、详细文档


| 想做的事              | 看这里                                                   |
| ----------------- | ----------------------------------------------------- |
| 从零完整搭环境           | [01 · 环境准备](./docs/zh/01-environment-setup.md)        |
| 安装报错 / 镜像问题       | [02 · 依赖安装](./docs/zh/02-installation.md)             |
| 推理报错 / Python 调用  | [03 · 推理快速上手](./docs/zh/03-inference-quickstart.md)   |
| 选模型 / 选 env       | [04 · 模型与环境选择](./docs/zh/04-models-and-envs.md)       |
| 纠结 PyTorch vs JAX | [05 · PyTorch vs JAX](./docs/zh/05-pytorch-vs-jax.md) |
| 自己数据微调            | [06 · 微调流程](./docs/zh/06-finetuning.md)               |
| 报错查询              | [99 · 常见问题](./docs/zh/99-troubleshooting.md)          |
| 总入口               | [docs/zh/README.md](./docs/zh/README.md)              |


