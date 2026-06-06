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

## 三、跑通微调（ALOHA Sim · 2 小时 smoke + 过夜 20k）

4090 上最适合入门的微调任务是 **`pi0_aloha_sim`**（transfer_cube）：

- 数据集仅 ~1.6 GB（vs LIBERO 60 GB）
- Smoke 3000 步 ~2 小时，能在仿真里**眼见为实**
- Full 20000 步过夜跑，成功率可达 70%+

**完整步骤、调参、评估脚本** → [07 · 4090 微调全流程实战](./docs/zh/07-4090-finetune-walkthrough.md)

最小命令（细节看 07）：

```bash
# 1. 算 norm stats（下 ~1.6 GB 数据，~5 分钟）
uv run scripts/compute_norm_stats.py --config-name pi0_aloha_sim_low_mem

# 2. Smoke 训练（~2 小时）
nohup env XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 WANDB_MODE=disabled \
  uv run scripts/train.py pi0_aloha_sim_low_mem \
    --exp-name=smoke3k --overwrite \
    --num-train-steps=3000 --batch-size=4 \
  > train_smoke3k.log 2>&1 &

# 3. 仿真评估
uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config=pi0_aloha_sim_low_mem \
  --policy.dir=checkpoints/pi0_aloha_sim_low_mem/smoke3k/3000
# 另一个终端：MUJOCO_GL=egl uv run examples/aloha_sim/main.py --host 127.0.0.1
```

⚠️ 4090 必须用 **`pi0_aloha_sim_low_mem`**（项目里加的 LoRA 版本，见 `src/openpi/training/config.py`）。
原版 `pi0_aloha_sim` 是全量微调，需要 ~48 GB 显存，4090 必 OOM。

> 想跑 LIBERO benchmark（60 GB 数据 + 20–40 小时训练）见 [06 · 微调流程](./docs/zh/06-finetuning.md)。

---

## 四、详细文档

| 想做的事 | 看这里 |
|---------|-------|
| 从零完整搭环境 | [01 · 环境准备](./docs/zh/01-environment-setup.md) |
| 安装报错 / 镜像问题 | [02 · 依赖安装](./docs/zh/02-installation.md) |
| 推理报错 / Python 调用 | [03 · 推理快速上手](./docs/zh/03-inference-quickstart.md) |
| 选模型 / 选 env | [04 · 模型与环境选择](./docs/zh/04-models-and-envs.md) |
| 纠结 PyTorch vs JAX | [05 · PyTorch vs JAX](./docs/zh/05-pytorch-vs-jax.md) |
| 自己数据微调 / 通用 | [06 · 微调流程](./docs/zh/06-finetuning.md) |
| **4090 微调全流程实战** | [**07 · 4090 微调全流程**](./docs/zh/07-4090-finetune-walkthrough.md) |
| 报错查询 | [99 · 常见问题](./docs/zh/99-troubleshooting.md) |
| 总入口 | [docs/zh/README.md](./docs/zh/README.md) |


