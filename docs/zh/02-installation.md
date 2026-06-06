# 02 · 依赖安装

> 前置：已完成 [01 · 环境准备](./01-environment-setup.md)。
>
> 目标：用 uv 装好 OpenPI 主体依赖（PyTorch + JAX + CUDA 等约 3GB+），并验证 GPU 可用。

---

## 2.1 一键安装

```bash
export UV_INDEX_URL=https://mirrors.cloud.tencent.com/pypi/simple/

cd /home/featurize/work/openpi
uv cache clean
rm -rf .venv

GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

---

## 2.2 注意事项

- 首次 sync 下载 **3GB+**（含 PyTorch、JAX CUDA、CUDNN、CUBLAS、NCCL 等），用腾讯云镜像约 **5–15 分钟**
- 请**耐心等待，不要中断**，否则 `.venv` 处于半装状态，下次 sync 行为不稳定
- `GIT_LFS_SKIP_SMUDGE=1` 是**必须**的，否则拉 LeRobot 依赖会卡在 LFS
- **不要**装系统级 CUDA，CUDA 库由 uv 自动管理；冲突时优先 `uv cache clean` 重装

---

## 2.3 验证安装

```bash
cd /home/featurize/work/openpi
uv run python -c "import torch, jax; \
  print('torch:', torch.__version__, 'cuda:', torch.cuda.is_available()); \
  print('jax devices:', jax.devices())"
```

期望输出：

```
torch: 2.7.1+cu126  cuda: True
jax devices: [CudaDevice(id=0)]
```

torch 版本可能是 `+cu126` 或 `+cu128`，**只要 `cuda: True` 且 jax devices 列出 `CudaDevice` 就算成功**。

---

## 2.4 PyTorch 版本额外步骤（可选）

如果要用 **PyTorch 实现**的 π₀ / π₀.₅（默认是 JAX 实现）：

```bash
cp -r ./src/openpi/models_pytorch/transformers_replace/* \
      .venv/lib/python3.11/site-packages/transformers/
```

⚠️ **副作用**：默认 uv 用 hardlink，这会修改 uv 缓存里的 transformers 源文件，影响其他用到 transformers 的项目。还原命令：

```bash
uv cache clean transformers
```

PyTorch 版本能力差异详见 [05 · PyTorch vs JAX](./05-pytorch-vs-jax.md)。

---

## 2.5 完整一键脚本

把整个 01 + 02 流程串起来：

```bash
#!/bin/bash
set -e

export UV_INDEX_URL=https://mirrors.cloud.tencent.com/pypi/simple/
export GIT_LFS_SKIP_SMUDGE=1

cd /home/featurize/work/openpi

# 1) 子模块（如不需要 LIBERO/ALOHA 可跳过）
GIT_PROGRESS=1 git submodule update --init --recursive --depth 1 --progress

# 2) uv（若未安装）
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh

# 3) 依赖
uv cache clean
rm -rf .venv
uv sync
uv pip install -e .

# 4) 验证
uv run python -c "import torch, jax; \
  print('torch:', torch.__version__, 'cuda:', torch.cuda.is_available()); \
  print('jax devices:', jax.devices())"

echo "OpenPI 安装完成 ✅"
```

---

## 下一步

依赖装好 → [03 · 推理快速上手](./03-inference-quickstart.md)
