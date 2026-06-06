# 05 · PyTorch vs JAX

> 适用：纠结用 `train.py`（JAX）还是 `train_pytorch.py`（PyTorch）的用户。
>
> 结论先放：**4090 用户默认选 JAX**，除非有非 PyTorch 不可的理由。

---

## 5.1 背景

OpenPI 维护两套实现（2025/09 加入 PyTorch）：

| 路径 | 实现 | 训练入口 |
|------|------|---------|
| `src/openpi/models/` | JAX（原始版，功能完整） | `scripts/train.py` |
| `src/openpi/models_pytorch/` | PyTorch（新加） | `scripts/train_pytorch.py` |

权重可互转：`examples/convert_jax_model_to_pytorch.py`。

---

## 5.2 PyTorch 端缺失的功能

### ① π₀-FAST 模型 ❌

**是什么**：π₀-FAST 是 OpenPI 的**自回归 VLA**，基于 [FAST action tokenizer](https://www.physicalintelligence.company/research/fast)。

| 模型 | 架构 | 推理速度 | 语言跟随 |
|------|------|---------|---------|
| π₀ | 流匹配（flow matching） | 快 | 一般 |
| **π₀-FAST** | **自回归（autoregressive）** | **慢** | **强** |
| π₀.₅ | 流匹配 + 知识隔离 | 快 | 强 |

**影响**：PyTorch 下用不了 `pi0_fast_*` 系列 config。想用这些 config 必须走 JAX 路径。

### ② 混合精度（Mixed Precision）❌

**JAX 端**：默认混合精度——**权重和梯度 float32，激活和计算 bfloat16**。大模型训练标配，能在不损失稳定性的前提下节省大量显存。

**PyTorch 端**：只能二选一（通过 `pytorch_training_precision` 切换）：

- **全 bfloat16**（默认）：显存最省，但 loss 比 float32 略高
- **全 float32**：显存翻倍，4090 上很容易 OOM

**没法像 JAX 那样精细控制每层精度**。

### ③ FSDP（Fully Sharded Data Parallelism）❌

**是什么**：FSDP 把模型权重、梯度、优化器状态在多张卡之间**分片**存储，每张卡只持有部分参数，显著降低单卡显存。

| 训练方式 | 单卡显存 | 多卡通信 |
|---------|---------|---------|
| DDP（数据并行） | 完整模型 × 1 | 少（只同步梯度） |
| **FSDP** | **模型 / N** | 多（每步需 all-gather） |

**JAX 端**：通过 config 的 `fsdp_devices` 控制，全量微调（70GB+）必须用。

**PyTorch 端**：**只支持 DDP**（`torchrun --nproc_per_node=N`），每张卡必须装得下完整模型 → 4090（24GB）上**没法全量微调**。

### ④ LoRA（Low-Rank Adaptation）❌

**是什么**：LoRA 在原始权重旁挂一对低秩矩阵 `ΔW = BA`（rank 通常 8/16/32），训练时**冻结原权重，只学这对小矩阵**。

**显存对比**（以 π₀ 为例）：

- 全量微调：~70 GB（需 A100）
- **LoRA 微调：~22.5 GB**（4090 刚好够）

**PyTorch 端缺这个功能 → 4090 用户基本只能用 JAX 跑 LoRA**。若硬要 PyTorch，只能用 `pi0_libero_low_mem_finetune` 这种**部分冻结**的 config（冻结大部分层），但不是真正的 LoRA。

### ⑤ EMA（Exponential Moving Average）❌

**是什么**：训练时维护一份权重的**滑动平均副本**

\[ \theta_{EMA} = \alpha \theta_{EMA} + (1-\alpha) \theta \]

推理时用 EMA 权重，通常更平滑、效果更好。

**代价**：额外存一份和模型同样大小的 float32 权重。

**JAX 端**：支持，config 开关；**PyTorch 端**：不支持训练时 EMA。

---

## 5.3 何时选哪个？

| 需求 | 推荐路径 |
|------|---------|
| 推理（任何模型） | PyTorch 或 JAX 都行 |
| **4090 上 LoRA 微调** | **JAX** |
| π₀-FAST 系列 | **必须 JAX** |
| 多卡全量微调 | **必须 JAX**（PyTorch 无 FSDP） |
| 单卡纯 PyTorch 玩 | π₀ / π₀.₅ 推理 + 全 bf16 微调（小数据集） |
| 对接现有 PyTorch 代码 / 生态 | PyTorch |
| EMA 想要 | **JAX** |

---

## 5.4 PyTorch 训练命令

### 单卡

```bash
uv run scripts/train_pytorch.py <config_name> --exp_name <run_name>
```

例：

```bash
uv run scripts/train_pytorch.py debug --exp_name pytorch_test
uv run scripts/train_pytorch.py debug --exp_name pytorch_test --resume
```

### 多卡 DDP（单节点）

```bash
uv run torchrun --standalone --nnodes=1 --nproc_per_node=<num_gpus> \
  scripts/train_pytorch.py <config_name> --exp_name <run_name>
```

### 多节点

```bash
uv run torchrun \
  --nnodes=<num_nodes> \
  --nproc_per_node=<gpus_per_node> \
  --node_rank=<rank> \
  --master_addr=<master_ip> \
  --master_port=<port> \
  scripts/train_pytorch.py <config_name> --exp_name=<run_name>
```

---

## 5.5 转换 JAX checkpoint 为 PyTorch

```bash
uv run examples/convert_jax_model_to_pytorch.py \
  --checkpoint_dir /path/to/jax/checkpoint \
  --config_name <config name> \
  --output_path /path/to/converted/pytorch/checkpoint
```

转换后用同样的 API 推理，只把 `checkpoint_dir` 指到转换后的目录即可。

---

## 下一步

- 选好实现想开始训练 → [06 · 微调流程](./06-finetuning.md)
