# 06 · 微调流程

> 前置：已完成 [02 · 依赖安装](./02-installation.md)；已选好实现路径（见 [05 · PyTorch vs JAX](./05-pytorch-vs-jax.md)）。
>
> 4090 用户能做的：**LoRA 微调**（用 JAX）、**小数据集 PyTorch 全 bf16 微调**。
>
> 4090 用户做不了：**全量微调**（需 70GB+ 显存，A100/H100）。

---

## 6.1 三步走

1. 数据 → LeRobot dataset 格式
2. 算 norm stats（一次性）
3. 训练 + 启动 server 推理

---

## 6.2 数据转换

OpenPI 用 [LeRobot dataset 格式](https://github.com/huggingface/lerobot)。LIBERO 的转换脚本可参考：

```bash
uv run examples/libero/convert_libero_data_to_lerobot.py \
  --data_dir /path/to/your/libero/data
```

> 仅想跑 LIBERO 不必转换：OpenPI 的 LIBERO config 已经指向预转换的数据集。

自己的数据，参考 `convert_libero_data_to_lerobot.py` 改写。

---

## 6.3 自定义 config 三要素

在 `src/openpi/` 下定义：

| 文件 | 作用 |
|------|------|
| `policies/<your>_policy.py` 中的 `Inputs / Outputs` | 训练 / 推理时数据 ⇄ 模型的映射 |
| `training/config.py` 中的 `LeRobot<Your>DataConfig` | 处理原始 LeRobot 数据用于训练 |
| `training/config.py` 中的 `TrainConfig` | 微调超参、data config、weight loader |

参考 LIBERO 的实现（`libero_policy.py`、`LeRobotLiberoDataConfig`、`pi05_libero`）。

---

## 6.4 算 norm stats（必做）

```bash
cd /home/featurize/work/openpi
uv run scripts/compute_norm_stats.py --config-name pi05_libero
```

**不算 norm stats 训练会失败**。算完会落地到 `assets/<config_name>/<repo_id>/norm_stats.json`。

> 若要复用 pre-training 的 norm stats（机器人在预训练 mixture 里），见 [`docs/norm_stats.md`](../norm_stats.md)。

---

## 6.5 训练

### JAX（推荐）

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
  uv run scripts/train.py pi05_libero \
    --exp-name=my_experiment \
    --overwrite
```

- `XLA_PYTHON_CLIENT_MEM_FRACTION=0.9`：让 JAX 用满 4090 显存（默认 0.75）
- `--exp-name`：实验名，决定 checkpoint 保存目录 `checkpoints/pi05_libero/my_experiment/`
- `--overwrite`：同名实验覆盖

### PyTorch（单卡）

```bash
uv run scripts/train_pytorch.py pi05_libero --exp_name pytorch_run
```

注意：

- PyTorch 不支持 LoRA / FSDP，全量微调在 4090 上**会 OOM**
- 用小数据 + 全 bf16 + 部分冻结 config（`_low_mem_finetune` 系列）有机会跑通

---

## 6.6 训练时长参考（4090 单卡）

LIBERO 微调（`pi0_libero_low_mem_finetune`，LoRA，`batch_size=32`，`num_train_steps=30000`）：

| 阶段 | 耗时 |
|------|------|
| 下载数据集（首次） | 30–90 分钟（LIBERO ~60 GB） |
| `compute_norm_stats.py`（一次性） | 10–30 分钟 |
| **训练完整 30k steps** | **~20–40 小时（约 1–2 天）** |
| 单 step | ~2–5 秒 |

**更快出可用 checkpoint** 的办法：

- `num_train_steps` 从 30k 降到 5k–10k，**5–15 小时**就能看到效果
- 默认 `save_interval=1000`，随时可中断用中间 checkpoint
- 进一步降显存/提速：`batch_size` 降到 16 或 8

> **不要用 `pi05_libero`（全量微调 config）**：`batch_size=256` + EMA，4090 必 OOM。

---

## 6.7 4090 显存优化（OOM 时排查）

按效果由强到弱：

| 措施 | 效果 | 副作用 |
|------|------|--------|
| `XLA_PYTHON_CLIENT_MEM_FRACTION=0.9` | 解锁更多显存 | 几乎无 |
| 选 `_low_mem_finetune` 系列 config | 部分冻结，省一大半显存 | 表达力受限 |
| LoRA 微调（JAX） | 22.5 GB 就够 | 表达力略受限 |
| 禁用 EMA（config 改 `ema_decay=None`） | 省一份模型权重 | 验证效果略降 |
| `--fsdp-devices N` | 多卡分片 | 单卡无效；需要多卡 |

> **FSDP 在单卡上无意义**，README 写 "也可以设置 `--fsdp-devices 1` 缓解开销" 实际改善有限。

---

## 6.8 训练后启动 server

```bash
uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config=pi05_libero \
  --policy.dir=checkpoints/pi05_libero/my_experiment/20000
```

`20000` 是迭代步数，按实际改。

---

## 6.9 配套监控

训练时自动用 Weights & Biases（需要 `wandb login`）。或者只看控制台日志。

checkpoint 默认存在 `checkpoints/<config>/<exp>/<step>/`，体积较大（10GB+），4090 实例建议指向本地盘而非 `~/work`：

```bash
# 在训练命令前
export OPENPI_CHECKPOINT_DIR=/home/featurize/checkpoints
```

（具体环境变量名按 OpenPI 实现为准，必要时直接软链 `checkpoints/` 到本地盘目录）

---

## 下一步

- 训练完想推理 → [03 · 推理快速上手](./03-inference-quickstart.md)（把 `--policy.dir` 改成你的 checkpoint）
- 报错没解决 → [99 · 常见问题](./99-troubleshooting.md)
