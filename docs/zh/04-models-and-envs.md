# 04 · 模型与环境选择

> 适用：想搞清楚 `--env` 到底代表什么、checkpoint 怎么选、自己的场景该用哪个 config。

---

## 4.1 EnvMode 是什么

`scripts/serve_policy.py --env` 指定**机器人平台**，server 和 client 必须用同一个 env。每个 env 对应：

1. 一个**默认 checkpoint**（model + 微调权重）
2. 一组**观测 schema**（输入图像 / 状态的 key 和形状）
3. 一组**动作 schema**（输出动作的维度）

源码位置：

- `scripts/serve_policy.py::DEFAULT_CHECKPOINT`
- `examples/simple_client/main.py::_random_observation_*`

---

## 4.2 四个内置 env

### `DROID` — Franka 单臂桌面机器人 ⭐ 推荐先跑

| 项目 | 值 |
|------|----|
| 默认 config | `pi05_droid` |
| 默认 checkpoint | `gs://openpi-assets/checkpoints/pi05_droid`（11.6 GB） |
| 平台 | [DROID dataset](https://droid-dataset.github.io/) 标准 Franka |
| 动作维度 | 8（7 关节 + 1 夹爪） |
| 用途 | 单臂抓取、桌面操作；最通用、泛化最好的 checkpoint |

**观测 schema**：

```python
{
  "observation/exterior_image_1_left": (224, 224, 3) uint8,
  "observation/wrist_image_left":      (224, 224, 3) uint8,
  "observation/joint_position":        (7,)   float,
  "observation/gripper_position":      (1,)   float,
  "prompt": "pick up the fork",
}
```

### `ALOHA` — 双臂遥操机器人（真机）

| 项目 | 值 |
|------|----|
| 默认 config | `pi05_aloha` |
| 默认 checkpoint | `gs://openpi-assets/checkpoints/pi05_base`（**base**，未任务化） |
| 平台 | Stanford [ALOHA](https://tonyzhaozh.github.io/aloha/) 双臂平台 |
| 动作维度 | 14（2 臂 × 7） |
| 用途 | 真实 ALOHA 机器人 |

**观测 schema**：

```python
{
  "state": (14,) float,
  "images": {
     "cam_high":        (3, 224, 224) uint8,
     "cam_low":         (3, 224, 224) uint8,
     "cam_left_wrist":  (3, 224, 224) uint8,
     "cam_right_wrist": (3, 224, 224) uint8,
  },
  "prompt": "fold the towel",
}
```

**任务特化的 ALOHA checkpoint**（不是 default env 自动加载的，需手动指定）：

- `pi0_aloha_towel` — 叠毛巾
- `pi0_aloha_tupperware` — 从保鲜盒拿食物
- `pi0_aloha_pen_uncap` / `pi05_aloha_pen_uncap` — 给笔脱帽

用法：

```bash
uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config=pi0_aloha_towel \
  --policy.dir=gs://openpi-assets/checkpoints/pi0_aloha_towel
```

### `ALOHA_SIM` — 双臂机器人的 MuJoCo 仿真

| 项目 | 值 |
|------|----|
| 默认 config | `pi0_aloha_sim`（**注意是 pi0，不是 pi05**） |
| 默认 checkpoint | `gs://openpi-assets/checkpoints/pi0_aloha_sim` |
| 平台 | [gym-aloha](https://github.com/huggingface/gym-aloha) MuJoCo 仿真 |
| 观测 schema | 与 `ALOHA` 相同 |
| 任务 | "transfer cube"（左臂抓方块给右臂） |
| 渲染依赖 | `MUJOCO_GL=egl`（headless）或 glx |

4090 服务器无显示器，用 egl：

```bash
MUJOCO_GL=egl uv run scripts/serve_policy.py --env ALOHA_SIM
```

具体跑法看 [`examples/aloha_sim/README.md`](../../examples/aloha_sim/README.md)。

### `LIBERO` — 标准 benchmark

| 项目 | 值 |
|------|----|
| 默认 config | `pi05_libero` |
| 默认 checkpoint | `gs://openpi-assets/checkpoints/pi05_libero` |
| 平台 | [LIBERO benchmark](https://libero-project.github.io/datasets) MuJoCo 仿真 |
| 动作维度 | 7 |
| 任务套件 | `libero_spatial / object / goal / 10 / 90` |

**观测 schema**：

```python
{
  "observation/state":        (8,) float,
  "observation/image":        (224, 224, 3) uint8,
  "observation/wrist_image":  (224, 224, 3) uint8,
  "prompt": "pick up the alphabet soup and place it in the basket",
}
```

**特点**：

- **官方 benchmark**，π₀.₅-LIBERO 是 SOTA
- 需要 `git submodule update --init --recursive` 拉 `third_party/libero`
- 推荐用 Docker（依赖较老：Python 3.8、CUDA 11.3）
- 适合**对比论文结果**、**评测微调效果**

详情：[`examples/libero/README.md`](../../examples/libero/README.md)

---

## 4.3 选哪个？

| 目的 | env |
|------|-----|
| 第一次跑通验证安装 | **DROID**（checkpoint 公开效果好，无需仿真依赖） |
| 想看仿真画面（双臂） | `ALOHA_SIM` |
| 跑标准 benchmark / 复现论文 | `LIBERO` |
| 真 Franka / DROID 机器人 | `DROID` |
| 真 ALOHA 机器人 | `ALOHA` |

---

## 4.4 完整 config 列表（不止 4 个 env）

`--env` 只暴露 4 个默认，但 `src/openpi/training/config.py` 里还有更多 config，用 `policy:checkpoint` 模式启动：

```bash
uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config=<config_name> \
  --policy.dir=<ckpt_path>
```

### 按模型族分组

**π₀ / π₀.₅ ALOHA 系列**：

- `pi0_aloha`、`pi05_aloha`
- `pi0_aloha_towel`、`pi0_aloha_tupperware`
- `pi0_aloha_pen_uncap`、`pi05_aloha_pen_uncap`
- `pi0_aloha_sim`

**π₀ / π₀.₅ DROID 系列**：

- `pi0_droid`、`pi05_droid`
- `pi0_fast_droid`（自回归 FAST 版）
- `pi05_droid_finetune`、`pi05_full_droid_finetune`
- `pi0_fast_full_droid_finetune`

**LIBERO 系列**：

- `pi0_libero`、`pi05_libero`
- `pi0_libero_low_mem_finetune`（小显存微调）
- `pi0_fast_libero`、`pi0_fast_libero_low_mem_finetune`

### 命名约定

| 后缀 | 含义 |
|------|------|
| `_finetune` | 完整微调配置 |
| `_full_*_finetune` | 在某数据集做完整微调（如 full DROID） |
| `_low_mem_finetune` | 小显存版本，**部分冻结**层 |
| `_fast_*` | π₀-FAST 模型（自回归，JAX only） |

---

## 4.5 模型族横向对比

| 模型 | 架构 | 推理速度 | 语言跟随 | 4090 LoRA |
|------|------|---------|---------|-----------|
| π₀ | 流匹配 flow matching | 快 | 一般 | ✅ |
| **π₀-FAST** | **自回归 autoregressive** | **慢** | **强** | ✅（JAX only） |
| π₀.₅ | 流匹配 + 知识隔离 | 快 | 强 | ✅ |

详情见 [05 · PyTorch vs JAX](./05-pytorch-vs-jax.md)（FAST 在 PyTorch 端不可用）。

---

## 下一步

- 想选 PyTorch 还是 JAX → [05 · PyTorch vs JAX](./05-pytorch-vs-jax.md)
- 想微调自己的数据 → [06 · 微调流程](./06-finetuning.md)
