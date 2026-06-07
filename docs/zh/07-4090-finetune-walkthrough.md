# 07 · 4090 微调全流程实战（ALOHA Sim · transfer_cube）

> 适用：Featurize 4090（24 GB）单卡，想**先 2 小时跑通**、再过夜出可用模型。
>
> 任务：ALOHA 双臂 transfer cube（左手抓方块，递给右手）。
>
> 为什么选这个：数据集仅 ~1.6 GB（vs LIBERO ~60 GB），仿真可直接看效果，是 OpenPI 上最适合 4090 的入门微调任务。

---

## 7.1 整体策略

分两阶段，避免「闷头跑 N 小时才发现链路有问题」：

| 阶段 | 步数 | 4090 实测耗时 | 预估成功率 | 目的 |
|------|------|--------------|----------|------|
| **Phase 1 · Smoke** | 3000 | **~40 分钟** | 5–25% | 验证管线 + 看仿真效果 + 看 loss 曲线 |
| **Phase 2 · Full** | 20000 | **~4–5 小时** | 40–60% | 一次跑完看完整效果 |
| Phase 2+（可选） | 80000 | ~16 小时 | 70–80% | 补 batch_size=4 的样本量缺口 |

> **实测数据**（pi0_aloha_sim_low_mem + bs=4 + 4090 LoRA）：
> - 稳态 **1.4 step/s**
> - 显存 22 GB / 24 GB（90% 占满）
> - GPU-Util 100%、功耗 95%
>
> ⚠️ 4090 装不下 bs=16/32 → "单步看到样本数少"是天花板：bs=4 × 20k 步 = 论文 bs=32 设置的 1/8 样本量。
> 想接近论文级 80%+，把步数加到 80k 补缺口（仍在 4090 单卡 16 小时可行范围）。

---

## 7.2 预检清单（Pre-flight）

跑训练前，10 分钟搞定下列环境：

### ① HF / OpenPI 缓存到本地盘（防止撑爆 work 配额）

```bash
cat >> ~/.zshrc << 'EOF'
export HF_HOME=/home/featurize/hf_cache
export HF_DATASETS_CACHE=/home/featurize/hf_cache/datasets
export HF_HUB_DISABLE_XET=1                # 关 Xet，省一半磁盘
export HF_ENDPOINT=https://hf-mirror.com   # 国内镜像
export OPENPI_DATA_HOME=/home/featurize/openpi_cache
EOF
source ~/.zshrc
mkdir -p /home/featurize/hf_cache /home/featurize/openpi_cache

# 若已有默认缓存，迁过去（避免重下）
if [ -d ~/.cache/huggingface ] && [ ! -L ~/.cache/huggingface ]; then
  mv ~/.cache/huggingface/* /home/featurize/hf_cache/ 2>/dev/null
  rmdir ~/.cache/huggingface
  ln -s /home/featurize/hf_cache ~/.cache/huggingface
fi
```

### ② checkpoint 软链到本地盘

训练 checkpoint 单份 ~10 GB，跑 20k 步会留多份，必须避开 30 GB 的 work 配额：

```bash
mkdir -p /home/featurize/openpi_checkpoints
ln -s /home/featurize/openpi_checkpoints /home/featurize/work/openpi/checkpoints
```

### ③ 登录 wandb（或显式禁用）

`pi0_aloha_sim` config 默认 `wandb_enabled=True`，**没登 wandb 会让训练直接崩**（`nohup` 后台没 tty 就抛 `api_key not configured`）。

**选项 A：登录 wandb**（推荐，可远程看进度）

```bash
# 1. 浏览器打开 https://wandb.ai/authorize ，复制【纯 40 位】API key
# 2. 直接传给 wandb（避免交互粘贴出错）
wandb login <你的40位key>

# 3. 验证
cat ~/.netrc | grep wandb   # 看到 password = <key> 即成功
```

> ⚠️ wandb API key **固定 40 位**。如果报 `API key must be 40 characters long, yours was 86`，是你粘错了内容（可能多带了 URL、邮箱或 PAT token）。**只复制页面上那串 40 位十六进制**。

**选项 B：禁用 wandb**（不想登）

在训练命令里加 `WANDB_MODE=disabled`：

```bash
nohup env XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
  WANDB_MODE=disabled \
  uv run scripts/train.py pi0_aloha_sim ...
```

只能本地 `tail -f train.log` 看进度，掉实例就丢了。

### ④ 验证 GPU 可用

```bash
cd /home/featurize/work/openpi
uv run python -c "import jax; print('jax devices:', jax.devices())"
# 期望: [CudaDevice(id=0)]
```

---

## 7.3 Phase 1 · Smoke Run（3000 步，~2 小时）

> ⚠️ **必须用 `pi0_aloha_sim_low_mem` 这个 config，不是 `pi0_aloha_sim`**。
> 原版是全量微调 + EMA，需要 **~48 GB** 显存，4090 必 OOM。
> `pi0_aloha_sim_low_mem` 是项目里加的 LoRA + 无 EMA 版本（见 `src/openpi/training/config.py`），约 22 GB 能装下。

### 1) 算 norm stats（一次性，~5 分钟）

```bash
cd /home/featurize/work/openpi
sudo apt-get update
sudo apt-get install -y ffmpeg
curl -LsSf https://astral.sh/uv/install.sh | sh
uv run scripts/compute_norm_stats.py --config-name pi0_aloha_sim_low_mem
```

会下载 `lerobot/aloha_sim_transfer_cube_human` 数据集（~1.6 GB），下完算统计量落地到 `assets/pi0_aloha_sim_low_mem/`。

### 2) 启动训练（后台 + 日志落地）

```bash
kill $(cat smoke3k.pid) 2>/dev/null
nohup env XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
  WANDB_MODE=disabled \
  uv run scripts/train.py pi0_aloha_sim_low_mem \
    --exp-name=smoke3k \
    --overwrite \
    --num-train-steps=3000 \
    --batch-size=4 \
  > train_smoke3k.log 2>&1 &
echo $! > smoke3k.pid
```

**关键参数解释**：


| 参数                                   | 含义         | 为什么这么设                       |
| ------------------------------------ | ---------- | ---------------------------- |
| `--batch-size=4`                     | 单步样本数      | LoRA + bs=4 占 ~22 GB；跑稳后可试 8 |
| `--num-train-steps=3000`             | 总迭代步数      | 2h 出可观察的曲线                   |
| `XLA_PYTHON_CLIENT_MEM_FRACTION=0.9` | JAX 占显存比例  | 默认 0.75 不够用，**必须加**          |
| `WANDB_MODE=disabled`                | 关 wandb 上报 | 未登录 wandb 时必加，否则 nohup 后台会崩  |
| `nohup ... &`                        | 后台运行       | SSH 断了也不中断                   |


### 3) 监控

```bash
# 看实时日志
tail -f train_smoke3k.log

# 看 GPU 占用
watch -n 5 nvidia-smi

# 看进度（每 100 步打印一次）
grep -E "step=|loss=" train_smoke3k.log | tail
```

或直接打开 wandb 网页看 loss 曲线。

### 4) 拿到中间 checkpoint 跑仿真评估

⚠️ **必须先在 config 里加 `keep_period=1000`**（已加入 `pi0_aloha_sim_low_mem`），否则只保留最后一份 checkpoint。
源码硬编码 `max_to_keep=1`，永久保留靠 `keep_period`；默认 5000，对 3000 步训练等于全删。

正常情况会有：`checkpoints/pi0_aloha_sim_low_mem/smoke3k/{1000,2000,3000}/`。

**跑评估的实际命令**（aloha_sim/main.py 一次只跑 1 个 episode，要 shell 循环换 seed）：

```bash
# 起 server（如已在跑跳过）
nohup uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config=pi0_aloha_sim_low_mem \
  --policy.dir=checkpoints/pi0_aloha_sim_low_mem/smoke3k/3000 \
  > server_3000.log 2>&1 &
sleep 90

# 跑 10 次评估（不同 seed），视频落地 data/aloha_sim/eval_videos/
mkdir -p data/aloha_sim/eval_videos
for seed in $(seq 0 9); do
  echo "=== seed $seed ==="
  MUJOCO_GL=egl uv run examples/aloha_sim/main.py \
    --args.host 127.0.0.1 --args.seed $seed \
    --args.out-dir data/aloha_sim/eval_videos 2>&1 | tee -a eval.log
done
```

**踩坑提醒**：

- ⚠️ `aloha_sim/main.py` 用 `tyro.cli(main)` 把 `args: Args` 当成子命名空间，**所有参数必须加 `--args.` 前缀**（`--args.host` / `--args.seed` / `--args.out-dir`），不是 `--host` / `--seed`
- `--args.host 0.0.0.0`（默认）在 Featurize 连不通，**必须显式 `--args.host 127.0.0.1`**
- `aloha_sim/main.py` **没有** `--num-episodes` 参数，一次只跑 1 个 episode，多次评估用 shell 循环换 `--args.seed`
- 它**也不输出成功率**，只存视频；要看成败**人眼数视频**，或改 `env.py` 暴露 `_episode_reward` 再 print

### 渲染后端选择

Featurize 容器通常 **没有 `/dev/dri/*` 权限 + 没装 NVIDIA EGL vendor 文件**，`MUJOCO_GL=egl` 会报 `Cannot initialize a headless EGL display`。

**先检查**：

```bash
ls /usr/share/glvnd/egl_vendor.d/
```

- 看到 `10_nvidia.json` → 可用 EGL，前面加 `export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json` 后用 `MUJOCO_GL=egl`
- 只有 `50_mesa.json` → 走下面 OSMesa（Featurize 大多数实例属于这种）

**OSMesa（推荐，CPU 软渲染）**：

完整三步搞定（每步都可能踩一个坑，按顺序做）：

```bash
# 1. 装 OSMesa
sudo apt-get update
sudo apt-get install -y libosmesa6-dev libosmesa6 libgl1-mesa-glx

# 2. 建版本号软链：Ubuntu 22.04 装的是 libOSMesa.so.8，PyOpenGL 硬编码找 .so.0
sudo ln -sf /usr/lib/x86_64-linux-gnu/libOSMesa.so.8 \
            /usr/lib/x86_64-linux-gnu/libOSMesa.so.0
sudo ldconfig

# 3. 启动评估时 LD_PRELOAD 系统 libstdc++
#    （conda 自带的 libstdc++ 太老，没 GLIBCXX_3.4.30，加载 libLLVM-15.so.1 会失败）
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 \
  MUJOCO_GL=osmesa \
  uv run examples/aloha_sim/main.py \
    --args.host 127.0.0.1 --args.seed 0
```

**永久生效**：

```bash
cat >> ~/.zshrc << 'EOF'
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
export MUJOCO_GL=osmesa
EOF
source ~/.zshrc
# 之后跑评估只需：uv run examples/aloha_sim/main.py --args.host 127.0.0.1 --args.seed 0
```

⚠️ OSMesa 比 GPU 渲染慢 3–10 倍，单 episode 30–90 秒，10 个 seed 评估 5–15 分钟。**不影响训练**，只影响仿真画面生成。

---

## 7.4 Phase 2 · Full Run（20000 步，~4–5 小时）

确认 Smoke 阶段一切正常后，可直接在 4090 上跑完（不再需要"过夜"）。

**想冲更高成功率**：把 `--num-train-steps=20000` 改成 `80000`，约 16 小时跑完，补 batch_size=4 的样本数缺口。

### 1) 启动训练

```bash
cd /home/featurize/work/openpi

nohup env XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
  WANDB_MODE=disabled \
  uv run scripts/train.py pi0_aloha_sim_low_mem \
    --exp-name=full20k \
    --overwrite \
    --num-train-steps=20000 \
    --batch-size=4 \
  > train_full20k.log 2>&1 &
echo $! > full20k.pid
```

### 2) 推荐保留 checkpoint 列表

`save_interval=1000` + `keep_period=5000`：

- 每 1000 步存一份临时 checkpoint（会循环覆盖）
- 每 5000 步存一份**永久保留**的 checkpoint（5000 / 10000 / 15000 / 20000）

最终目录预期：

```
checkpoints/pi0_aloha_sim_low_mem/full20k/
├── 5000/
├── 10000/
├── 15000/
├── 19000/   (最近的临时)
└── 20000/
```

### 3) 完整曲线评估

对永久 checkpoint 都跑评估，画出"训练步数 vs 成功率"曲线：

```bash
for step in 5000 10000 15000 20000; do
  nohup uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi0_aloha_sim_low_mem \
    --policy.dir=checkpoints/pi0_aloha_sim_low_mem/full20k/$step \
    > server_$step.log 2>&1 &
  SERVER_PID=$!
  sleep 90

  MUJOCO_GL=egl uv run examples/aloha_sim/main.py \
    --host 127.0.0.1 --num-episodes=50 \
    2>&1 | tee eval_full_$step.log

  kill $SERVER_PID
done
```

`--num-episodes=50` 取代 smoke 阶段的 20，统计上更可信。

---

## 7.5 调参速查（OOM / 太慢 / 想再省时）


| 症状           | 调整                                                                      |
| ------------ | ----------------------------------------------------------------------- |
| **OOM**      | `--batch-size=8`，或进一步 `=4`；仍 OOM 就改 `config.py` 把 `ema_decay` 设为 `None` |
| 想再快          | `--num-train-steps` 减半；但成功率会同比下降                                        |
| 想再准          | `--batch-size=32`（需要 A100/H100）；或多卡 `--fsdp-devices=N`                  |
| 单 step 慢于 5s | 检查 `nvidia-smi` GPU-Util 是否上 80%+；CPU 数据加载瓶颈可调 `--num-workers=4`        |
| loss 不降      | 检查是否漏了 `compute_norm_stats.py`；wandb 看 lr 曲线是否正确 warmup                 |


---

## 7.6 收尾 & 清理

```bash
# 训练结束后，把最终 checkpoint 备份回 work（10GB，注意配额）
cp -r /home/featurize/openpi_checkpoints/pi0_aloha_sim/full20k/20000 \
      ~/work/openpi_my_best/

# 临时日志清理
rm train_*.log server_*.log eval_*.log

# 训练进程兜底 kill
kill $(cat full20k.pid) 2>/dev/null
```

---

## 7.7 实测数据反馈

跑完后欢迎补充实测表（替换我现在的估计）：

| 步数    | 训练耗时        | 评估成功率（10 seed） | 行为定性描述 |
| ----- | ----------- | ---------------- | -------- |
| 3000  | ~40 分钟（实测） | **0/10**（实测）    | **全部主动尝试抓取**（task 结构正确）；夹爪一致性地落在距方块一段距离的桌面，**精度不足** |
| 5000  | —           | —                |          |
| 10000 | —           | —                |          |
| 20000 | ~4 小时（估）   | —                |          |
| 80000 | ~16 小时（估）  | —                |          |

### 质性结论（3000 步 / bs=4 / 10 seed）

**强一致性**：10/10 都"伸手 + 张爪 + 另一臂等待"，0/10 抓到。失败模式高度一致（不是随机乱动），属于**典型欠拟合但方向正确**。

| 学会 | 缺什么 |
|------|--------|
| ✅ 任务目标（伸向方块） | ⚠️ 末端 3D 定位精度 |
| ✅ 动作语义（夹爪张开时机） | |
| ✅ 双臂分工（等待方应等待） | |

→ 增加训练步数能直接见效。Phase 2（20k）保守预计 40–60%。


---

## 下一步

- 想用自己的数据微调 → [06 · 微调流程](./06-finetuning.md)（讲数据格式 + 自定义 config）
- 训练时报错 → [99 · 常见问题](./99-troubleshooting.md)
- 想换 LIBERO 全 benchmark → 见 [06 · 微调流程](./06-finetuning.md#62-数据转换)（注意需 ~60 GB 数据 + 20–40h 训练）

