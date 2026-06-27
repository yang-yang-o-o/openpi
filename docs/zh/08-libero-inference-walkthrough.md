# 08 · LIBERO 推理实战（π₀.₅ · 对论文 + 看视频）

> 适用：已完成 [02 · 依赖安装](./02-installation.md)、[03 · 推理快速上手](./03-inference-quickstart.md)，想在云上跑 **LIBERO benchmark** 闭环推理。
>
> 目标：加载 `pi05_libero` → MuJoCo 仿真 rollout → 看成功率与录像，可与论文数字对照。

---

## 8.1 和 DROID simple_client 的区别

| | DROID `simple_client` | LIBERO `examples/libero/main.py` |
|--|----------------------|----------------------------------|
| 观测来源 | 随机假数据 | **MuJoCo 仿真真环境** |
| 动作执行 | 无（只测延迟） | **逐步 step 进仿真** |
| 能否对论文 | ❌ | ✅（官方表见 §8.8） |
| Python 环境 | 主 `uv` 环境 | **单独 Python 3.8 venv** |
| 额外权重 | `pi05_droid` ~11.6 GB | `pi05_libero` ~12 GB（需另下） |

LIBERO 是 **两终端 + 双环境** 架构：

```
终端 1（主 uv 环境）  serve_policy.py  ← GPU 推理
终端 2（libero .venv） examples/libero/main.py  ← CPU/GPU 仿真 + websocket 客户端
```

---

## 8.2 预检清单

```bash
cd /home/featurize/work/openpi

# ① 子模块（推理需要 third_party/libero）
git submodule update --init --recursive --depth 1 third_party/libero

# ② 确认 submodule 在
test -d third_party/libero/libero/libero/bddl_files && echo "LIBERO submodule OK"

# ③ 权重目录（与 03 文档一致，自行管理）
ls openpi_cache/openpi-assets/checkpoints/pi05_libero 2>/dev/null \
  || echo "尚未下载 pi05_libero，首次启动 server 会自动拉取（见 §8.5）"
```

---

## 8.3 一次性准备

### ① LIBERO 路径配置（避免交互式提问）

首次 `import libero` 会往 `~/.libero/config.yaml` 写配置并 **阻塞等待输入**。云上请预先写好：

```bash
cd /home/featurize/work/openpi
mkdir -p .libero

cat > .libero/config.yaml << 'EOF'
benchmark_root: /home/featurize/work/openpi/third_party/libero/libero/libero
bddl_files: /home/featurize/work/openpi/third_party/libero/libero/libero/bddl_files
init_states: /home/featurize/work/openpi/third_party/libero/libero/libero/init_files
datasets: /home/featurize/work/openpi/third_party/libero/libero/datasets
assets: /home/featurize/work/openpi/third_party/libero/libero/libero/assets
EOF

# 写入 shell 配置（推理 client 终端需要）
grep -q 'LIBERO_CONFIG_PATH' ~/.zshrc 2>/dev/null || cat >> ~/.zshrc << 'EOF'
export LIBERO_CONFIG_PATH=/home/featurize/work/openpi/.libero
EOF
```

> `datasets` 路径在**纯推理评估**时不会用到（没有训练数据也能跑 benchmark），若出现 `[Warning]: datasets path ... does not exist` 可忽略。

### ② 创建 LIBERO 客户端虚拟环境（Python 3.8）

LIBERO / robosuite 依赖较老，**不能**和主 OpenPI 环境混用。

```bash
cd /home/featurize/work/openpi
uv venv --python 3.8 examples/libero/.venv
source examples/libero/.venv/bin/activate
```

#### 装依赖：Featurize 踩坑汇总

> **三个硬性约定**（后面所有 `pip` 命令都适用）：
>
> 1. **一律 `python -m pip`**，不要用裸 `pip`——Featurize 默认 `pip` 可能指向 conda 3.11，cp38 wheel 会报 `not supported on this platform`。
> 2. **先 `unset all_proxy ALL_PROXY`**——实例常带 socks5 代理，会导致 pip 连国内镜像超时。
> 3. **不要 `pip install -r examples/libero/requirements.txt`**——该文件 pin 了 `torch==1.11.0+cu113`，国内 PyPI 没有；且文件是 uv 编译产物，含大量缩进注释行（`    # via mujoco`），用 `grep -v '^#'` 过滤后会报 `Invalid requirement: '#'`。

`examples/libero/requirements.txt` 过滤辅助函数（下文复用）：

```bash
filter_reqs() {
  grep -E '^[a-zA-Z0-9]' "$1" | grep -vE '^(torch|torchvision|torchaudio)'
}
```

---

**方案 A · CPU 版 torch（推荐，Featurize 已验证）**

LIBERO **仿真 client 不做 GPU 推理**（模型在终端 1 的 OpenPI server 上跑），client 里 `import torch` 只为满足依赖，**CPU 版即可**（~750 MB，比 cu113 的 1.6 GB 小很多）：

```bash
cd /home/featurize/work/openpi
source examples/libero/.venv/bin/activate
unset all_proxy ALL_PROXY

PIP_MIRROR="-i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com"

# ① 编译 egl_probe 需要 cmake
sudo apt-get install -y cmake build-essential

# ② torch 三件套：从阿里云 pytorch-wheels 下 wheel（带进度条，可断点续传）
mkdir -p ~/data/pytorch-cpu && cd ~/data/pytorch-cpu
wget -c --progress=bar:force \
  https://mirrors.aliyun.com/pytorch-wheels/cpu/torch-1.11.0%2Bcpu-cp38-cp38-linux_x86_64.whl
wget -c --progress=bar:force \
  https://mirrors.aliyun.com/pytorch-wheels/cpu/torchvision-0.12.0%2Bcpu-cp38-cp38-linux_x86_64.whl
wget -c --progress=bar:force \
  https://mirrors.aliyun.com/pytorch-wheels/cpu/torchaudio-0.11.0%2Bcpu-cp38-cp38-linux_x86_64.whl

cd /home/featurize/work/openpi
source examples/libero/.venv/bin/activate
python -m pip install ~/data/pytorch-cpu/torch-1.11.0+cpu-*.whl \
  ~/data/pytorch-cpu/torchvision-0.12.0+cpu-*.whl \
  ~/data/pytorch-cpu/torchaudio-0.11.0+cpu-*.whl

# ③ examples/libero 其余依赖（过滤 torch 三件套 + 缩进注释行）
filter_reqs examples/libero/requirements.txt \
  | xargs python -m pip install $PIP_MIRROR

# ④ third_party/libero 额外依赖（bddl / future / cloudpickle 等；robosuite 单独 pin）
python -m pip install egl_probe bddl==1.0.1 future $PIP_MIRROR
sed 's/^[[:space:]]*//' third_party/libero/requirements.txt \
  | grep -vE '^(#|robosuite)' | grep -v '^$' \
  | xargs python -m pip install $PIP_MIRROR
python -m pip install 'robosuite==1.4.1' $PIP_MIRROR   # 必须 1.4.1，否则会拉到 1.5.x

# ⑤ openpi-client + libero 本体
python -m pip install -e packages/openpi-client -e third_party/libero
```

---

**方案 B · 对齐官方 cu113（可选，体积大）**

与 `requirements.txt` pin 一致，torch 仍 ~1.6 GB，走阿里云 CDN：

```bash
mkdir -p ~/data/pytorch-cu113 && cd ~/data/pytorch-cu113
wget -c --progress=bar:force \
  https://mirrors.aliyun.com/pytorch-wheels/cu113/torch-1.11.0%2Bcu113-cp38-cp38-linux_x86_64.whl
wget -c --progress=bar:force \
  https://mirrors.aliyun.com/pytorch-wheels/cu113/torchvision-0.12.0%2Bcu113-cp38-cp38-linux_x86_64.whl
wget -c --progress=bar:force \
  https://mirrors.aliyun.com/pytorch-wheels/cu113/torchaudio-0.11.0%2Bcu113-cp38-cp38-linux_x86_64.whl

cd /home/featurize/work/openpi
source examples/libero/.venv/bin/activate
unset all_proxy ALL_PROXY
python -m pip install ~/data/pytorch-cu113/torch-1.11.0+cu113-*.whl \
  ~/data/pytorch-cu113/torchvision-0.12.0+cu113-*.whl \
  ~/data/pytorch-cu113/torchaudio-0.11.0+cu113-*.whl

# 其余步骤同方案 A 的 ③–⑤
```

| 镜像 | 用途 |
|------|------|
| `https://mirrors.aliyun.com/pypi/simple/` | 普通 PyPI 包 |
| `https://mirrors.aliyun.com/pytorch-wheels/cpu/` | CPU 版 `torch*` wheel（方案 A） |
| `https://mirrors.aliyun.com/pytorch-wheels/cu113/` | cu113 版 `torch*` wheel（方案 B） |

---

```bash
export PYTHONPATH=$PYTHONPATH:$PWD/third_party/libero
export LIBERO_CONFIG_PATH=/home/featurize/work/openpi/.libero
export MUJOCO_GL=osmesa
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
```

验证（需先完成 §8.3③ OSMesa 安装）：

```bash
python -c "
from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv
print('ALL OK')
"
```

### ③ Featurize 无头渲染（OSMesa）

与 [07 · ALOHA Sim 评估](./07-4090-finetune-walkthrough.md#74-仿真评估踩坑-featurize-专版) 相同：容器通常没有 DRI/EGL，用 OSMesa CPU 软渲染。

```bash
# 必须先 update，否则 mesa 包可能 404（Featurize 上常见）
sudo apt-get update
sudo apt-get install -y libosmesa6-dev libosmesa6 libgl1-mesa-glx

# PyOpenGL 找 libOSMesa.so.0，Ubuntu 22.04 实际是 .so.8
sudo ln -sf /usr/lib/x86_64-linux-gnu/libOSMesa.so.8 \
            /usr/lib/x86_64-linux-gnu/libOSMesa.so.0
sudo ldconfig

# conda 自带 libstdc++ 过旧时（ImportError / GL 相关崩溃）
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
export MUJOCO_GL=osmesa
```

建议把 `MUJOCO_GL` / `LD_PRELOAD` / `LIBERO_CONFIG_PATH` 一并写入 `~/.zshrc`。

---

## 8.4 Smoke 测试（约 10–15 分钟）

先跑 **`libero_spatial` 全部 10 个 task，每个 task 1 次 trial**（共 10 条 episode），确认 server ↔ 仿真 ↔ 录像全链路。

> Featurize 4090 实测（2025-06）：10/10 成功，总耗时 **~9 分钟**（平均每 episode **~50 秒**，含 OSMesa 渲染 + websocket 推理）。首次 `infer` 可能额外等 **30–60 秒**（JAX 编译），属正常。

### 终端 1 — 策略服务（主 uv 环境）

```bash
cd /home/featurize/work/openpi
export OPENPI_DATA_HOME=/home/featurize/work/openpi/openpi_cache

# 方式 A：本地权重（推荐，跳过 gs:// 自动下载）
uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config=pi05_libero \
  --policy.dir=/home/featurize/work/openpi/openpi_cache/openpi-assets/checkpoints/pi05_libero
```

若 `pi05_libero` 尚未下载，可临时：

```bash
export OPENPI_DATA_HOME=/home/featurize/work/openpi/openpi_cache
uv run scripts/serve_policy.py --env LIBERO   # 仅首次拉权重，下完改回方式 A
```

成功标志：`Restoring checkpoint from .../pi05_libero/params`，`server listening on 0.0.0.0:8000`。

> ⚠️ **server 必须前台常驻运行**：
>
> - 不要用 **Ctrl+Z** 挂起——进程会 `suspended`，client 连上 8000 后 `infer()` 会一直卡在 `recv()`。
> - 终端 2 用 libero venv 前，终端 1 建议 **`conda deactivate`**，避免环境变量干扰。
> - 用 `ss -tlnp | grep 8000` 确认 python 在 **LISTEN**，且状态不是 `T`（stopped）。

### 终端 2 — LIBERO 仿真客户端（`examples/libero/.venv`）

```bash
cd /home/featurize/work/openpi
source examples/libero/.venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$PWD/third_party/libero
export LIBERO_CONFIG_PATH=/home/featurize/work/openpi/.libero
export MUJOCO_GL=osmesa
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6

# Smoke：spatial 套件 10 个 task，每个 1 次 trial
python examples/libero/main.py \
  --args.host 127.0.0.1 \
  --args.task-suite-name libero_spatial \
  --args.num-trials-per-task 1
```

> ⚠️ **必须** `--args.host 127.0.0.1`（默认 `0.0.0.0` 在 Featurize 连不上本机 server，见 [03 §3.3](./03-inference-quickstart.md#33--必须显式加-host-127001)）。

每个 task 开始时会打印 `[Warning]: datasets path ... does not exist!`——**纯推理可忽略**，不影响 rollout（Featurize 实测 10/10 均在此警告下成功）。

### 预期日志（成功）

```text
INFO:root:Task suite: libero_spatial
INFO:root:Waiting for server at ws://127.0.0.1:8000...
INFO:root:Task: pick up the black bowl between the plate and the ramekin and place it on the plate
INFO:root:Starting episode 1...
INFO:root:Success: True
INFO:root:# successes: 1 (100.0%)
...
INFO:root:Total success rate: 1.0
INFO:root:Total episodes: 10
```

### 预期产出

| 产出 | 位置 / 说明 |
|------|-------------|
| 成功率 | 日志末尾 `Total success rate`；Smoke **不能**直接对标论文（见 §8.7） |
| 成功录像 | `data/libero/videos/rollout_<任务英文描述>_success.mp4` |
| 失败录像 | `..._failure.mp4`（若某 episode 未达成任务） |

`libero_spatial` Smoke 成功时应有 **10 个 `_success.mp4`**（每个 task 一条），体积约 **50–75 KB**（agentview 低帧率）。在 IDE 或下载到本地直接播放即可。

**Smoke vs 论文**：论文 `libero_spatial` 为 **98.8%**（每 task **50 trials**）。Smoke 的 10/10 只验证链路通了、策略能工作；要对论文需跑默认 `num-trials-per-task=50`。

---

## 8.5 权重与 tokenizer

| 资源 | 路径 |
|------|------|
| checkpoint | `openpi_cache/openpi-assets/checkpoints/pi05_libero/` |
| tokenizer | `openpi_cache/big_vision/paligemma_tokenizer.model` |

方式 A 启动时 checkpoint 不走网络；tokenizer 若缺失仍可能从 `gs://big_vision/...` 下到默认缓存。可预先：

```bash
export OPENPI_DATA_HOME=/home/featurize/work/openpi/openpi_cache
# 任意一次加载 pi05_libero 会把 tokenizer 也落到 openpi_cache
```

或从已有 `~/.cache/openpi/big_vision/` 拷到 `openpi_cache/big_vision/`。

---

## 8.6 推理闭环在做什么（读代码用）

`examples/libero/main.py` 核心循环：

1. 按 task suite 遍历每个 LIBERO 任务与初始状态
2. 前 `num_steps_wait` 步发 dummy action，等物体落稳
3. 从仿真取 **agentview + wrist** 图像，**旋转 180°**（与训练预处理一致）
4. 拼 `observation/state`（eef pos + axis-angle quat + gripper）
5. 通过 websocket 调 server 的 `policy.infer()`，拿到 **action chunk**
6. 每 `replan_steps=5` 步重新规划一次（默认）
7. `env.step(action)` 执行 7 维动作，直到 `done` 或超时
8. 把 agentview 帧写成 mp4

观测 → 模型的映射在 `src/openpi/policies/libero_policy.py`（`LiberoInputs` / `LiberoOutputs`）。

想看每步输入输出，server 加 `--record`，再用 `examples/policy_records.ipynb` 分析（与 DROID 相同）。

---

## 8.7 完整 benchmark（对论文数字）

官方评估配置：`libero_spatial` 等每个 task **50 trials**（`num_trials_per_task=50`，默认即是）。

```bash
# 终端 2（完整 spatial，10 tasks × 50 trials，耗时长）
python examples/libero/main.py \
  --args.host 127.0.0.1 \
  --args.task-suite-name libero_spatial
```

其他套件：`libero_object`、`libero_goal`、`libero_10`、`libero_90`。

| Suite | task 数 | 默认 trials | 论文 π₀.₅ 成功率 |
|-------|--------|------------|-----------------|
| libero_spatial | 10 | 50 | 98.8% |
| libero_object | 10 | 50 | 98.2% |
| libero_goal | 10 | 50 | 98.0% |
| libero_10 | 10 | 50 | 92.4% |
| **平均** | | | **96.85%** |

完整 4 套件 × 50 trials 在 OSMesa 上可能要 **数小时到一天**（Smoke 10 episode 约 9 分钟，可按比例估算单个 suite 全量约 **4–5 小时**）；建议先 Smoke，再 overnight 跑单个 suite。

---

## 8.8 参数速查

| 参数 | 默认 | 说明 |
|------|------|------|
| `--args.host` | `0.0.0.0` | Featurize 上改为 **`127.0.0.1`** |
| `--args.task-suite-name` | `libero_spatial` | 任务套件名 |
| `--args.num-trials-per-task` | `50` | Smoke 用 `1` |
| `--args.replan-steps` | `5` | 每 N 步重新向模型要 action chunk |
| `--args.video-out-path` | `data/libero/videos` | rollout 录像输出目录 |
| `--args.seed` | `7` | 随机种子 |

---

## 8.9 常见问题

| 现象 | 处理 |
|------|------|
| `Invalid requirement: '#'` | `requirements.txt` 有缩进注释行，不能用 `grep -v '^#'`；用 `grep -E '^[a-zA-Z0-9]'`（§8.3②） |
| `wheel is not supported on this platform` | 改用 `python -m pip`（venv 3.8），不要用裸 `pip`（可能走 conda 3.11） |
| pip 连镜像超时 | `unset all_proxy ALL_PROXY` |
| `No matching distribution found for torch==1.11.0+cu113` | 国内 PyPI 无此包；CPU 版走方案 A，或从 `mirrors.aliyun.com/pytorch-wheels/` 下 wheel |
| `No module named 'future'` / `easydict` / `cloudpickle` | 补装 `third_party/libero/requirements.txt`（§8.3② 步骤 ④） |
| `egl_probe` 编译失败 | `sudo apt-get install -y cmake build-essential` |
| `pip install yaml` 失败 | PyPI 包名是 **`PyYAML`**，不是 `yaml` |
| robosuite 行为异常 | 强制 `robosuite==1.4.1`（requirements 里未 pin 版本会拉到 1.5.x） |
| apt 装 OSMesa 报 404 | 先 `sudo apt-get update` 再装（§8.3③） |
| client `infer()` 卡在 `recv()` / `KeyboardInterrupt` | ① server 被 **Ctrl+Z** 挂起 → `fg` 恢复或重启 server；② 首次 infer 等 JAX 编译（30–60s）；③ 确认 `pi05_libero` 已加载完成 |
| client `InvalidMessage` / `EOFError` | `--args.host 127.0.0.1` |
| `Do you want to specify a custom path...` 卡住 | 设好 `LIBERO_CONFIG_PATH` + `config.yaml`（§8.3①） |
| EGL / `/dev/dri` 报错 | `MUJOCO_GL=osmesa` + OSMesa 依赖（§8.3③） |
| `libOSMesa.so.0` not found | 建 `.so.8` → `.so.0` 软链 |
| `libstdc++.so.6` / GL 崩溃 / `glGetError` | `LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6` + `MUJOCO_GL=osmesa` |
| server 又在下 11 GB | 改用 `policy:checkpoint` + 本地 `--policy.dir`；`pi05_libero` 与 `pi05_droid` 是**不同**权重 |
| `datasets path does not exist` 警告 | 纯推理可忽略；Featurize Smoke 10/10 在此警告下正常完成 |

更多见 [99 · 常见问题](./99-troubleshooting.md)。

---

## 8.10 Docker 方案（可选）

若本机 venv 依赖冲突严重，可用官方 Docker（需 GPU + 常需 X11）：

```bash
sudo xhost +local:docker
SERVER_ARGS="policy:checkpoint --policy.config pi05_libero --policy.dir /home/featurize/work/openpi/openpi_cache/openpi-assets/checkpoints/pi05_libero" \
CLIENT_ARGS="--args.host 127.0.0.1 --args.num-trials-per-task 1" \
  docker compose -f examples/libero/compose.yml up --build
```

Featurize 上 Docker + EGL 同样可能踩坑；**无 Docker 的 OSMesa 方案通常更省事**。

---

## 下一步

- 理解 LIBERO 在 4 env 中的定位 → [04 · 模型与环境选择](./04-models-and-envs.md)
- 想在 LIBERO 上微调自己的 ckpt → [06 · 微调流程](./06-finetuning.md)
- 只想快速验证 GPU 推理、不看仿真 → [03 · 推理快速上手](./03-inference-quickstart.md)
