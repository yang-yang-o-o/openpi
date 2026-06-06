# 01 · 环境准备

> 前置：拿到一台 Featurize 4090 实例（或同等配置），代码已 `git clone` 到 `~/work/openpi`。
>
> 目标：装好 uv、配好 PyPI 镜像、拉好子模块，为下一步 `uv sync` 做准备。

---

## 1.1 硬件与适用范围

参考实例：

- GPU: **NVIDIA GeForce RTX 4090**，24 GB
- CUDA Driver: 580.95.05 / CUDA 13.0（系统自带，不要再装）
- OS: Ubuntu 22.04（OpenPI 官方支持）
- Python: 3.11.8（conda base）

| 模式 | 显存需求 | 4090 可用？ |
|------|----------|-------------|
| 推理 | > 8 GB | ✅ |
| LoRA 微调 | > 22.5 GB | ✅（接近上限，慎用） |
| 全量微调 | > 70 GB | ❌（需 A100/H100） |

---

## 1.2 Featurize 存储路径约定

| 目录 | 性能 | 用途 |
|------|------|------|
| `/home/featurize/work` | **慢**（云同步盘，但跨实例持久） | **仅放代码** |
| `/home/featurize/data` | 快（本地盘，实例销毁清空） | 数据集 |
| 其他本地目录 | 快（实例销毁清空） | checkpoint 缓存、临时文件 |

### 建议

- **代码**：放 `~/work/openpi`（已在此）
- **checkpoint / HF / uv 缓存**：用环境变量指向本地盘，避免拖慢训练、**避免撑爆 work 配额**：

  ```bash
  cat >> ~/.zshrc << 'EOF'
  export OPENPI_DATA_HOME=/home/featurize/openpi_cache
  export HF_HOME=/home/featurize/hf_cache
  export HF_DATASETS_CACHE=/home/featurize/hf_cache/datasets
  export HF_HUB_DISABLE_XET=1   # 关 Xet，省一半磁盘
  export HF_ENDPOINT=https://hf-mirror.com  # 国内镜像加速
  EOF
  source ~/.zshrc
  ```

- `~/work` 配额通常 **30 GB**，超出会计费。用 `du -sh ~/work/` 查用量。
- `.venv` 也较大（5–10 GB），建议软链到本地盘：

  ```bash
  # 没装过 .venv 时执行；已装则先 rm -rf .venv
  mkdir -p /home/featurize/openpi_venv
  ln -s /home/featurize/openpi_venv /home/featurize/work/openpi/.venv
  ```

### 数据集大小预估（避免下载惊吓）

| 数据集 | 文件数 | 总大小 | 用 Xet 默认占用 |
|--------|--------|--------|----------------|
| LIBERO（`physical-intelligence/libero`） | ~1699 parquet | **~60 GB** | **~120 GB** |
| DROID 子集 | 视具体 config | 数十 GB | 翻倍 |
| OpenPI checkpoint (`pi05_droid`) | 1 个 | 11.6 GB | — |

→ **首次下载前务必配好 `HF_HOME` 指向本地盘 + 关 Xet**，否则会偷偷占 100GB+。

---

## 1.3 安装 uv

OpenPI 使用 [uv](https://docs.astral.sh/uv/) 管理 Python 依赖。

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.zshrc   # 或 ~/.bashrc
uv --version       # 期望 ≥ 0.11
```

---

## 1.4 配置 PyPI 镜像（关键！）

### 推荐：腾讯云

Featurize 实测最快：

```bash
export UV_INDEX_URL=https://mirrors.cloud.tencent.com/pypi/simple/
```

### 镜像对比

| 镜像 | uv 兼容性 | Featurize 实测速度 |
|------|-----------|-------------------|
| **腾讯云** `https://mirrors.cloud.tencent.com/pypi/simple/` | ✅ | ⭐⭐⭐⭐⭐ 推荐 |
| 阿里云 `https://mirrors.aliyun.com/pypi/simple/` | ✅ | ⭐⭐⭐⭐ |
| 中科大 `https://pypi.mirrors.ustc.edu.cn/simple/` | ✅ | ⭐⭐⭐ |
| **清华 HTTPS** `https://pypi.tuna.tsinghua.edu.cn/simple` | ❌ `invalid Content-Range header` | — |
| 官方 PyPI | ✅ | ⭐ 很慢 |

### ⚠️ 清华源踩坑

清华 HTTPS 镜像 + uv 的分片下载（HTTP Range）请求不兼容，会在某些小 wheel（如 `tensorboard-data-server`）上报：

```
error: Failed to unzip wheel: tensorboard_data_server-0.7.2-py3-none-any.whl
  Caused by: invalid Content-Range header: bytes 18446744073709543972-...
```

这是清华 HTTPS + uv 的已知 bug（非 OpenPI 问题）。**不要用清华 HTTPS 源。** 必须用就改 HTTP（不推荐）：

```bash
export UV_INDEX_URL=http://pypi.tuna.tsinghua.edu.cn/simple
```

### 永久生效

```bash
cat >> ~/.zshrc << 'EOF'
export UV_INDEX_URL=https://mirrors.cloud.tencent.com/pypi/simple/
export GIT_LFS_SKIP_SMUDGE=1
EOF
source ~/.zshrc
```

### 换源前清缓存

之前下载失败的 wheel 可能已损坏：

```bash
uv cache clean
rm -rf /home/featurize/work/openpi/.venv
```

---

## 1.5 拉取子模块

OpenPI 依赖两个 git 子模块：
- `third_party/aloha`（~55 MB）
- `third_party/libero`（~293 MB）

### 是否必须拉？

| 用途 | 是否需要 |
|------|---------|
| DROID 推理 / 训练 | ❌ 可跳过 |
| LIBERO 仿真评估 | ✅ 需要 |
| ALOHA 真机 | ✅ 需要 |

### 推荐命令（带进度 + 浅克隆）

```bash
cd /home/featurize/work/openpi
GIT_PROGRESS=1 git submodule update --init --recursive --depth 1 --progress
```

`--depth 1` 只拉最新提交，速度更快、体积更小，OpenPI 使用足够。

### 若之前中断过，先清理

```bash
cd /home/featurize/work/openpi
git submodule deinit -f third_party/aloha third_party/libero
rm -rf third_party/aloha third_party/libero \
       .git/modules/third_party/aloha .git/modules/third_party/libero

GIT_PROGRESS=1 git submodule update --init --recursive --depth 1 --progress
```

### 看起来卡住？

- 默认 `git submodule update` 不显示进度，libero 仓库较大，需 3–5 分钟，**用 `--progress` 才能看到**
- Featurize 拉 GitHub 速度 1–8 MB/s 波动正常
- 若仍很慢，配置 ghproxy：

  ```bash
  git config --global url."https://ghproxy.com/https://github.com/".insteadOf "https://github.com/"
  ```

---

## 下一步

环境准备完毕 → [02 · 依赖安装](./02-installation.md)
