# OpenPI 中文文档（Featurize 4090 实践版）

基于 Featurize 平台 RTX 4090 (24GB) 实例，记录 OpenPI 项目从零部署、推理、微调的完整流程，沉淀踩坑与解决方案。

---

## 阅读顺序

| 编号 | 文档 | 适用场景 |
|------|------|---------|
| 01 | [环境准备](./01-environment-setup.md) | 第一次拿到服务器，配 uv / 镜像 / 子模块 |
| 02 | [依赖安装](./02-installation.md) | `uv sync` 装 OpenPI 主体依赖 |
| 03 | [推理快速上手](./03-inference-quickstart.md) | 启 server + 跑 simple_client 验证 |
| 04 | [模型与环境选择](./04-models-and-envs.md) | 4 个 EnvMode、完整 config 列表、怎么选 |
| 05 | [PyTorch vs JAX](./05-pytorch-vs-jax.md) | 两套实现差异，4090 用户应该选哪个 |
| 06 | [微调流程](./06-finetuning.md) | 自己数据微调 + 显存优化（通用） |
| 07 | [**4090 微调全流程实战**](./07-4090-finetune-walkthrough.md) | **ALOHA Sim · 2h smoke + 过夜 20k，含评估脚本** |
| 99 | [常见问题](./99-troubleshooting.md) | 全量踩坑速查表 |

---

## 场景速查

> **我想……**

- **跑通推理验证** → 01 → 02 → 03
- **4090 上 2 小时跑通微调** → 01 → 02 → **07** ⭐
- **跑标准 benchmark（LIBERO）** → 01 → 02 → 04（选 LIBERO） → 06
- **自己数据微调** → 01 → 02 → 06
- **接真机（DROID/ALOHA）** → 01 → 02 → 03 → 04（看对应 env）
- **遇到错误** → 99

---

## 文档之外

- 项目根目录 [`setup.md`](../../setup.md)：4 步极简安装速查
- 官方英文文档：[`docs/`](../)（`docker.md`、`norm_stats.md`、`remote_inference.md`）
- 上游 README：[`README.md`](../../README.md)
