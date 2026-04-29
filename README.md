# MTHFVRP / VaP-CSMV Research Codebase

[English](#english) | [中文](#中文)

## English

### Overview

This repository provides research code for solving the Heterogeneous Fleet Vehicle Routing Problem (HFVRP) and its constrained variants with Deep Reinforcement Learning.

### Repository Structure

| Folder | Description |
| :--- | :--- |
| `framework/` | Generic abstractions for environments, algorithms, models, and utilities. |
| `implement/` | Core implementation of the paper method: generator, environment, model, RL algorithm, trainer, and evaluator. |
| `quickstarts/` | Jupyter notebooks for walkthroughs and quick onboarding. |
| `train.py` | Main training entrypoint (env/model/algo/trainer assembly). |
| `requirements.txt` | Dependency list. |

### Key Modules

- `implement/generator.py`: instance generation and `variant_preset` configuration.
- `implement/environment.py`: transition logic, legal action masking, and reward-related flow.
- `implement/model.py`: Transformer-based policy network.
- `implement/reinforce_alg.py`: REINFORCE training loop and baseline/sampling strategy.
- `implement/evaluation.py`: inference-time evaluation and metric aggregation.
- `implement/trainer.py`: callbacks, validation schedule, checkpoints, and logging.

### Variant Support

Configured via `variant_preset`, including (or extendable to):

- HF: Heterogeneous Fleet
- TW: Time Windows
- O: Open Routes
- L: Distance Limits
- B: Backhaul

### Installation and Usage

1. Recommended environment: Python 3.9+, PyTorch 2.x, CUDA (optional).
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run training:

```bash
python train.py
```

4. Suggested notebook reading order:

- `quickstarts/data_generation_guide.ipynb`
- `quickstarts/environment_guide.ipynb`
- `quickstarts/model_architecture_guide.ipynb`
- `quickstarts/train.ipynb`
- `quickstarts/eval.ipynb`

## 中文

### 项目简介

本仓库是面向异构车队车辆路径问题（HFVRP）及其复杂变体的深度强化学习研究代码。

### 目录结构

| 目录 | 说明 |
| :--- | :--- |
| `framework/` | 通用抽象层：环境、算法、模型与工具基类。 |
| `implement/` | 论文方法的核心实现：数据生成、环境、模型、算法、训练与评估。 |
| `quickstarts/` | Notebook 形式的快速上手与模块讲解。 |
| `train.py` | 训练入口脚本（含环境构建、模型组装、Trainer 配置）。 |
| `requirements.txt` | 依赖列表。 |

### 主要模块

- `implement/generator.py`: 问题实例生成与变体配置（`variant_preset`）。
- `implement/environment.py`: 环境状态转移、合法动作掩码、奖励相关逻辑。
- `implement/model.py`: 基于 Transformer 的策略网络实现。
- `implement/reinforce_alg.py`: REINFORCE 训练流程与基线/采样策略。
- `implement/evaluation.py`: 推理评估与指标统计。
- `implement/trainer.py`: 训练回调、验证、checkpoint 与日志管理。

### 支持的变体

通过 `variant_preset` 进行组合配置，覆盖（或可扩展到）以下典型约束：

- HF: Heterogeneous Fleet（异构车队）
- TW: Time Window（时间窗）
- O: Open Route（开放路径）
- L: Distance Limit（里程限制）
- B: Backhaul（回程取货）

### 安装与运行

1. 建议环境：Python 3.9+，PyTorch 2.x，CUDA（可选）。
2. 安装依赖：

```bash
pip install -r requirements.txt
```

3. 启动训练：

```bash
python train.py
```

4. 快速理解代码（建议顺序）：

- `quickstarts/data_generation_guide.ipynb`
- `quickstarts/environment_guide.ipynb`
- `quickstarts/model_architecture_guide.ipynb`
- `quickstarts/train.ipynb`
- `quickstarts/eval.ipynb`
