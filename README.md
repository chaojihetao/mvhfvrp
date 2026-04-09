# Multi-Type VRP (MTHFVRP) Deep Reinforcement Learning Framework

本项目是一个基于深度强化学习 (Deep RL) 解决**多类型异构车队车辆路径问题 (MTHFVRP)** 的研究框架。框架采用了基于 Transformer 的注意力模型 (AM) 和 REINFORCE 算法，支持多种 VRP 变体（如 CVRP, VRPTW, OVRP, HFVRP 等）的组合求解。

## 1. 项目架构 (Project Architecture)

项目采用分层架构设计，将通用框架与具体实现分离，便于扩展新的算法或问题类型。

### 目录结构说明

| 目录 | 说明 | 核心文件 |
| :--- | :--- | :--- |
| `framework/` | **通用基础设施**。定义了 RL 算法、环境和数据生成的抽象基类，以及通用的工具函数（日志、IO 等）。 | `env.py`, `alg/rlbase.py` |
| `implement/` | **具体业务逻辑**。针对 MTHFVRP 问题的具体实现，包括状态转移逻辑、神经网络架构、REINFORCE 算法细节等。 | `environment.py`, `model.py`, `trainer.py` |
| `examples/` | **生产级脚本**。包含用于长期训练和大型评估的脚本。 | `train/`, `eval/` |
| `quickstarts/` | **交互式教程**。用于快速理解代码逻辑的 Jupyter Notebooks。 | `data_gen.ipynb`, `environment_guide.ipynb` |
| `data/` | **数据存储**。存放生成的数据集。 | `data_genrator.py` |

---

## 2. 快速入门 (Quickstart)

我们准备了一系列 Jupyter Notebook 帮助你深入理解本项目的各个组件。建议按以下顺序阅读：

### 📚 1. 数据生成 ([data_generation_guide.ipynb](quickstarts/data_generation_guide.ipynb))
*   **内容**: 学习如何使用 `MTHFVRPGenerator` 生成不同变体的数据（如 VRPTW, HFCVRP）。
*   **关键点**: 理解 `variant_preset` 参数，学习数据保存格式 (JSONL)。

### 🎮 2. 环境交互 ([environment_guide.ipynb](quickstarts/environment_guide.ipynb))
*   **内容**: 深入 `MTHFVRPEnv` 内部。
*   **关键点**: 学习 `reset` 和 `step` 方法，理解 `legal_action_mask` 的计算逻辑，以及如何解析和可视化环境输出的路径。

### 🧠 3. 模型架构 ([model_architecture_guide.ipynb](quickstarts/model_architecture_guide.ipynb))
*   **内容**: 详细解析 Transformer 模型结构。
*   **关键点**: 理解 Encoder-Decoder 架构，掌握 Global Feature 与 Current Context Feature 的输入流向。

### 🚀 4. 训练流程 ([train.ipynb](quickstarts/train.ipynb))
*   **内容**: 完整的训练 Pipeline 演示。
*   **关键点**: 结合 Lightning 框架，组装 Environment, Model 和 Algorithm，启动一个 Mini-Training 任务。

### 📊 5. 模型评估 ([eval.ipynb](quickstarts/eval.ipynb))
*   **内容**: 加载训练好的模型进行推理。
*   **关键点**: 演示多起点 (Multi-Start) 采样评估策略，计算测试集上的平均奖励。

---

## 3. 功能特性 (Key Features)

### 🧩 支持多种 VRP 变体
通过 `variant_preset` 参数，可以灵活组合以下约束特征：
*   **HF (Heterogeneous Fleet)**: 异构车队（不同容量、成本）。
*   **TW (Time Window)**: 节点服务时间窗。
*   **O (Open Route)**: 开放式路径（车辆不回车场）。
*   **L (Distance Limit)**: 车辆最大行驶距离限制。
*   **B (Backhaul)**: 回程取货需求。

### ⚡ 高效的训练策略
*   **RL Algorithm**: 改进的 REINFORCE 算法，带有 Baseline 减小方差。
*   **Entropy & Covariance Control**: 引入协方差控制策略 (Covariance-based Strategy) 动态调整探索力度。
*   **Multi-GPU Support**: 基于 PyTorch Lightning，原生支持 DDP 分布式训练。

### 🛠️ 完善的工具链
*   **Tensordict 集成**: 全面使用 `tensordict` 管理状态数据，简化了 Batch 处理和设备管理。
*   **自动化评估**: `trainer.py` 中集成了详细的 `MyCallback`，支持验证集定期评估、模型保存和早停机制。

## 4. 安装 (Installation)

1.  **环境要求**: Linux, Python 3.9+, PyTorch 2.0+
2.  **依赖安装**:
    ```bash
    pip install torch tensordict lightning matplotlib tqdm
    ```

## 5. 引用 (Citation)

如果这套代码对您的研究有帮助，请引用相关工作或保留本项目链接。
