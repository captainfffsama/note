#具身智能 #强化学习 #论文AI速读

[toc]

# IN-RIL: Interleaved Reinforcement and Imitation Learning for Policy Fine-Tuning

时间：May 16, 2025

### 1. 核心动机与背景 (Motivation)

#### 解决的痛点：微调阶段的“性能崩塌”与样本效率低下

在机器人操作（Manipulation）领域，主流范式通常是“两阶段法”：先用行为克隆（BC）预训练策略，再用强化学习（RL）进行微调。然而，这种范式面临两个致命问题：

1. **性能崩塌（Performance Collapse）：** 启动 RL 微调的初期，由于 RL 依赖随机探索（Exploration），策略往往会迅速偏离预训练的专家分布，导致原本已经学会的技能（如抓取姿态）丢失，甚至导致训练发散。

2. **稀疏奖励下的探索困境：** 在长视距（Long-horizon）或稀疏奖励（Sparse Reward）任务中，如果策略偏离了演示分布，RL 很难再次通过随机探索触碰到奖励信号，导致样本效率极低。

#### 与现有方法的本质区别 (Differentiation)

现有的解决方案通常是：

- **添加正则项（Regularization）：** 在 Loss 中加入约束项（如 L2 距离或 KL 散度）限制策略偏离。但这引入了极难调节的超参数，容易导致“过约束”无法提升，或“欠约束”发生崩塌 。
- **数据混合（Replay Buffer Mixing）：** 将专家数据放入 RL 的 Buffer 中。这通常要求专家数据带有奖励标签（这在现实中很难获得），且需要复杂的采样策略 。

**IN-RIL 的核心洞察**在于打破了“两阶段”或“混合数据”的思维，转而采用**时间交织（Interleaved）**的优化策略。它并不修改 RL 算法本身，而是将 IL 更新作为一种周期性的“纠偏”机制注入到 RL 训练过程中，同时利用**梯度分离（Gradient Separation）**机制解决目标冲突 6。

### 2. 方法论框架 (Methodology)

#### 2.1 整体架构与流程

IN-RIL 是一个与模型无关（Model-Agnostic）的插件式框架。它在微调阶段引入了一个循环机制：每进行 $m(t)$ 次 RL 更新，就插入 1 次 IL 更新 7。

- **输入 (Input):** 专家演示数据 $\mathcal{D}_{exp}$（无奖励），环境交互数据 $\mathcal{D}_{env}$（有奖励）。
- **参数更新:**
    - IL Update: $\theta \leftarrow \theta - \alpha_{IL} \nabla \mathcal{L}_{IL}$
    - RL Update: $\theta \leftarrow \theta - \alpha_{RL} \nabla \mathcal{L}_{RL}$ （连续执行 $m$ 次）
#### 2.2 核心机制：梯度分离 (Gradient Separation)

这是该论文最核心的工程贡献。由于 IL 试图最小化模仿误差，而 RL 试图最大化长期回报，两者的梯度方向在优化景观（Optimization Landscape）中往往是不一致甚至冲突的（Destructive Interference）。

论文提出了两种具体的实现路径来解决这个问题：

A. 梯度手术 (Gradient Surgery) 

如果直接更新同一个网络，IN-RIL 会计算 $\nabla \mathcal{L}_{IL}$ 和 $\nabla \mathcal{L}_{RL}$，并将它们投影到独立的子空间（orthogonal subspaces）或对冲突分量进行剔除（Project onto dual cone），确保更新方向不会互相抵消。

B. 网络分离 (Network Separation / Residual Policy)

这是在复杂任务（如 FurnitureBench）中更有效的方案。

- **Base Policy ($\pi_{base}$):** 仅接受 **IL 梯度**更新。它负责保持专家行为的“底色”。
- **Residual Policy ($\pi_{res}$):** 仅接受 **RL 梯度**更新。它是一个额外的 MLP 网络，输出动作的残差 $\Delta a$。
- **最终动作:** $a = \pi_{base}(s) + \pi_{res}(s)$ 。
- **物理意义:** 这种设计将“保持稳定性”和“探索新策略”在**网络结构层面**进行了解耦，彻底避免了梯度冲突。

#### 2.3 数学表达与理论支撑

论文通过理论分析证明了交织更新的有效性。

- **最优比率 $m(t)$:** 理论推导表明，RL 更新次数 $m(t)$ 应当是动态的，取决于梯度对齐程度 $\rho(t)$（余弦相似度）。当梯度冲突严重（$\rho > 0$）时，需要更多的 RL 更新来克服阻力；当梯度一致时，可以减少 RL 更新 。
- **双重下降 (Double Descent) 现象:** 实验发现，纯 RL 微调会导致 IL Loss 飙升。而 IN-RIL 中的 IL Loss 会呈现“先升后降”的趋势，这表明 RL 的探索实际上帮助 IL 跳出了局部的次优解（Local Minima），进入了更好的盆地 。

### 3. 数据流水线 (Data Pipeline)
- **数据来源:**
    - **Offline Data:** 少量专家演示（如 Robomimic 中的 300 条，FurnitureBench 中的 50 条）。注意：IN-RIL 不需要这些数据包含 Reward 标签 。
    - **Online Data:** Agent 在仿真环境中实时交互产生的 Trajectory。
- **预处理:**
    - 使用 Diffusion Policy 或 Gaussian Policy 进行标准的 IL 预训练 16。
    - 采用了 **Action Chunking** 技术来增强时序一致性 17。
        

### 4. 训练与推理细节 (Training & Inference)
#### 训练策略
1. **预训练 (Pre-training):** 使用 IL 把 Policy 训练到收敛（Loss Plateau）。
2. **微调 (Fine-tuning):**
    - 加载预训练权重。
    - 如果是 **Residual 模式**：冻结或仅用 IL 更新 Base Policy，初始化一个零输出的 MLP 作为 Residual Policy，用 RL 算法（如 PPO, IDQL）更新 Residual 部分。
    - 如果是 **Full Network 模式**：按照 $1:m$ 的比例交替执行 IL 和 RL 的 Backward pass。对于 $m$ 的选择，实验表明 $m \in [5, 15]$ 是一个鲁棒的区间 。
#### 推理部署
- 在推理时，如果是 Network Separation 架构，由于 Base Policy 和 Residual Policy 是解耦的，前向传播并不复杂：$a_{final} = \text{Diffusion\_Net}(s) + \text{MLP}(s)$。
- **实时性:** 由于 Residual 部分只是简单的 MLP，其推理开销极小，主要延迟仍取决于 Base Policy（特别是如果是 Diffusion Policy，去噪步数是瓶颈）。

### 5. 技术评价与启发
#### 局限性 (Limitations)
1. **计算开销：** 即使在 RL 微调阶段，仍然需要不断从磁盘读取专家数据并计算 IL 梯度，这增加了训练时的 I/O 和计算负担。
2. **超参数敏感性：** 虽然论文给出了 $m$ 的推荐范围，但在极端梯度冲突的任务中，静态的 $m$ 可能不是最优的，需要实现理论部分提到的动态调整 $m(t)$ 。
3. **Residual 的局限：** 残差策略假设 RL 的修正量是加性的（Additive）。如果最优策略需要完全改变动作的模式（例如从“抓取”变为“推”），简单的加法残差可能难以拟合。
    
#### 工程迁移指南 (Engineering Migration)

如果你计划在 **Isaac Lab** 或 **AGX Orin** 上复现或部署：

1. **架构选择建议：** 强烈建议采用 **Network Separation (Residual Policy)** 方案。
    - **原因：** 在 Isaac Lab 这种基于 GPU 的并行仿真环境中，PPO 等 On-policy 算法非常高效。将复杂的 Diffusion Policy（Base）冻结或仅作低频更新，而让高频 RL 专注于训练轻量级的 MLP Residual，可以极大提升 FPS（Frames Per Second）和训练吞吐量。
2. **坑点预警：**
    - **Observation Space 对齐：** 确保 Base Policy 和 Residual Policy 看到的 Observation 是完全一致的（包括 Normalization 参数）。
    - **Action Scale：** Residual Policy 的输出通常需要限制在一个较小的范围（如 $\pm 0.1$），否则初始阶段的随机探索可能会完全破坏 Base Policy 的行为。论文中并未详细提及具体的 Scale 限制，但在工程实践中这是必须的。
