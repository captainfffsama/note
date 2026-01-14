#具身智能 #强化学习 #双足 #alphaXiv 

[toc]

以下回来来自 gemini 3 pro

# Learning Locomotion Skills for Cassie_Iterative Design and Sim-to-Real
## 1. 核心动机与背景 (Motivation)

### 试图解决的痛点

- **"One-shot" RL 的不可行性**：在实际机器人控制中，很难一次性定义完美的奖励函数（Reward Function）来直接训练出理想策略。通常需要反复修改奖励项来消除抖动、改善脚部轨迹等。
- **灾难性遗忘与重训练成本**：每次修改奖励函数后重新从头训练（From Scratch）效率极低。
- **Sim-to-Real 的复杂性**：主流方法（如 OpenAI Dactyl, ETH 的 ANYmal 工作）严重依赖大规模域随机化或复杂的执行器网络建模，增加了训练的算力负担和不确定性。

### 与现有方法的本质区别

本文的核心差异在于提出了 **DASS (Deterministic Action Stochastic State)** 数据采样策略，配合 **混合策略梯度（Mixed Policy Gradients）**。这允许研究人员像“做手术”一样微调策略（例如：保持行走能力的同时，修正在原地踏步时的骨盆抖动），而不是每次都推倒重来。此外，文章证明了**精确的系统辨识（System ID）** 加上 **状态估计器（State Estimator）** 的闭环训练，足以替代域随机化。

## 2. 方法论框架 (Methodology)

该论文的方法论由三个核心模块构成：运动追踪（Motion Tracking）、策略微调（Policy Refinement）和策略蒸馏（Policy Distillation）。

### A. 模型结构 (Architecture)

- **网络类型**：采用标准的 **Actor-Critic** 架构，使用 PPO 算法 。
- **输入张量 (Input)**：85 维向量。包含机器人状态 $X$（关节位置、速度、骨盆姿态、速度等）和参考运动 $\hat{X}$（Reference Motion）。注意，参考运动是基于时钟相位的（Phase-based）。
- **输出张量 (Output)**：关节 PD 控制器的**残差目标位置 (Residual PD target angles)**。
    - 公式：$\theta_{target} = \theta_{ref} + \pi(s)$
    - **物理意义**：策略输出的不是力矩，而是对参考轨迹的“微调量”。这种残差结构为 Sim-to-Real 提供了极强的先验稳定性。
        
### B. 核心技术：DASS (Deterministic Action Stochastic State)

这是论文最硬核的数学贡献。为了在迭代中传递策略，普通的行为克隆（Behavior Cloning）会失效，因为它只采样了专家策略的极限环（Limit Cycle），导致学生策略在遇到扰动时无法恢复（Covariate Shift 问题）。

- **DASS 采样逻辑**：
    1. 在状态 $s$ 处，使用**随机策略**（Stochastic Policy）采样动作 $a \sim \pi_{expert}(\cdot|s)$ 并执行，以此进入这一稍有偏离的状态 $s'$（模拟扰动）。
    2. 在记录数据用于训练学生网络时，记录该状态 $s'$ 对应的**确定性动作均值** $\mu_{expert}(s')$。
    - **直观理解**：这相当于教会学生网络：“当你稍微偏离轨道时（由随机探索导致），专家会用什么最精准的动作（确定性均值）把你拉回来”。这捕捉了反馈控制的流形（Manifold）。
        

### C. 数学表达：混合策略梯度

在策略微调阶段（如从普通行走微调为高抬腿行走），损失函数由两部分组成 ：

$$\nabla_{\theta} = \underbrace{\nabla_{\theta} J_{RL}}_{\text{新任务的奖励}} - w \underbrace{\nabla_{\theta} J_{SL}}_{\text{旧策略的约束}}$$

- $J_{RL} = \mathbb{E}[\sum \gamma^t r_{new}(s_t, a_t)]$：强化学习目标，最大化新定义的奖励（如更稳定的骨盆）。
- $J_{SL} = \mathbb{E}_{s \sim \mathcal{D}} [(\mu_{\theta}(s) - \mu_{expert}(s))^2]$：监督学习约束，利用 DASS 采样的数据 $\mathcal{D}$，强迫新策略不要偏离旧策略太远。


## 3. 数据流水线 (Data Pipeline)

- **参考运动 (Reference Motion)**：使用简单的运动学“草图”（Sketch）。例如，直接让骨盆匀速平移，不管脚是否打滑。这种物理上不可行（Unphysical）的参考轨迹被证明足以引导 RL 学习。
- **状态估计器的引入 (State Estimation)**：
    - **关键 Trick**：训练时不使用仿真器提供的 Ground Truth 状态，而是模拟机载状态估计器（Extended Kalman Filter）的输出作为 Policy 的输入。
    - **目的**：让 Policy 适应估计器的噪声和延迟，消除 Sim-to-Real 中的观测分布差异。

## 4. 训练与推理细节 (Training & Inference)

### 训练策略

1. **Phase 1 (Bootstrapping)**：使用简单的轨迹追踪奖励 $r = w_j r_j + w_{rp} r_{rp}$ 训练出一个初始行走的 Policy 。
2. **Phase 2 (Refinement)**：发现初始策略骨盆抖动严重。引入设计奖励 $r_{design}$（惩罚骨盆角速度），结合 DASS 约束进行微调 。
3. **Phase 3 (Compression/Distillation)**：将多个技能（前走、后退、侧移）蒸馏到一个更小的网络（如 64x64 或 16x16 隐层）中，不仅压缩了模型，还提高了鲁棒性。
    
### 推理与部署

- **硬件平台**：Cassie 机载电脑（Ubuntu + PyTorch）。
- **控制频率**：策略运行在 **33Hz** (30ms)，而底层 PD 控制器运行在 **2kHz**。这种高低频分离是保证实时性的关键 。
- **无域随机化**：依靠精确的系统辨识，特别是**反射惯量 (Reflected Inertia)** 的建模。如果在 MuJoCo 中忽略电机转子的反射惯量（armature 参数），策略在真机上必挂无疑。


## 5. 技术评价与工程启发
### 局限性

- **参考轨迹依赖**：该方法严重依赖参考轨迹（Reference Motion）。如果想生成非周期性、非结构化的动作（如跌倒恢复、跑酷），这种基于残差追踪的框架限制较大。
- **相位变量 (Phase Variable)**：策略输入包含了时钟信号（Phase），这意味着机器人必须严格按照周期执行动作，丧失了一定的非周期灵活性。
    

### 💡 工程迁移指南 (Isaac Lab / 实际部署建议)

如果你计划复现此论文或将其应用到自己的机器人（如 Unitree H1/Go2）上，请注意以下深层细节：

1. **Reflected Inertia (Armature) 是隐形杀手**：
    - 在 URDF/MJCF 文件中，不要只填连杆（Link）的惯量。对于高减速比的关节（Cassie 是谐波/摆线针轮），**电机转子的转动惯量 $\times$ 减速比平方** 必须加到模型中 。在 MuJoCo 中通过 `<joint armature="0.x">` 设置。
2. **残差动作空间 (Residual Action Space)**：
    - 初学者做 Sim-to-Real，不要直接输出力矩（Torque）。输出 `PD_target = q_ref + action` 是最稳妥的起步方式，因为 `q_ref` 提供了基础的重力补偿和构型维持 。
3. **DASS 的现代应用**：
    - 当你发现训练好的策略在真机上有一点小瑕疵（如脚抬得不够高），不要调整 Reward 重训。使用本文的 **RL + SL (DASS)** 方法，把旧策略当老师，加上新的约束项进行微调，能极大节省时间并保证稳定性。
4. **状态估计器闭环**：
    - 在仿真中，必须把观测噪声（Sensor Noise）和估计器延迟（Latency）加进去。如果直接用 `sim.data.qpos` 训练，部署时大概率会因为观测数据的分布漂移（Distribution Shift）而失败。

# DASS 采样时确定确定性动作均值 $u_{expert}(s')$ 的方式

根据论文《Learning Locomotion Skills for Cassie: Iterative Design and Sim-to-Real》中的描述，特别是在 **4.2 Policy Transfer Using Sample Tuples** 章节和 **Algorithm 1** 中，确定状态 $s'$ 的确定性动作均值 $\mu_{expert}(s')$ 的具体过程如下：

**核心机制：提取高斯分布的均值**
1. 前提：策略的定义
    论文中采用的策略 $\pi_{e}(\cdot|s)$ 是一个 高斯分布 (Gaussian Policy) 。神经网络接收状态 $s'$ 作为输入，输出该分布的 均值 (Mean) $\mu_{e}(s')$。在实际执行或探索时，会在这个均值上叠加高斯噪声（noise）来生成随机动作 $a$。
2. 具体操作步骤
    在 DASS (Deterministic Action Stochastic State) 采样循环中：
    - **状态获取**：机器人处于某个状态 $s'$。这个状态通常是由于上一步执行了带有噪声的随机动作（Stochastic Action）而到达的“扰动状态” 。
    - **确定性均值的获取**：对于这个通过随机探索到达的状态 $s'$，算法直接**取专家策略网络输出的均值向量**作为 $\mu_{expert}(s')$ 。
    - **关键点**：这里**不包含**任何探索噪声。它代表了专家策略认为在当前（可能略微偏离的）状态下，最理想、无噪声的恢复动作 。
3. 算法流程 (Algorithm 1)
    根据论文中的 Algorithm 1 伪代码 ：
    - 在每一步 $i$，对于当前状态 $s_i$：
    - **记录数据**：将元组 $(s_i, \mu_{e}(s_i))$ 加入数据集 $\mathcal{D}$。这里的 $\mu_{e}(s_i)$ 就是确定性动作均值。
    - **执行动作**：从分布 $\pi_{e}(\cdot|s_i)$ 中采样一个随机动作 $a_i$（包含噪声），并作用于环境，从而进入下一个状态 $s_{i+1}$。
        

总结

简单来说，$\mu_{expert}(s')$ 就是专家策略神经网络在输入为 $s'$ 时直接输出的“纯净”动作指令（即高斯分布的中心/均值），去掉了用于探索的随机噪声项。这样做是为了让学生策略学习到专家在面对扰动状态时试图将系统拉回稳定极限环（Limit Cycle）的“意图”，而不是学习包含随机噪声的具体执行动作 。