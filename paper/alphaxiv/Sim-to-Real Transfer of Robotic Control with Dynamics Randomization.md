#具身智能 #sim2real #强化学习

**sim2real 奠基作品**

# 1. 核心动机与背景 (Motivation)

## 试图解决的痛点
- **Reality Gap（现实差距）：** 深度强化学习（DeepRL）在仿真中表现优异，但直接迁移到真机时，由于摩擦力、质量分布、传感器噪声和控制延迟等物理参数的建模误差，策略往往失效。
- **采样效率与安全性：** 在真机上直接训练 DeepRL 需要数百万次交互，不仅时间成本极高，且探索过程中的随机动作可能损坏机器人或环境。
- **系统辨识的局限性：** 传统方法依赖高精度的系统辨识（System Identification）来校准仿真器，但这对于复杂的接触动力学（如推物体）非常困难且耗时。

## 与现有方法的本质区别

- **显式 vs. 隐式适应：** 之前的方法（如 Yu et al. 4）尝试显式地预测物理参数（如质量、摩擦系数），然后作为输入传给策略。本文提出，**不需要显式辨识参数**，而是通过训练一个**递归神经网络（LSTM）**，让网络从过去的状态和动作历史中**隐式地（Implicitly）** 推断当前环境的动力学特征 。
- **域随机化（Domain Randomization）：** 核心思想不是让仿真器“更逼真”，而是让仿真器“更多变”。通过在训练中剧烈随机化动力学参数，迫使策略学会一种能够适应各种动力学环境的“鲁棒元策略” 。


# 2. 方法论框架 (Methodology)

## 模型结构 (Architecture)

该方法采用 Actor-Critic 架构，针对 Policy（策略网络）和 Value（价值网络）进行了特殊设计 。

- **策略网络 (Policy Network - LSTM):**
    - **输入流：** 分为两部分。
        1. **Feedforward Branch:** 处理当前状态 $s_t$ 和目标 $g$。
        2. **Recurrent Branch (LSTM):** 处理当前状态 $s_t$ 和上一时刻的动作 $a_{t-1}$ 。
    - **核心逻辑：** LSTM 的隐藏状态 $z_t$ 充当了“内部记忆”，捕捉了物体运动对动作的响应历史。这实际上是在进行在线的系统辨识。
    - **输出：** 动作 $a_t$（7 维，对应关节角度的增量）。
        
- **价值网络 (Omniscient Critic - Value Network):**
    - **特殊设计：** 这是一个**全知 Critic**。除了状态和动作外，它还接收**真实的动力学参数 $\mu$**（如摩擦力、质量等）作为额外输入 。
    - **目的：** 在训练时，Critic 知道当前的物理环境是怎样的，因此能给出更准确的梯度指导 Policy 学习；而在推理（部署）时，只需要 Policy，不需要 $\mu$。
        
## 核心算法与数学表达

- **算法：** 使用 **RDPG (Recurrent Deterministic Policy Gradient)** 配合 **HER (Hindsight Experience Replay)** 。
- 目标函数：
    旨在最大化不同动力学模型分布 $\rho_{\mu}$ 下的期望回报：

    $$J(\pi) = \mathbb{E}_{\mu \sim \rho_{\mu}} [ \mathbb{E}_{\tau \sim p(\tau|\pi, \mu)} [ \sum_{t=0}^{T-1} r(s_t, a_t) ] ]$$

- 稀疏奖励 (Sparse Reward):

    $$r(s, g) = 0 \quad \text{if } \| \text{object} - \text{target} \| [cite_start]< \epsilon, \quad \text{else } -1$$

    由于奖励是二值的且非常稀疏，必须引入 HER 来通过“事后诸葛亮”的方式（将失败的轨迹标记为实现了其最终到达的目标）来加速学习 。    

# 3. 数据流水线 (Data Pipeline)

## 数据来源

- 完全来自 **MuJoCo** 物理引擎仿真，未使用任何真机数据进行训练。


## 关键的动力学随机化 (Dynamics Randomization)

这是该论文的“工程灵魂”。在每一集（Episode）开始时，从分布中采样一组物理参数 $\mu$ 并固定 。随机化参数列表如下（共 95 个参数）：

1. **质量 (Mass):** 机器人连杆 (0.25x - 4x)、物体质量。
2. **阻尼 (Damping):** 关节阻尼 (0.2x - 20x)。
3. **接触属性:** 物体摩擦系数 (0.1 - 5.0)、物体阻尼。
4. **环境几何:** 桌子高度。
5. **控制器:** PID 增益 (0.5x - 2x)。
6. **延迟 (Latency):** **非常关键**。动作的执行步长 $\Delta t$ 是随机的，模拟控制回路的延迟和抖动。
7. **观测噪声 (Observation Noise):** 给状态加上高斯噪声 。

# 4. 训练与推理细节 (Training & Inference)

## 训练策略
- **优化器:** ADAM，学习率 $5 \times 10^{-4}$ 。
- **Batch Size:** 128 个 Episodes 。
- **HER 设置:** 80% 的概率重采样目标 (Goal) 。
- **计算量:** 在 100 核集群上训练约 8 小时，约 1 亿次采样步数 。

## 推理与部署 (Inference)
- **真机状态:** Fetch 机器人的 7-DOF 手臂。
- **感知:** 使用 PhaseSpace 动捕系统获取物体位置（在实际应用中通常会被视觉模块替代，但本文专注于控制迁移）。
- **闭环控制:** 策略网络在真机上运行时，需要维护 LSTM 的 hidden state。虽然训练时参数 $\mu$ 是静态的，但在真机上，网络会根据 sensor feedback 动态调整 hidden state，从而适应真实的摩擦力和质量。
    

## 鲁棒性验证
- 为了验证鲁棒性，作者甚至在真机的物体（冰球）底部粘了一包薯片（Chips bag），极大地改变了接触面形状和摩擦系数，策略依然能保持 ~91% 的成功率 。


# 5. 技术评价与启发

## 局限性

1. **保守性 (Conservativeness):** 为了适应极端的物理参数（如极低的摩擦力或极高的延迟），策略可能会变得过于保守（动作缓慢、抖动），无法达到最优的时间效率。
2. **分布选择困难:** 随机化的范围（Range）需要领域知识手动调节。如果范围太宽，模型可能无法收敛；如果太窄，则无法覆盖真机情况。
3. **状态依赖:** 论文假设能通过动捕系统获得完美的物体位姿 $(x, y, z, \theta)$。如果结合视觉（Pixel-to-Control），会引入新的视觉域差异问题（Visual Reality Gap）。

## 工程迁移与复现建议 (Sim2Real Checklist)

如果你计划在 NVIDIA Isaac Lab 或类似环境中复现此工作并部署到硬件：

1. 必须随机化延迟 (Latency Randomization):
    论文特别提到，如果不随机化动作的时间步长（模拟通讯和计算延迟），真机成功率会从 ~89% 骤降至 ~29% 。在 Isaac Lab 中，可以通过随机设置 decimation 或在 buffer 中随机取历史动作来实现。

2. 观测历史 vs. LSTM:
    如果不想使用 LSTM（训练较慢），可以使用 Frame Stacking（历史观测堆叠）。论文对比实验显示，Feedforward + History (FF+Hist) 也能达到 ~70% 的真机成功率，虽然不如 LSTM 的 ~89%，但工程实现更简单 。

3. 特权信息的利用 (Privileged Information):
    复现 Critic 时，务必将随机化的物理属性（Physics scale, friction, mass）输入给 Critic，这能显著降低 Value Function 学习的方差（Variance Reduction），加速收敛。

4. 噪声注入:
    不要只随机化物理参数，必须给 Observation 加入高斯噪声。论文显示，去掉观测噪声会导致真机成功率降至 ~25% 。真机的传感器永远是不完美的。

    

下一步建议：

如果您正在进行相关的复现工作，我可以为您提供 " 基于 Isaac Lab/Gym 的 Domain Randomization 配置代码片段 "，或者详细解释 "Omniscient Critic 在代码中如何通过非对称 Actor-Critic 实现 "。您需要哪方面的具体帮助？