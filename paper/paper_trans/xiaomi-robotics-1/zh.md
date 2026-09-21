# Xiaomi-Robotics-1: 利用超 10 万小时真实世界轨迹拓展视觉-语言-动作模型

**小米机器人 (Xiaomi Robotics)**  
*arXiv:2607.15330v2 \[cs.RO\] 22 Jul 2026*  
项目主页: [https://robotics.xiaomi.com/xiaomi-robotics-1.html](https://robotics.xiaomi.com/xiaomi-robotics-1.html)  
联系邮箱: `mi-robotics@xiaomi.com`

---

## 摘要 (Abstract)

我们推出了 **Xiaomi-Robotics-1**，这是一个基石视觉-语言-动作 (VLA) 模型，具备以下能力：

1. 能够遵循多样化的自然语言指令，在未见过的环境中开箱即用 (out-of-the-box) 地执行广泛的移动操作任务；  
2. 仅需极少的微调数据，即可高效适配全新的下游任务。

我们提出了一种由预训练 (pre-training) 和后训练 (post-training) 组成的两阶段训练配方。在预训练阶段，我们通过在覆盖海量环境和任务的超 10 万小时真实世界操作轨迹（通过 UMI 设备采集）上进行训练，赋予模型广泛且具泛化性的动作生成能力。至关重要的是，我们开发了一套可扩展的自动标注流水线，该流水线使用描述场景状态转移的自然语言对轨迹片段进行标注，为动作学习提供了丰富且精确的条件指导。在后训练阶段，我们致力于将这些能力与机器人本体以及人类自然用于提示机器人的祈使式任务指令进行对齐，有效地将描述性状态转移理解映射为可执行的任务提示。

广泛的实验表明了强大的扩展定律 (scaling behavior) 表现：在预训练期间，Xiaomi-Robotics-1 随着数据规模和模型尺寸的增加而持续提升。这种扩展特性可直接迁移至后训练阶段——更强大的预训练模型在未见过的真实机器人评测环境中展现出更优异的开箱即用性能。此外，Xiaomi-Robotics-1 可作为强有力的机器人基础策略，以极高的数据效率在复杂、灵巧的任务上高效微调。在多个仿真基准测试中，Xiaomi-Robotics-1 均超越了现有最先进的方法。值得注意的是，它在 RoboCasa365 上取得了 **57.4%** 的成功率，大幅刷新了此前最佳记录 (46.6%)；在 RoboDojo 上取得了 **20.07** 的平均分，显著优于先前的最先进水平 (13.07)。相关代码和模型权重将被开源。

---

![Figure 1: Overview of Xiaomi-Robotics-1]() *图 1：概览。Xiaomi-Robotics-1 在超 10 万小时带有自动标注的状态转移语言提示的真实世界 UMI 轨迹上进行预训练。随后通过跨本体后训练与机器人本体和祈使式指令提示对齐。Xiaomi-Robotics-1 能随数据量和模型规模有效扩展。它能够在未见过的环境中开箱即用执行多种任务，并能高效学习新任务。*

---

## 1\. 引言 (Introduction)

现代大模型的卓越能力从根本上是由规模驱动的，其中大规模且多样化的训练语料库为大型语言模型 \[7, 22, 27, 45\] 和视觉-语言模型 \[1, 14, 62, 63\] 的性能带来了前所未有的跃升。近期关于视觉-语言-动作 (VLA) 模型 \[4, 5, 24, 25, 54, 71, 76\] 以及世界-动作模型 (WAM) \[36, 81, 83\] 的研究在机器人操作领域展现出愈发令人鼓舞的成果，早期证据表明，随着训练数据规模和多样性的增长，策略的能力和泛化性也会同步增强。因此，遵循大模型的这一扩展路径是机器人学自然且极具吸引力的发展方向。

然而，机器人学面临着独特的数据瓶颈限制。主流的数据采集范式——真实机器人遥操作 (teleoperation)，速度慢、成本高昂且受限于硬件，难以实现规模化扩展。此外，遥操作数据往往具有高度冗余性，集中在狭窄的任务和环境子集中，限制了数据的多样性。

为此，我们推出了 **Xiaomi-Robotics-1**（图 1），这是一个在大规模真实世界操作轨迹上训练的基石视觉-语言-动作 (VLA) 模型。受大语言模型训练范式的启发，我们提出了一种包含预训练和后训练的两阶段训练配方。

在预训练阶段，我们利用在体量和多样性上都极易扩展的数据源，赋予模型稳健且可泛化的动作生成能力。具体而言，我们使用 UMI 设备 \[17\] 构建了一个包含超 10 万小时真实世界操作轨迹的数据集，涵盖了极为广泛的环境和任务。传统的轨迹标注通常需要按照任务语义进行手动分段和语言注释，这种劳动密集型流程在此规模下是不可行的。为了解决这一挑战，我们开发了一套可扩展的自动标注流水线，利用预训练视觉-语言模型 (VLM) \[70\] 为固定长度的轨迹片段生成详细描述场景状态转移的语言说明。这些注释提供了精确且充分的语义监督。在这些数据上训练后，模型学会了生成使场景从当前状态转变为语言指定目标状态的动作（图 6）。

在后训练阶段，我们利用超过 1 万小时的跨本体 (cross-embodiment) 数据来对齐预训练期间获得的强大动作生成能力。该阶段弥合了两个差距：

1. 将模型从为 UMI 夹爪生成动作适配为为具体机器人本体生成动作；  
2. 从状态转移提示转换为人类通常用于提示机器人的祈使式指令。

完成微调对齐后，Xiaomi-Robotics-1 能够遵循指令并在未见过的环境中执行广泛的任务。此外，它还可以作为强大的机器人基础策略，仅需少量数据即可高效微调以学习全新任务。

我们进行了广泛的实验来研究 Xiaomi-Robotics-1 的扩展特性：

- Xiaomi-Robotics-1 在预训练阶段表现出有效的扩展性，随着数据和模型规模的扩大，验证集动作误差持续降低。  
- 预训练中观察到的扩展规律直接迁移至后训练，更强的预训练模型在未见环境中开箱即用的真实机器人评估中取得了更高的后训练成功率。  
- 当在四个具有挑战性的下游任务上使用极少量数据（平均每项任务 \<10 小时）进行微调时，Xiaomi-Robotics-1 取得了 **75%** 的平均成功率，显著优于 $\\pi\_{0.5}$ 的 40%。  
- 在仿真基准测试中，Xiaomi-Robotics-1 在 RoboCasa \[52\]、RoboCasa365 \[53\]、VLABench \[87\] 和 RoboDojo \[12\] 上均创下了新的 SOTA 成果。尤其在 RoboCasa365 上达到了 **57.4%**（此前最佳为 46.6%），在 RoboDojo 上达到了 **20.07**（此前最佳为 13.07）。  
- 在真实机器人移动操作中，它自主完成了一项持续时间超过 10 分钟的长视距行李箱打包任务。

---

## 2\. Xiaomi-Robotics-1

Xiaomi-Robotics-1 是一个端到端的视觉-语言-动作 (VLA) 模型，在异构数据源（包括 UMI 轨迹、跨本体机器人轨迹和视觉-语言数据）上进行了规模化训练。给定当前观测 $o\_t$ 和语言指令 $l$，训练模型 $\\pi\_\\theta$ 通过最大化在训练数据集 $\\mathcal{D}$ 上的对数似然来预测动作块 (action chunk) $a\_{t:t+H}$：

$$\\max\_{\\theta} \\mathbb{E}*{(o\_t, l, a*{t:t+H}) \\sim \\mathcal{D}} \\log \\pi\_\\theta(a\_{t:t+H} \\mid o\_t, l)$$

我们采用由预训练和后训练组成的两阶段训练配方。预训练利用具备丰富开放世界多样性的可扩展非机器人数据集，赋予模型用于动作生成的广泛且可泛化的表征。后训练则使用高质量的跨本体数据集，将这些表征对齐到机器人本体以及以指令为条件的动作生成上。

![Figure 2: Model Architecture of Xiaomi-Robotics-1]() *图 2：模型架构。Xiaomi-Robotics-1 采用了 Transformer 混合体 (Mixture-of-Transformers, MoT) \[44\] 架构，将预训练 VLM 与 DiT 耦合。VLM 对观测和语言指令进行编码，并通过选择策略 (Choice Policies) \[59\] 额外预测动作块以加速训练收敛。以机器人状态以及 VLM 中观测和语言标记的 KV 缓存为条件，DiT 通过流匹配生成动作块。注意，来自 VLM 的动作相关标记被排除在 DiT 的注意力计算之外。*

### 2.1 模型结构 (Model)

如图 2 所示，Xiaomi-Robotics-1 采用了 Transformer 混合体 (Mixture-of-Transformers, MoT) \[44\] 架构，由预训练视觉-语言模型 (VLM)（即 Qwen3-VL \[3\]）和扩散 Transformer (DiT) \[57\] 组成。DiT 在层数上与 VLM 匹配，但使用较小的隐藏层维度以实现更快的推理速度。表 1 详细列出了 Xiaomi-Robotics-1 不同规模变体的模型参数。

| 模型 | 层数 (\# Layers) | VLM 隐藏层维度 | VLM 参数量 | DiT 隐藏层维度 | DiT 参数量 | 总参数量 |
| :---- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Xiaomi-Robotics-1-2B** | 28 | 2048 | 2.1B | 1024 | 470M | **2.6B** |
| **Xiaomi-Robotics-1-5B** | 36 | 2560 | 4.4B | 1024 | 604M | **5.1B** |
| **Xiaomi-Robotics-1-10B** | 36 | 4096 | 8.8B | 2048 | 1.5B | **10.5B** |

*表 1：Xiaomi-Robotics-1 不同规模变体的模型配置。*

VLM 将当前观测 $o\_t$ 和语言指令 $l$ 作为输入。以机器人本体感知状态 $s\_t$ 以及 VLM 生成的 KV 缓存为条件，DiT 通过流匹配 (flow-matching) \[49\] 生成动作块：

$$\\mathcal{L}*{\\text{Flow}}(\\theta) \= \\left| v*\\theta\\left(o\_t, l, s\_t, \\tilde{a}*{t:t+H}^\\tau, \\tau\\right) \- u\\left(\\tilde{a}*{t:t+H}^\\tau, a\_{t:t+H}, \\tau\\right) \\right|\_2^2$$

其中 $\\tau$ 为流匹配时间步。$\\tilde{a}*{t:t+H}^\\tau \= \\tau a*{t:t+H} \+ (1 \- \\tau)\\epsilon$ 为加噪动作，其中 $\\epsilon \\sim \\mathcal{N}(0, I)$。遵循 \[5\]，我们从 Beta 分布中采样时间步 $\\tau$，在训练期间赋予噪声更大的时间步更高的权重：

$$u \\sim \\text{Beta}(1.5, 1), \\quad \\tau \= (1 \- u) \\times 0.999 \\in \[0, 0.999\]$$

与 \[8\] 类似，我们利用自适应归一化层 (adaLN) \[57\] 将流匹配时间步条件注入 DiT 中以进行动作生成。在推理过程中，我们将预测的动作块初始化为随机噪声 $a\_{t:t+H}^{\\tau=0} \\sim \\mathcal{N}(0, I)$。无噪的动作块通过 5 步欧拉积分进行恢复：

$$a\_{t:t+H}^{\\tau \+ \\Delta\\tau} \= a\_{t:t+H}^\\tau \+ \\Delta\\tau \\cdot v\_\\theta\\left(o\_t, l, s\_t, a\_{t:t+H}^\\tau, \\tau\\right)$$

其中步长设置为 $\\Delta\\tau \= 0.2$。

为了加速收敛 \[58\]，我们在 VLM 上引入了辅助动作生成监督。具体来说，我们利用选择策略 (Choice Policies) \[59\] 在 VLM 框架内直接实现动作生成 \[8\]。我们使用多层感知机 (MLP) 将机器人状态编码为一个标记，并将其与动作和评分查询标记一同附加到视觉-语言标记序列的末尾。对应于动作和评分查询标记的输出分别预测 $K$ 个候选动作块及其关联的 $K$ 个得分。我们采用与 \[59\] 中相同的胜者通吃 (winner-takes-all) 范式，其中仅将具有最小 $L\_1$ 损失的候选样本纳入动作损失计算：

$$\\mathcal{L}*{\\text{Regression}}(\\theta) \= \\left| \\hat{a}*{t:t+H}^\* \- a\_{t:t+H} \\right|*1 \+ \\sum*{k=1}^K \\left| \\hat{s}\_k \- s\_k \\right|\_2^2$$

设 $\\hat{a}*{t:t+H}^k$ 表示第 $k$ 个预测的候选动作块，则 $\\hat{a}*{t:t+H}^\*$ 为与真实值 $a\_{t:t+H}$ 具有最小 $L\_1$ 距离的候选动作块。$\\hat{s}*k$ 是第 $k$ 个候选动作块的预测得分，其回归目标 $s\_k$ 定义为 $s\_k \= \\left| \\hat{a}*{t:t+H}^k \- a\_{t:t+H} \\right|\_1$。也就是说，$K$ 个预测动作块与真实动作块之间的 $L\_1$ 距离用作评分预测的目标标签。

直接在 VLM 上应用动作生成监督，可以引导其表征趋向于更好地支持动作生成的特征，从而使 DiT 的学习更加有效。然而，我们在实验中经验性地观察到，允许 DiT 标记关注这些动作相关标记的 KV 缓存会导致性能下降。我们推测这是由于 DiT 形成了一条捷径 (shortcut)——简单地复制 VLM 生成的动作，而不是有效地将自身生成扎根于视觉和文本上下文中。为了缓解该问题，我们在 DiT 的注意力计算中排除了这些动作相关标记，限制 DiT 标记仅关注语言指令和视觉观测的表征。

---

### 2.2 训练与数据 (Training & Data)

![Figure 3: Pre-training Dataset]() *图 3：预训练数据集。Xiaomi-Robotics-1 的预训练数据集包含使用 UMI 设备采集的超 10 万小时真实世界操作轨迹。*

#### 2.2.1 预训练 (Pre-training)

在预训练阶段，我们的主要目标是赋予模型广泛且可泛化的表征，使其能够在多样的操作场景中迁移。为此，我们构建了一个包含**超过 100,000 小时**真实世界操作轨迹的数据集，这些轨迹是使用通用操作接口 (Universal Manipulation Interface, UMI) 手持夹爪 \[17\] 和第一人称相机采集的（图 3）。该数据集涵盖了在海量环境中采集的多样化任务，包括家庭、商业场所、工业厂房、办公室和室外空间。传统的机器人轨迹注释需要根据任务语义手动分割轨迹，并为每个片段标注一条语言指令——这一劳动密集型过程在此规模下是难以承受的。

为了实现语言注释的规模化扩展，我们开发了一套自动标注流水线：首先将每条轨迹划分为等长片段，并利用 Qwen3.5-27B \[70\] 在每个片段内为夹爪和场景中交互对象的状态转移生成描述性标题（示例见图 11）。为了加速注释过程，我们开发了解耦切片分割与标题标注的生产者-消费者流水线：当 CPU 工作线程将各片段切片切入内存文件系统时，客户端线程保持数百个标题生成请求处于运行状态。这种极高效的流水线使我们能够在大约两周内完成对整个超 10 万小时语料库的标注。在该数据集上训练后，模型学会了生成动作以驱动场景从当前观测中的状态转换为语言标注所描述的目标状态。

通过联合最小化 DiT 的流匹配损失 $\\mathcal{L}*{\\text{Flow}}$ 和 VLM 选择策略的回归损失 $\\mathcal{L}*{\\text{Regression}}$ 来优化模型以预测动作。为了保留预训练 VLM 的视觉-语言能力，我们进一步在之前工作 \[8\] 中构建的高质量视觉-语言数据集上联合训练该模型，训练目标为下一个标记预测 (Next-Token Prediction) 损失 $\\mathcal{L}\_{\\text{NTP}}$。总体训练目标公式化为：

$$\\mathcal{L} \= \\mathcal{L}*{\\text{Flow}} \+ \\mathcal{L}*{\\text{Regression}} \+ \\lambda \\mathcal{L}\_{\\text{NTP}}$$

在我们的实验中，$\\lambda$ 设置为 0.1。视觉-语言数据和 UMI 轨迹按 1:9 的比例采样。为了最大化训练吞吐量，我们将一个批次内的所有视觉-语言标记打包成单个序列以进行 VLM 前向传递。由于 VLM 在计算上比 DiT 更昂贵，我们通过每个样本采样四个流匹配时间步来平摊其成本。所得到的四个 DiT 输入同样被打包并在一次 DiT 传递中处理，以解包后的相应 VLM KV 缓存为条件。

![Figure 4: Post-training Dataset]() *图 4：后训练数据集。Xiaomi-Robotics-1 的后训练数据集包含约 1 万小时的跨本体轨迹，包括使用移动操作机器人和双臂机器人采集的超 7,200 小时内部机器人数据、超 1,000 小时带有指令标注的 UMI 数据，以及开源机器人数据集。*

#### 2.2.2 后训练 (Post-training)

后训练的目标具有双重性：

1. 将预训练期间获得的 UMI 夹爪动作生成能力迁移至机器人本体。  
2. 将语言条件从预训练中使用的状态转移描述转变为人类在提示机器人执行任务时通常发出的祈使式指令。

我们使用通过 UMI 设备、静态机械臂和移动操作机器人采集的跨本体操作轨迹来构建后训练数据集。具体而言，我们在多样化的家庭环境和任务中，使用移动操作机器人和双臂机器人收集了**超过 7,200 小时**的机器人数据（图 4）。我们利用 Qwen3.5 \[70\] 为人工分段的视频片段标注语言指令。此外，我们整合了**超过 1,000 小时**的人工注释 UMI 数据，这些数据标注有时序分段和语言指令。与预训练中使用的状态转移描述不同，这些语言指令与人类提示机器人执行任务的方式高度贴合，直接符合我们在后训练阶段的对齐目标（对比见图 11 和图 12）。最后，我们纳入了开源机器人数据集，包括 Bridge V2 \[74\]、RT-1 \[6\] 和 DROID \[28\]。我们过滤掉了轨迹内的空闲片段，以防止模型学到无信息量或带有噪声的信号。总计，我们的后训练数据集包含了约 **10,000 小时**的轨迹数据。

对于机械臂动作，我们采用相对于当前状态的末端执行器 (EE) 相对增量位姿：

$$a\_{t+i} \= \\left( {}^{\\text{Base}}*{\\text{EE}} T\_t \\right)^{-1} {}^{\\text{Base}}*{\\text{EE}} \\hat{T}\_{t+i}$$

其中 ${}^{\\text{Base}}*{\\text{EE}} T\_t$ 表示在当前时间步 $t$ 末端执行器相对于基座的位姿，而 ${}^{\\text{Base}}*{\\text{EE}} \\hat{T}\_{t+i}$ 代表时间步 $t+i$ 处的末端执行器目标位姿。为了对齐不同本体之间的机械臂动作空间，我们在预训练和后训练数据集中统一了所有机器人数据和 UMI 数据的末端执行器坐标系方向。因此，无论底层硬件平台如何，相似的机械臂运动（例如相对于末端执行器坐标系向前或向后移动）都会产生一致的动作值。

对于移动机器人数据，我们分别使用底盘速度和腰部位置的相对增量来表示底盘和腰部动作。为了适应异构本体，我们对所有轨迹数据采用统一的动作向量。尽管机械臂动作在本体之间进行了对齐，但不同机器人的动作空间在维度上仍然存在差异。我们在损失计算期间对缺失动作分量对应的维度进行掩蔽 (mask out)。

我们采用与预训练相同的目标函数来训练模型。视觉-语言数据、开源机器人数据、指令标注的 UMI 数据和内部机器人数据按 0.5 : 0.5 : 0.5 : 8.5 的比例采样。在后训练之后，模型可以通过语言指令提示，在未见环境中开箱即用执行广泛的任务。此外，它还可以用极少量的数据高效适配新的下游任务。

---

## 3\. 实验 (Experiments)

我们在设计 Xiaomi-Robotics-1 时充分考虑了可扩展性。在本节中，我们通过广泛的实验探究其扩展特性：

- Xiaomi-Robotics-1 在预训练期间是否能随着数据规模和模型尺寸的增加有效扩展？  
- 更强的预训练模型在全新环境中进行开箱即用评估时，是否能转化为更优的后训练性能？  
- Xiaomi-Robotics-1 是否能够仅利用极少量的数据适配极具挑战性的新任务？  
- Xiaomi-Robotics-1 与其他机器人基石模型在真实机器人实验和仿真基准测试中的对比表现如何？

### 3.1 预训练：数据与模型扩展 (Pre-training: Data and Model Scaling)

![Figure 5: Scaling of Pre-training]() *图 5：预训练的扩展特性。我们展示了数据扩展和模型扩展预训练实验中的验证集动作误差 (MSE)。在数据扩展实验中，我们对 12.5% 和 25% 数据的训练提前终止，因为验证集损失显示出过拟合迹象。*

**数据扩展 (Data Scaling)。** 我们使用 Xiaomi-Robotics-1-5B 进行了数据扩展实验。由于计算预算限制，我们分别在约 2 万小时 UMI 数据的 12.5%、25%、50% 和 100% 子集上对模型进行预训练。每个模型均在保留的验证集上进行评估，使用流匹配预测动作与真实动作之间的均方误差 (MSE) 作为指标。如图 5 所示，Xiaomi-Robotics-1 的验证动作误差随着数据规模的增加而降低。在使用 12.5% 和 25% 数据时，验证动作误差在训练过程中先下降后上升，表明存在过拟合。相比之下，50% 和 100% 数据带来了损失的单调下降，其中 20k 小时设置呈现出更陡峭的下降趋势。图 6 展示了验证数据上的定性动作预测结果。

![Figure 6: Qualitative Action Prediction on Pre-training Validation Clips]() *图 6：预训练定性结果。经过预训练后，Xiaomi-Robotics-1 能够根据状态转移的语言描述，在保留验证集上预测 UMI 夹爪的动作轨迹。*

**模型扩展 (Model Scaling)。** 我们在表 1 所列的 Xiaomi-Robotics-1 三种规模变体（2B、5B 和 10B）上进行了模型扩展实验。所有三个模型都在相同的 2 万小时数据上进行训练，并在相同的保留验证集上进行评估。如图 5 所示，随着模型尺寸的增加，Xiaomi-Robotics-1 在动作预测精度上展现出持续的提升。然而，不同模型大小之间的性能差距不如跨数据规模观察到的差距显著。这表明数十亿参数规模的模型容量可能已足以捕获当前数据集的分布，使得数据体量成为进一步泛化的主要瓶颈。

---

### 3.2 后训练：在全新环境中的开箱即用评估 (Post-training: Out-of-the-Box Evaluation in Novel Environments)

我们在跨本体后训练数据集上进行后训练实验，并研究在训练期间未见过的全新环境中的开箱即用性能。为了缓解过拟合，针对内部机器人数据，我们采样了一个多样化的子集进行后训练。模型在未见环境中无需任何针对特定任务或环境的微调，直接对 4 个任务进行开箱即用评测（图 7）：鞋子收纳 (Shoe Storage)、装包 (Bag Packing)、桌面整理 (Table Organization) 和沙发整理 (Sofa Tidying)。这些任务在后训练数据集中属于已见类别，但评估时的环境和物体实例完全未曾见过。

![Figure 7: Post-training Evaluation on Four Tasks in Unseen Environments]() *图 7：后训练评估。我们在全新环境中对后训练模型在四项任务上进行开箱即用评估。至关重要的是，评估时的环境和物体实例在训练期间均未见过。*

#### 3.2.1 扩展预训练数据的有效性

我们通过 5B 变体检验扩展预训练数据带来的收益是否能迁移至后训练阶段。采用完全一致的训练配方，我们对分别基于 2 万小时预训练数据的 12.5%、25%、50% 和 100% 预训练权重初始化的模型进行后训练，并设立了一个从没有动作预训练的 Qwen3-VL 预训练权重初始化作为基线 (0%)。

如图 8 所示，整体成功率随着预训练数据规模呈单调增长趋势，从无动作预训练时的 26% 提升至 100% 预训练数据时的 75%：

- **平均成功率 (Average)**: 100% 数据: **75%**，50%: 69%，25%: 56%，12.5%: 53%，0%: 26%  
- **鞋子收纳 (Shoe Storage)**: 100%: **75%**，50%: 83%，25%: 42%，12.5%: 42%，0%: 0%  
- **装包 (Bag Packing)**: 100%: **63%**，50%: 63%，25%: 56%，12.5%: 30%，0%: 7%  
- **桌面整理 (Table Organization)**: 100%: **82%**，50%: 67%，25%: 56%，12.5%: 63%，0%: 48%  
- **沙发整理 (Sofa Tidying)**: 100%: **80%**，50%: 70%，25%: 67%，12.5%: 72%，0%: 33%

扩展预训练数据带来的增益在需要丰富接触操作的任务上尤为显著（例如鞋子收纳：无预训练时为 0%，使用 100% 数据时达 75%）。值得注意的是，仅使用 12.5% 的预训练数据，基线的整体成功率就实现了一倍以上的提升（53% vs. 26%）。将数据从 50% 翻倍至 100% 可带来额外的 6 个百分点提升，且未显现出饱和趋势。

![Figure 8: Quantitative Results of Post-training]() *图 8：后训练定量结果。我们展示了在不同预训练数据规模和模型尺寸下后训练模型的成功率。*

#### 3.2.2 扩展模型规模的有效性

我们在 2B、5B 和 10B 变体上探究了后训练期间模型规模的影响，所有模型均基于在 2 万小时 UMI 数据上预训练的检查点初始化（图 8）。整体成功率随着模型规模单调递增：

- **平均成功率 (Average)**: 10B: **79%**，5B: 75%，2B: 61%  
- **鞋子收纳 (Shoe Storage)**: 10B: **92%**，5B: 75%，2B: 58%  
- **装包 (Bag Packing)**: 10B: **67%**，5B: 63%，2B: 56%  
- **桌面整理 (Table Organization)**: 10B: **89%**，5B: 82%，2B: 70%  
- **沙发整理 (Sofa Tidying)**: 10B: **77%**，5B: 80%，2B: 60%

模型扩展带来的收益在鞋子收纳任务上最为显著，从 58% (2B) 跃升至 75% (5B) 和 92% (10B)。预训练数据规模和模型尺寸构成了提升分布外泛化性能的两个互补维度。

---

### 3.3 下游微调：向新任务的高效适配 (Downstream Fine-tuning: Efficient Adaptation to New Tasks)

![Figure 9: Downstream Fine-tuning Tasks Suite]() *图 9：下游微调评估。我们利用极少量的数据在四个具有挑战性的新任务上微调后训练模型。*

我们在从内部数据集中完全保留的四项全新挑战性任务上微调后训练模型（图 9）：

- **手机包装 (Phone Packing)**：需要双臂协同操作。  
- **放入洗衣机 (Laundry Loading)**：涉及多步指令遵循的长视距移动操作。  
- **打印机加纸 (Printer Refilling)**：处理高度易变形的单张纸张。  
- **装盒打包 (Box Packing)**：跨多个物体的语言定位与接地 (language grounding)。

评估了两种设置：

1. **大数据设置 (High-data setting)**：所有任务累计 144 小时（每项任务 \<40 小时）。  
2. **小数据设置 (Low-data setting)**：25% 的子集（总计 36 小时，平均每项任务 \<10 小时；打印机加纸仅有 10.3 小时）。

我们采用异步训练 \[8\] 进行微调，并与 $\\pi\_{0.5}$ \[5\]（官方 OpenPi 协议）和 Xiaomi-Robotics-0 \[8\] 进行对比。每项任务评估 10 次试验。我们报告了基于里程碑完成度的成功率和进度得分（表 6）。

![Figure 10: Quantitative Results of Downstream Fine-tuning]() *图 10：下游微调定量结果。我们汇报了不同模型在四项任务上的成功率与进度得分。*

| 设置 | 指标 | 方法 | 综合整体 | 手机包装 | 打印机加纸 | 放入洗衣机 | 装盒打包 |
| :---- | :---- | :---- | :---: | :---: | :---: | :---: | :---: |
| **小数据** (\<10h/任务) | 成功率 (%) | **Xiaomi-Robotics-1 (本文方法)** | **75%** | **70%** | **70%** | **80%** | **80%** |
|  |  | $\\pi\_{0.5}$ \[5\] | 40% | 30% | 20% | 40% | 70% |
|  |  | Xiaomi-Robotics-0 \[8\] | 15% | 0% | 0% | 0% | 60% |
|  | 进度得分 (%) | **Xiaomi-Robotics-1 (本文方法)** | **90%** | **89%** | **82%** | **96%** | **94%** |
|  |  | $\\pi\_{0.5}$ \[5\] | 66% | 74% | 42% | 64% | 84% |
|  |  | Xiaomi-Robotics-0 \[8\] | 36% | 31% | 22% | 0% | 92% |
| **大数据** (\<40h/任务) | 成功率 (%) | **Xiaomi-Robotics-1 (本文方法)** | **85%** | **80%** | **60%** | **100%** | **100%** |
|  |  | $\\pi\_{0.5}$ \[5\] | 52% | 40% | 20% | 50% | 100% |
|  |  | Xiaomi-Robotics-0 \[8\] | 65% | 40% | 40% | 80% | 100% |
|  | 进度得分 (%) | **Xiaomi-Robotics-1 (本文方法)** | **94%** | **91%** | **86%** | **100%** | **100%** |
|  |  | $\\pi\_{0.5}$ \[5\] | 78% | 80% | 46% | 90% | 100% |
|  |  | Xiaomi-Robotics-0 \[8\] | 82% | 80% | 52% | 98% | 100% |

Xiaomi-Robotics-1 在两种设置下均显著超越基线。在 \<10h/任务的数据量下，它达到了 **75%** 的成功率和 **90%** 的进度得分（对比 $\\pi\_{0.5}$ 的 40% 成功率与 66% 进度得分）。在打印机加纸任务中，它将成功率从 20% 提高至 70%。在放入洗衣机任务中，它达到了 80% 成功率和 96% 进度，而 Xiaomi-Robotics-0 在此任务上完全失败。

---

### 3.4 仿真基准测试 (Simulation Benchmarks)

#### 1\. RoboCasa 基准测试

RoboCasa \[52\] 专注于真实厨房环境中的单臂操作，涵盖 24 个日常厨房任务。评估测试了未见过的物体实例和 2 种未见过的厨房场景风格。遵循标准协议，我们在 5 个场景中对每项任务评测 100 个 episode。

| 方法 | 平均成功率 (%) |
| :---- | :---: |
| UVA \[41\] | 50.0 |
| UWM \[96\] | 60.8 |
| $\\pi\_{0.5}$ \[5\] | 62.1 |
| $\\pi\_0$-FAST \[58\] | 63.6 |
| GR00T N1.6 \[54\] | 66.2 |
| Cosmos Policy \[32\] | 67.1 |
| RLDX-1 \[29\] | 70.6 |
| World2Act \[73\] | 72.6 |
| **Xiaomi-Robotics-1 (本文方法)** | **74.5** |

*表 2：RoboCasa 基准测试结果。平均成功率 (%)。*

#### 2\. RoboCasa365 基准测试

RoboCasa365 \[53\] 将 RoboCasa 扩展至横跨 2,500 个程序化生成厨房和 3,200 个物体实例的 365 项任务，评测原子技能 (atomic skills)、已见复合任务 (seen composite tasks) 和零样本未见复合任务 (zero-shot unseen composite tasks)。评估涵盖 50 个基准任务（18 个原子任务、16 个已见复合任务、16 个未见复合任务）。

| 方法 | 平均 (Average) | 原子任务 (Atomic) | 已见复合 (Comp.-Seen) | 未见复合 (Comp.-Unseen) |
| :---- | :---: | :---: | :---: | :---: |
| Diffusion Policy \[16\] | 6.1 | 15.7 | 0.2 | 1.3 |
| $\\pi\_{0.5}$ \[5\] | 16.9 | 39.6 | 7.1 | 1.2 |
| GigaWorld-Policy 0.1 \[79\] | 20.7 | 44.4 | 11.8 | 2.9 |
| GR00T-N1.6 \[54\] | 21.9 | 51.1 | 9.4 | 1.7 |
| WorldDreamer \[75\] | 35.3 | 66.3 | 26.7 | 9.0 |
| Qwen-RobotManip \[71\] | 35.9 | 68.6 | 20.1 | 14.9 |
| RLDX-1 \[29\] | 36.0 | 67.6 | 27.9 | 8.5 |
| ABot-M0.5 \[11\] | 40.4 | 75.9 | 38.3 | 2.7 |
| ABot-M0.6 \[11\] | 46.6 | 79.4 | 48.3 | 7.9 |
| **Xiaomi-Robotics-1 (本文方法)** | **57.4** | **80.2** | **57.1** | **32.1** |

*表 3：RoboCasa365 基准测试结果。任务成功率 (%)。*

Xiaomi-Robotics-1 创下了新的 SOTA 纪录，平均成功率达到 **57.4%**（相比此前最佳提升 \+10.8%）。至关重要的是，在零样本**未见复合任务 (Composite-Unseen)** 划分上，它取得了 **32.1%** 的成绩，达到最接近竞争对手 (14.9%) 的两倍以上（提升 \+17.2%）。

#### 3\. VLABench 基准测试

VLABench \[87\] 评估了在 100 个类别和 2,000 个物体上的以语言为条件的操作，涵盖五个赛道：分布内 (In-distribution)、跨类别 (Cross-Category)、常识 (Commonsense)、指令 (Instruction) 和纹理 (Texture)。我们仅在分布内演示数据（10 个任务，每个 500 条演示）上训练，并带有思维链 (CoT) 辅助损失 \[61\]。总计评测 2,500 次 rollout。

| 方法 | 分布内 (In-dist.) | 跨类别 (Cross Category) | 常识 (Commonsense) | 指令 (Instruction) | 纹理 (Texture) | 平均 (Avg.) |
| :---- | :---: | :---: | :---: | :---: | :---: | :---: |
|  | **SR / PS / IS** | **SR / PS / IS** | **SR / PS / IS** | **SR / PS / IS** | **SR / PS / IS** | **SR / PS / IS** |
| $\\pi\_0$-FAST \[58\] | 56.2 / 72.4 / 67.8 | 31.0 / 47.8 / 25.1 | 48.6 / 56.8 / 48.2 | 35.0 / 59.4 / 56.8 | 39.0 / 56.8 / 54.6 | 41.8 / 58.6 / 51.1 |
| X-VLA \[92\] | 66.8 / 76.5 / 67.8 | 38.2 / 54.1 / 38.9 | 38.0 / 52.3 / 46.3 | 45.0 / 63.1 / 39.6 | 49.0 / 74.6 / 44.5 | 47.4 / 63.5 / 47.4 |
| ACOT-VLA \[94\] | 66.1 / 79.8 / 69.4 | 45.2 / 58.6 / 42.1 | 42.6 / 54.8 / 47.2 | 48.2 / 65.4 / 45.3 | 52.4 / 68.2 / 48.6 | 50.9 / 65.4 / 50.5 |
| $\\pi\_{0.5}$ \[5\] | 77.8 / 80.4 / 65.4 | 49.7 / 52.0 / 38.2 | 60.0 / 57.3 / 43.9 | 64.2 / 67.0 / 48.2 | 62.3 / 65.0 / 44.9 | 62.8 / 64.3 / 48.1 |
| ERVLA \[61\] | 69.7 / 81.1 / 84.2 | 47.0 / 61.0 / 66.4 | 44.0 / 55.0 / 57.2 | 73.8 / 70.2 / 58.0 | 47.4 / 62.3 / 70.6 | 56.4 / 65.9 / 67.3 |
| **Xiaomi-Robotics-1 (本文方法)** | **75.6 / 85.0 / 79.8** | **53.0 / 66.6 / 66.4** | **58.2 / 68.4 / 58.3** | **55.8 / 66.8 / 70.2** | **62.6 / 74.9 / 74.8** | **61.0 / 72.3 / 69.9** |

*表 4：VLABench 基准测试结果。SR: 成功率 (%), PS: 进度得分 (%), IS: 意图得分 (Intention Score, %)。*

#### 4\. RoboDojo 仿真基准测试

RoboDojo \[12\] 是一个综合基准测试，在五个维度上评估 42 个仿真任务中的通用操作能力：泛化性 (Generalization)、精确度 (Precision)、长视距 (Long-Horizon)、记忆 (Memory) 和开放性 (Open)。

| 方法 | 泛化性 (Generalization) | 精确度 (Precision) | 长视距 (Long-Horizon) | 记忆 (Memory) | 开放性 (Open) | 平均 (Average) |
| :---- | :---: | :---: | :---: | :---: | :---: | :---: |
| GalaxeaVLA (G0) \[26\] | 4.53 / 2.83% | 8.10 / 3.83% | 12.60 / 5.58% | 3.17 / 1.89% | 0.70 / 0.67% | 5.82 / 2.96% |
| GigaWorld-Policy \[79\] | 5.34 / 2.89% | 6.15 / 1.83% | 15.51 / 8.92% | 3.46 / 2.22% | 0.54 / 0.50% | 6.20 / 3.27% |
| StarVLA-α \[80\] | 3.93 / 2.33% | 9.90 / 4.33% | 14.15 / 6.50% | 3.34 / 2.44% | 0.68 / 0.58% | 6.40 / 3.24% |
| Xiaomi-Robotics-0 \[8\] | 7.43 / 5.56% | 8.42 / 4.58% | 13.51 / 6.92% | 5.07 / 3.67% | 0.22 / 0.17% | 6.93 / 4.18% |
| X-WAM \[21\] | 7.39 / 3.33% | 6.72 / 1.83% | 17.47 / 9.08% | 6.32 / 4.67% | 0.57 / 0.25% | 7.69 / 3.83% |
| X-VLA \[92\] | 10.48 / 6.78% | 18.32 / 12.00% | 16.53 / 9.75% | 4.76 / 3.56% | 0.55 / 0.50% | 10.13 / 6.52% |
| $\\pi\_{0.5}$ \[5\] | 13.37 / 8.17% | 12.40 / 5.50% | 23.54 / 14.67% | 5.78 / 4.56% | 1.98 / 1.67% | 11.41 / 6.91% |
| Spatial Forcing \[35\] | 14.12 / 9.33% | 17.33 / 10.58% | 23.26 / 14.58% | 5.43 / 4.11% | 1.78 / 1.58% | 12.38 / 8.04% |
| Hy-Embodied-0.5-VLA \[85\] | 11.77 / 8.39% | 13.81 / 8.00% | 25.74 / 14.92% | **13.37 / 12.11%** | 0.65 / 0.58% | 13.07 / 8.80% |
| **Xiaomi-Robotics-1 (本文方法)** | **23.55 / 17.00%** | **26.69 / 18.83%** | **38.39 / 23.67%** | 7.81 / 6.56% | **3.94 / 3.58%** | **20.07 / 13.93%** |

*表 5：RoboDojo 仿真基准测试结果。每项汇报为 得分 / 成功率 (%)。*

Xiaomi-Robotics-1 取得了 **20.07** 的得分和 **13.93%** 的成功率（相较于先前 SOTA 提升 \+7.0 分与 \+5.13% 成功率）。它在泛化性（23.55 vs. 14.12）、精确度（26.69 vs. 18.32）、长视距（38.39 vs. 25.74）和开放式指令遵循（3.94 vs. 1.98）四个维度上均位列第一。

---

## 4\. 相关工作 (Related Work)

### 机器人学习的扩展 (Scaling for Robot Learning)

对大语言模型 (LLM) 扩展定律的研究表明，当数据、算力和模型容量协同扩展时，性能会实现可预测的提升 \[22, 27\]。大语言模型 \[7, 72\] 和多模态基础模型 \[1–3, 63\] 进一步展示了由扩大数据和模型规模带来的实质性能力飞跃。受这些进展的启发，机器人学习越来越多地拥抱这种扩展范式 \[4–6, 8, 31, 42, 54, 58, 65–67, 97\]。

然而，机器人学习的规模扩展在根本上有别于网络规模数据：真实机器人轨迹需要昂贵且繁重的遥操作，导致数据仅局限于狭窄的环境和任务子集。为了缓解该瓶颈，近期研究利用便携式 UMI 设备 \[17\] 在无需物理机器人本体的情况下进行野外 (in-the-wild) 操作数据采集 \[17, 46, 65, 66, 77, 90\]。作为补充，第一人称人类操作视频通过表征对齐或运动重定向，提供了跨任务、物体和环境的丰富多样性 \[15, 40, 50\]。在本文中，我们利用超过 10 万小时的真实世界 UMI 轨迹，系统地探索了基础 VLA 模型的扩展特性。

### 机器人基础模型 (Robot Foundation Models)

机器人基础模型能够在不同环境中实现稳健的泛化，并能高效适配下游任务。世界-动作模型 (WAM) \[21, 32, 36, 39, 51, 56, 68, 78, 81, 83, 86\] 和视觉-语言-动作 (VLA) 模型 \[4, 5, 8–10, 30, 38, 42, 60, 82, 92\] 代表了两种主要范式。

WAM 构建在预训练视频模型之上，建模未来观测或环境动力学以指导动作生成 \[18, 23, 32, 33, 36, 37, 41, 43, 95, 96\]，融合 3D 几何结构 \[21, 39, 88, 91\]，并在异构视频-动作数据上进行预训练以实现零样本迁移 \[81, 86\]。

VLA 模型利用预训练视觉-语言模型，借助通用的语义知识进行动作预测 \[4, 5, 8–10, 24, 25, 30, 47, 54, 97\]。近期 VLA 的进展包括：

1. 具身推理标记和视觉思维链 \[13, 19, 34, 64, 84, 89, 93\]，  
2. 通过学习的分词器和流匹配构建表达能力丰富的连续动作表征 \[4, 20, 48, 58, 82\]，以及  
3. 跨异构硬件平台的跨本体预训练 \[8, 30, 54, 55, 69, 82\]。

我们的工作沿袭了 VLA 范式，重点关注扩展定律，并依托于可扩展的数据采集和状态转移自动标注基础设施。

---

## 5\. 结论 (Conclusions)

在这项工作中，我们介绍了 **Xiaomi-Robotics-1**，这是一个基石视觉-语言-动作 (VLA) 模型，能够在未见环境中开箱即用遵循指令执行多样的移动操作任务，并能利用极少的数据高效适配全新的挑战性任务。在预训练期间，我们利用了超过 10 万小时的真实世界操作轨迹，赋予模型广泛且可泛化的操作能力。为了有效扩展训练，我们提出了一套自动标注流水线，用详细的场景状态转移描述作为语言提示来标注大规模数据集。在后训练阶段，我们使用跨本体数据集将预训练中获得的强大能力与机器人本体及祈使式指令提示对齐。

大量实验表明，Xiaomi-Robotics-1 的性能随着预训练中数据规模和模型大小的增加而持续提高。更重要的是，这一扩展特性在后训练后可直接转化为未见环境中的开箱即用表现。Xiaomi-Robotics-1 还可以作为强大的机器人基础模型，用极少的数据适配全新的挑战性真实机器人任务。此外，它在四个极具挑战性的仿真基准测试中均取得了强劲的 SOTA 表现。我们希望这项工作能为未来探索可开箱即用部署于真实世界的、具可扩展性的机器人策略奠定基础。

---

## 贡献与致谢 (Contributions & Acknowledgment)

作者按姓氏字母顺序排列。

**核心贡献者 (Core Contributors):**  
Jun Guo, Piaopiao Jin, Jason Li, Peiyan Li, Yingyan Li, Futeng Liu, Wanli Peng, Optimus Qin, Yifei Su, Nan Sun, Qiao Sun, Runze Suo, Heyun Wang, Yunhong Wang, Rujie Wu, Caoyu Xia, Lina Zhang, Jack Zhao.

**贡献者 (Contributors):**  
Guoliang Chen, Wenlong Chen, Xinze He, Bin Li, Qing Li, Zhuorong Li, Heng Qu, Wenxuan Song, Diyun Xiang, Yifan Xie, Peiran Xu, Hangjun Ye, Wen Ye, Han Zhao, Quanyun Zhou.

**致谢 (Acknowledgment):**  
我们向给予巨大支持的更广泛团队成员表达诚挚的谢意，包括未列于上方的成员：Li Jiang, Xiaohan Yu, Meichen Mu, Xiaoke Xilinjueluo, Qingyi Li, Qi Liu, Yayun Liu, Jun Xia, Feng Qiu, Donghao Wang, Yan Hou, Dong Wang, Liangliang He, Jiaxin Liu, Kang Zhou, Rui Cai, Shuoxue Bi, Yingchao Zhou, Kun Ma, Yiwei Zhou, 以及 Dongsheng Li。

---

## 附录 (Appendix)

### 下游微调的进度里程碑 (Progress Milestones for Downstream Fine-tuning)

| 任务 | 进度里程碑 | 进度百分比 (%) |
| :---- | :---- | :---: |
| **手机包装 (Phone Packing)** | 抓取手机；将手机放入盒中；抓取说明书；将说明书放入盒中；抓取盒盖；成功盖上盒盖。 | 10, 10, 30, 10, 10, 30 |
| **打印机加纸 (Printer Refilling)** | 抓取纸张；完成双臂交接；成功将纸张一端插入打印机托盘；将整叠纸完全插入打印机托盘；双臂复位至初始休息姿态。 | 20, 20, 20, 30, 10 |
| **放入洗衣机 (Laundry Loading)** | 打开洗衣机门；将洗衣篮移至门前；将衣服转运至洗衣机内；移开洗衣篮；关闭洗衣机门。 | 每项各占 20 |
| **装盒打包 (Box Packing)** | 根据语言指令抓取并将每个指定的目标物体放入盒中。每次评测共评估 5 个目标物体。 | 每项各占 20 |

*表 6：新任务高效适配评测的进度定义。每次评测根据完成的任务里程碑被赋予 0 至 100% 的进度得分。*

---

### 数据集样本与可视化 (Dataset Samples and Visualizations)

![Figure 11: Examples of UMI data in the Pre-training Dataset]() *图 11：预训练数据集中的 UMI 数据示例。轨迹片段通过自动生成的场景状态转移进行描述，记录了夹爪动作以及物体状态的变化。*

![Figure 12: Examples of UMI data in the Post-training Dataset]() *图 12：后训练数据集中的 UMI 数据示例。由人工注释的轨迹片段，标有祈使式自然语言指令。*

---

## 参考文献 (References)

1. J. Achiam, S. Adler, S. Agarwal, L. Ahmad, I. Akkaya, F. L. Aleman, D. Almeida, J. Altenschmidt, S. Altman, S. Anadkat, et al. GPT-4 technical report. *arXiv preprint arXiv:2303.08774*, 2023\.  
2. N. Agarwal, A. Ali, J. Allen, M. Antolini, A. Aubame, A. Azzolini, J. Bai, M. Bala, Y. Balaji, J. Bapst, et al. Cosmos 3: Omnimodal world models for physical AI. *arXiv preprint arXiv:2606.02800*, 2026\.  
3. S. Bai, Y. Cai, R. Chen, K. Chen, X. Chen, Z. Cheng, L. Deng, W. Ding, C. Gao, C. Ge, et al. Qwen3-VL technical report. *arXiv preprint arXiv:2511.21631*, 2025\.  
4. K. Black, N. Brown, D. Driess, A. Esmail, M. Equi, C. Finn, N. Fusai, L. Groom, K. Hausman, B. Ichter, et al. $\\pi\_0$: A vision-language-action flow model for general robot control. *arXiv preprint arXiv:2410.24164*, 2024\.  
5. K. Black, N. Brown, J. Darpinian, K. Dhabalia, D. Driess, A. Esmail, M. Equi, C. Finn, N. Fusai, et al. $\\pi\_{0.5}$: A vision-language-action model with open-world generalization. *arXiv preprint arXiv:2504.16054*, 2025\.  
6. A. Brohan, N. Brown, J. Carbajal, Y. Chebotar, J. Dabis, C. Finn, K. Gopalakrishnan, K. Hausman, A. Herzog, J. Hsu, et al. RT-1: Robotics transformer for real-world control at scale. *arXiv preprint arXiv:2212.06817*, 2022\.  
7. T. Brown, B. Mann, N. Ryder, M. Subbiah, J. D Kaplan, P. Dhariwal, A. Neelakantan, P. Shyam, G. Sastry, A. Askell, et al. Language models are few-shot learners. *Advances in Neural Information Processing Systems*, 33:1877–1901, 2020\.  
8. R. Cai, J. Guo, X. He, P. Jin, J. Li, B. Lin, F. Liu, W. Liu, F. Ma, K. Ma, et al. Xiaomi-robotics-0: An open-sourced vision-language-action model with real-time execution. *arXiv preprint arXiv:2602.12684*, 2026\.  
9. C.-L. Cheang, G. Chen, Y. Jing, T. Kong, H. Li, Y. Li, Y. Liu, H. Wu, J. Xu, Y. Yang, et al. GR-2: A generative video-language-action model with web-scale knowledge for robot manipulation. *arXiv preprint arXiv:2410.06158*, 2024\.  
10. C. Cheang, S. Chen, Z. Cui, Y. Hu, L. Huang, T. Kong, H. Li, Y. Li, Y. Liu, X. Ma, et al. GR-3 technical report. *arXiv preprint arXiv:2507.15493*, 2025\.  
11. R. Chen, Y. Yang, Z. Tang, D. Huo, T. Lin, H. Wu, H. Liu, Y. Chen, L. Zheng, B. Yuan, T. Li, M. Wang, D. Qi, B. Hu, W. Mei, Y. Xuan, H. Yang, Y. Zhu, M. Xu, Z. Ma, and X. Chang. ABot-M0.5: Unified mobility-and-manipulation world action model. *arXiv preprint arXiv:2607.00678*, 2026\.  
12. T. Chen, Y. Chen, Z. Li, J. Tang, K. Su, H. Lu, W. Wan, B. Chen, S. Liu, H. Yan, H. Su, Z. Dou, K. Wang, D. Zhang, Y. Liu, Y. Qin, Q. Liang, Q. Wu, Z. Lin, W. Lin, Y. Wang, M. He, T. Wu, R. Wu, J. Zhou, K.-C. Lei, H. Yu, Y. Ji, W. Jin, G. Lin, X. Li, Q. Xiong, R. Xu, Z. Li, W. Chai, E. Xie, Z. Wang, Y. Mu, H. Dong, W. Matusik, M. Ding, W. Ding, P. Luo, and M. Tomizuka. RoboDojo: A unified sim-and-real benchmark for comprehensive evaluation of generalist robot manipulation policies. *arXiv preprint arXiv:2607.04434*, 2026\.  
13. W. Chen, S. Belkhale, S. Mirchandani, O. Mees, D. Driess, K. Pertsch, and S. Levine. Training strategies for efficient embodied reasoning. *arXiv preprint arXiv:2505.08243*, 2025\.  
14. X. Chen, X. Wang, S. Changpinyo, A. J. Piergiovanni, P. Padlewski, D. Salz, S. Goodman, A. Grycner, B. Mustafa, L. Beyer, et al. PaLI: A jointly-scaled multilingual language-image model. *arXiv preprint arXiv:2209.06794*, 2022\.  
15. Y. Chen, Z. Chen, P. Wang, Y.-L. Li, J. Huo, J. Shi, and Y. Gao. WHO: Generative world models as scalable sources of egocentric human hand manipulation data. *arXiv preprint arXiv:2606.22136*, 2026\.  
16. C. Chi, Z. Xu, S. Feng, E. Cousineau, Y. Du, B. Burchfiel, R. Tedrake, and S. Song. Diffusion policy: Visuomotor policy learning via action diffusion. *The International Journal of Robotics Research*, 2024\.  
17. C. Chi, Z. Xu, C. Pan, E. Cousineau, B. Burchfiel, S. Feng, R. Tedrake, and S. Song. Universal manipulation interface: In-the-wild robot teaching without in-the-wild robots. *arXiv preprint arXiv:2402.10329*, 2024\.  
18. Y. Du, S. Yang, B. Dai, H. Dai, O. Nachum, J. Tenenbaum, D. Schuurmans, and P. Abbeel. Learning universal policies via text-guided video generation. *Advances in Neural Information Processing Systems*, 36:9156–9172, 2023\.  
19. H. Fang, J. Duan, D. Clay, S. Wang, S. Liu, W. Huang, X. Fan, W.-C. Tsai, S. Chen, Y. R. Wang, et al. MolmoAct2: Action reasoning models for real-world deployment. *arXiv preprint arXiv:2605.02881*, 2026\.  
20. Galaxea Team. Galaxea G0.5 technical report. 2026\. [https://opengalaxea.github.io/G05/](https://opengalaxea.github.io/G05/).  
21. J. Guo, Q. Li, P. Li, Z. Chen, N. Sun, Y. Su, H. Wang, Y. Zhang, X. Li, and H. Liu. Unified 4D world action modeling from video priors with asynchronous denoising. *arXiv preprint arXiv:2604.26694*, 2026\.  
22. J. Hoffmann, S. Borgeaud, A. Mensch, E. Buchatskaya, T. Cai, E. Rutherford, D. de Las Casas, L. A. Hendricks, J. Welbl, A. Clark, et al. Training compute-optimal large language models. *arXiv preprint arXiv:2203.15556*, 2022\.  
23. Y. Hu, Y. Guo, P. Wang, X. Chen, Y.-J. Wang, J. Zhang, K. Sreenath, C. Lu, and J. Chen. Video prediction policy: A generalist robot policy with predictive visual representations. *arXiv preprint arXiv:2412.14803*, 2024\.  
24. Physical Intelligence, A. Amin, R. Aniceto, A. Balakrishna, K. Black, K. Conley, G. Connors, J. Darpinian, K. Dhabalia, J. DiCarlo, et al. $\\pi\_{0.6}$: A VLA that learns from experience. *arXiv preprint arXiv:2511.14759*, 2025\.  
25. Physical Intelligence, B. Ai, A. Amin, R. Aniceto, A. Balakrishna, G. Balke, K. Black, G. Bokinsky, S. Cao, T. Charbonnier, et al. $\\pi\_{0.7}$: A steerable generalist robotic foundation model with emergent capabilities. *arXiv preprint arXiv:2604.15483*, 2026\.  
26. T. Jiang, T. Yuan, Y. Liu, C. Lu, J. Cui, X. Liu, S. Cheng, J. Gao, H. Xu, and H. Zhao. Galaxea open-world dataset and G0 dual-system VLA model. *arXiv preprint arXiv:2509.00576*, 2025\.  
27. J. Kaplan, S. McCandlish, T. Henighan, T. B. Brown, B. Chess, R. Child, S. Gray, A. Radford, J. Wu, and D. Amodei. Scaling laws for neural language models. *arXiv preprint arXiv:2001.08361*, 2020\.  
28. A. Khazatsky, K. Pertsch, S. Nair, A. Balakrishna, S. Dasari, S. Karamcheti, S. Nasiriany, M. K. Srirama, L. Y. Chen, K. Ellis, et al. DROID: A large-scale in-the-wild robot manipulation dataset. *arXiv preprint arXiv:2403.12945*, 2024\.  
29. D. Kim, H. Jang, M. Koo, S. Jang, T. Kim, B. Kim, B. Yoon, C. Jang, D. Choi, D. Han, et al. RLDX-1 technical report. *arXiv preprint arXiv:2605.03269*, 2026\.  
30. M. J. Kim, K. Pertsch, S. Karamcheti, T. Xiao, A. Balakrishna, S. Nair, R. Rafailov, E. Foster, G. Lam, P. Sanketi, et al. OpenVLA: An open-source vision-language-action model. *arXiv preprint arXiv:2406.09246*, 2024\.  
31. M. J. Kim, C. Finn, and P. Liang. Fine-tuning vision-language-action models: Optimizing speed and success. *arXiv preprint arXiv:2502.19645*, 2025\.  
32. M. J. Kim, Y. Gao, T.-Y. Lin, Y.-C. Lin, Y. Ge, G. Lam, P. Liang, S. Song, M.-Y. Liu, C. Finn, et al. Cosmos Policy: Fine-tuning video models for visuomotor control and planning. *arXiv preprint arXiv:2601.16163*, 2026\.  
33. P.-C. Ko, J. Mao, Y. Du, S.-H. Sun, and J. B. Tenenbaum. Learning to act from actionless videos through dense correspondences. In *ICLR*, pages 40938–40958, 2024\.  
34. J. Lee, J. Duan, H. Fang, Y. Deng, S. Liu, B. Li, B. Fang, J. Zhang, Y. R. Wang, S. Lee, et al. MolmoAct: Action reasoning models that can reason in space. *arXiv preprint arXiv:2508.07917*, 2025\.  
35. F. Li, W. Song, H. Zhao, J. Wang, P. Ding, D. Wang, L. Zeng, and H. Li. Spatial Forcing: Implicit spatial representation alignment for vision-language-action model. *arXiv preprint arXiv:2510.12276*, 2025\.  
36. L. Li, Q. Zhang, Y. Luo, S. Yang, R. Wang, F. Han, M. Yu, Z. Gao, N. Xue, X. Zhu, et al. Causal world modeling for robot control. *arXiv preprint arXiv:2601.21998*, 2026\.  
37. P. Li, H. Wu, Y. Huang, C. Cheang, L. Wang, and T. Kong. GR-MG: Leveraging partially-annotated data via multi-modal goal-conditioned policy. *IEEE Robotics and Automation Letters*, 10(2):1912–1919, 2025\.  
38. P. Li, Y. Chen, H. Wu, X. Ma, X. Wu, Y. Huang, L. Wang, T. Kong, and T. Tan. BridgeVLA: Input-output alignment for efficient 3D manipulation learning with vision-language models. *Advances in Neural Information Processing Systems*, 38:63635–63673, 2026\.  
39. P. Li, Y. Chen, Y. Xu, J. Yang, X. Wu, J. Guo, N. Sun, L. Qian, X. Li, X. Xiao, et al. Multi-view video diffusion policy: A 3D spatio-temporal-aware video action model. *arXiv preprint arXiv:2604.03181*, 2026\.  
40. Q. Li, Y. Deng, Y. Liang, L. Luo, L. Zhou, C. Yao, L. Zeng, Z. Feng, H. Liang, S. Xu, et al. Scalable vision-language-action model pretraining for robotic manipulation with real-life human activity videos. *arXiv preprint arXiv:2510.21571*, 2025\.  
41. S. Li, Y. Gao, D. Sadigh, and S. Song. Unified video action model. *arXiv preprint arXiv:2503.00200*, 2025\.  
42. X. Li, P. Li, M. Liu, D. Wang, J. Liu, B. Kang, X. Ma, T. Kong, H. Zhang, and H. Liu. Towards generalist robot policies: What matters in building vision-language-action models. *arXiv preprint arXiv:2412.14058*, 2024\.  
43. J. Liang, R. Liu, E. Ozguroglu, S. Sudhakar, A. Dave, P. Tokmakov, S. Song, and C. Vondrick. Dreamitate: Real-world visuomotor policy learning via video generation. *arXiv preprint arXiv:2406.16862*, 2024\.  
44. W. Liang, L. Yu, L. Luo, S. Iyer, N. Dong, C. Zhou, G. Ghosh, M. Lewis, W.-t. Yih, L. Zettlemoyer, et al. Mixture-of-transformers: A sparse and scalable architecture for multi-modal foundation models. *arXiv preprint arXiv:2411.04996*, 2024\.  
45. A. Liu, B. Feng, B. Xue, B. Wang, B. Wu, C. Lu, C. Zhao, C. Deng, C. Zhang, C. Ruan, et al. DeepSeek-V3 technical report. *arXiv preprint arXiv:2412.19437*, 2024\.  
46. F. Liu, C. Li, Y. Qin, J. Xu, P. Abbeel, and R. Chen. VITAMIN: Learning contact-rich tasks through robot-free visuo-tactile manipulation interface. *arXiv preprint arXiv:2504.06156*, 2025\.  
47. S. Liu, L. Wu, B. Li, H. Tan, H. Chen, Z. Wang, K. Xu, H. Su, and J. Zhu. RDT-1B: A diffusion foundation model for bimanual manipulation. In *ICLR*, pages 29982–30009, 2025\.  
48. S. Liu, B. Li, K. Ma, L. Wu, H. Tan, X. Ouyang, H. Su, and J. Zhu. RDT2: Exploring the scaling limit of UMI data towards zero-shot cross-embodiment generalization. *arXiv preprint arXiv:2602.03310*, 2026\.  
49. X. Liu, C. Gong, and Q. Liu. Flow straight and fast: Learning to generate and transfer data with rectified flow. *arXiv preprint arXiv:2209.03003*, 2022\.  
50. H. Luo, Y. Feng, W. Zhang, S. Zheng, Y. Wang, H. Yuan, J. Liu, C. Xu, Q. Jin, and Z. Lu. Being-H0: Vision-language-action pretraining from large-scale human videos. *arXiv preprint arXiv:2507.15597*, 2025\.  
51. T. Ma, J. Zheng, Z. Wang, C. Jiang, A. Cui, J. Liang, and S. Yang. DiT4DiT: Jointly modeling video dynamics and actions for generalizable robot control. *arXiv preprint arXiv:2603.10448*, 2026\.  
52. S. Nasiriany, A. Maddukuri, L. Zhang, A. Parikh, A. Lo, A. Joshi, A. Mandlekar, and Y. Zhu. RoboCasa: Large-scale simulation of everyday tasks for generalist robots. *arXiv preprint arXiv:2406.02523*, 2024\.  
53. S. Nasiriany, S. Nasiriany, A. Maddukuri, and Y. Zhu. RoboCasa365: A large-scale simulation framework for training and benchmarking generalist robots. *arXiv preprint arXiv:2603.04356*, 2026\.  
54. NVIDIA, J. Bjorck, N. Cherniadev, F. Castañeda, X. Da, R. Ding, L. Fan, Y. Fang, D. Fox, F. Hu, S. Huang, J. Jang, Z. Jiang, J. Kautz, K. Kundalia, L. Lao, Z. Li, Z. Lin, K. Lin, G. Liu, E. Llontop, L. Magne, A. Mandlekar, A. Narayan, S. Nasiriany, S. Reed, Y. L. Tan, G. Wang, Z. Wang, J. Wang, Q. Wang, J. Xiang, Y. Xie, Y. Xu, Z. Xu, S. Ye, Z. Yu, A. Zhang, H. Zhang, Y. Zhao, R. Zheng, and Y. Zhu. GR00T N1: An open foundation model for generalist humanoid robots. *arXiv preprint*, March 2025\.  
55. A. O'Neill, A. Rehman, A. Maddukuri, A. Gupta, A. Padalkar, A. Lee, A. Pooley, A. Gupta, A. Mandlekar, A. Jain, et al. Open X-Embodiment: Robotic learning datasets and RT-X models. In *ICRA*, pages 6892–6903, 2024\.  
56. J. Pai, L. Achenbach, V. Montesinos, B. Forrai, O. Mees, and E. Nava. MIMIC-Video: Video-action models for generalizable robot control beyond VLAs. *arXiv preprint arXiv:2512.15692*, 2025\.  
57. W. Peebles and S. Xie. Scalable diffusion models with transformers. In *ICCV*, pages 4195–4205, 2023\.  
58. K. Pertsch, K. Stachowicz, B. Ichter, D. Driess, S. Nair, Q. Vuong, O. Mees, C. Finn, and S. Levine. FAST: Efficient action tokenization for vision-language-action models. *arXiv preprint arXiv:2501.09747*, 2025\.  
59. H. Qi, Y.-J. Wang, T. Lin, B. Yi, Y. Ma, K. Sreenath, and J. Malik. Coordinated humanoid manipulation with choice policies. *arXiv preprint arXiv:2512.25072*, 2025\.  
60. D. Qu, H. Song, Q. Chen, Y. Yao, X. Ye, Y. Ding, Z. Wang, J. Gu, B. Zhao, D. Wang, et al. SpatialVLA: Exploring spatial representations for visual-language-action model. *arXiv preprint arXiv:2501.15830*, 2025\.  
61. N. Sun, Y. Zhang, Y. Yang, W. Zhao, P. Li, J. Guo, W. Song, P. Ding, R. Suo, Y. Su, et al. Revisiting embodied chain-of-thought for generalizable robot manipulation. *arXiv preprint arXiv:2606.03784*, 2026\.  
62. Gemini Team, R. Anil, S. Borgeaud, J.-B. Alayrac, J. Yu, R. Soricut, J. Schalkwyk, A. M. Dai, A. Hauth, K. Millican, et al. Gemini: A family of highly capable multimodal models. *arXiv preprint arXiv:2312.11805*, 2023\.  
63. Gemini Team, P. Georgiev, V. I. Lei, R. Burnell, L. Bai, A. Gulati, G. Tanzer, D. Vincent, Z. Pan, S. Wang, et al. Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context. *arXiv preprint arXiv:2403.05530*, 2024\.  
64. Gemini Robotics Team, S. Abeyruwan, J. Ainslie, J.-B. Alayrac, M. G. Arenas, T. Armstrong, A. Balakrishna, R. Baruch, M. Bauza, M. Blokzijl, et al. Gemini robotics: Bringing AI into the physical world. *arXiv preprint arXiv:2503.20020*, 2025\.  
65. Generalist Team. Gen-0: Embodied foundation models that scale with physical interaction. *Generalist AI Blog*, 2025\.  
66. Generalist Team. Gen-1: Scaling embodied foundation models to mastery. *Generalist AI Blog*, 2026\.  
67. Genesis AI Team. Gene-26.5: Advancing robotic manipulation to human level. *Genesis AI Blog*, May 2026\.  
68. MotuBrain Team, C. Xiang, F. Bao, H. Liu, H. Tan, H. Bi, J. Li, J. Liu, J. Pang, K. Jing, et al. MotuBrain: An advanced world action model for robot control. *arXiv preprint arXiv:2604.27792*, 2026\.  
69. Octo Model Team, D. Ghosh, H. Walke, K. Pertsch, K. Black, O. Mees, S. Dasari, J. Hejna, T. Kreiman, C. Xu, et al. Octo: An open-source generalist robot policy. *arXiv preprint arXiv:2405.12213*, 2024\.  
70. Qwen Team. Qwen3.5: Accelerating productivity with native multimodal agents, February 2026\.  
71. Qwen Team. Qwen-RobotManip technical report: Alignment unlocks scale for robotic manipulation foundation models. 2026\.  
72. H. Touvron, T. Lavril, G. Izacard, X. Martinet, M.-A. Lachaux, T. Lacroix, B. Rozière, N. Goyal, E. Hambro, F. Azhar, et al. LLaMA: Open and efficient foundation language models. *arXiv preprint arXiv:2302.13971*, 2023\.  
73. A. D. Vuong, T. V. Vo, A. Sohail, H. Ding, L. Ma, X. Liang, A. Duan, I. Laptev, and I. Reid. World2Act: Latent action post-training from world model dynamics. *arXiv preprint arXiv:2603.10422*, 2026\.  
74. H. R. Walke, K. Black, T. Z. Zhao, Q. Vuong, C. Zheng, P. Hansen-Estruch, A. W. He, V. Myers, M. J. Kim, M. Du, et al. BridgeData V2: A dataset for robot learning at scale. In *CoRL*, pages 1723–1736, 2023\.  
75. X. Wang, Z. Zhu, G. Huang, B. Wang, X. Chen, and J. Lu. WorldDreamer: Towards general world models for video generation via predicting masked tokens. *arXiv preprint arXiv:2401.09985*, 2024\.  
76. W. Wu, F. Lu, Y. Wang, S. Yang, S. Liu, F. Wang, Q. Zhu, H. Sun, Y. Wang, S. Ma, et al. A pragmatic VLA foundation model. *arXiv preprint arXiv:2601.18692*, 2026\.  
77. M. Xu, H. Zhang, Y. Hou, Z. Xu, L. Fan, M. Veloso, and S. Song. DexUMI: Using human hand as the universal manipulation interface for dexterous manipulation. *arXiv preprint arXiv:2505.21864*, 2025\.  
78. S. Yang, J. Mu, T. Wei, C. Lu, X. Li, L. Xu, Z. Xue, Z. Yuan, D. Lin, J. Pang, et al. MemoryWAM: Efficient world action modeling with persistent memory. *arXiv preprint arXiv:2606.20562*, 2026\.  
79. A. Ye, B. Wang, C. Ni, G. Huang, G. Zhao, H. Li, H. Li, J. Li, J. Lv, J. Liu, M. Cao, P. Li, Q. Deng, W. Mei, X. Wang, X. Chen, X. Zhou, Y. Wang, Y. Chang, Y. Li, Y. Zhou, Y. Ye, Z. Liu, and Z. Zhu. GigaWorld-Policy: An efficient action-centered world-action model. *arXiv preprint arXiv:2603.17240*, 2026\.  
80. J. Ye, N. Gao, S. Yang, J. Zheng, Z. Wang, Y. Chen, P. Chen, Y. Chen, S. Liu, and J. Jia. StarVLA-α: Reducing complexity in vision-language-action systems. In *ECCV*, 2026\.  
81. S. Ye, Y. Ge, K. Zheng, S. Gao, S. Yu, G. Kurian, S. Indupuru, Y. L. Tan, C. Zhu, J. Xiang, et al. World action models are zero-shot policies. *arXiv preprint arXiv:2602.15922*, 2026\.  
82. R. Yu, P. Zhang, S. Liu, B. Liu, M. Kang, S. Li, L. Shi, E. Ma, P. Yang, C. Pan, et al. WALL-OSS-0.5 technical report. *arXiv preprint arXiv:2605.30877*, 2026\.  
83. T. Yuan, Z. Dong, Y. Liu, and H. Zhao. Fast-WAM: Do world action models need test-time future imagination? *arXiv preprint arXiv:2603.16666*, 2026\.  
84. M. Zawalski, W. Chen, K. Pertsch, O. Mees, C. Finn, and S. Levine. Robotic control via embodied chain-of-thought reasoning. *arXiv preprint arXiv:2407.08693*, 2024\.  
85. H. Zhang, L. Xiang, H. Lin, Z. Huang, M. Wang, D. Zhong, Y. Dong, Y. Wu, Y. Rao, D. Zhang, et al. Hy-Embodied-0.5-VLA: From vision-language-action models to a real-world robot learning stack. *arXiv preprint arXiv:2606.14409*, 2026\.  
86. Q. Zhang, L. Li, L. Zhang, S. Yang, Y. Luo, S. Li, R. Wang, J. Wang, J. Shao, G. Xu, et al. Native video-action pretraining for generalizable robot control. *arXiv preprint arXiv:2607.08639*, 2026\.  
87. S. Zhang, Z. Xu, P. Liu, X. Yu, Y. Li, Q. Gao, Z. Fei, Z. Yin, Z. Wu, Y.-G. Jiang, et al. VLABench: A large-scale benchmark for language-conditioned robotics manipulation with long-horizon reasoning tasks. In *ICCV*, pages 11142–11152, 2025\.  
88. H. Zhao, X. Zhao, S. Huang, X. Li, D. Zhao, and Z. Li. RynnWorld-4D: 4D embodied world models for robotic manipulation. *arXiv preprint arXiv:2607.06559*, 2026\.  
89. Q. Zhao, Y. Lu, M. J. Kim, Z. Fu, Z. Zhang, Y. Wu, Z. Li, Q. Ma, S. Han, C. Finn, et al. CoT-VLA: Visual chain-of-thought reasoning for vision-language-action models. In *CVPR*, pages 1702–1713, 2025\.  
90. Z. Zhaxizhuoma, K. Liu, C. Guan, Z. Jia, Z. Wu, X. Liu, T. Wang, S. Liang, P. Chen, P. Zhang, et al. FastUMI: A scalable and hardware-independent universal manipulation interface with dataset. In *CoRL*, pages 3069–3093, 2025\.  
91. H. Zhen, Q. Sun, H. Zhang, J. Li, S. Zhou, Y. Du, and C. Gan. Tesseract: Learning 4D embodied world models. *arXiv preprint arXiv:2504.20995*, 2025\.  
92. J. Zheng, J. Li, Z. Wang, D. Liu, X. Kang, Y. Feng, Y. Zheng, J. Zou, Y. Chen, J. Zeng, et al. X-VLA: Soft-prompted transformer as scalable cross-embodiment vision-language-action model. *arXiv preprint arXiv:2510.10274*, 2025\.  
93. R. Zheng, Y. Liang, S. Huang, J. Gao, H. Daumé III, A. Kolobov, F. Huang, and J. Yang. TraceVLA: Visual trace prompting enhances spatial-temporal awareness for generalist robotic policies. In *ICLR*, pages 54277–54296, 2025\.  
94. L. Zhong, Y. Liu, Y. Wei, Z. Xiong, M. Yao, S. Liu, and G. Ren. ACOT-VLA: Action chain-of-thought for vision-language-action models. *arXiv preprint arXiv:2601.11404*, 2026\.  
95. S. Zhou, Y. Du, J. Chen, Y. Li, D.-Y. Yeung, and C. Gan. RoboDreamer: Learning compositional world models for robot imagination. *arXiv preprint arXiv:2404.12377*, 2024\.  
96. C. Zhu, R. Yu, S. Feng, B. Burchfiel, P. Shah, and A. Gupta. Unified world models: Coupling video and action diffusion for pretraining on large robotic datasets. *arXiv preprint arXiv:2504.02792*, 2025\.  
97. B. Zitkovich, T. Yu, S. Xu, P. Xu, T. Xiao, F. Xia, J. Wu, P. Wohlhart, S. Welker, A. Wahid, et al. RT-2: Vision-language-action models transfer web knowledge to robotic control. In *CoRL*, pages 2165–2183, 2023\.

&nbsp;