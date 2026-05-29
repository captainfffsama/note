#manipulation #alphaXiv #具身智能 

# Vision-based Manipulation from Single Human Video with Open-World Object Graphs
- 时间： 2025.9.4
- 文章：[[2405.20321] Vision-based Manipulation from Single Human Video with Open-World Object Graphs](https://arxiv.org/abs/2405.20321)
# 引言

机器人在开放世界环境中的操作是机器人学中最具挑战性的问题之一，它要求系统能够适应超出受控实验室条件的新颖物体、环境和情况。传统的机器人技能习得方法通常依赖于大量的人工工程、远程操作数据收集或复杂的仿真环境——所有这些方法都成本高昂且耗时，并且难以泛化到其特定的训练领域之外。

本文介绍了 ORION（Open-woRld video ImitatiON），一种使机器人能够从单个人类演示视频中学习复杂操作技能的方法。与传统方法不同，ORION 可以使用移动设备捕获、从互联网获取甚至合成生成的现成视频，极大地降低了机器人技能习得的障碍，同时在多样化的环境中实现了强大的泛化能力。

![](../../Attachments/Vision-based%20Manipulation%20from%20Single%20Human%20Video%20with%20Open-World%20Object%20Graphs_fig1.png)

 _图 1：ORION 从单个人类演示视频中学习可泛化的机器人操作策略。该系统利用视觉基础模型提取开放世界对象图（OOGs），捕捉以对象为中心的时空信息，从而能够在具有不同视觉背景、摄像机角度和空间布局的各种测试环境中进行部署。_

# 核心方法论

ORION 通过两阶段方法解决开放世界操作的挑战：从演示视频中提取操作计划和合成机器人策略以执行动作。

## 操作计划生成

第一阶段将单个人类演示视频转换为一系列开放世界对象图（OOGs）——一种捕获任务相关对象及其随时间交互的结构化表示。此过程利用了多个视觉基础模型：

**对象识别和跟踪**：ORION 首先使用像 GPT-4o 这样的视觉 - 语言模型（VLM）分析视频帧并识别任务相关对象，区分被操作对象和参考对象。然后，这些对象使用 Grounded-SAM 进行分割，并使用专门的模型（如用于 RGB-D 视频的 Cutie 或用于仅 RGB 视频的 FoundationPose）在视频帧中进行跟踪。

**关键帧发现**：系统自动识别演示中对象接触关系或运动模式发生显著变化的关键时刻。对于 RGB-D 视频，这涉及使用 Co-Tracker 跟踪 3D 关键点并对速度统计数据应用无监督变化点检测。对于仅 RGB 视频，该方法分析 6-DoF 姿态轨迹以识别这些过渡点。

**开放世界对象图构建**：在每个关键帧处，ORION 构建一个 OOG，通过两级图结构表示场景状态：

*   **高级节点**：对象节点，存储彩色 3D 点云或网格表示；抓取节点，包含接触点和抓取状态等交互线索。
*   **低级节点**：对应于对象关键点的点节点，存储描述关键帧之间运动的以对象为中心的特征轨迹。
*   **边**：节点之间的连接，具有指示接触关系的二进制属性。

由此产生的一系列 OOGs，表示为 $\{G_0,G_1\}$ ，构成了从演示中提取的操作计划。

### 机器人策略合成

第二阶段将提取的 OOG 序列转换为可执行的机器人动作，通过一种强调对象运动相似性而非直接模仿人类动作的轨迹变形方法。

**OOG 匹配**：在执行过程中，机器人的当前观察结果会生成一个 OOG，该 OOG 根据接触关系与演示计划进行匹配。这标识了当前状态并确定了操作序列中的下一个目标状态。

**以物体为中心的轨迹扭曲**：ORION 对演示中的物体轨迹进行扭曲，使其适应机器人当前的环境。对于 RGB-D 视频，这涉及平移扭曲，该扭曲对齐点云并缩放轨迹，同时保留方向模式。对于仅有 RGB 的视频，使用球形线性插值（SLERP）等技术进行平移和旋转扭曲，以实现平滑的姿态过渡。

扭曲过程可以数学表示为：

$$
\tau_{target}^{Ro} = W(\tau_{target}^V, o, \hat{o})
$$

其中 $\tau^V_{target}$ 是原始轨迹， $o$ 和 $\hat{o}$ 是当前和演示物体配置，而 $W$ 代表扭曲函数。

**动作优化**：扭曲后的轨迹指导机器人末端执行器 SE(3) 变换的优化，最大限度地减少预测轨迹点与末端执行器运动之间的差异，同时结合来自抓取节点的抓取信息。

![](../../Attachments/Vision-based%20Manipulation%20from%20Single%20Human%20Video%20with%20Open-World%20Object%20Graphs_fig2.png)

 _图 2：ORION 规划提取阶段的详细流程图，展示了视觉基础模型如何处理 RGB 或 RGB-D 演示，以创建捕获物体轨迹和交互点的结构化 OOG 表示。_

# 实验验证

作者对多项操作任务进行了广泛的实验，以评估 ORION 的有效性和鲁棒性。实验设置包括在不同环境中执行的短时任务（单步操作）和长时任务（多步序列）。

## 任务表现

ORION 在所有测试任务中取得了 74.4% 的平均成功率，展示了其从少量人类输入中学习并泛化到未见环境的能力。该系统在需要精确物体放置和组装操作的任务中表现出特别的优势。

![](../../Attachments/Vision-based%20Manipulation%20from%20Single%20Human%20Video%20with%20Open-World%20Object%20Graphs_fig3.png)

 _图 3：RGB-D 演示任务的性能结果。ORION 在各种操作场景中取得了持续的成功率，从简单的放置任务到复杂的组装操作。_

对于仅有 RGB 的演示，ORION 取得了更高的平均成功率，达到 85.3%，这表明该方法可以通过先进的单目深度估计和 6 自由度姿态跟踪模型有效补偿深度信息的缺失。

![](../../Attachments/Vision-based%20Manipulation%20from%20Single%20Human%20Video%20with%20Open-World%20Object%20Graphs_fig4.png)

 _图 4：来自作者录制、YouTube 视频和合成 Veo 2 内容等多种来源的仅有 RGB 演示任务的结果。持续的高性能表明 ORION 能够处理现成的视频源。_

## 对比分析

消融研究揭示了 ORION 设计选择的关键重要性。与直接模仿人类手部轨迹的手部运动模仿基线相比，ORION 显著优于该替代方法，验证了以物体为中心的抽象策略。

![](../../Attachments/Vision-based%20Manipulation%20from%20Single%20Human%20Video%20with%20Open-World%20Object%20Graphs_fig5.png)

 _图 5：ORION 与基线进行比较的消融研究。以物体为中心的方法显著优于手部运动模仿和密集对应方法。_

与使用光流的密集对应基线进行比较进一步强调了通过 Co-Tracker 进行鲁棒关键点跟踪的重要性。光流方法在关键帧识别方面存在困难，导致不正确的操作计划和糟糕的任务执行。

## 环境鲁棒性

一个特别重要的发现是 ORION 对多样化视觉条件的鲁棒性。在各种截然不同的环境中（厨房、办公室、户外环境）录制的演示视频中测试“杯子放在杯垫上”任务，结果显示策略性能没有统计学上的显著差异，证实了该方法的开放世界能力。

![](../../Attachments/Vision-based%20Manipulation%20from%20Single%20Human%20Video%20with%20Open-World%20Object%20Graphs_fig6.png)

 _图 6：跨多样化视觉环境的鲁棒性评估。尽管背景、光照和摄像机视角存在显著差异，ORION 仍保持一致的性能。_

# 技术贡献与意义

ORION 引入了几项关键技术贡献，推动了机器人操作学习的发展：

**开放世界对象图（OOGs）**：本文提出了一种结构化的、基于图的表示，它能有效地从演示视频中抽象出与任务相关的信息，同时保持对象之间的空间和时间关系。这种表示对于跨不同环境和对象实例的泛化至关重要。

**基础模型集成**：ORION 展示了如何有效地协调多个视觉基础模型（VLM、分割模型、跟踪系统、3D 重建工具），以构建一个鲁棒的机器人学习感知管道。这种集成使得系统能够利用通过互联网规模训练发展起来的广泛视觉理解能力。

**以对象为中心的轨迹扭曲**：该方法通过扭曲对象轨迹而非直接模仿人类动作，实现了跨不同实体和空间配置的有效迁移。其数学公式在适应新的环境约束的同时，保留了基本的运动特征。

这项工作的意义超越了其技术贡献。通过使机器人能够从现成的人类视频中学习，ORION 有可能使机器人技能获取民主化，并为通过在线海量人类演示数据库扩展机器人能力开辟了道路。

# 局限性与未来方向

作者承认了一些局限性，这些局限性指明了未来的研究方向：

**目标规范**：当前方法主要处理由接触状态定义的目标，避免了空间关系推断的复杂性（例如，“放置在旁边”的关系）。扩展到更细致的目标理解是未来工作的一个重要方向。

**摄像机运动**：ORION 假设演示期间摄像机是静止的，这限制了其对所有可用视频内容的适用性。将该方法扩展以处理动态摄像机运动将显著拓宽其潜在的视频来源。

**基础模型依赖**：虽然利用基础模型带来了显著优势，但系统的性能与这些模型的能力及其潜在的故障模式密切相关。视觉基础模型的持续改进将直接提高 ORION 的鲁棒性和适用性。

尽管存在这些局限性，ORION 代表了通过人类观察实现实用和可扩展机器人操作的重大进展，为机器人如何从现代世界中丰富的视觉数据中快速获取新技能提供了一个引人注目的愿景。

# 相关引用
## 基于观察的模仿学习

本论文定义了核心研究问题，即“基于观测的模仿学习”，其中策略是从仅包含状态的演示中学习的。ORION 论文明确地将自身工作置于此背景下，使其成为其问题表述的基础性引用。

F. Torabi. Imitation Learning from Observation. PhD thesis, University of Texas at Austin, 2021. PhD Thesis.

## Okami ：通过单个视频模仿教授人形机器人操作技能

本文被引证为解决同一问题的最直接的现有方法。作者以 OKAMI 作为关键的比较点，突出他们自身方法独特的面向对象方法，与 OKAMI 以示教者为中心的关注点形成对比，从而阐明了他们的独特贡献。

J. Li, Y. Zhu, Y. Xie, Z. Jiang, M. Seo, G. Pavlakos, and Y. Zhu. Okami: Teaching humanoid robots manipulation skills through single video imitation, 2024. URL https://arxiv. org/abs/2410.11792.

## 协同追踪器 ：共同追踪更佳

CoTracker 是 ORION 管线中的一个关键技术组件，用作跟踪任意点 (TAP) 模型，以在整个视频中跟踪对象的关键点。这种跟踪对于发现关键帧以及生成 ORION 方法核心的以对象为中心的操作计划至关重要。

N. Karaev, I. Rocco, B. Graham, N. Neverova, A. Vedaldi, and C. Rupprecht. Cotracker: It is better to track together. arXiv preprint arXiv:2307.07635, 2023.

## Foundationpose ：统一的新颖物体 6D 姿态估计与跟踪

本研究提供了 6 自由度姿态估计模型，使 ORION 系统能够仅凭 RGB 视频进行操作，这是其一项关键功能，也是实验评估的重要组成部分。FoundationPose 是使 ORION 无需深度信息即可跟踪物体轨迹的专用工具。

B. Wen, W. Yang, J. Kautz, and S. Birchfield. Foundationpose: Unified 6d pose estimation and tracking of novel objects. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 17868–17879, 2024.