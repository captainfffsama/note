#VLA #具身智能 

以下内容均来自 gemini3 pro

# 关于视觉 - 语言 - 动作（VLA）模型最优末端执行器（EEF）姿态表示的技术报告

## 引言

本报告旨在为视觉 - 语言 - 动作（VLA）模型中表示末端执行器（EEF）姿态增量（delta）的三维旋转形式提供一个全面的技术分析。在机器人深度学习领域，旋转表示的选择是一项关键的架构决策，它深刻影响着模型的学习能力、性能表现和实现复杂性。报告将首先对三种主要的候选表示方法——欧拉角（Euler angles）、轴角表示（axis-angle representation）和四元数（quaternions）——进行基础的数学特性梳理。随后，分析的重点将从经典运动学转向现代基于梯度的深度学习框架所提出的特定且往往与直觉相悖的要求。这包括深入探讨“表示连续性”（representation continuity）的概念，这是衡量神经网络回归任务成功与否的关键指标，并揭示所有低维表示方法在该维度上的理论缺陷。作为应对这一挑战的先进理论方案，本报告将引入连续的六维（6D）旋转表示。

至关重要的是，本报告将通过剖析包括谷歌的 RT-1、RT-2 以及开源的 Octo 模型在内的开创性 VLA 模型的动作空间，来弥合理论与实践之间的鸿沟。这些案例研究将旨在解决一个看似矛盾的问题：为何在学术界普遍推崇连续表示的背景下，大规模部署的先进系统（如 RT-1）却成功地采用了经离散化的欧拉角。通过综合理论原则与实证发现，本报告最终将提出一个明确的、基于证据的建议，以指导开发稳健且高性能的 VLA 模型。

## 第 1 节：三维旋转形式的数学回顾

本节为后续的比较分析奠定数学基础，详细阐述每种旋转表示方法的基本原理。

### 1.1 欧拉角：直观但存在奇异性的参数化方法

#### 核心原理

欧拉角通过一系列连续的三次元旋转来描述任意三维姿态，这些旋转围绕一个坐标系的坐标轴进行 1。该方法因其与物理万向节和人类可理解的概念（如横滚 - 俯仰 - 偏航，roll-pitch-yaw）之间的直观联系而广受欢迎 3。

#### 数学公式

最终的姿态是三个旋转矩阵的乘积，例如，一个典型的 Tait-Bryan 序列（XYZ）可以表示为：

$$
R=R_z​(\gamma)R_y​(\beta)R_x​(\alpha)
$$

其中， $R_x​,R_y​,R_z$ ​ 分别代表绕 x、y、z 轴的元旋转矩阵 4。这些旋转的顺序至关重要；不同的序列会产生完全不同的最终姿态 6。存在十二种不同的旋转序列，分为六种“正常欧拉角”（如 ZXZ）和六种“泰特 - 布莱恩角”（如 XYZ）。因此，在任何应用中都必须明确定义所使用的约定 1。

#### 参数化

欧拉角是旋转群 $SO(3)$ 的一种最小化的三参数表示。这种简约性在存储上具有吸引力，但它也正是其主要缺陷的根源。

### 1.2 轴角表示法：旋转的几何基础

#### 核心原理

该表示法基于欧拉旋转定理，该定理指出，任何三维旋转都可以等效为绕某一固定轴 e 旋转一个角度 $\theta$ 7。这提供了一种高度直观的几何解释。

#### 数学公式

轴角通常表示为一个四维向量 $[e_x, e_y, e_z, \theta ]$ ，或者更简洁地表示为一个三维旋转向量（也称欧拉向量） $v = \theta e$ ，其中向量的方向是旋转轴，其模长是旋转角度 8。它到旋转矩阵的转换通过罗德里格斯旋转公式（Rodrigues' rotation formula）实现：  

$$
R = I + (\sin\theta)K + (1 - \cos\theta)K^2
$$

其中，$I$ 是单位矩阵，$K$ 是轴向量 $e$ 的叉乘矩阵 8。

#### 模糊性

这种表示并非唯一。绕轴 e 旋转 $\theta与绕轴-e旋转-\theta是等效的。此外，对于任意整数k$，旋转角度 $\theta与\theta + 2\pi k$ 也是等效的 8。

### 1.3 四元数：代数上优雅且无奇异性的替代方案

#### 核心原理

四元数是复数在四维空间上的扩展，通常表示为 $q=w+x_i+y_j+z_k$ ，其中 w 是标量部分，$(x, y, z)$ 是向量部分 [1, 7, 11]。一个单位四元数（满足 $w^2 + x^2 + y^2 + z^2 = 1$ ）可以表示任何三维旋转。

#### 数学公式

绕单位轴 $e = (e_x, e_y, e_z)$ 旋转角度 $\\theta$ 的操作，可以由单位四元数 q 编码：  

$$ 
q = \left(\cos\frac{\theta}{2}, e_x\sin\frac{\theta}{2}, e_y\sin\frac{\theta}{2}, e_z\sin\frac{\theta}{2}\right) 
$$

这个公式明确显示了四元数是对轴角参数的一种紧凑编码 7。旋转的复合可以通过对应四元数的汉密尔顿积（Hamilton product）来实现，这在计算上非常高效 1。

#### 双重覆盖特性

四元数的一个关键特性是，q 和 -q 表示完全相同的三维旋转。这被称为 SO(3) 的“双重覆盖”（double cover），是其表示模糊性的来源，在学习任务中必须妥善处理 13。

#### 本节要点

这三种表示方法之间存在着深刻的内在联系。欧拉旋转定理确立了旋转的几何本质是“轴”和“角”，这使得轴角表示法成为最“几何纯粹”的形式 7。然而，直接使用轴角来复合旋转非常繁琐 1。四元数通过提供一个简单的代数乘积（ $q_c​=q_b​ \cdot q_a$ ​）解决了这个问题，该乘积直接对应于旋转的复合，并且其数学形式本身就是对轴角参数的有效编码 7。欧拉角则是将一个最终的旋转分解为三个连续的步骤。我们可以从四元数或轴角计算出最终的旋转矩阵，然后反解出一组对应的欧拉角 4。这个反解过程暴露了欧拉角的固有问题：解不唯一且存在奇异点。这揭示了一个抽象层次的递进关系：轴角是几何真理，四元数是高效操作它的代数工具，而欧拉角是对最终结果的一种直观但有问题的参数化描述。

## 第 2 节：经典运动学与动画领域的比较分析

本节基于传统的工程学和计算机图形学标准对各种旋转表示进行评估，从而确立了偏爱四元数的传统观点。

### 2.1 万向节死锁问题：理解欧拉角的运动学奇异性

#### 定义

万向节死锁（Gimbal Lock）是指当三个万向节（或元旋转）中的两个的旋转轴变得共线时，导致系统丧失一个旋转自由度的现象 15。对于一个典型的泰特 - 布莱恩序列（横滚、俯仰、偏航），当俯仰角达到±90 度时，就会发生这种情况 17。

#### 后果

在这种状态下，横滚的变化和偏航的变化会产生相同的旋转效果，使得物体无法被唯一地定向。这是一个欧拉角参数化中的数学奇异性，而非机械故障 17。

#### 影响

万向节死锁会导致运动的抖动和不可预测性，以及动画中的视觉瑕疵。这是因为在奇异点附近，一个微小的期望姿态变化可能需要角度值发生巨大且突然的改变（例如，偏航角瞬间翻转 180 度）17。这种不稳定性使得欧拉角对于需要在整个三维姿态空间中导航的系统而言是不可靠的 20。

### 2.2 插值的黄金标准：四元数的球面线性插值 (Slerp)

#### 线性插值（Lerp）的问题

对欧拉角或旋转矩阵的各个分量进行线性插值，并不能产生匀速的旋转，且可能导致不自然的运动路径。对于四元数，简单的线性插值会导致路径偏离单位超球面，这意味着插值结果在重新归一化之前不是有效的旋转 13。

#### Slerp 的解释

球面线性插值（Spherical Linear Interpolation, Slerp）计算的是在两个单位四元数之间的四维单位超球面上的最短大圆弧上的匀速运动 13。这在三维空间中对应于围绕一个固定轴的平滑、匀速旋转 13。

#### 优越性

Slerp 被认为是旋转动画的黄金标准，因为它能在两个姿态之间生成最短、最自然且扭矩最小的路径 13。它还通过检查四元数的点积来优雅地处理双重覆盖的模糊性，并在必要时翻转其中一个四元数，以确保总是沿着“最短路径”进行插值 13。

### 2.3 实用性比较：计算成本、存储与数值稳定性

#### 存储

欧拉角和轴角（作为旋转向量）是最紧凑的（3 个浮点数）。四元数稍大（4 个浮点数）。旋转矩阵是最大的（9 个浮点数）7。

#### 计算

使用四元数复合旋转速度最快（一次四元数乘法）。欧拉角需要多次三角函数调用和三次矩阵乘法。旋转矩阵需要一次矩阵乘法 1。

#### 数值稳定性

四元数通常具有更好的数值稳定性。由于浮点误差的累积，旋转矩阵可能会逐渐失去其正交性，需要进行昂贵的重正交化处理。而一个长度偏离 1 的四元数可以非常简单且低成本地被重新归一化 1。

#### 本节要点

从经典运动学的角度看，存在一个清晰的优劣层次：欧拉角对人类来说直观，但由于万向节死锁和糟糕的插值特性，对于机器来说存在严重缺陷。四元数在数学上更为优越，提供了无奇异性、计算高效且插值平滑的解决方案。这一传统观点明确指出，对于任何通用的三维旋转系统，万向节死锁是欧拉角的一个根本性、决定性的缺陷 15。同时，Slerp 作为平滑插值的解决方案，与四元数表示法紧密相连，使其成为动画和任何需要平滑轨迹生成的应用领域的默认选择 13。对计算成本和稳定性的进一步分析，更巩固了四元数在重复计算方面相对于欧拉角和旋转矩阵的优越性 7。因此，在深度学习时代之前，工程上的标准建议是使用四元数进行内部计算和存储，仅在绝对必要时（如用户显示或输入）才转换为欧拉角。

## 第 3 节：深度学习范式：为何连续性至关重要

本节引入了以学习为中心的现代视角，围绕一个新的关键属性——表示连续性——重构了整个辩论。

### 3.1 不连续性对基于梯度的优化的影响

#### 学习问题

VLA 模型及其他用于机器人的深度学习系统，通常将动作生成问题建模为一个回归问题。神经网络必须学习一个从高维观测空间（图像、文本）到动作空间（例如，EEF 姿态增量）的连续映射 14。

#### 连续性要求

神经网络作为连续函数（如 ReLU、Sigmoid、线性层）的复合体，天生擅长逼近其他连续函数，但在处理不连续性时会遇到极大困难 25。一个理想的旋转表示应该满足：期望输出旋转的微小变化，对应于其表示值的微小变化。

#### 模糊性带来的问题

像欧拉角（例如，(0, 0, 0\) vs. (0, 0, 360)) 和四元数（q vs. \-q）这样的表示，对于同一个旋转存在多个不同的表示值。如果训练数据中包含在旋转空间中非常接近但在表示空间中相距甚远的样本，标准的 L2 损失函数将产生巨大且相互矛盾的梯度信号。这会严重破坏学习过程的稳定性，并减慢收敛速度 14。

### 3.2 拓扑学障碍：低维表示的内在不连续性

#### 核心定理

Zhou 等人的一篇开创性论文从拓扑学上证明，对于三维旋转群 SO(3)，在四维或更低维的实数欧几里得空间中，所有的表示都必然是不连续的 14。

#### 蕴含

这并非某个特定公式的缺陷，而是一个根本性的拓扑约束。它意味着，如果需要覆盖完整的旋转范围，欧拉角、轴角甚至四元数，在本质上都不适合直接、简单地用于神经网络的回归任务。

#### 实证证据

实验一致表明，与连续的替代方案相比，训练用于回归这些不连续表示的网络收敛更慢，并且误差显著更高，尤其是在表示的不连续点附近 24。

### 3.3 现代神经网络回归的解决方案：连续六维（6D）旋转表示

#### 推导

一个连续的旋转表示可以在六维空间中构建。最常见的形式是提取 3x3 旋转矩阵 $R = [c_1, c_2, c_3]$ 的前两列向量，并将 $[c_1, c_2]$（一个 3x2 的矩阵）作为其六维表示 25。

#### 重构

为了将这个六维表示转换回一个有效的 3x3 旋转矩阵，使用了一种类似格拉姆 - 施密特（Gram-Schmidt）正交化的过程：

1. 将第一列 c1​归一化，得到新的第一列 b1​。  
2. 将第二列 c2​相对于 b1​正交化，然后归一化，得到 b2​。  
3. 第三列 b3​通过计算 b1​和 b2​的叉乘得到 25。

#### 优势

这种表示是连续的。六维向量空间的微小变化会导致最终旋转矩阵的微小变化。它没有奇异点或模糊性问题。实证研究表明，在姿态估计和逆运动学等回归任务中，它的性能始终优于低维表示 14。

#### 本节要点

旋转表示的选择不仅仅是一个输出格式的决策，它从根本上定义了神经网络必须导航的损失函数景观的几何形状。不连续的表示在这个景观中制造了“悬崖”和“瞬移”，使得梯度下降法效率极低。

这个问题的核心在于神经网络如何通过梯度下降来最小化损失函数，通常是在表示空间中的 L2 距离，即 $L=∣∣y_{pred}​−y_{true}​∣∣^2$ 14。设想两个真实旋转

RA​和 RB​无限接近。一个好的表示应该使得它们的表示值 yA​和 yB​也无限接近。然而，对于四元数，可能存在 yA​≈q 而 yB​≈−q 的情况。尽管 RA​≈RB​，但表示空间中的距离 $||q - (-q)||^2$ 却很大。这会给网络一个巨大且错误的梯度信号，指示它进行一次巨大的调整，而实际上只需要一次微小的调整 14。类似地，对于接近万向节死锁点的欧拉角，姿态的微小变化可能要求其中一个角度翻转 180 度，这同样会使损失景观变得病态 27。

相比之下，六维表示通过其从旋转矩阵的构造方式以及通过格拉姆 - 施密特过程的平滑重构，确保了如果 RA​≈RB​，那么它们的六维表示 yA​和 yB​也会很接近。这创造了一个平滑、行为良好的损失景观，非常适合基于梯度的优化。因此，问题的关键不仅仅在于表示本身，还在于该表示如何与损失函数和优化算法相互作用。六维表示是“梯度友好”的。

## 第 4 节：理论与实践的交汇：VLA 模型中的动作表示

本节分析了现实世界中的 VLA 模型，揭示了务实的工程决策如何导致对第 3 节中提出的理论理想的偏离。

### 4.1 RT-1 案例研究：离散化欧拉角的分析

#### RT-1 的动作空间

RT-1 的论文明确将其机械臂的动作空间定义为 7 个连续维度，其中包括用于 EEF 姿态增量的 (roll, pitch, yaw) 31。

#### 关键细节：离散化

至关重要的是，RT-1 并不直接回归这些连续值。相反，这 7 个维度中的每一个都被离散化为 256 个均匀的区间（bins）32。

#### 学习即分类

这一操作将问题从回归（regression）转换为了分类（classification）。Transformer 模型的输出是针对每个动作维度的 256 个区间的概率分布。最终的动作由选择概率最高的区间来确定。

#### 为何这能规避连续性问题

通过将空间离散化，模型不再需要学习一个跨越整个 $SO(3)$ 流形的平滑映射。万向节死锁和表示模糊性等问题，虽然在底层的连续空间中依然存在，但对于分类器而言，它们不会造成病态的损失景观。网络只需学习将一个观测与一个特定的、离散的动作“桶”关联起来。这是一个远比回归更简单、更稳定的学习问题 33。

### 4.2 演进中的动作空间：从 RT-2 的词元化到 Octo 的扩散策略

#### RT-2 的方法

RT-2 直接建立在 RT-1 的基础之上。它将 RT-1 的离散动作区间表示为文本词元（text tokens）37。这使得一个预训练的视觉 - 语言模型（VLM）可以通过微调来输出动作，就好像它们是句子中的单词一样，从而统一了视觉、语言和动作领域 40。其底层的物理动作表示仍然是来自 RT-1 的离散化欧拉角增量。

#### Octo 的飞跃：回归连续控制

Octo 模型代表了一次重大的演进。它摒弃了离散化，回归到连续的动作空间。它通过使用一个*扩散策略*（diffusion policy）作为其动作头（action head）来实现这一点 43。扩散模型是一种生成模型，它学习如何逆转一个添加噪声的过程。在机器人技术中，它们可以学习在观测条件的约束下，从一个噪声先验中生成动作。

#### 扩散模型的意义

扩散模型能够学习复杂的多模态分布。这比一个带有 L2 损失的简单回归模型更适合 $SO(3)$ 的非欧几里得拓扑结构。它可以隐式地学习流形的结构，从而可能使其对底层表示的选择具有鲁棒性（例如，它可以学会处理四元数的双重覆盖特性或避免万向节死锁区域）。Octo 的设计强调灵活性，允许它被微调以适应新的动作空间，这表明其架构并未硬编码为单一的表示方法 45。

### 4.3 综合证据：为何实际实现会偏离理论

#### “增量”的优势

用户的查询是关于 EEF 的*增量*。对于以合理频率运行的控制回路（RT-1 以 3Hz 运行 31），这些增量通常很小。与指令大幅度的绝对姿态变化相比，欧拉角空间的病态区域（即接近万向节死锁的区域）对于小的、增量式的运动来说，造成灾难性失败的可能性较小。

#### 简单性与可解释性

欧拉角实现简单，并且易于人类工程师调试和解释 23。当对其进行离散化时，会得到一个非常直观的动作空间，例如“向前旋转一小点”。

#### 务实的权衡

RT-1 中选择离散化的欧拉角是一个出色的工程折衷。它将一个困难的回归问题简化为一个易于处理的分类问题，从而首次展示了一个大规模、通用的机器人 Transformer。它用连续控制的精度换取了分类的稳定性和简单性，这在当时是一个成功的策略 36。向 Octo 的扩散策略的演进表明，随着模型和训练技术的进步，该领域现在能够更直接地解决更困难的连续控制问题。

#### 本节要点

VLA 模型中动作表示的选择并非一个静态的决定，而是反映了当前学习技术水平以及在模型复杂性、数据需求和期望性能之间权衡的结果。核心的因果关系在于，*离散化*动作空间的决策是使得*欧拉角*成为 RT-1 可行甚至合乎逻辑选择的主要原因，尽管欧拉角在连续回归设置中存在已知的理论缺陷。

我们可以循着一个逻辑链条来理解这一决策过程：首先，面临的问题是如何让 Transformer 输出一个机器人动作，最直接的方式是回归动作参数。然而，第 3 节已经表明，由于不连续性，回归低维旋转表示非常困难。一个解决方案是，如果不做回归，而是做分类呢？通过离散化动作空间 49，问题就转化为了深度网络极其擅长的任务。那么，应该离散化哪种表示呢？旋转需要 3 个自由度，欧拉角提供了三个可以独立离散化的参数（横滚、俯仰、偏航），这既简单又模块化 32。在分类的背景下，离散化一个六维向量或四维四元数不仅不直观，也带不来明显的好处。因此，决策顺序很可能是：

需要稳定的学习信号 \-\> 选择分类而非回归 \-\> 离散化动作空间 \-\> 使用欧拉角作为最简单的独立离散化基础。这解释了 RT-1 的设计。

接下来的问题是：在证明这种方法有效之后，下一步是什么？主要缺点是精度的损失 48。如何恢复精度？需要回到连续的动作空间。新的障碍是：如何在不遇到回归问题的情况下实现连续控制？答案是使用更强大的生成模型。扩散模型（如 Octo 所用 43）非常适合此目的，因为它们可以学习复杂的 $SO(3)$ 流形上的有效动作的底层分布。这解释了该领域的演进方向。

## 第 5 节：最终建议与实施指南

本节综合所有先前的分析，为用户提供一套具体、可操作的建议，并以一个总结性表格作为快速参考。

### 5.1 旋转表示总结表

此表旨在对四种关键表示方法在 VLA 模型开发相关指标上进行密集、一目了然的比较，作为本报告发现的执行摘要。

表 1：用于 VLA 模型的旋转表示方法的比较分析

| 指标 | 欧拉角 (横滚, 俯仰, 偏航) | 轴角 (旋转向量) | 单位四元数 | 6D 连续表示 |
| :---- | :---- | :---- | :---- | :---- |
| **维度** | 3 | 3 (或 4\) | 4 | 6 |
| **奇异性问题** | 是 (万向节死锁) | 是 (零角度时) | 否 | 否 |
| **神经网络回归的连续性** | 不连续 | 不连续 | 不连续 | **连续** |
| **插值质量** | 差 / 不唯一 | 复杂 | **优秀 (Slerp)** | 简单 (Lerp) |
| **复合复杂度** | 高 (矩阵乘积) | 高 | **低 (汉密尔顿积)** | 高 (需矩阵转换) |
| **模糊性** | 多种有效序列 | (θ, e) vs. (-θ, \-e) | q vs. \-q (双重覆盖) | 无 |
| **对离散化分类的适用性** | **优秀** (轴独立) | 中等 | 中等 | 差 (维度高) |
| **对连续回归的适用性** | 差 (奇异性, 不连续) | 差 (不连续) | 差 (不连续, 模糊性) | **优秀** (连续性) |

### 5.2 关于表示 EEF 姿态增量的最终建议

最优选择直接取决于 VLA 模型的动作空间是**连续的**还是**离散的**。

* **建议 1：对于离散动作空间（分类方法）**  
  * **首选：欧拉角（横滚、俯仰、偏航）。**  
  * **理由：** 正如 RT-1 和 RT-2 的成功所证明的，将每个欧拉角轴离散化为一组区间是一种简单、稳定且高效的策略 32。它将一个在非欧几里得流形上进行回归的难题，转化为了一个标准的分类问题，而 Transformer 架构对此类问题非常擅长。万向节死锁和插值的固有问题在很大程度上被缓解了，因为模型并非在学习一个跨越这些病态区域的连续函数。各轴的独立性也使得实现过程非常直接。  
* **建议 2：对于连续动作空间（回归/生成方法）**  
  * **理论上最优选择：6D 连续表示。**  
  * **理由：** 如果目标是直接回归姿态增量，那么 6D 表示是本文讨论的唯一一种被证明是连续的表示形式，从而避免了困扰低维表示的病态损失景观 14。对于任何使用标准回归损失（如 L1 或 L2）来预测姿态的学习任务，它都是最先进的选择。  
  * **务实的高性能选择：四元数与先进生成模型的结合。**  
  * **理由：** 以 Octo 为代表的行业演进趋势揭示了另一条路径 43。一个强大的生成模型（如扩散模型）可以学习 $SO(3)$ 群的底层拓扑结构，包括四元数的双重覆盖特性。在这种情况下，使用四元数可以非常有效。它们比 6D 表示更紧凑，并且可以受益于现有的庞大机器人工具和库生态系统（例如，用于 Slerp 和各种转换）50。这里的选择取决于开发者是倾向于在  
    *表示层面*（使用 6D）解决连续性问题，还是在*模型层面*（使用扩散模型）解决。

### 5.3 VLA 框架中的实施指南

* **数据记录：** 无论最终选择哪种动作表示，都建议**使用四元数来存储原始轨迹数据**。它们紧凑，对于存储而言没有歧义（如果强制执行某种约定，如标量部分为正），并且可以无损地转换为训练所需的任何其他表示。这为未来的实验提供了最大的灵活性。  
* **动作增量 vs. 绝对姿态：** 推荐使用离散化欧拉角的一个重要前提是模型预测的是*增量*。小的、渐进式的变化不易受到奇异性的最坏影响。如果模型被要求预测绝对的全局姿态，那么即使在离散化设置中，使用连续、无奇异性的表示（如 6D 或四元数）的理由也会变得更加充分。  
* **网络头设计：** 如果使用 6D 表示，网络的输出层应有 6 个神经元。格拉姆 - 施密特过程应被实现为一个可微分的层（例如，在 PyTorch 或 JAX 中），在计算与真实旋转矩阵的损失之前，将 6D 输出转换为 3x3 矩阵。

### 5.4 未来展望：具身智能中动作表示的发展轨迹

* 行业趋势，如从 RT-1 到 Octo 的转变所例证的，是从简单的离散化动作空间向更具表现力的连续动作空间演进。这得益于基础模型（Transformers）能力的日益增强以及复杂生成技术（扩散策略）的发展 43。  
* 随着模型理解和生成复杂数据分布的能力越来越强，通过离散化来简化问题的需求可能会减少。未来的 VLA 模型几乎肯定会在连续动作空间中运行。  
* 在这样的未来，关于四元数和 6D 表示的辩论仍将具有现实意义。6D 表示为标准回归提供了数学上的优雅和简单性，而四元数则提供了紧凑性和丰富的工具生态系统，其主要缺点（模糊性）可通过更先进的模型架构来解决。最终的选择可能取决于计算效率和与现有物理及控制栈集成的便利性等因素 52。

#### 引用的著作

1. Euler angles \- Wikipedia, 访问时间为 八月 27, 2025， [https://en.wikipedia.org/wiki/Euler\_angles](https://en.wikipedia.org/wiki/Euler_angles)  
2. en.wikipedia.org, 访问时间为 八月 27, 2025， [https://en.wikipedia.org/wiki/Euler\_angles\#:\~:text=Euler%20angles%20can%20be%20defined,to%20reach%20any%20target%20frame.](https://en.wikipedia.org/wiki/Euler_angles#:~:text=Euler%20angles%20can%20be%20defined,to%20reach%20any%20target%20frame.)  
3. Frame Rotations and Representations \- Autonomous Robots Lab, 访问时间为 八月 27, 2025， [https://www.autonomousrobotslab.com/frame-rotations-and-representations.html](https://www.autonomousrobotslab.com/frame-rotations-and-representations.html)  
4. Computing Euler angles from a rotation matrix, 访问时间为 八月 27, 2025， [https://eecs.qmul.ac.uk/\~gslabaugh/publications/euler.pdf](https://eecs.qmul.ac.uk/~gslabaugh/publications/euler.pdf)  
5. Euler Angles \-- from Wolfram MathWorld, 访问时间为 八月 27, 2025， [https://mathworld.wolfram.com/EulerAngles.html](https://mathworld.wolfram.com/EulerAngles.html)  
6. Old Gimbal Rotation \- Create 3d Characters, 访问时间为 八月 27, 2025， [https://create3dcharacters.com/old-gimbal-rotation/](https://create3dcharacters.com/old-gimbal-rotation/)  
7. Quaternions and spatial rotation \- Wikipedia, 访问时间为 八月 27, 2025， [https://en.wikipedia.org/wiki/Quaternions\_and\_spatial\_rotation](https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation)  
8. Axis–angle representation \- Wikipedia, 访问时间为 八月 27, 2025， [https://en.wikipedia.org/wiki/Axis%E2%80%93angle\_representation](https://en.wikipedia.org/wiki/Axis%E2%80%93angle_representation)  
9. AxisAngleRotation, 访问时间为 八月 27, 2025， [https://homepages.inf.ed.ac.uk/rbf/CVonline/LOCAL\_COPIES/AV0405/REDSTONE/AxisAngleRotation.html](https://homepages.inf.ed.ac.uk/rbf/CVonline/LOCAL_COPIES/AV0405/REDSTONE/AxisAngleRotation.html)  
10. 3DRotations, 访问时间为 八月 27, 2025， [http://motion.cs.illinois.edu/RoboticSystems/3DRotations.html](http://motion.cs.illinois.edu/RoboticSystems/3DRotations.html)  
11. Quaternions vs Axis \+ angle \- Stack Overflow, 访问时间为 八月 27, 2025， [https://stackoverflow.com/questions/9417246/quaternions-vs-axis-angle](https://stackoverflow.com/questions/9417246/quaternions-vs-axis-angle)  
12. Slerp \- Wikipedia, 访问时间为 八月 27, 2025， [https://en.wikipedia.org/wiki/Slerp](https://en.wikipedia.org/wiki/Slerp)  
13. Better rotation representations for accurate pose estimation \- Towards Data Science, 访问时间为 八月 27, 2025， [https://towardsdatascience.com/better-rotation-representations-for-accurate-pose-estimation-e890a7e1317f/](https://towardsdatascience.com/better-rotation-representations-for-accurate-pose-estimation-e890a7e1317f/)  
14. en.wikipedia.org, 访问时间为 八月 27, 2025， [https://en.wikipedia.org/wiki/Gimbal\_lock\#:\~:text=In%20a%20three%2Ddimensional%20three,a%20degenerate%20two%2Ddimensional%20space.](https://en.wikipedia.org/wiki/Gimbal_lock#:~:text=In%20a%20three%2Ddimensional%20three,a%20degenerate%20two%2Ddimensional%20space.)  
15. Gimbal lock \- Wikipedia, 访问时间为 八月 27, 2025， [https://en.wikipedia.org/wiki/Gimbal\_lock](https://en.wikipedia.org/wiki/Gimbal_lock)  
16. Understanding Gimbal Lock and how to prevent it \- Movella.com, 访问时间为 八月 27, 2025， [https://base.movella.com/s/article/Understanding-Gimbal-Lock-and-how-to-prevent-it](https://base.movella.com/s/article/Understanding-Gimbal-Lock-and-how-to-prevent-it)  
17. Singularity in 3D rotation angle sequences \- Robot Academy, 访问时间为 八月 27, 2025， [https://robotacademy.net.au/lesson/singularity-in-3d-rotation-angle-sequences/](https://robotacademy.net.au/lesson/singularity-in-3d-rotation-angle-sequences/)  
18. Euler (gimbal lock) Explained \- YouTube, 访问时间为 八月 27, 2025， [https://www.youtube.com/watch?v=zc8b2Jo7mno](https://www.youtube.com/watch?v=zc8b2Jo7mno)  
19. Robotic Assembly Using a Singularity-Free Orientation Representation Based on Quaternions \- Lund University Publications, 访问时间为 八月 27, 2025， [https://lup.lub.lu.se/search/files/3341879/3127293.pdf](https://lup.lub.lu.se/search/files/3341879/3127293.pdf)  
20. Spherical Linear Interpolation (Slerp) — splines, version 0.3.3 \- Read the Docs, 访问时间为 八月 27, 2025， [https://splines.readthedocs.io/en/latest/rotation/slerp.html](https://splines.readthedocs.io/en/latest/rotation/slerp.html)  
21. SLERP interpolation – MRPT, 访问时间为 八月 27, 2025， [https://www.mrpt.org/tutorials/programming/maths-and-geometry/slerp-interpolation/](https://www.mrpt.org/tutorials/programming/maths-and-geometry/slerp-interpolation/)  
22. Quaternions vs. Euler Angles \- Stack Overflow, 访问时间为 八月 27, 2025， [https://stackoverflow.com/questions/6002516/quaternions-vs-euler-angles](https://stackoverflow.com/questions/6002516/quaternions-vs-euler-angles)  
23. 16720 Project Report: Rotation Representations in Deep Learning \- Zhengyi Luo, 访问时间为 八月 27, 2025， [https://www.zhengyiluo.com/assets/pdf/Rotation\_DL.pdf](https://www.zhengyiluo.com/assets/pdf/Rotation_DL.pdf)  
24. On the Continuity of Rotation Representations in Neural Networks \- Yi Zhou, 访问时间为 八月 27, 2025， [https://zhouyisjtu.github.io/project\_rotation/rotation.html](https://zhouyisjtu.github.io/project_rotation/rotation.html)  
25. On the Continuity of Rotation Representations in Neural Networks \- ICT Vision & Graphics Lab, 访问时间为 八月 27, 2025， [https://vgl.ict.usc.edu/Publications/2019/On%20the%20Continuity%20of%20Rotation%20Representations%20in%20Neural%20Networks.pdf](https://vgl.ict.usc.edu/Publications/2019/On%20the%20Continuity%20of%20Rotation%20Representations%20in%20Neural%20Networks.pdf)  
26. On the Continuity of Rotation Representations in Neural Networks \- Hao Li, 访问时间为 八月 27, 2025， [https://www.hao-li.com/publications/papers/cvpr2019OCRRNN.pdf](https://www.hao-li.com/publications/papers/cvpr2019OCRRNN.pdf)  
27. On Representation of 3D Rotation in the Context of Deep Learning \- ResearchGate, 访问时间为 八月 27, 2025， [https://www.researchgate.net/publication/384756533\_On\_Representation\_of\_3D\_Rotation\_in\_the\_Context\_of\_Deep\_Learning](https://www.researchgate.net/publication/384756533_On_Representation_of_3D_Rotation_in_the_Context_of_Deep_Learning)  
28. On the Continuity of Rotation Representations in Neural Networks \- ResearchGate, 访问时间为 八月 27, 2025， [https://www.researchgate.net/publication/338509732\_On\_the\_Continuity\_of\_Rotation\_Representations\_in\_Neural\_Networks](https://www.researchgate.net/publication/338509732_On_the_Continuity_of_Rotation_Representations_in_Neural_Networks)  
29. (PDF) Learning rotations \- ResearchGate, 访问时间为 八月 27, 2025， [https://www.researchgate.net/publication/363475740\_Learning\_rotations](https://www.researchgate.net/publication/363475740_Learning_rotations)  
30. Robotics Transformer: RT-1, 访问时间为 八月 27, 2025， [https://robotics-transformer1.github.io/](https://robotics-transformer1.github.io/)  
31. RT-1: Robotics Transformer for real-world control at scale \- Google Research, 访问时间为 八月 27, 2025， [https://research.google/blog/rt-1-robotics-transformer-for-real-world-control-at-scale/](https://research.google/blog/rt-1-robotics-transformer-for-real-world-control-at-scale/)  
32. DCOB: Action space for reinforcement learning of high DoF robots \- Akihiko Yamaguchi, PhD, 访问时间为 八月 27, 2025， [http://akihikoy.net/info/wdocs/Yamaguchi,Ogasawara,2013-DCOB..RL%20of%20high%20DoF%20robots-AURO34-4.pdf](http://akihikoy.net/info/wdocs/Yamaguchi,Ogasawara,2013-DCOB..RL%20of%20high%20DoF%20robots-AURO34-4.pdf)  
33. arxiv.org, 访问时间为 八月 27, 2025， [https://arxiv.org/html/2507.05251v1\#:\~:text=Discrete%20action%20spaces%20simplify%20learning,of%20throttle%20and%20steering%20values.](https://arxiv.org/html/2507.05251v1#:~:text=Discrete%20action%20spaces%20simplify%20learning,of%20throttle%20and%20steering%20values.)  
34. Action Space Reduction Strategies for Reinforcement Learning in Autonomous Driving This work was supported in part by the National Science Foundation (NSF) under Grant MRI 2214830\. \- arXiv, 访问时间为 八月 27, 2025， [https://arxiv.org/html/2507.05251v1](https://arxiv.org/html/2507.05251v1)  
35. Continuous Control with Action Quantization from Demonstrations, 访问时间为 八月 27, 2025， [https://google-research.github.io/aquadem/](https://google-research.github.io/aquadem/)  
36. RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control, 访问时间为 八月 27, 2025， [https://proceedings.mlr.press/v229/zitkovich23a.html](https://proceedings.mlr.press/v229/zitkovich23a.html)  
37. RT-2: The Vision-Language-Action Model | by shashank Jain | Medium, 访问时间为 八月 27, 2025， [https://medium.com/@jain.sm/came-across-this-paper-from-google-deepmind-of-visual-language-action-model-269dbbdcf4b2](https://medium.com/@jain.sm/came-across-this-paper-from-google-deepmind-of-visual-language-action-model-269dbbdcf4b2)  
38. Vision-Language-Action Models Transfer Web Knowledge to Robotic Control \- RT-2, 访问时间为 八月 27, 2025， [https://robotics-transformer2.github.io/assets/rt2.pdf](https://robotics-transformer2.github.io/assets/rt2.pdf)  
39. Vision Language Action Models (VLA) Overview: LeRobot Policies Demo, 访问时间为 八月 27, 2025， [https://learnopencv.com/vision-language-action-models-lerobot-policy/](https://learnopencv.com/vision-language-action-models-lerobot-policy/)  
40. RT 2 | PDF | Robotics | Computing \- Scribd, 访问时间为 八月 27, 2025， [https://www.scribd.com/document/724547947/rt2](https://www.scribd.com/document/724547947/rt2)  
41. RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control \- arXiv, 访问时间为 八月 27, 2025， [https://arxiv.org/pdf/2307.15818](https://arxiv.org/pdf/2307.15818)  
42. An Open-Source Generalist Robot Policy \- Octo, 访问时间为 八月 27, 2025， [https://octo-models.github.io/paper.pdf](https://octo-models.github.io/paper.pdf)  
43. Octo: An Open-Source Generalist Robot Policy, 访问时间为 八月 27, 2025， [https://octo-models.github.io/](https://octo-models.github.io/)  
44. Octo \- AI Agent Index \- MIT, 访问时间为 八月 27, 2025， [https://aiagentindex.mit.edu/octo/](https://aiagentindex.mit.edu/octo/)  
45. Octo: An Open-Source Generalist Robot Policy \- arXiv, 访问时间为 八月 27, 2025， [https://arxiv.org/html/2405.12213v1](https://arxiv.org/html/2405.12213v1)  
46. Octo: An Open-Source Generalist Robot Policy \- YouTube, 访问时间为 八月 27, 2025， [https://www.youtube.com/watch?v=99667VDGWMg](https://www.youtube.com/watch?v=99667VDGWMg)  
47. Decisiveness in Imitation Learning for Robots \- Google Research, 访问时间为 八月 27, 2025， [https://research.google/blog/decisiveness-in-imitation-learning-for-robots/](https://research.google/blog/decisiveness-in-imitation-learning-for-robots/)  
48. Discretizing Continuous Action Space for On-Policy Optimization, 访问时间为 八月 27, 2025， [https://ojs.aaai.org/index.php/AAAI/article/view/6059/5915](https://ojs.aaai.org/index.php/AAAI/article/view/6059/5915)  
49. Quaternion transformations in Python \- ros \- Robotics Stack Exchange, 访问时间为 八月 27, 2025， [https://robotics.stackexchange.com/questions/53148/quaternion-transformations-in-python](https://robotics.stackexchange.com/questions/53148/quaternion-transformations-in-python)  
50. \[ROS Q\&A\] 044 \- How to convert quaternions to Euler angles \- YouTube, 访问时间为 八月 27, 2025， [https://m.youtube.com/watch?v=mFpH9KK7GvI](https://m.youtube.com/watch?v=mFpH9KK7GvI)  
51. PyBullet Quickstart Guide \- GitHub, 访问时间为 八月 27, 2025， [https://raw.githubusercontent.com/bulletphysics/bullet3/master/docs/pybullet\_quickstartguide.pdf](https://raw.githubusercontent.com/bulletphysics/bullet3/master/docs/pybullet_quickstartguide.pdf)  
52. pydrake.multibody.tree, 访问时间为 八月 27, 2025， [https://drake.mit.edu/pydrake/pydrake.multibody.tree.html](https://drake.mit.edu/pydrake/pydrake.multibody.tree.html)