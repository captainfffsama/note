---
title: "Flow Matching 与 DDIM 小记"
source: "https://duanyll.com/2025/6/21/Flow/"
author:
  - "[[duanyll]]"
published: 2025-06-21
created: 2026-11-10
description: "背景故事 2022 年 10 月左右，同期有三篇关于 Flow 模型用于图像生成的论文发表，并都被 ICLR 2023 录用：  Flow Matching for Generative Modeling Learning to Generate and Transfer Data with Rectified Flow Building Normalizing Flows with Stocha"
tags:
  - "clippings"
---
## 背景故事

2022 年 10 月左右，同期有三篇关于 Flow 模型用于图像生成的论文发表，并都被 ICLR 2023 录用：

- [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747)
- [Learning to Generate and Transfer Data with Rectified Flow](https://arxiv.org/abs/2209.03003)
- [Building Normalizing Flows with Stochastic Interpolants](https://arxiv.org/pdf/2209.15571)

这三篇文章从不同角度出发得推导出了同一个生成模型，下面再整理一下三篇论文的故事讲法。

### Flow Matching

背景：连续归一化流

连续归一化流（Continuous Normalizing Flows, CNF）的概念最早由 [Chen et al. (2018)](https://arxiv.org/abs/1806.07366) 提出，CNF 通过求解常微分方程（ODE）来实现数据的变换。时间依赖的概率密度路径 满足 , 流的动态由时间依赖的向量场 定义：

CNF 可以通过推前操作将先验分布 转换为目标分布 （理论上的结论，并不能用于采样）：

同时连续性方程可用于验证向量场 是否生成满足以上性质的概率路径：

CNF 可以用于建模任意的概率路径，然而此前不存在高效的方法 [^1] 来学习向量场 。Flow Matching 论文提出了条件流匹配方法（Conditional Flow Matching, CFM）实现从数据集中采样学习向量场 。首先引入 Flow Matching 损失函数，用神经网络 来回归近似向量场 ：

再来进一步考虑如何构造可从数据集中采样求得的 和 。将数据集中的样本 视作条件，那么 作为条件概率路径 的边缘分布可以通过积分得到：

类似的，向量场 可以视作条件向量场 的边缘分布，通过积分得到:

原文证明了条件向量场 确实能生成条件概率密度路径 。进一步，论文证明了条件流匹配损失函数

的梯度等价于 FM 损失函数 的梯度，故可以使用 CFM 损失函数训练 CNF 模型。此结论并未依赖于具体的条件概率路径 或条件向量场 的设计。

在图像生成模型中，一般把高斯噪声作为先验分布，即 。流映射在 条件下总是可以写成

可以证明此时的条件速度场具有唯一的形式

DDIM 的均值 和方差 可以直接带入到上式中，得到 Flow Matching 框架下的条件速度场。

论文又提出 OT 速度场的设计，使用线性插值的均值 和方差 。通过实验发现 OT 速度场训练收敛更快且 FID 更低。

### Rectified Flow

背景：传输映射问题

给定 上两个可以观测的分布 ，目标是寻找传输映射 将 映射到 。

在生成建模中， 是未知但可观测目标分布（图像数据集），而 是简单的先验分布（如高斯分布）。还可以考虑传输建模，即 和 都是未知的可观测分布。

最优传输（Optimal Transport, OT）目标希望最小化某个代价函数 ，即

其中 可以是 L2 范数或则其他度量。一般的 OT 问题非常难解，在图像生成任务中也不太关心最小化代价。Rectified Flow 和 Reflow 是此背景下一种实用的近似方法。

Rectified Flow 使用一个简单的 ODE 来学习传输映射 。从 出发，沿速度场 进行演化，得到 ：

问题转换为学习速度场 。直观的想法（先前的工作）是想最小化 ODE 采样结果 的分布 和目标分布 的距离 ，如 KL 散度，但多次模拟求解 ODE 的代价太高了。发现此问题中，速度场 实际上是过参数化的：我们只关心开头和结尾的分布应当是 和 ，而中间的分布并不重要，可以人为地设置非常强的先验。简单的选择是让中间分布是开头结尾的某种线性插值，走直线路径，这样不仅能更接近 OT 目标，还能减少 ODE 的求解次数。

对源分布和目标分布中观测到的样本 和 ，定义线性插值

则 满足以下 ODE：

还不能直接用这个 ODE 作为速度场 ，因为它不是因果的， 依赖于最终状态 。论文提出 Rectify 过程来因果化速度场 。思路如下：

Rectify 过程的 Rewire 特点

1. 不同的插值轨迹 之间可能存在相交点，在同一个 处会有多个可能的 取值，因为不知道要沿哪条轨迹走。
2. 另一方面，在 ODE 中， 必须由 唯一确定，于是不同的 轨迹是不能相交的。
3. 于是在相交点处， 可以取值为 在所有可能的 取值上的平均值。
4. 导致原本的直线轨迹 必须在相交点处弯曲来避免相交，并改变原分布和目标分布的映射关系。

以上 Rectify 过程更新了传输映射 ，但可证明保持了边缘分布 和 不变。训练目标为

代入线性插值轨迹得到训练目标

> 此训练目标就可以直接用在代码里了。

Rectify 过程将 轨迹转换为因果的 轨迹，不仅保持边缘分布不变，还可以证明对于凸的代价函数 ，传输代价不增

Reflow

Reflow 拉直轨迹

原本 轨迹是直的，经过 Rectify 过程后 会在相交点弯曲，起不到一开始选择线性插值轨迹时，保持直线的优势。论文提出 Reflow 过程来修正这个问题，使用 Rectify 过程得到的 配对耦合重新训练速度场 并反复迭代：

论文证明了对于下面的 “直线程度” 度量

满足

随 收敛到 0。也就是说，经过 Reflow 过程后， 轨迹会越来越接近直线。

Reflow 其实就是从 OT 角度解释了先前的时间步蒸馏技巧。常见的预训练模型都没有做 Reflow 蒸馏。

### Stochastic Interpolants

WIP.

### Diffusion

DDPM 和 DDIM 也有很多种推导的方法，下面摘自 [苏神博客](https://spaces.ac.cn/archives/9164) 中贝叶斯角度的推导，比较接近 DDIM 论文的推导方式。

#### DDPM

与 GAN 类似，DDPM 模型将生成定义为将一个随机噪声 变换成真实图像 的过程。然而，像 GAN 一样一步将噪声映射到真实图像的分布是困难的，因此 DDPM 考虑逐步地将噪声 变换为真实图像。首先定义一个前向过程，即将真实图像样本 逐步添加噪声，直到完全成为高斯噪声，则生成图像的过程即为将噪声 逐步变为样本数据的反向过程。前向过程的每个步骤可以表示为

其中 是真实图像样本数据， 是完全随机的高斯噪声， 是设置的时间步数。 和 是控制每一步添加噪声强度的系数。前向过程也可以写成条件概率分布的形式

其中 代表在前向过程中 到 的条件概率分布。

注意到如果规定 ，则可以从 直接推导出

于是 中包含 个独立的高斯噪声 的线性组合。代入 的条件，得到

将 记为 ， 记为 ${\_t}，则 关于 的条件概率分布为

也可以写成

只要 ，经过充分多的时间步后， 就会足够接近标准高斯分布 。利用 ，可以快速从 采样到 。接下来考虑反向过程，关键在于求条件概率分布 以便从噪音多的时间步还原噪音少的时间步。根据贝叶斯公式

然而， 和 是未知的。考虑引入 作为条件变量，则

代入 和 ，可以写出 服从的分布

使用此分布采样 需要已知 ，然而 正是最终要生成的图像。DDPM 引入可学习的函数 来近似估计未知的 ，训练的目标为最小化损失函数 。DDPM 实际上没有使用神经网络 直接预测原始图像 ，而是用于预测添加的噪声 ，这样在每一步上神经网络的输出都服从标准高斯分布，有助于提高训练的稳定性。根据 ，可以重参数化 为

进而得到完整的损失函数

最后，用 近似 代入 ，得到与 无关的反向过程条件概率分布

并进一步得到 DDPM 的迭代采样公式

其中 ，。

在 DDPM 原始的实验中， 取值为 1000， 根据经验设置为 . 观察 ，可以发现每一步的采样都需要计算神经网络 ，因此计算量非常大。同时 DDPM 的每一步采样都包括对随机噪声 的采样，结果具有不确定性。

#### DDIM

DDIM（Denoising Diffusion Implicit Models）模型在 DDPM 的基础上进一步推导了采样公式，发展了允许跳过时间步和确定性的采样方法，并允许将扩散模型的采样过程与常微分方程和随机微分方程的求解算法结合起来。本节将介绍 DDIM 的主要推导过程和结论。

注意到在 DDPM 的神经网络 的训练过程 中，只依赖了关于 的假定 ，而与具体的 的设置 无关，因此在求解贝叶斯公式 时，不再代入 ，而是假设 服从高斯分布

则关于 的采样公式可以写成

比较上式与 的系数，消去 和 ，得到

是关于 服从的分布的通解，具有自由参数 . 沿用 DDPM 训练的神经网络模型，用 近似 代入 ，得到 DDIM 的迭代采样公式

公式中的三项可分别视作从 预测的 ，从当前步指向 的方向和随机噪声。注意到 是可自由选取的超参数，当 时， 变为确定性的采样方式

另外，在推导 的过程中并不依赖于 ，同样的过程也适用于推导任意两个时间步 和 之间的采样公式

这实质上允许了以任意的时间步序列 进行采样，不仅可以跳过时间步节约计算量，还允许确定性地从图像样本 完全或者部分地反推对应的带噪声图像 ，为带参考图像作为条件的图像生成和编辑提供了注入参考图像信息的方法，称为 DDIM 反转。 也可以视作使用欧拉法求解常微分方程，可以推导出对应的常微分方程形式或者随机微分方程的形式，然后使用 Heun 法、Runge-Kutta 法等效率更高的数值求解方法进行采样。

## Flow 与 DDIM 联系

三篇提出 Flow 模型的论文都分析了 DDIM 模型在 Flow 框架中的地位。在 FM 论文中，可以使用 DDIM 的均值和方差特性来设计条件速度场 ，从而得到 Flow Matching 的条件流匹配损失函数 。在 Rectified Flow 框架和 Stochastic Interpolants 框架中，DDIM 相当于使用球面插值取代了线性插值，并应用了时间步缩放。

后续有一些博客文章整理和可视化了 Flow Matching 和 DDIM 的关系：

- [Diffusion Meets Flow Matching](https://diffusionflow.github.io/) - DeepMind
- [Let us Flow Together](https://rectifiedflow.github.io/) - UT Austin

Flow 和 Diffusion 的关系可以从重参数化、插值方式和 ODE/SDE 的角度来讨论。

### 重参数化

此处重参数化指的是在模型中神经网络预测的对象不一样。DDIM 采样公式可以写成以下简单形式：

其中 是从当前步对最终干净图像的预测， 是当前步对噪声的预测。一般 Diffusion 模型中神经网络预测的是 ，也可以选择预测 。

Flow 模型的采样公式为

神经网络预测速度场 。于是发现无论神经网络预测的是 ， 还是 ，采样公式都可以统一地写成

其中 是神经网络， 和 是

|  |  |  |
| --- | --- | --- |
|  |  |  |
|  |  |  |
|  |  |  |
|  |  |  |

表中 指一般的 Flow Matching 模型， 特指 Rectified Flow 模型的线性插值速度场。

> 此处跳过 DeepMind 博客中关于采样时重参数化和缩放的讨论，在下一小节讨论插值时会提到。

考虑重参数化对训练过程的影响。常用的 DDIM 的损失函数为

其中 是对数信噪比， 是人为选定的时间步权重函数。

在 Stable Diffusion 3 中，CFM 训练的损失函数中也引入了人为选择的时间步权重函数：

为了便于比较，我们把这些损失函数都重参数化使用 来表达

|  |  | 说明 |
| --- | --- | --- |
|  |  | 小时，过于关注 的误差 |
|  |  | 大时对 的误差不敏感 |
|  |  |  |
|  |  | 在两种情形之间取得平衡 |

表明神经网络的不同预测对象意味着在预测 的基础上，对损失函数施加了不同的时间步权重。

不同模型中等效的时间步权重

结合考虑重参数化的影响，SD3 的损失函数时间步权重实际上很接近 EDM.

### 插值

Rectified Flow 使用线性插值构建 的轨迹：

而 DDIM 和其他概率流 ODE 可以写成更通用的仿射插值形式

> 具体来说, 时间步均匀的球面插值可以写成
> 
> 而 DDIM 相当于在球面插值的基础上缩放了时间步

看起来在训练之前就必须选择一种插值形式，并且在采样时也必须使用同样的插值形式，因为速度场 是在特定的插值形式上训练的。

实际上，在一个很宽松的条件下，可以证明，使用任何插值形式训练并使用不同的插值形式采样，使用相同的训练集数据对 经过 Rectify 过程后，（在 **理想** 的速度场上）采样得到的样本对 是相同的。

不同插值方式改变走过的路径但不影响端点

以下是一种感性的理解方式：如果两种插值过程能以一种可微的方式互相变换，意味着插值的轨迹也可以按照相同的方式互相变换。考虑两条相交的 轨迹 ，Rectify 过程只是在轨迹相交的点把轨迹重新连接为 从而避免轨迹相交，因此无论怎么平滑地变换轨迹，Rectify 之后都会得到相同的连接关系。也就是 Rectify 后的 Coupling （采样结果）与插值方式无关，只与数据集中的样本对 有关。

形式化来说, 对于两条插值路径 和 , 如果存在可微分的映射

并且 可逆, 满足

则称其为逐点可变换的. 如果两组插值路径 和 来自相同的 Coupling, 起止点相同 且逐点可变换, 和 由 Rectify 过程得到:

则可证明以下结论:

1. 和 可以使用同一个映射来变换
2. 和 的 Coupling 相同，即采样结果相同
3. 设 和 分别是 和 的速度场，则

可以交换 Rectify 和变换操作

具体对仿射插值 (直线, 球面, DDIM) 来说: 对于两种仿射插值轨迹 和 ，存在标量函数 和 满足

其中 和 是方程组

对于实际使用的 一般难以写出显式的 和 ，但可以通过数值方法求解。 相当于缩放时间步, 相当于缩放向量的模长.

用数值方法解出直线, 球面插值和 DDIM 之间的变换关系

进一步得到适用于仿射变换的速度场变换公式

从直线插值 变换到仿射插值 的情况可以得到解析解

采样时使用不同插值方式得到相同结果

这个演示实验在训练时使用线性插值, 但在采样时使用了按照 变换的球面插值速度场. 可以看到使用球面插值速度场时, 经过的路径不同, 但采样的结果是几乎一致的.

> 看起来直线插值的路径并不直? 首先是因为 Rectify 过程本身就不会产生直线轨迹,

减少时间步后不同插值方法的差异

左为直线插值, 右为线性插值

尽管理论上经过正确的速度场变换, 不同插值方式应该得到相同的结果, 然而实际应用中由于采样步数有限, ODE 离散化带来的误差会导致采样结果存在差异. 上图为仅使用 4 步 Euler 法采样的结果, 两种插值方式表现出了明显的区别.

增加 Euler 法的步数, 两者差异逐渐收敛到零.

Deepmind 博客中的实验

[DeepMind 博客](https://diffusionflow.github.io/) 中展示 DDIM 和 Flow Matching 采样时差异的实验也是只使用 6 步 Euler 法采样. 上面的分析表明此处展示的区别主要是由步数少造成的, 而不是本质上训练目标的不同.

插值方式的选择影响训练

如上一节介绍的, 训练时不同的插值方式相当于施加了时间步权重. 如果手动增加时间步权重项抵消插值方式不同的影响, 同时使用正确的方式在采样时变换速度场, 则可以得到相近的采样结果.

### ODE / SDE

<svg xmlns="http://www.w3.org/2000/svg" width="702.775" height="339.772" viewBox="-72 -72 527.081 254.829"><g stroke="#000" stroke-miterlimit="10" stroke-width=".4"><g stroke-width=".8"><path fill="none" d="M129.965-71.87H-67.87a4 4 0 0 0-4 4v83.049a4 4 0 0 0 4 4h197.835a4 4 0 0 0 4-4V-67.87a4 4 0 0 0-4-4ZM-71.87 19.179"></path><g stroke="none"><text x="84.272" y="-26.346" font-family="cmbx10" font-size="10" transform="translate(-99.585 -7.7)">OT</text> <text x="104.244" y="-26.346" font-family="cmbx10" font-size="10" transform="translate(-99.585 -7.7)">V</text> <text x="111.98" y="-26.346" font-family="cmbx10" font-size="10" transform="translate(-99.585 -7.7)">elo</text> <text x="126.514" y="-26.346" font-family="cmbx10" font-size="10" transform="translate(-99.585 -7.7)">cit</text> <text x="138.973" y="-26.346" font-family="cmbx10" font-size="10" transform="translate(-99.585 -7.7)">y</text> <text x="148.375" y="-26.346" font-family="cmbx10" font-size="10" transform="translate(-99.585 -7.7)">Field</text> <path d="M-68.537-27.101h199.17v.4h-199.17z"></path><text x="69.135" y="-6.501" font-family="cmmi10" font-size="10" transform="translate(-99.585 -7.7)">Z</text> <text x="75.962" y="-5.001" font-family="cmmi7" font-size="7" transform="translate(-99.585 -7.7)">s</text> <text x="88.304" y="-6.501" font-family="cmr10" font-size="10" transform="translate(-99.585 -7.7)">=</text> <text x="104.149" y="-6.501" font-family="cmmi10" font-size="10" transform="translate(-99.585 -7.7)">Z</text> <text x="110.975" y="-5.001" font-family="cmmi7" font-size="7" transform="translate(-99.585 -7.7)">t</text> <text x="118.834" y="-6.501" font-family="cmr10" font-size="10" transform="translate(-99.585 -7.7)">+</text> <text x="130.95" y="-6.501" font-family="cmmi10" font-size="10" transform="translate(-99.585 -7.7)">v</text> <text x="135.797" y="-5.001" font-family="cmr7" font-size="7" transform="translate(-99.585 -7.7)">OT</text> <text x="152.483" y="-6.501" font-family="cmsy10" font-size="10" transform="translate(-99.585 -7.7)">¢</text> <text x="159.599" y="-6.501" font-family="cmr10" font-size="10" transform="translate(-99.585 -7.7)">(</text><text x="163.488" y="-6.501" font-family="cmmi10" font-size="10" transform="translate(-99.585 -7.7)">s</text> <text x="172.513" y="-6.501" font-family="cmsy10" font-size="10" transform="translate(-99.585 -7.7)">¡</text> <text x="184.629" y="-6.501" font-family="cmmi10" font-size="10" transform="translate(-99.585 -7.7)">t</text><text x="188.24" y="-6.501" font-family="cmr10" font-size="10" transform="translate(-99.585 -7.7)">)</text></g></g> <g stroke-width=".8"><path fill="none" d="M450.411-71.87H252.576a4 4 0 0 0-4 4v83.049a4 4 0 0 0 4 4h197.835a4 4 0 0 0 4-4V-67.87a4 4 0 0 0-4-4ZM248.576 19.179"></path><g stroke="none"><text x="76.133" y="-26.346" font-family="cmbx10" font-size="10" transform="translate(220.862 -7.7)">DDIM</text> <text x="112.383" y="-26.346" font-family="cmbx10" font-size="10" transform="translate(220.862 -7.7)">V</text> <text x="120.119" y="-26.346" font-family="cmbx10" font-size="10" transform="translate(220.862 -7.7)">elo</text> <text x="134.653" y="-26.346" font-family="cmbx10" font-size="10" transform="translate(220.862 -7.7)">cit</text> <text x="147.112" y="-26.346" font-family="cmbx10" font-size="10" transform="translate(220.862 -7.7)">y</text> <text x="156.514" y="-26.346" font-family="cmbx10" font-size="10" transform="translate(220.862 -7.7)">Field</text> <path d="M251.91-27.101h199.169v.4h-199.17z"></path><text x="65.199" y="-6.501" font-family="cmmi10" font-size="10" transform="translate(220.862 -7.7)">Z</text> <text x="72.026" y="-5.001" font-family="cmmi7" font-size="7" transform="translate(220.862 -7.7)">s</text> <text x="83.821" y="-6.501" font-family="cmr10" font-size="10" transform="translate(220.862 -7.7)">=</text> <text x="99.119" y="-6.501" font-family="cmmi10" font-size="10" transform="translate(220.862 -7.7)">Z</text> <text x="105.946" y="-5.001" font-family="cmmi7" font-size="7" transform="translate(220.862 -7.7)">t</text> <text x="113.586" y="-6.501" font-family="cmr10" font-size="10" transform="translate(220.862 -7.7)">+</text> <text x="125.483" y="-6.501" font-family="cmmi10" font-size="10" transform="translate(220.862 -7.7)">v</text> <text x="130.331" y="-5.001" font-family="cmr7" font-size="7" transform="translate(220.862 -7.7)">DDIM</text> <text x="157.075" y="-6.501" font-family="cmsy10" font-size="10" transform="translate(220.862 -7.7)">¢</text> <text x="163.972" y="-6.501" font-family="cmr10" font-size="10" transform="translate(220.862 -7.7)">(</text><text x="167.861" y="-6.501" font-family="cmmi10" font-size="10" transform="translate(220.862 -7.7)">s</text> <text x="176.668" y="-6.501" font-family="cmsy10" font-size="10" transform="translate(220.862 -7.7)">¡</text> <text x="188.565" y="-6.501" font-family="cmmi10" font-size="10" transform="translate(220.862 -7.7)">t</text><text x="192.176" y="-6.501" font-family="cmr10" font-size="10" transform="translate(220.862 -7.7)">)</text></g></g> <g stroke-width=".8"><path fill="none" d="M450.411 91.11H252.576a4 4 0 0 0-4 4v83.05a4 4 0 0 0 4 4h197.835a4 4 0 0 0 4-4V95.11a4 4 0 0 0-4-4Zm-201.835 91.05"></path><g stroke="none"><text x="76.091" y="-26.346" font-family="cmbx10" font-size="10" transform="translate(220.862 134.446)">DDIM</text> <text x="112.341" y="-26.346" font-family="cmbx10" font-size="10" transform="translate(220.862 134.446)">ODE</text> <text x="140.688" y="-26.346" font-family="cmbx10" font-size="10" transform="translate(220.862 134.446)">Sampler</text> <path d="M251.91 115.045h199.169v.4h-199.17z"></path><text x="107.464" y="7.999" font-family="cmmi10" font-size="10" transform="translate(220.862 134.446)">Z</text> <text x="114.29" y="9.499" font-family="cmmi7" font-size="7" transform="translate(220.862 134.446)">s</text> <text x="121.342" y="7.999" font-family="cmr10" font-size="10" transform="translate(220.862 134.446)">=</text> <text x="131.898" y="7.999" font-family="cmmi10" font-size="10" transform="translate(220.862 134.446)">®</text> <text x="138.295" y="9.499" font-family="cmmi7" font-size="7" transform="translate(220.862 134.446)">s</text> <text x="145.437" y="5.471" font-family="cmr10" font-size="10" transform="translate(220.862 134.446)">^</text> <text x="142.569" y="7.999" font-family="cmmi10" font-size="10" transform="translate(220.862 134.446)">X</text> <text x="153.861" y="7.999" font-family="cmr10" font-size="10" transform="translate(220.862 134.446)">+</text> <text x="163.861" y="7.999" font-family="cmmi10" font-size="10" transform="translate(220.862 134.446)">¾</text> <text x="169.575" y="9.499" font-family="cmmi7" font-size="7" transform="translate(220.862 134.446)">s</text> <text x="173.934" y="7.999" font-family="cmr10" font-size="10" transform="translate(220.862 134.446)">^</text> <text x="173.849" y="7.999" font-family="cmmi10" font-size="10" transform="translate(220.862 134.446)">²</text><text x="177.908" y="7.999" font-family="cmmi10" font-size="10" transform="translate(220.862 134.446)">;</text><text x="80.579" y="29.308" font-family="cmr10" font-size="10" transform="translate(220.862 134.446)">where</text> <text x="112.363" y="26.781" font-family="cmr10" font-size="10" transform="translate(220.862 134.446)">^</text> <text x="109.495" y="29.308" font-family="cmmi10" font-size="10" transform="translate(220.862 134.446)">X</text> <text x="121.342" y="29.308" font-family="cmr10" font-size="10" transform="translate(220.862 134.446)">=</text> <text x="133.098" y="22.543" font-family="cmmi10" font-size="10" transform="translate(220.862 134.446)">Z</text> <text x="139.924" y="24.043" font-family="cmmi7" font-size="7" transform="translate(220.862 134.446)">t</text> <text x="145.667" y="22.543" font-family="cmsy10" font-size="10" transform="translate(220.862 134.446)">¡</text> <text x="155.667" y="22.543" font-family="cmmi10" font-size="10" transform="translate(220.862 134.446)">¾</text> <text x="161.381" y="24.043" font-family="cmmi7" font-size="7" transform="translate(220.862 134.446)">t</text> <text x="164.987" y="22.543" font-family="cmr10" font-size="10" transform="translate(220.862 134.446)">^</text> <text x="164.902" y="22.543" font-family="cmmi10" font-size="10" transform="translate(220.862 134.446)">²</text> <path d="M353.96 161.054h35.863v.4H353.96z"></path><text x="146.07" y="36.168" font-family="cmmi10" font-size="10" transform="translate(220.862 134.446)">®</text> <text x="152.467" y="37.668" font-family="cmmi7" font-size="7" transform="translate(220.862 134.446)">t</text></g></g> <g stroke-width=".8"><path fill="none" d="M129.965 91.11H-67.87a4 4 0 0 0-4 4v83.05a4 4 0 0 0 4 4h197.835a4 4 0 0 0 4-4V95.11a4 4 0 0 0-4-4ZM-71.87 182.16"></path><g stroke="none"><text x="75.466" y="-26.346" font-family="cmbx10" font-size="10" transform="translate(-99.585 134.78)">DDPM</text> <text x="115.216" y="-26.346" font-family="cmbx10" font-size="10" transform="translate(-99.585 134.78)">SDE</text> <text x="141.313" y="-26.346" font-family="cmbx10" font-size="10" transform="translate(-99.585 134.78)">Sampler</text> <path d="M-68.537 115.38h199.17v.4h-199.17z"></path><text x="42.549" y="11.499" font-family="cmmi10" font-size="10" transform="translate(-99.585 134.78)">d</text> <text x="47.198" y="11.499" font-family="cmmi10" font-size="10" transform="translate(-99.585 134.78)">Z</text> <text x="54.025" y="12.999" font-family="cmmi7" font-size="7" transform="translate(-99.585 134.78)">s</text> <text x="61.077" y="11.499" font-family="cmr10" font-size="10" transform="translate(-99.585 134.78)">=</text> <text x="71.632" y="-2.601" font-family="cmex10" font-size="10" transform="translate(-99.585 134.78)">∙</text> <text x="76.91" y="11.499" font-family="cmmi10" font-size="10" transform="translate(-99.585 134.78)">f</text> <text x="82.882" y="11.499" font-family="cmr10" font-size="10" transform="translate(-99.585 134.78)">(</text><text x="86.771" y="11.499" font-family="cmmi10" font-size="10" transform="translate(-99.585 134.78)">s</text><text x="91.459" y="11.499" font-family="cmr10" font-size="10" transform="translate(-99.585 134.78)">)</text> <text x="95.348" y="11.499" font-family="cmmi10" font-size="10" transform="translate(-99.585 134.78)">Z</text> <text x="102.174" y="12.999" font-family="cmmi7" font-size="7" transform="translate(-99.585 134.78)">s</text> <text x="108.671" y="11.499" font-family="cmr10" font-size="10" transform="translate(-99.585 134.78)">+</text> <text x="118.671" y="11.499" font-family="cmmi10" font-size="10" transform="translate(-99.585 134.78)">g</text> <text x="123.799" y="11.499" font-family="cmr10" font-size="10" transform="translate(-99.585 134.78)">(</text><text x="127.688" y="11.499" font-family="cmmi10" font-size="10" transform="translate(-99.585 134.78)">s</text><text x="132.376" y="11.499" font-family="cmr10" font-size="10" transform="translate(-99.585 134.78)">)</text> <text x="136.264" y="7.37" font-family="cmr7" font-size="7" transform="translate(-99.585 134.78)">2</text> <text x="145" y="4.734" font-family="cmr10" font-size="10" transform="translate(-99.585 134.78)">^</text> <text x="144.915" y="4.734" font-family="cmmi10" font-size="10" transform="translate(-99.585 134.78)">²</text> <path d="M42.366 143.58h9.988v.4h-9.988z"></path><text x="141.951" y="18.358" font-family="cmmi10" font-size="10" transform="translate(-99.585 134.78)">¾</text> <text x="147.665" y="19.858" font-family="cmmi7" font-size="7" transform="translate(-99.585 134.78)">s</text> <text x="153.139" y="-2.601" font-family="cmex10" font-size="10" transform="translate(-99.585 134.78)">¸</text> <text x="160.083" y="11.499" font-family="cmmi10" font-size="10" transform="translate(-99.585 134.78)">ds</text> <text x="172.198" y="11.499" font-family="cmr10" font-size="10" transform="translate(-99.585 134.78)">+</text> <text x="182.198" y="11.499" font-family="cmmi10" font-size="10" transform="translate(-99.585 134.78)">g</text> <text x="187.326" y="11.499" font-family="cmr10" font-size="10" transform="translate(-99.585 134.78)">(</text><text x="191.215" y="11.499" font-family="cmmi10" font-size="10" transform="translate(-99.585 134.78)">s</text><text x="195.903" y="11.499" font-family="cmr10" font-size="10" transform="translate(-99.585 134.78)">)</text> <text x="199.792" y="11.499" font-family="cmmi10" font-size="10" transform="translate(-99.585 134.78)">dW</text> <text x="214.441" y="12.999" font-family="cmmi7" font-size="7" transform="translate(-99.585 134.78)">s</text> <text x="58.299" y="33.399" font-family="cmr8" font-size="8" transform="translate(-99.585 134.78)">(Score</text> <text x="84.778" y="33.399" font-family="cmr8" font-size="8" transform="translate(-99.585 134.78)">is</text> <text x="93.825" y="33.399" font-family="cmsy10" font-size="10" transform="translate(-99.585 134.78)">r</text> <text x="102.159" y="34.899" font-family="cmmi7" font-size="7" transform="translate(-99.585 134.78)">Z</text> <text x="107.607" y="35.899" font-family="cmmi5" font-size="5" transform="translate(-99.585 134.78)">s</text> <text x="113.625" y="33.399" font-family="cmr10" font-size="10" transform="translate(-99.585 134.78)">log</text> <text x="128.208" y="33.399" font-family="cmmi10" font-size="10" transform="translate(-99.585 134.78)">p</text> <text x="133.239" y="34.899" font-family="cmmi7" font-size="7" transform="translate(-99.585 134.78)">s</text> <text x="137.514" y="33.399" font-family="cmr10" font-size="10" transform="translate(-99.585 134.78)">(</text><text x="141.402" y="33.399" font-family="cmmi10" font-size="10" transform="translate(-99.585 134.78)">Z</text> <text x="148.229" y="34.899" font-family="cmmi7" font-size="7" transform="translate(-99.585 134.78)">s</text><text x="152.503" y="33.399" font-family="cmr10" font-size="10" transform="translate(-99.585 134.78)">)</text> <text x="159.17" y="33.399" font-family="cmsy10" font-size="10" transform="translate(-99.585 134.78)">¼</text> <text x="169.725" y="33.399" font-family="cmsy10" font-size="10" transform="translate(-99.585 134.78)">¡</text> <text x="177.588" y="33.399" font-family="cmr10" font-size="10" transform="translate(-99.585 134.78)">^</text> <text x="177.503" y="33.399" font-family="cmmi10" font-size="10" transform="translate(-99.585 134.78)">²</text> <text x="181.562" y="33.399" font-family="cmmi10" font-size="10" transform="translate(-99.585 134.78)">=¾</text> <text x="192.276" y="34.899" font-family="cmmi7" font-size="7" transform="translate(-99.585 134.78)">s</text><text x="196.551" y="33.399" font-family="cmr8" font-size="8" transform="translate(-99.585 134.78)">)</text></g></g> <g stroke-width=".8"><path fill="none" d="M140.565-26.346h101.411"></path><path d="M136.014-26.346c.56.139 3.016.924 4.551 1.779v-3.557c-1.535.854-3.992 1.64-4.55 1.778ZM246.527-26.346c-.559-.138-3.015-.924-4.55-1.778v3.557c1.535-.855 3.991-1.64 4.55-1.779Z"></path><g fill="#fff"><path stroke="none" d="M159.132-19.901h64.278v-12.89h-64.278Z"></path><g fill="#000" stroke="none" font-family="cmr10" font-size="10"><text x="31.048" y="-26.346" transform="translate(130.084 2.5)">A±ne</text> <text x="60.214" y="-26.346" transform="translate(130.084 2.5)">Scaling</text></g></g></g> <g stroke-width=".8"><path fill="none" d="M351.494 25.779V84.51"></path><path d="M351.494 21.228c-.139.559-.924 3.015-1.779 4.55h3.557c-.854-1.535-1.64-3.991-1.778-4.55ZM351.494 89.061c.138-.559.924-3.015 1.778-4.55h-3.557c.855 1.535 1.64 3.991 1.779 4.55Z"></path><g fill="#fff"><path stroke="none" d="M306.619 61.533h89.75V48.756h-89.75Z"></path><text x="31.048" y="-26.346" fill="#000" stroke="none" font-family="cmr10" font-size="10" transform="translate(277.571 83.935)">Reparameterization</text></g></g> <g stroke-width=".8"><path fill="none" d="M241.976 136.635h-101.41"></path><path d="M246.527 136.635c-.559-.139-3.015-.924-4.55-1.779v3.558c1.535-.855 3.991-1.64 4.55-1.78ZM136.014 136.635c.56.139 3.016.924 4.551 1.779v-3.558c-1.535.855-3.992 1.64-4.55 1.779Z"></path><g fill="#fff"><path stroke="none" d="M155.062 142.107h72.417v-10.944h-72.417Z"></path><g fill="#000" stroke="none" font-family="cmr10" font-size="10"><text x="31.048" y="-26.346" transform="translate(126.015 166.453)">Zero</text> <text x="53.853" y="-26.346" transform="translate(126.015 166.453)">noise</text> <text x="78.909" y="-26.346" transform="translate(126.015 166.453)">limit</text></g></g></g> <g stroke-width=".8"><path fill="none" d="M31.048 25.779V84.51"></path><path d="M31.048 21.228c-.139.559-.924 3.015-1.779 4.55h3.557c-.854-1.535-1.64-3.991-1.778-4.55ZM31.048 89.061c.138-.559.924-3.015 1.778-4.55H29.27c.855 1.535 1.64 3.991 1.779 4.55Z"></path><g fill="#fff"><path stroke="none" d="M-13.147 60.617h88.39V49.672h-88.39Z"></path><g fill="#000" stroke="none" font-family="cmr10" font-size="10"><text x="31.048" y="-26.346" transform="translate(-42.195 84.963)">Twdddie's</text> <text x="79.437" y="-26.346" transform="translate(-42.195 84.963)">F</text> <text x="85.131" y="-26.346" transform="translate(-42.195 84.963)">orm</text> <text x="102.103" y="-26.346" transform="translate(-42.195 84.963)">ula</text></g></g></g></g></svg>

结合之前的讨论, 已经可将 Rectified Flow 的速度场通过仿射缩放变换为 DDIM 的速度场, 再重参数化成常见的 DDIM ODE 采样公式 形式, 随后根据 DDIM 论文中的推导就可以在 ODE 和 SDE 采样之间转换. 接下来从 Flow 速度场这一侧出发, 讨论在 Flow 或 Diffusion 中使用 SDE 采样的性质.

Flow Matching 的 ODE 形式已经很美好了: 能保持 的边际分布, 还能实现更少时间步的采样. 为什么要在这个过程中引入 SDE 要素呢? 一种动机是为了对抗 ODE 离散化求解的累积误差. 可以使用 SDE 引入一种 “反馈机制” 来修正误差.

朗之万动力学

这是朗之万动力学的随机微分方程, 描述随机过程 在时间 上的演化.

- 漂移项 是方程的确定性部分, 表示系统受到一个与概率密度 有关的力的影响. 指向概率密度 增加的方向. 是参数控制漂移的幅度.
- 扩散项 是方程的随机部分, 是一个布朗运动. 是一个服从正态分布的随机噪声, 是控制噪声强度的参数. 加入随机性有助于系统脱离局部机制, 收敛到全局高概率区域.

这个朗之万动力学过程满足 Fokker-Planck 方程

因此满足平稳性条件, 即 不随着时间 的演化而改变.

在标准的 RF ODE 中加入朗之万动力学项

其中 是一个独立于 ODE 时间步 的时间尺度, 我们假设在内层的朗之万动力学过程收敛后再在外层的 RF 过程前进. 加入朗之万动力学后, RF 的边际分布不改变, 但相当于引入了一种负反馈调节机制, 无偏差地促进系统向 高的方向演化. 鉴于 RF 求解过程中, 轨迹的误差也没那么大, 可以认为朗之万动力学的时间尺度 只用一步就能收敛, 将它与 合并

漂移项促进系统向概率密度高的方向移动

下面的实验中训练时速度场 处于欠拟合的状态, 发现在这个条件下, SDE 采样器的表现比 ODE 采样器更好.

为了写出 的具体表达式, 可以使用 [Tweedie 公式](https://en.wikipedia.org/wiki/Maurice_Tweedie#Tweedie's_formula). 当使用仿射插值 且 服从与 无关的标准正态分布时有

代入 RF 速度场的定义式

可以得到

代入 得到 SDE

使用线性插值 , ，得到

而取 和 时得到 DDPM 的 SDE

增加噪声尺度的效果

反直觉的结果是增加噪声尺度 并没有给结果带来更多的 “随机性”, 而是让结果更接近均值, 方差变小了. 原因在于在 的表达式 中, 当 时, , 也就是说 应该发散到无穷, 但实际计算中 的值是有限的. 因此在 时不适用 Tweedie 公式. 另外, 也会放大 的误差.

在图像生成中过大的噪声尺度会生成过于平滑的图像

> [EDM 论文](https://arxiv.org/pdf/2206.00364) 的附录 E 中观察到了类似的现象, 但没有给出原因. 这里的分析也不够充分说明 “过度集中” 现象发生的原因，可能与神经网咯拟合速度场的误差有关。
> 
> [Qiang Liu 组的最新工作](https://arxiv.org/pdf/2506.15864) 尝试解决这个问题。

[^1]: 关于 CNF 的对数似然求解方法可以参考 CNF 原文。 [这篇知乎文章](https://zhuanlan.zhihu.com/p/10764455963) 也有介绍。对数似然训练 Neural ODE 在高维空间效果差且不稳定。