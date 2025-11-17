---
title: "扩散过程与流动过程的匹配（Diffusion Meets Flow Matching） --- Diffusion Meets Flow Matching"
source: "https://diffusionflow.github.io/"
author:
  - "[[Ruiqi Gao  高瑞琪]]"
  - "[[Emiel Hoogeboom  埃米尔·胡格布姆（Emiel Hoogeboom）]]"
  - "[[Jonathan Heek  乔纳森·希克（Jonathan Heek）]]"
  - "[[Valentin De Bortoli瓦伦丁·德·博尔托利（Valentin De Bortoli）]]"
  - "[[Kevin P. Murphy凯文·P·墨菲（Kevin P. Murphy）]]"
  - "[[Tim Salimans  蒂姆·萨利曼斯（Tim Salimans）]]"
published:
created: 2025-11-11
description: "Flow matching and diffusion models are two popular frameworks in generative modeling. Despite seeming similar, there is some confusion in the community about their exact connection. In this post, we aim to clear up this confusion and show that <i>diffusion models and Gaussian flow matching are the same</i>, although different model specifications can lead to different network outputs and sampling schedules. This is great news, it means you can use the two frameworks interchangeably."
tags:
  - "clippings"
---
## Diffusion Meets Flow Matching: Two Sides of the Same Coin扩散作用与流动匹配：同一枚硬币的两面

Flow matching and diffusion models are two popular frameworks in generative modeling. Despite seeming similar, there is some confusion in the community about their exact connection. In this post, we aim to clear up this confusion and show that *diffusion models and Gaussian flow matching are the same*, although different model specifications can lead to different network outputs and sampling schedules. This is great news, it means you can use the two frameworks interchangeably.  
“流匹配（Flow Matching）”与“扩散模型（Diffusion Models）”是生成式建模（Generative Modeling）领域中两种非常流行的技术框架。尽管这两种技术看起来很相似，但在学术界中仍存在关于它们之间具体关系的困惑。在这篇文章中，我们旨在澄清这种困惑，并说明：扩散模型与高斯流匹配（Gaussian Flow Matching）其实本质上是相同的；只不过不同的模型配置可能会导致不同的网络输出结果或采样策略。这无疑是个好消息——因为这意味着你可以将这两种技术框架互换使用。

![](https://diffusionflow.github.io/assets/img/2025-04-28-distill-example/twotrees-480.webp)

Flow matching has gained popularity recently, due to the simplicity of its formulation and the “straightness” of its induced sampling trajectories. This raises the commonly asked question:  
由于流匹配（Flow Matching）的数学模型较为简单，且其生成的采样轨迹具有“线性”特征（即采样点之间的相对位置关系非常规律），因此这种方法最近变得非常流行。这就引出了一个大家经常问的问题：

*"Which is better, diffusion or flow matching?"  
“哪种方法更好：扩散（diffusion）还是流动匹配（flow matching）？”*

As we will see, diffusion models and flow matching are *equivalent* (for the common special case that the source distribution used with flow matching corresponds to a Gaussian), so there is no single answer to this question. In particular, we will show how to convert one formalism to another. But why does this equivalence matter? Well, it allows you to mix and match techniques developed from the two frameworks. For example, after training a flow matching model, you can use either a stochastic or deterministic sampling method (contrary to the common belief that flow matching is always deterministic).  
正如我们将会看到的那样，扩散模型（diffusion models）与流匹配（flow matching）在某种特定情况下是等价的（即：当流匹配所使用的源分布（source distribution）服从高斯分布（Gaussian distribution）时），因此这个问题并没有一个唯一的答案。具体来说，我们会展示如何将这两种方法相互转换。那么，这种等价性究竟有什么重要性呢？其实，它允许我们将来自这两个不同框架的技术结合起来使用。例如，在训练完流匹配模型之后，你可以选择使用随机采样方法或确定性采样方法来进行后续的数据处理（这与人们普遍认为的“流匹配过程总是确定性的”这一观点相反）。

We will focus on the most commonly used flow matching formalism with the optimal transport path  
我们将重点研究最常用的流量匹配方法（即用于确定最优运输路径的数学模型）, which is closely related to rectified flow  
这与“整流流”（rectified flow）有着密切的关系 and stochastic interpolants  
以及随机插值方法（stochastic interpolation methods）. Our purpose is not to recommend one approach over another (both frameworks are valuable, each rooted in distinct theoretical perspectives, and it’s actually even more encouraging that they lead to the same algorithm in practice), but rather to help practitioners understand and feel confident about using these frameworks interchangeably, while understanding the true degrees of freedom one has when tuning the algorithm—regardless of what it’s called.  
我们的目的并不是要推荐某种特定的方法而排斥另一种（这两种框架都非常有价值，它们各自基于不同的理论视角；实际上，更令人欣慰的是：在实践中，这两种框架最终都能引导出相同的算法结果）。我们的目标是帮助从业者理解如何灵活地使用这两种框架，并让他们在使用这些框架时充满信心；同时，我们也希望他们能够清楚地认识到：在调整算法参数时，自己实际上拥有多少真正的“自由度”（无论这个算法被称作什么名称）。

Check this [Google Colab](https://colab.research.google.com/drive/13lAveB3qwjkgyILWW-9qiOOSHG0U5_O6?usp=sharing) for code used to produce plots and animations in this post.  
请查看这个 Google Colab 文档，其中包含了用于生成本文中所示图表和动画的代码。

## Overview 概述

We start with a quick overview of the two frameworks.  
我们首先会对这两个框架进行简要的介绍。

### Diffusion models 扩散模型（Diffusion Models）

A diffusion process gradually destroys an observed datapoint $x$ (such as an image) over time $t$ , by mixing the data with Gaussian noise. The noisy data at time $t$ is given by a forward process: $(1)zt=αtx+σtϵ,whereϵ∼N(0,I).$  $αt$ and $σt$ define the **noise schedule**. A noise schedule is called variance-preserving if $αt2+σt2=1$ . The noise schedule is designed in a way such that $z0$ is close to the clean data, and $z1$ is close to a Gaussian noise.  
这种扩散过程会随着时间的推移（ $t$ ），通过将原始数据与高斯噪声混合，逐渐破坏被观测到的数据点（例如图像）。在时间 $t$ 时，所得到的数据就是被添加了噪声后的数据； $(1)zt=αtx+σtϵ,whereϵ∼N(0,I).$ 、 $αt$ 和 $σt$ 分别定义了噪声生成的规则（即“噪声生成方案”）。如果该噪声生成方案能够“保持数据的方差特性”（即数据的统计特性不会被改变），那么它就被称为“方差保持型”（variance-preserving）噪声生成方案。该噪声生成方案的设计目的是：使得经过处理后的数据 $z0$ 与原始数据尽可能接近，同时确保生成的高斯噪声 $z1$ 具有理想的高斯噪声特性。

To generate new samples, we can “reverse” the forward process: We initialize the sample $z1$ from a standard Gaussian. Given the sample $zt$ at time step $t$ , we predict what the clean sample might look like with a neural network (a.k.a. denoiser model) $x^=x^(zt;t)$ , and then we project it back to a lower noise level $s$ with the same forward transformation:  
为了生成新的样本，我们可以“反向”执行原始的数据处理流程：首先从一个标准的高斯分布中生成初始样本 $z1$ ；然后根据当前时间步长 $t$ 中已有的样本 $zt$ ，使用神经网络（也就是去噪模型 $x^=x^(zt;t)$ ）来预测原始样本可能的样子；最后，再使用相同的处理流程将数据投影回较低的噪声水平 $s$ 。

$(2)zs=αsx^+σsϵ^,$ where $ϵ^=(zt−αtx^)/σt$ . (Alternatively we can train a neural network to predict the noise $ϵ^$ .) We keep alternating between predicting the clean data, and projecting it back to a lower noise level until we get the clean sample. This is the DDIM sampler  
$(2)zs=αsx^+σsϵ^,$ 在这里， $ϵ^=(zt−αtx^)/σt$ 表示某个特定的数据点。（或者，我们也可以训练一个神经网络来预测噪声成分 $ϵ^$ 。）我们不断在“预测原始数据”和“将数据投影回较低的噪声水平”这两个步骤之间循环进行，直到最终得到一个纯净的样本。这就是所谓的 DDIM 采样器。. The randomness of samples only comes from the initial Gaussian sample, and the entire reverse process is deterministic. We will discuss the stochastic samplers later.  
样本的随机性仅来源于初始的高斯分布；整个反向处理过程实际上是确定性的（即每次处理的结果都是可预测的）。关于随机采样器的相关内容，我们稍后会讨论。

### Flow matching \*\*流匹配（Flow Matching）\*\*

In flow matching, we view the forward process as a linear interpolation between the data $x$ and a noise term $ϵ$ : $(3)zt=(1−t)x+tϵ.$   
在流匹配（flow matching）的过程中，我们将数据点 $x$ 与噪声项 $ϵ$ 之间的变化视为一种线性插值过程，其结果表示为 $(3)zt=(1−t)x+tϵ.$

This corresponds to the diffusion forward process if the noise is Gaussian (a.k.a. Gaussian flow matching) and we use the schedule $αt=1−t,σt=t$ .  
如果噪声服从高斯分布（即所谓的“高斯流匹配”），那么这种情况就对应于扩散过程；同时，我们使用的是 $αt=1−t,σt=t$ 这一调度方案。

Using simple algebra, we can derive that $zt=zs+u⋅(t−s)$ for $s<t$ , where $u=ϵ−x$ is the “velocity”, “flow”, or “vector field”. Hence, to sample $zs$ given $zt$ , we reverse time and replace the vector field with our best guess at time $t$ : $u^=u^(zt;t)=ϵ^−x^$ , represented by a neural network, to get  
通过简单的代数运算，我们可以得出：对于 $s<t$ 来说， $zt=zs+u⋅(t−s)$ 的值与 $u=ϵ−x$ 有关；其中 $u=ϵ−x$ 代表“速度”、“流动”或“矢量场”。因此，为了根据 $zt$ 的数据来计算 $zs$ 的值，我们需要倒回时间顺序，并用我们在时间点 $t$ 时对矢量场的最佳猜测值（即由神经网络计算出的结果 $u^=u^(zt;t)=ϵ^−x^$ ）来替换原来的矢量场。

$$
(4)zs=zt+u^⋅(s−t).
$$

Initializing the sample $z1$ from a standard Gaussian, we keep getting $zs$ at a lower noise level than $zt$ , until we obtain the clean sample.  
我们从标准高斯分布中初始化样本 $z1$ ，结果得到的样本 $zs$ 的噪声水平始终低于样本 $zt$ ，直到我们获得一个“干净”的（即噪声较少的）样本为止。

### Comparison 比较

So far, we can already discern the similar essences in the two frameworks:  
到目前为止，我们已经可以发现这两个框架之间的相似之处：

1\. **Same forward process**, if we assume that one end of flow matching is Gaussian, and the noise schedule of the diffusion model is in a particular form.  
1\. 前向处理过程是相同的；如果我们假设流匹配过程的一端遵循高斯分布，并且扩散模型的噪声变化规律具有某种特定的形式的话。

2\. **"Similar" sampling processes**: both follow an iterative update that involves a guess of the clean data at the current time step. (Spoiler: below we will show they are exactly the same!)  
2\. “相似”的采样过程：这两种方法都采用迭代更新的方式，即在当前时间步对“干净数据”进行猜测。（剧透一下：下面我们会证明这两种方法其实完全相同！）

## Sampling 抽样（Sampling）

It is commonly thought that the two frameworks differ in how they generate samples: Flow matching sampling is deterministic with “straight” paths, while diffusion model sampling is stochastic and follows “curved paths”. Below, we clarify this misconception. We will focus on deterministic sampling first, since it is simpler, and will discuss the stochastic case later on.  
人们普遍认为这两种框架在生成样本的方式上存在差异：Flow Matching 采样是一种确定性的采样方法，其生成的样本路径是“直线”；而扩散模型采样则是随机性的，生成的样本路径是“曲线”。下面我们将澄清这种误解。我们首先讨论确定性的采样方法，因为它更简单；随机性采样的内容则会在后面再详细说明。

Imagine you want to use your trained denoiser model to transform random noise into a datapoint. Recall that the DDIM update is given by $zs=αsx^+σsϵ^$ . Interestingly, by rearranging terms it can be expressed in the following formulation, with respect to several sets of network outputs and reparametrizations:  
假设你想使用训练好的去噪模型将随机噪声转换为数据点。请记住，DDIM（Diffusion Denoising Model）的更新公式是由 $zs=αsx^+σsϵ^$ 给出的。有趣的是，通过重新排列公式中的各项，我们可以用另一种方式来表达这个更新过程——这种方式涉及到多个网络输出以及参数的重新设置。

$$
(5)z~s=z~t+Networkoutput⋅(ηs−ηt)
$$

| Network Output 网络输出（Network Output） | Reparametrization 重新参数化（Reparametrization） |
| --- | --- |
| $x^$ -prediction $x^$ – 预测结果 | $z~t=zt/σt$ and $ηt=αt/σt$ $z~t=zt/σt$ 和 $ηt=αt/σt$ |
| $ϵ^$ -prediction $ϵ^$ – 预测结果 | $z~t=zt/αt$ and $ηt=σt/αt$ $z~t=zt/αt$ 和 $ηt=σt/αt$ |
| $u^$ -flow matching vector field   $u^$ – 流量匹配向量场（Flow Matching Vector Field） | $z~t=zt/(αt+σt)$ and $ηt=σt/(αt+σt)$ $z~t=zt/(αt+σt)$ 和 $ηt=σt/(αt+σt)$ |

Remember the flow matching update in Equation (4)? This should look similar. If we set the network output as $u^$ in the last line and let $αt=1−t$ , $σt=t$ , we have $z~t=zt$ and $ηt=t$ , which is the flow matching update! More formally, the flow matching update is a Euler sampler of the sampling ODE (i.e., $dzt=u^dt$ ), and with the flow matching noise schedule,  
还记得方程（4）中的“流量匹配更新”（flow matching update）吗？这个过程应该与此类似。如果我们在最后一行将网络输出设置为 $u^$ ，同时让 $αt=1−t$ 和 $σt=t$ 保持不变，那么就会得到 $z~t=zt$ 和 $ηt=t$ ，这些值其实就是流量匹配更新的结果！更正式地说，流量匹配更新其实是一种用于采样常微分方程（sampling ODE）的算法（即 $dzt=u^dt$ ）；而这个算法的具体实现还需要依赖于相应的“流量匹配噪声调度”（flow matching noise schedule）。

*Diffusion with DDIM sampler == Flow matching sampler (Euler).  
使用 DDIM 采样器进行扩散处理时，其扩散过程实际上等同于使用“流匹配采样器”（即欧拉（Euler）方法）来模拟物质在流体中的运动。*

Some other comments on the DDIM sampler:  
关于 DDIM 采样器的其他一些评论：

1. The DDIM sampler *analytically* integrates the reparametrized sampling ODE (i.e., $dz~t=[Networkoutput]⋅dηt$ ) if the network output is a *constant* over time. Of course the network prediction is not constant, but it means the inaccuracy of DDIM sampler only comes from approximating the intractable integral of the network output (unlike the Euler sampler of the probability flow ODE  
	当网络输出随时间保持不变时，DDIM 采样器会通过解析方法对重新参数化后的采样微分方程（即 $dz~t=[Networkoutput]⋅dηt$ ）进行积分计算。当然，网络的实际输出并不会保持恒定；这意味着 DDIM 采样器的误差仅来源于对网络输出值的近似计算（这与用于处理概率流微分方程的 Euler 采样器的情况不同）。 which involves an additional linear term of $zt$ ). The DDIM sampler can be considered a first-order Euler sampler of the repamemetrized sampling ODE, which has the same update rule for different network outputs. However, if one uses a higher-order ODE solver, the network output can make a difference, which means the $u^$ output proposed by flow matching can make a difference from diffusion models.  
	该算法中包含了一个额外的线性项（即 $zt$ ）。DDIM 采样器可以被视为一种用于对采样过程进行控制的“一阶欧拉采样器”（first-order Euler sampler）；这种采样器适用于那些采用了重新参数化（repametrized）的采样微分方程（sampling ODE），并且对于不同的网络输出，其更新规则是相同的。然而，如果使用更高阶的微分方程求解器（higher-order ODE solver），网络输出的结果可能会发生变化；这意味着由“流匹配”（flow matching）算法计算出的 $u^$ 值可能会与传统的扩散模型（diffusion models）计算出的结果有所不同。
2. The DDIM sampler is *invariant* to a linear scaling applied to the noise schedule $αt$ and $σt$ , as scaling does not affect $z~t$ and $ηt$ . This is not true for other samplers e.g. Euler sampler of the probability flow ODE.  
	DDIM 采样器对于施加在噪声生成规则（ $αt$ 和 $σt$ ）上的线性缩放操作具有“不变性”（即缩放操作不会影响采样器的输出结果）；因为这种线性缩放不会影响到变量 $z~t$ 和 $ηt$ 。然而，对于其他类型的采样器来说，情况并非如此——例如用于求解概率流微分方程（ODE）的 Euler 采样器，其输出结果会受到线性缩放的影响。

To validate Claim 2, we present the results obtained using several noise schedules, each of which follows a flow-matching schedule ( $αt=1−t,σt=t$ ) with different scaling factors. Feel free to change the slider below the figure. At the left end, the scaling factor is $1$ , which is exactly the flow matching schedule (FM), while at the right end, the scaling factor is $1/[(1−t)2+t2]$ , which corresponds to a variance-preserving schedule (VP). We see that DDIM (and flow matching sampler) always gives the same final data samples, regardless of the scaling of the schedule. The paths bend in different ways as we are showing $zt$ (but not $z~t$ ), which is scale-dependent along the path. For the Euler sampler of the probabilty flow ODE, the scaling makes a true difference: we see that both the paths and the final samples change.  
为了验证“主张 2”的正确性，我们展示了使用多种不同的噪声生成方案所得到的实验结果。这些方案均遵循“流匹配算法”（ $αt=1−t,σt=t$ ），但各自的缩放因子各不相同；您可以根据需要调整图下方的滑块来改变这些缩放因子。在左侧，缩放因子为 $1$ ，这正好对应于标准的“流匹配算法”（Flow Matching Algorithm）；而在右侧，缩放因子为 $1/[(1−t)2+t2]$ ，这对应于“方差保持算法”（Variance-Preserving Algorithm）。无论采用哪种缩放方案，DDIM（以及基于该算法的采样器）始终能够生成相同的数据样本。如图所示，数据样本的生成路径会以不同的方式发生弯曲（具体表现为 $zt$ ，而 $z~t$ 则不会发生这种变化）；这种路径的弯曲程度会受到缩放因子的影响。对于基于概率流微分方程（Probabilistic Flow ODE）的 Euler 采样器来说，缩放因子确实会对实验结果产生显著影响：此时，数据样本的生成路径以及最终的样本值都会发生变化。

Wait a second! People often say flow matching results in *straight* paths, but in the above figure, the sampling trajectories look *curved*.  
等一下！人们常常认为“流程匹配”（flow matching）会导致路径呈直线，但在上面的图中，采样轨迹看起来却是弯曲的。

Well first, why do they say that? If the model would be perfectly confident about the data point it is moving to, the path from noise to data will be a straight line, with the flow matching noise schedule. Straight line ODEs would be great because it means that there is no integration error whatsoever. Unfortunately, the predictions are not for a single point. Instead they average over a larger distribution. And flowing *straight to a point!= straight to a distribution*.  
首先，他们为什么会这么说呢？如果模型对它要移动到的数据点有完全的信心，那么从噪声状态到数据点的路径应该是一条直线，且这个过程应该与噪声的变化规律（即“噪声分布”）完全一致。如果可以用线性微分方程（ODE）来描述这个过程，那就再好不过了，因为这样就不会产生任何积分误差。不幸的是，模型的预测结果并不是针对单个数据点的；实际上，预测结果是对一个更大的数据分布进行平均后得出的。而且，从噪声状态“直接”移动到一个数据点，并不等同于从噪声状态“直接”移动到一个数据分布。

In the interactive graph below, you can change the variance of the data distribution on the right hand side by the slider. Note how the variance preserving schedule is better (straighter paths) for wide distributions, while the flow matching schedule works better for narrow distributions.  
在下面的交互式图表中，您可以通过滑块来调整数据分布的方差。请注意：对于分布范围较广的数据（即数据分布的“曲线”较为平缓的情况），“保持方差不变”的调度方式效果更好（即数据流动的路径更加平滑）；而对于分布范围较窄的数据（即数据分布的“曲线”较为陡峭的情况），则“匹配数据流动规律”的调度方式效果更佳。

Finding such straight paths for real-life datasets like images is of course much less straightforward. But the conclusion remains the same: The optimal integration method depends on the data distribution.  
对于像图像这样的实际数据集来说，找到这样的“直线路径”（即数据分布的理想模型）当然要困难得多。不过结论仍然是一样的：最佳的数据整合方法取决于数据本身的分布情况。

Two important takeaways from deterministic sampling:  
从确定性抽样方法中可以得出两个重要的结论：

1\. **Equivalence in samplers**: DDIM is equivalent to the flow matching sampler, and is invariant to a linear scaling to the noise schedule.  
1\. 采样器之间的等价性：DDIM（Deep Density Imprinting）与“flow matching sampler”具有等价性；同时，它对噪声调度中的线性缩放操作也不敏感（即其性能不会因此受到影响）。

2\. **Straightness misnomer**: Flow matching schedule is only straight for a model predicting a single point. For realistic distributions, other schedules can give straighter paths.  
2\. “直线性”这一说法其实并不准确：对于那些仅用于预测单个点的模型来说，其训练过程（即数据流匹配的顺序）确实呈现出直线状；但对于那些用于处理现实数据分布的模型而言，其他数据流匹配方式反而能够生成更加“平滑”（即路径更加连续）的训练过程。

## Training 训练

Diffusion models 扩散模型（Diffusion Models） are trained by estimating $x^=x^(zt;t)$ , or alternatively $ϵ^=ϵ^(zt;t)$ with a neural net.Learning the model is done by minimizing a weighted mean squared error (MSE) loss: $(6)L(x)=Et∼U(0,1),ϵ∼N(0,I)[w(λt)⋅dλdt⋅‖ϵ^−ϵ‖22],$ where $λt=log⁡(αt2/σt2)$ is the log signal-to-noise ratio, and $w(λt)$ is the **weighting function**, balancing the importance of the loss at different noise levels. The term $dλ/dt$ in the training objective seems unnatural and in the literature is often merged with the weighting function. However, their separation helps *disentangle* the factors of training noise schedule and weighting function clearly, and helps emphasize the more important design choice: the weighting function.  
这些模型的训练过程是通过使用神经网络来估计参数 $x^=x^(zt;t)$ 或 $ϵ^=ϵ^(zt;t)$ 来完成的。模型的学习过程是通过最小化加权均方误差（MSE）损失函数来实现的；其中， $λt=log⁡(αt2/σt2)$ 表示信号与噪声的对数比值， $w(λt)$ 则是用于平衡不同噪声水平下损失重要性的权重函数。训练目标中的 $dλ/dt$ 这一项在数学表达上显得有些“不自然”（即其含义不太直观），因此在相关文献中通常会被与权重函数合并在一起。不过，将这两者分开处理有助于更清晰地理解训练过程中的各个因素（如噪声调度机制与权重函数的作用），同时也能够更突出权重函数这一更为重要的设计要素。

Flow matching also fits in the above training objective. Recall below is the conditional flow matching objective used by  
“流量匹配”（Flow Matching）也符合上述的训练目标。下面介绍的是被某些系统所采用的条件性流量匹配（Conditional Flow Matching）算法。:

$$
(7)LCFM(x)=Et∼U(0,1),ϵ∼N(0,I)[‖u^−u‖22]
$$

Since $u^$ can be expressed as a linear combination of $ϵ^$ and $zt$ , the CFM training objective can be rewritten as mean squared error on $ϵ$ with a specific weighting.  
由于 $u^$ 可以表示为 $ϵ^$ 和 $zt$ 的线性组合，因此 CFM 的训练目标可以重新表述为对 $ϵ$ 的均方误差（mean squared error），并且这个误差的计算会使用特定的权重进行加权处理。

### How do we choose what the network should output?我们该如何决定网络应该输出什么结果呢？

Below we summarize several network outputs proposed in the literature, including a few versions used by diffusion models and the one used by flow matching. They can be derived from each other given the current data $zt$ . One may see the training objective defined with respect to MSE of different network outputs in the literature. From the perspective of training objective, they all correspond to having some additional weighting in front of the $ϵ$ -MSE that can be absorbed in the weighting function.  
下面我们总结了文献中提出的几种网络输出方式，其中包括扩散模型使用的一些版本，以及流匹配（flow matching）模型使用的一种输出方式。根据现有的数据，这些输出方式其实是可以相互推导出来的。在文献中，训练目标通常是通过衡量不同网络输出结果的均方误差（MSE）来定义的。从训练目标的角度来看，这些方法都涉及到在计算误差时对某些参数（ $ϵ$ ）进行额外的加权处理；这种加权效果可以通过相应的权重函数来实现。

| Network Output 网络输出（Network Output） | Formulation 配方设计 | MSE on Network Output   网络输出上的均方误差（MSE）： |
| --- | --- | --- |
| $ϵ^$ -prediction $ϵ^$ – 预测结果 | $ϵ^$ | $‖ϵ^−ϵ‖22$ |
| $x^$ -prediction $x^$ – 预测结果 | $x^=(zt−σtϵ^)/αt$ | $‖x^−x‖22=e−λ‖ϵ^−ϵ‖22$ |
| $v^$ -prediction $v^$ – 预测结果 | $v^=αtϵ^−σtx^$ | $‖v^−v‖22=αt2(e−λ+1)2‖ϵ^−ϵ‖22$ |
| $u^$ -flow matching vector field   $u^$ – 流量匹配向量场（Flow Matching Vector Field） | $u^=ϵ^−x^$ | $‖u^−u‖22=(e−λ/2+1)2‖ϵ^−ϵ‖22$ |

In practice, however, the model output might make a difference. For example,  
然而，在实际应用中，模型输出的结果可能会有所不同。例如，

- $ϵ^$ -prediction can be problematic at high noise levels, because any error in $ϵ^$ will get amplified in $x^=(zt−σtϵ^)/αt$ , as $αt$ is close to 0. It means that small changes create a large loss under some weightings.  
	$ϵ^$ - 在高噪声环境下，预测结果可能会出现问题；因为 $ϵ^$ 中的任何误差都可能在 $x^=(zt−σtϵ^)/αt$ 中被放大（由于 $αt$ 的值接近 0），从而导致较大的损失（即模型的预测结果出现较大偏差）。
- Following the similar reason, $x^$ -prediction is problematic at low noise levels, because $x$ as a target is not informative when added noise is small, and the error gets amplified in $ϵ^$ .  
	出于类似的原因，在低噪声环境下，预测结果同样也会出现问题：当添加的噪声较小时，作为目标的 $x$ 本身所包含的信息量就很有限，其误差会在 $ϵ^$ 中被进一步放大，从而导致预测结果不准确。

Therefore, a heuristic is to choose a network output that is a combination of $x^$ - and $ϵ^$ -predictions, which applies to the $v^$ -prediction and the flow matching vector field $u^$ .  
因此，一种启发式方法是选择一种网络输出，该输出结合了 $x^$ 和 $ϵ^$ 的预测结果，并将这些预测结果应用于 $v^$ 的预测结果以及流匹配向量场 $u^$ 。

### How do we choose the weighting function?我们如何选择权重函数呢？

The weighting function is the most important part of the loss. It balances the importance of high frequency and low frequency components in perceptual data such as images, videos and audo  
权重函数是损失函数（loss function）中最关键的部分。它用于平衡图像、视频和音频等感知数据中高频成分与低频成分的重要性。. This is crucial, as certain high frequency components in those signals are not perceptible to humans, and thus it is better not to waste model capacity on them when the model capacity is limited. Viewing losses via their weightings, one can derive the following non-obvious result:  
这一点非常重要，因为这些信号中包含的一些高频成分是人类无法察觉的；因此，当模型的容量有限时，最好不要将这些高频成分纳入模型的处理范围（即不要浪费模型的计算资源）。通过分析这些高频成分在模型中的“权重”（即它们对模型输出结果的影响程度），我们可以得出一个并不显而易见的结论：

*Flow matching weighting == diffusion weighting of {\\bf v} -MSE loss + cosine noise schedule.  
“流匹配加权”（Flow Matching Weighting）实际上就是 {\\bf v} 提出的“扩散加权”（Diffusion Weighting）算法；该算法的核心损失函数为 MSE（Mean Squared Error），同时还会结合余弦函数（cosine function）来控制噪声的强度（即噪声的变化规律）。*

That is, the conditional flow matching objective in Equation (7) is the same as a commonly used setting in diffusion models! See Appendix D.2-3 in  
也就是说，方程（7）中提到的“条件流匹配”目标，其实与扩散模型中常用的设置是完全相同的！详见附录 D.2-3。 for a detailed derivation. Below we plot several commonly used weighting functions in the literature, as a function of $λ$ .  
有关详细推导过程，请参见相关文献。下面，我们绘制了几种在学术文献中常用的权重函数，这些权重函数都是作为变量 $λ$ 的函数来表示的。

The flow matching weighting (also $v$ -MSE + cosine schdule weighting) decreases exponentially as $λ$ increases. Empirically we find another interesting connection: The Stable Diffusion 3 weighting  
当参数 $λ$ 的值增加时，用于流量匹配的权重（也称为 $v$ -MSE + 余弦函数权重）会呈指数级下降。通过实验，我们还发现了另一个有趣的规律：即 Stable Diffusion 3 算法中使用的权重计算方式。, a reweighted version of flow matching, is very similar to the EDM weighting  
这是一种经过重新加权处理的流量匹配（flow matching）算法；它的原理与 EDM（Energy-Directed Matching）算法中的加权机制非常相似 that is popular for diffusion models.  
这种方法在扩散模型（diffusion models）中非常受欢迎。

### How do we choose the training noise schedule?我们该如何选择训练过程中的噪声（即干扰信号）的生成方式（即噪声生成的时间表或频率分布）呢？

We discuss the training noise schedule last, as it should be the least important to training for the following reasons:  
我们最后再讨论训练过程中的“噪声”（即那些可能干扰训练过程的因素）的处理方式。其实，这些因素对训练效果的影响应该是最小的，原因如下：

1. The training loss is *invariant* to the training noise schedule. Specifically, the loss fuction can be rewritten as $L(x)=∫λminλmaxw(λ)Eϵ∼N(0,I)[‖ϵ^−ϵ‖22]dλ$ , which is only related to the endpoints ( $λmax$ , $λmin$ ), but not the schedule $λt$ in between. In practice, one should choose $λmax$ , $λmin$ such that the two ends are close enough to the clean data and Gaussian noise respectively. $λt$ might still affect the variance of the Monte Carlo estimator of the training loss. A few heuristics have been proposed in the literature to automatically adjust the noise schedules over the course of training. [This blog post](https://sander.ai/2024/06/14/noise-schedules.html#adaptive) has a nice summary.  
	训练损失（training loss）对训练过程中所使用的噪声分布（noise distribution）是“不变的”；也就是说，训练损失函数的具体形式与噪声分布的细节无关。具体来说，该损失函数可以重新表示为某种形式（ $L(x)=∫λminλmaxw(λ)Eϵ∼N(0,I)[‖ϵ^−ϵ‖22]dλ$ ），这种形式仅与数据集的边界点（ $λmax$ 、 $λmin$ ）有关，而与训练过程中噪声分布的变化规律（ $λt$ ）无关。在实际应用中，应选择合适的参数 $λmax$ 、 $λmin$ ，以确保数据集的边界点分别足够接近“干净的数据”（即没有噪声的数据）和高斯噪声（Gaussian noise）。不过，噪声分布的变化仍可能影响用于估计训练损失的蒙特卡洛（Monte Carlo）方法的精度。文献中提出了一些启发式方法，用于在训练过程中自动调整噪声分布的参数。这篇博客文章对这些方法进行了很好的总结。
2. Similar to sampling noise schedule, the training noise schedule is invariant to a linear scaling, as one can easily apply a linear scaling to $zt$ and an unscaling at the network input to get the equivalence. The key defining property of a noise schedule is the log signal-to-noise ratio $λt$ .  
	与采样噪声调度类似，训练噪声调度也具有线性缩放不变性（即：对训练过程中的噪声参数进行线性缩放后，系统的性能不会发生变化）。这是因为我们可以通过对输入数据应用线性缩放，同时对网络输入端的数据进行相应的“反缩放”操作，从而实现两者之间的等效性。噪声调度的一个关键定义性参数是对数信噪比（log signal-to-noise ratio）。
3. One can choose completely different noise schedules for training and sampling, based on distinct heuristics: For training, it is desirable to have a noise schedule that minimizes the variance of the Monte Carlo estimator, whereas for sampling the noise schedule is more related to the discretization error of the ODE / SDE sampling trajectories and the model curvature.  
	人们可以根据不同的启发式方法，为训练和采样过程选择完全不同的噪声生成策略。对于训练来说，理想的噪声生成策略应该是能够最小化蒙特卡洛估计量的方差；而对于采样过程而言，噪声生成策略则更多地与常微分方程（ODE）或随机微分方程（SDE）采样轨迹的离散化误差以及模型的曲率有关。

### Summary 摘要：

A few takeaways for training of diffusion models / flow matching:  
关于扩散模型（diffusion models）或流匹配（flow matching）的训练，有以下几条重要的经验或结论：

1\. **Equivalence in weightings**: The weighting function is important for training, which balances the importance of different frequency components of perceptual data. Flow matching weightings coincidentlly match commonly used diffusion training weightings in the literature.  
1\. 权重值的均衡性：权重函数在训练过程中起着关键作用，它用于平衡感知数据中不同频率成分的重要性。Flow Matching 算法所使用的权重值与文献中常见的扩散训练（diffusion training）算法所使用的权重值完全一致。

2\. **Insignificance of training noise schedule**: The noise schedule is far less important to the training objective, but can affect the training efficiency.  
2\. 训练过程中使用的“噪声”（即随机干扰因素）对训练结果的影响其实很小；不过这些“噪声”可能会影响训练的效率。

3\. **Difference in network outputs**: The network output proposed by flow matching is new, which nicely balances \\hat{\\bf x} - and \\hat{\\epsilon} -prediction, similar to \\hat{\\bf v} -prediction.  
3\. 网络输出结果的差异：流量匹配（flow matching）所提出的网络输出方法是一种创新性的方法，它能够很好地平衡 \\hat{\\bf x} 和 \\hat{\\epsilon} 的预测结果，其效果与 \\hat{\\bf v} 的预测结果类似。

## Diving deeper into samplers更深入地了解这些采样器（samplers）……

In this section, we discuss different kinds of samplers in more detail.  
在本节中，我们将更详细地讨论各种类型的采样器（sampler）。

### Reflow operator “Reflow Operator”（重排运算符）

The Reflow operation in flow matching connects noise and data points in a straight line. One can obtain these (data, noise) pairs by running a deterministic sampler from noise. A model can then be trained to directly predict the data given the noise avoiding the need for sampling. In the diffusion literature, the same approach was the one of the first distillation techniques  
在“流匹配”（flow matching）技术中，所谓的“Reflow”操作会将噪声数据点与真实数据点通过一条直线连接起来。这些（噪声数据对）可以通过从噪声数据中提取样本来获得；之后，可以利用这些样本训练模型，使模型能够在仅知道噪声数据的情况下直接预测出真实数据，从而无需再进行额外的采样操作。在扩散模型（diffusion models）的研究领域中，这种处理方法也被视为最早的“数据提取”（data extraction）技术之一。.

### Deterministic sampler vs. stochastic sampler确定性采样器（Deterministic Sampler）与随机采样器（Stochastic Sampler）

So far we have just discussed the deterministic sampler of diffusion models or flow matching. An alternative is to use stochastic samplers such as the DDPM sampler  
到目前为止，我们只讨论了扩散模型或流匹配中的确定性采样方法。另一种选择是使用随机采样器，例如 DDPM（Diffusion Diffusion Propagation Method）采样器。.

Performing one DDPM sampling step going from \\lambda\_t to \\lambda\_t + \\Delta\\lambda is exactly equivalent to performing one DDIM sampling step to \\lambda\_t + 2\\Delta\\lambda, and then renoising to \\lambda\_t + \\Delta\\lambda by doing forward diffusion. That is, the renoising by doing forward diffusion reverses exactly half the progress made by DDIM. To see this, let’s take a look at a 2D example. Starting from the same mixture of Gaussians distribution, we can take either a small DDIM sampling step with the sign of the update reversed (left), or a small forward diffusion step (right):  
从 \\lambda\_t 到 \\lambda\_t + \\Delta\\lambda 进行一次 DDPM（Diffusion Denoising with Permutation）采样，其实等同于先从 \\lambda\_t 到 \\lambda\_t + 2\\Delta\\lambda 进行一次 DDIM（Diffusion Denoising with Inversion）采样，然后再通过前向扩散（forward diffusion）操作将图像噪声重新添加到 \\lambda\_t + \\Delta\\lambda 。换句话说，这种通过前向扩散进行的“去噪”操作会完全抵消 DDIM 所带来的去噪效果（即会“撤销” DDIM 所产生的去噪效果）。为了更好地理解这一点，我们来看一个二维示例：从相同的高斯分布（Gaussian distribution）开始，我们可以选择执行一次 DDIM 采样（但此时采样的方向被反转了），或者选择执行一次前向扩散操作。

![](https://diffusionflow.github.io/assets/img/2025-04-28-distill-example/particle_movement.gif-480.webp)

For individual samples, these updates behave quite differently: the reversed DDIM update consistently pushes each sample away from the modes of the distribution, while the diffusion update is entirely random. However, when aggregating all samples, the resulting distributions after the updates are identical. Consequently, if we perform a DDIM sampling step (without reversing the sign) followed by a forward diffusion step, the overall distribution remains unchanged from the one prior to these updates.  
对于单个样本而言，这两种更新方式的效果截然不同：反向 DDIM 更新会持续将每个样本“推离”分布的“中心”（即分布的众数或模式）；而扩散更新则完全是随机的。然而，当对所有样本进行合并处理后，更新后的分布结果是完全相同的。因此，如果我们先执行 DDIM 抽样步骤（此时不改变数值的符号），然后再执行扩散步骤，那么整个分布的状态将与更新前的状态保持一致。

The fraction of the DDIM step to undo by renoising is a hyperparameter which we are free to choose (i.e. does not have to be exact half of the DDIM step), and which has been called the level of *churn* by  
通过添加噪声来“撤销” DDIM（Deep Denoising Image Modeling）处理效果的比例，其实是一个我们可以自由选择的超参数（也就是说，这个比例并不一定非得正好是 DDIM 处理效果的一半）；这个超参数被称作 “churn level”（即 “数据扰动程度”）。. Interestingly, the effect of adding churn to our sampler is to diminish the effect on our final sample of our model predictions made early during sampling, and to increase the weight on later predictions. This is shown in the figure below:  
有趣的是，将“数据 churn”（即数据中的随机变化或异常值）引入我们的采样过程后，会减弱采样初期所得到的模型预测结果对最终模型预测结果的影响；同时，会提高后期模型预测结果的权重（即这些预测结果在模型决策中的重要性）。这一点在下面的图表中得到了体现。

Here we ran different samplers for 100 sampling steps using a cosine noise schedule and $v^$ -prediction  
在这里，我们使用了不同的采样器进行了 100 次采样；采样过程遵循余弦噪声分布（cosine noise distribution）的规律，并采用了 $v^$ 预测（ $v^$ prediction）算法. Ignoring nonlinear interactions, the final sample produced by the sampler can be written as a weighted sum of predictions $v^t$ made during sampling and a Gaussian noise $e$ : $z0=∑thtv^t+∑tcte$ . The weights $ht$ of these predictions are shown on the y-axis for different diffusion times $t$ shown on the x-axis. DDIM results in an equal weighting of $v^$ -predictions for this setting, as shown in  
忽略非线性相互作用后，采样器生成的最终样本可以表示为采样过程中得到的所有预测结果（ $v^t$ ）与高斯噪声（ $e$ ）的加权和： $z0=∑thtv^t+∑tcte$ 。这些预测结果的权重（ $ht$ ）被绘制在 y 轴上，而不同的扩散时间（ $t$ ）则被绘制在 x 轴上。在这种设置下，DDIM 算法会使得所有预测结果（ $v^$ ）被赋予相同的权重（即所有预测结果都被平等地考虑在内），如图所示。, whereas DDPM puts more emphasis on predictions made towards the end of sampling. Also see  
而 DDPM 更注重在采样过程接近尾声时所做的预测。另请参阅： for analytic expressions of these weights in the $x^$ - and $ϵ^$ -predictions.  
以获取这些权重在 $x^$ 和 $ϵ^$ 预测中的解析表达式。

## SDE and ODE Perspective从常微分方程（SDE, Stochastic Differential Equations）和偏微分方程（ODE, Ordinary Differential Equations）的角度来看……

We’ve observed the practical equivalence between diffusion models and flow matching algorithms. Here, we formally describe the equivalence of the forward and sampling processes using ODE and SDE, as a completeness in theory.  
我们观察到扩散模型（diffusion models）与流动匹配算法（flow matching algorithms）在功能上是等价的。在此，我们通过常微分方程（ODE）和随机微分方程（SDE）的形式，正式阐述了这两种算法在“正向计算过程”（forward calculation processes）及“采样过程”（sampling processes）方面的等价性，从而从理论层面证明了这种等价性的完备性。

### Diffusion models 扩散模型（Diffusion Models）

The forward process of diffusion models which gradually destroys a data over time can be described by the following stochastic differential equation (SDE):  
扩散模型的“正向过程”（即模型如何随时间逐渐“破坏”数据）可以通过以下随机微分方程（Stochastic Differential Equation, SDE）来描述：

$$
(8)dzt=ftztdt+gtdz,
$$

where $dz$ is an *infinitesimal Gaussian* (formally, a Brownian motion).f\_t and g\_t decide the noise schedule. The generative process is given by the reverse of the forward process, whose formula is given by  
其中， $dz$ 表示一个无限小的高斯过程（严格来说，是一个布朗运动）； f\_t 和 g\_t 则用于决定噪声的生成规则/频率。该生成过程实际上是通过“反向过程”来实现的，而这个“反向过程”的具体公式则是通过“正向过程”的公式反推得到的。

$$
(9)dzt=(ftzt−1+ηt22gt2∇log⁡pt(zt))dt+ηtgtdz,
$$

where \\nabla \\log p\_t is the *score* of the forward process.  
其中， \\nabla \\log p\_t 表示前向处理过程的得分（即前向处理步骤所产生的结果或输出值）。

Note that we have introduced an additional parameter \\eta\_t which controls the amount of stochasticity at inference time. This is related to the *churn* parameter introduced before. When discretizing the backward process we recover DDIM in the case \\eta\_t = 0 and DDPM in the case \\eta\_t = 1.  
请注意，我们引入了一个额外的参数 \\eta\_t ，该参数用于控制推理过程中的随机性程度。这个参数与之前提到的“客户流失率”（churn rate）相关。在离散化反向传播过程（backward propagation process）时，当参数 \\eta\_t = 0 被设置为特定值时，系统会恢复到传统的 DDIM（Discrete Diffusion Inference Model）算法；而当参数 \\eta\_t = 1 被设置为特定值时，系统则会恢复到 DDPM（Discrete Diffusion Propagation Model）算法。

### Flow matching 流量匹配（Flow Matching）

The interpolation between $x$ and $ϵ$ in flow matching can be described by the following ordinary differential equation (ODE):  
在流量匹配过程中， $x$ 与 $ϵ$ 之间的插值过程可以用以下常微分方程（ODE）来描述：

$$
(10)dzt=utdt.
$$

Assuming the interpolation is $zt=αtx+σtϵ$ , then $ut=α˙tx+σ˙tϵ$ .  
假设插值函数为 $zt=αtx+σtϵ$ ，那么 $ut=α˙tx+σ˙tϵ$ 也是如此。

The generative process is simply reversing the ODE in time, and replacing $ut$ by its conditional expectation with respect to $zt$ . This is a specific case of *stochastic interpolants*  
生成过程其实只是将常微分方程（ODE）在时间轴上的演化过程“倒过来”，并将符号 “ $ut$ ” 替换为该符号关于变量 “ $zt$ ” 的条件期望值。这种处理方式属于随机插值方法（stochastic interpolation methods）的一种具体应用形式。, in which case it can be generalized to an SDE:  
在这种情况下，它可以被推广为一个随机微分方程（SDE）：

$(11)dzt=(ut−12εt2∇log⁡pt(zt))dt+εtdz,$ where $εt$ controls the amount of stochasticity at inference time.  
$(11)dzt=(ut−12εt2∇log⁡pt(zt))dt+εtdz,$ ：其中， $εt$ 用于控制推理过程中的随机性（即数据或模型的不确定性）程度。

### Equivalence of the two frameworks这两个框架之间的等价性（即它们在功能或效果上的相似性）

Both frameworks are defined by three hyperparameters respectively: f\_t, g\_t, \\eta\_t for diffusion, and \\alpha\_t, \\sigma\_t, \\varepsilon\_t for flow matching. We can show the equivalence by deriving one set of hyperparameters from the other. From diffusion to flow matching:  
这两个框架分别由三个超参数来控制： f\_t, g\_t, \\eta\_t 用于控制扩散过程， \\alpha\_t, \\sigma\_t, \\varepsilon\_t 用于控制流体流动的匹配过程。我们可以通过从其中一个框架推导出另一个框架所需的超参数参数集，从而证明这两个框架之间的等价性。具体来说，可以从扩散过程（即使用 f\_t, g\_t, \\eta\_t 这个超参数控制的模型）推导出用于控制流体流动匹配过程（即使用 \\alpha\_t, \\sigma\_t, \\varepsilon\_t 这个超参数控制的模型）所需的超参数参数集。

$$
αt=exp⁡(∫0tfsds),σt=(∫0tgs2exp⁡(−2∫0sfudu)ds)1/2,εt=ηtgt.
$$

From flow matching to diffusion:  
从“流量匹配”到“扩散”：

$$
ft=∂tlog⁡(αt),gt2=2αtσt∂t(σt/αt),ηt=εt/(2αtσt∂t(σt/αt))1/2.
$$

In summary, aside from training considerations and sampler selection, diffusion and Gaussian flow matching exhibit no fundamental differences.  
总之，除了训练相关的问题以及采样器的选择之外，扩散模型（diffusion models）与高斯流模型（Gaussian flow models）在本质上并没有任何显著的区别。

## Closing takeaways 总结要点：

If you’ve read this far, hopefully we’ve convinced you that diffusion models and Gaussian flow matching are equivalent. However, we highlight two new model specifications that Gaussian flow matching brings to the field:  
如果你已经读到了这里，希望我们能够说服你：扩散模型（diffusion models）与高斯流匹配（Gaussian flow matching）实际上是等价的。不过，我们还想强调高斯流匹配为该领域带来的两个新的模型特性/优势。

- **Network output**: Flow matching proposes a vector field parametrization of the network output that is different from the ones used in diffusion literature. The network output can make a difference when higher-order samplers are used. It may also affect the training dynamics.  
	网络输出：Flow Matching 提出了一种用于描述网络输出的向量场参数化方法，这种方法与扩散理论（diffusion theory）中使用的参数化方法有所不同。当使用更高阶的采样器（higher-order samplers）时，网络输出的结果可能会发生变化；此外，这种参数化方法还可能影响模型的训练过程（training dynamics）。
- **Sampling noise schedule**: Flow matching leverages a simple sampling noise schedule $αt=1−t$ and $σt=t$ , with the same update rule as DDIM.  
	采样噪声调度方案：Flow Matching 使用了一种简单的采样噪声调度方案（即 $αt=1−t$ 和 $σt=t$ ），其更新规则与 DDIM（Deep Density Imputation）相同。

It would be interesting to investigate the importance of these two model specifications empirically in different real world applications, which we leave to future work. It is also an exciting research area to apply flow matching to more general cases where the source distribution is non-Gaussian, e.g. for more structured data like protein  
从实证角度研究这两种模型规格在不同实际应用中的重要性会非常有意义；这部分内容我们将留待未来的研究工作来完成。此外，将“流匹配”（flow matching）技术应用于那些源数据分布非高斯分布的更一般性场景（例如结构化程度更高的数据，如蛋白质数据）也是一个极具挑战性的研究方向。.

## Acknowledgements 致谢

Thanks to our colleagues at Google DeepMind for fruitful discussions. In particular, thanks to Sander Dieleman, Ben Poole and Aleksander Hołyński.  
感谢 Google DeepMind 的同事们与我们进行了富有成果的讨论，特别要感谢 Sander Dieleman、Ben Poole 和 Aleksander Hołyński。

### References 参考文献

[^1]: Flow matching for generative modeling  
用于生成式建模的流量匹配（Flow Matching）技术  
Lipman, Y., Chen, R.T., Ben-Hamu, H., Nickel, M. and Le, M., 2022. arXiv preprint arXiv:2210.02747.  
Lipman, Y., Chen, R.T., Ben-Hamu, H., Nickel, M. 和 Le, M., 2022. arXiv 预印本：arXiv:2210.02747.

[^2]: Flow straight and fast: Learning to generate and transfer data with rectified flow  
数据流动应当顺畅且快速：学习如何使用“修正后的数据流”（rectified data flow）来生成和传输数据  
Liu, X., Gong, C. and Liu, Q., 2022. arXiv preprint arXiv:2209.03003.  
刘 X、龚 C、刘 Q，2022 年。预印本：arXiv:2209.03003。

[^3]: Building normalizing flows with stochastic interpolants  
使用随机插值方法来构建用于数据归一化的流程（即用于将数据转换到标准范围内的流程）  
Albergo, M.S. and Vanden-Eijnden, E., 2022. arXiv preprint arXiv:2209.15571.  
Albergo, M.S. 和 Vanden-Eijnden, E., 2022. arXiv 预印本：arXiv:2209.15571.

[^4]: Stochastic interpolants: A unifying framework for flows and diffusions  
随机插值方法：一种适用于流体运动与扩散过程的统一理论框架  
Albergo, M.S., Boffi, N.M. and Vanden-Eijnden, E., 2023. arXiv preprint arXiv:2303.08797.  
Albergo, M.S., Boffi, N.M. 和 Vanden-Eijnden, E., 2023. arXiv 预印本：arXiv:2303.08797.

[^5]: Denoising diffusion implicit models  
去噪扩散隐式模型（Denoising Diffusion Implicit Models）  
Song, J., Meng, C. and Ermon, S., 2020. arXiv preprint arXiv:2010.02502.  
Song, J., Meng, C. 和 Ermon, S., 2020. arXiv 预印本：arXiv:2010.02502.

[^6]: Score-based generative modeling through stochastic differential equations  
基于分数的生成建模方法：通过随机微分方程来实现  
Song, Y., Sohl-Dickstein, J., Kingma, D.P., Kumar, A., Ermon, S. and Poole, B., 2020. arXiv preprint arXiv:2011.13456.  
Song, Y., Sohl-Dickstein, J., Kingma, D.P., Kumar, A., Ermon, S. 和 Poole, B., 2020. arXiv 预印本：arXiv:2011.13456.

[^7]: Understanding diffusion objectives as the elbo with simple data augmentation  
将扩散（diffusion）算法的目标理解为通过简单的数据增强（data augmentation）技术来实现这些目标  
Kingma, D. and Gao, R., 2024. Advances in Neural Information Processing Systems, Vol 36.  
Kingma, D. 和 Gao, R., 2024. 《神经信息处理系统进展》（Advances in Neural Information Processing Systems），第 36 卷。

[^8]: Diffusion is spectral autoregression [\[HTML\]](https://sander.ai/2024/09/02/spectral-autoregression.html)  
扩散过程本质上是一种谱自回归（Spectral Autoregression）\[HTML\]  
Dieleman, S., 2024.

[^9]: Scaling rectified flow transformers for high-resolution image synthesis  
用于高分辨率图像合成的整流式流式变压器的优化设计（即：如何改进整流式流式变压器的结构或参数，以使其更适合高分辨率图像合成应用）  
Esser, P., Kulal, S., Blattmann, A., Entezari, R., Muller, J., Saini, H., Levi, Y., Lorenz, D., Sauer, A., Boesel, F. and others,, 2024. Forty-first International Conference on Machine Learning.  
Esser, P., Kulal, S., Blattmann, A., Entezari, R., Müller, J., Saini, H., Levi, Y., Lorenz, D., Sauer, A., Boesel, F. 等人，2024 年。第 41 届国际机器学习会议（41st International Conference on Machine Learning）。

[^10]: Elucidating the design space of diffusion-based generative models  
阐明基于扩散机制的生成模型的设计框架（即这些模型的设计可能性与实现方式）  
Karras, T., Aittala, M., Aila, T. and Laine, S., 2022. Advances in neural information processing systems, Vol 35, pp. 26565--26577.  
Karras, T., Aittala, M., Aila, T. 和 Laine, S., 2022. 《神经信息处理系统进展》（Advances in Neural Information Processing Systems），第 35 卷，第 26565–26577 页。

[^11]: Knowledge distillation in iterative generative models for improved sampling speed [\[PDF\]](https://arxiv.org/pdf/2101.02388.pdf)  
在迭代生成模型中运用知识蒸馏技术以提高采样速度 \[PDF\]  
Luhman, E. and Luhman, T., 2021. arXiv preprint arXiv:2101.02388.  
Luhman, E. 和 Luhman, T., 2021. arXiv 预印本：arXiv:2101.02388.

[^12]: Denoising diffusion probabilistic models  
去噪扩散概率模型（Denoising Diffusion Probabilistic Models）  
Ho, J., Jain, A. and Abbeel, P., 2020. Advances in neural information processing systems, Vol 33, pp. 6840--6851.  
Ho, J., Jain, A. 和 Abbeel, P., 2020. 《神经信息处理系统进展》（Advances in Neural Information Processing Systems），第 33 卷，第 6840–6851 页。

[^13]: Progressive Distillation for Fast Sampling of Diffusion Models  
用于快速采样扩散模型的渐进式蒸馏（Progressive Distillation）方法  
Salimans, T. and Ho, J., 2022. International Conference on Learning Representations.  
Salimans, T. 和 Ho, J., 2022. 国际学习表示会议（International Conference on Learning Representations）。

[^14]: Dpm-solver++: Fast solver for guided sampling of diffusion probabilistic models  
Dpm-solver++：一种用于引导扩散概率模型采样的快速求解器  
Lu, C., Zhou, Y., Bao, F., Chen, J., Li, C. and Zhu, J., 2022. arXiv preprint arXiv:2211.01095.  
陆超、周勇、鲍飞、陈杰、李超、朱军，2022 年。预印本：arXiv:2211.01095。

[^15]: Se (3)-stochastic flow matching for protein backbone generation  
用于生成蛋白质骨架的 Se(3)随机流匹配算法  
Bose, A.J., Akhound-Sadegh, T., Huguet, G., Fatras, K., Rector-Brooks, J., Liu, C., Nica, A.C., Korablyov, M., Bronstein, M. and Tong, A., 2023. arXiv preprint arXiv:2310.02391.  
Bose, A.J., Akhound-Sadegh, T., Huguet, G., Fatras, K., Rector-Brooks, J., Liu, C., Nica, A.C., Korablyov, M., Bronstein, M. 和 Tong, A., 2023. 《arXiv 预印本》：arXiv:2310.02391.