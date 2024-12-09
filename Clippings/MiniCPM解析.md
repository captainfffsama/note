---
title: "Notion – The all-in-one workspace for your notes, tasks, wikis, and databases."
source: "https://shengdinghu.notion.site/MiniCPM-c805a17c5c8046398914e47f0542095a"
author:
  - "[[Notion]]"
published:
created: 2024-12-04
description: "A new tool that blends your everyday work apps into one. It's the all-in-one workspace for you and your team"
tags:
  - "clippings"
---

原文网址: <https://shengdinghu.notion.site/MiniCPM-c805a17c5c8046398914e47f0542095a>

作者：胡声鼎、涂宇鸽、韩旭\*、崔淦渠、贺超群、赵威霖、龙翔、郑直、方晔玮、张开活、黄宇翔、戴振宁、龚柏涛、王崇屹、姚远、周界、蔡杰、张新荣、翟忠武、丁宁、贾超、曾国洋、李大海、刘知远\*、孙茂松等

MiniCPM 是一系列端侧语言大模型，主体语言模型 MiniCPM-2B 具有 2.4B 的非词嵌入参数量。在综合性榜单上与 Mistral-7B 相近（中文、数学、代码能力更优），整体性能超越 Llama2-13B、MPT-30B、Falcon-40B 等模型。在当前最接近用户体感的榜单 MTBench 上，MiniCPM-2B 也超越了 Llama2-70B-Chat、Vicuna-33B、Mistral-7B-Instruct-v0.1、Zephyr-7B-alpha 等众多代表性开源大模型。

我们将完全开源 MiniCPM-2B 的模型参数供学术研究和有限商用，以及训练过程中的所有 Checkpoint 和大部分非专有数据 (需要一定时间准备）给模型机理研究。

基于 MiniCPM-2B 的指令微调与人类偏好对齐的 MiniCPM-2B-SFT/DPO。

基于 MiniCPM-2B 的多模态模型 MiniCPM-V，能力超越基于 Phi-2 的同参数级别多模态模型。

MiniCPM-2B-SFT/DPO 的 Int4 量化版 MiniCPM-2B-SFT/DPO-Int4。

基于 MLC-LLM、LLMFarm 开发的 MiniCPM 手机端程序，文本及多模态模型均可在手机端进行推理。

受限于模型规模，模型可能出现幻觉性问题。其中由于 DPO 模型生成的回复内容更长，更容易出现幻觉。我们也将持续进行 MiniCPM 模型的迭代改进；

为了保证在学术研究用途上模型的通用性，我们未对模型进行任何身份认同训练。同时由于我们用 ShareGPT 开源语料作为部分训练数据，模型可能会输出类似 GPT 系列模型的身份认同信息；

受限于模型规模，模型的输出受到提示词（prompt）的影响较大，可能多次尝试产生不一致的结果；

受限于模型容量，模型的知识记忆较不准确，后续我们将结合 RAG 方法来增强模型的知识记忆能力。

大模型的实验成本高昂，难以在不进行配置调优的情况下得到最优秀的大模型性能。

借鉴 $\mu P$﻿等优秀的前人工作，我们提出在小模型上进行广泛的实验，通过可迁移的配置，获得大模型的最优训练方法。MiniCPM 本身，即为模型沙盒实验的成果。

我们进行了 Hyper-parameters、Batch size、Learning Rate、Learning Rate Scheduler、Data Strategy 五个方面的模型沙盒研究。

超参数对模型的性能具有重大影响，在传统训练方法中，需要对每个模型进行超参数调整，这对于大模型并不现实。借鉴 $\mu P$﻿的方法，我们对模型的各参数模块之间进行了连接权重的调整、以及对模型初始化的调整。部分调整接近 Cerebras-GPT。

上述操作的具体参数由近 400 次在 0.009B 模型规模上的贝叶斯参数搜索得到。

Batchsize 决定了模型的收敛速度和消耗计算资源的平衡。Batchsize 过大，达到一定的损失消耗的数据量和计算量都会很大，而 batchsize 过小，则需要消耗过多的训练步数，且有可能损失函数下降有限。在 2020 年 OpenAI 的开山之作中，OpenAI 研究了损失函数随 token 数变化的规律。在他们的实验中，他们将认为消耗更多的步数等价于消耗更多的时间，在这种假设下，OpenAI 定义了临界 Batchsize（Critical Batchsize），使得达到一定的损失，既不消耗过多 step，也不消耗过多 token。然而我们观察到在利用当前以 A100 为主的计算资源，结合 gradient checkpointing 策略进行训练时，通常计算速度（而不是显存）是瓶颈，这意味着在相同机器数量下，多一倍 Batchsize 几乎等同于慢一倍的单步时间。基于这个观察，我们取消了对“不消耗过多 step”的追求，而转向追求用最少的 token 量达到最低的 loss。

我们在 0.009B，0.036B，0.17B 的模型上分别进行了 6 个 batchsize 的训练实验，将结果记录如图下。

我们观察到了最优 batchsize 随着 C4 数据集上的 loss 的偏移规律（图中的红线）。

![](https://shengdinghu.notion.site/image/https%3A%2F%2Fprod-files-secure.s3.us-west-2.amazonaws.com%2F30c36155-a603-469f-957f-b0854b6e2372%2F7b4c335a-2cda-4fdd-8de2-c1405d3df6a7%2FUntitled.png?table=block&id=fd2c4c6b-dba8-4422-8648-b5de22be9352&spaceId=30c36155-a603-469f-957f-b0854b6e2372&width=1540&userId=&cache=v2)

![](https://shengdinghu.notion.site/image/https%3A%2F%2Fprod-files-secure.s3.us-west-2.amazonaws.com%2F30c36155-a603-469f-957f-b0854b6e2372%2F0086e773-a7c6-4e94-aa08-84c7a1936ff2%2FUntitled.png?table=block&id=a5bbe015-1311-440a-b284-e49a95583065&spaceId=30c36155-a603-469f-957f-b0854b6e2372&width=770&userId=&cache=v2)

将这三个图的红线进行连接，并进行拟合，我们得到了如下 Batchsize 关于 C4 Loss 的规律：

$$$
BS = \frac{1.2110 \times 10^9}{L^{6.2393}}
$$$

根据这个规律，我们预估了 2B 模型达到 C4 损失 2.5 左右，4M 是比较合适的 Batchsize。

由于我们使用了超参稳定的参数化方案，我们预期模型的最关键超参数: 学习率，不会因为模型规模扩大有大幅度的改变，因此我们在 0.04B, 0.1B, 0.3B, 0.5B 上分别做了 6 组学习率实验，我们发现虽然模型大小扩大了 10 倍，但是最优学习率偏移并不明显，均在 0.01 左右，我们在 2.1B 的规模上进行了简单验证，发现在 0.01 的学习率确实能取得最低的 Loss。

![](https://shengdinghu.notion.site/image/https%3A%2F%2Fprod-files-secure.s3.us-west-2.amazonaws.com%2F30c36155-a603-469f-957f-b0854b6e2372%2F44a3d57c-6131-4c93-8ddd-7d7f718b110f%2Floss_vs_lr.png?table=block&id=0a9c1f96-0a98-40d5-8cf6-f192dd97cf55&spaceId=30c36155-a603-469f-957f-b0854b6e2372&width=1060&userId=&cache=v2)

学习率调度器，即训练不同阶段使用不同学习率的调整策略，对模型性能影响很关键。当前通用的学习率策略是 Cosine 图像，即在学习率从 Warmup 阶段升高到最高点之后，开始呈现余弦函数的降低。几乎所有大模型都使用了 Cosine Learning Rate Scheduler (简称 Cosine LRS）的方式。

为了研究为什么 Cosine 的 Scheduler 表现优异，我们进行了大量实验。我们对 0.036B 的模型，设置不同的 Learning Rate Scheduler 的截止步数 $T$﻿，进行了持续训练。结果如下图：

![](https://shengdinghu.notion.site/image/https%3A%2F%2Fprod-files-secure.s3.us-west-2.amazonaws.com%2F30c36155-a603-469f-957f-b0854b6e2372%2F44dbc039-68b0-4a17-88ae-10f2939eeece%2FUntitled.png?table=block&id=0a274cda-127e-4c4a-9e15-6f9cc504972a&spaceId=30c36155-a603-469f-957f-b0854b6e2372&width=1150&userId=&cache=v2)

从图中可以看出，对于训练至 $S$﻿步的模型，将 Cosine LRS 的截止步数 $T$﻿设置为 $S$﻿步总是能获得最优的性能，而设置为更多或者更少性能都不是最优。

当我们考虑持续训练的场景，会发现 Cosine 调度器的问题有更多问题。如果我们在 Cosine 的截止步数之后继续沿用 0.1 倍的最大学习率（通常做法），则继续训练收敛非常缓慢；如果我们在 Cosine 的截止步数之后重启 Cosine LRS（即再次从最大学习率开始下降，或者是逐渐上升到最大学习率，再开始下降）则会发现损失会经历长时间的上升周期，而这段时间，模型处于不可用状态。

我们猜想 Cosine LRS 在预先指定步数的时候性能优异的原因有两点：

T=S 下的 Cosine LRS，相对于 Linear LRS、Noam LRS、以及 T<S 的 Cosine LRS，有更长时间的大学习率训练。这一阶段可能有助于模型寻找更好的全局最优解。

T=S 下的 Cosine LRS ，相对于 T>S 的 Cosine LRS、Constant LRS，有更充分的学习率下降的退火阶段，这一阶段可能发生了较为特别的动力学现象，导致模型可以找到更好的局部最优解。

结合这两点，我们提出了一种新的学习率调度策略，Warmup-Stable-Decay（WSD）调度器。这种学习率调度器分为三个阶段，warmup 阶段（用 W 表示 warmup 阶段结束时的步数/训练量），稳定训练阶段（用 S 表示稳定训练阶段结束时的步数/训练量），退火阶段（用 D 表示退火阶段的训练量）。这种调度器可以写为：

$$$
lr(s)=\begin{cases}\frac{s}{W}*\eta, s< W\\\eta, W<s<S\\f(s-S)*\eta, S<s<S+D\end{cases}
$$$

其中 $0< f(s-S)\leq 1$﻿ 是一个关于 $s$﻿的减函数， $\eta$﻿是最大学习率。这种策略有以下四个好处：

![](https://shengdinghu.notion.site/image/https%3A%2F%2Fprod-files-secure.s3.us-west-2.amazonaws.com%2F30c36155-a603-469f-957f-b0854b6e2372%2Fb837e282-7a3c-47dd-aca5-7aac4b5c072f%2FUntitled.png?table=block&id=cd4faecf-079e-4c2f-a3b8-0c6cbeaf099b&spaceId=30c36155-a603-469f-957f-b0854b6e2372&width=1060&userId=&cache=v2)

图中显示了 CosineWSD(W, S, D) 学习率调度器和 Cosine 的对比。可以看到，在 Cosine 调度器结束之后需要持续保持最低学习率，以保证 loss 不上升，而 WSD 调度器则可以从退火之前开始继续用最大学习率训练，经过更长的训练再开始退火。

我们发现如我们所设想的，在 Decay 阶段（退火阶段），随着学习率的变小，损失有大幅度的快速下降，在步数 S 时迅速降低至和 T=S 的 Cosine LRS 相等或更低。与此同时，我们可以复用 Decay 前的模型，进行大学习率的继续训练，在更多的步数 S’之后进行退火，取得和 T’=S’ 的 Cosine LRS 相等的效果。

![](https://shengdinghu.notion.site/image/https%3A%2F%2Fprod-files-secure.s3.us-west-2.amazonaws.com%2F30c36155-a603-469f-957f-b0854b6e2372%2Fcc95e459-7aec-4f53-94a4-7aa80ee4fdfc%2FUntitled.png?table=block&id=69f8c645-1acc-4ec7-ad86-50dd577eb1a7&spaceId=30c36155-a603-469f-957f-b0854b6e2372&width=1060&userId=&cache=v2)

图：图中最下方的橙色线为 Cosine LRS，上方绿色线为 WSD LRS。尽管 WSD LRS 在学习率恒定的 stable 阶段表现差于 Cosine，但是在最后的退火阶段，会形成快速下降的趋势，最终达到或者超越 Cosine LRS。我们尝试了不同的退火步长，例如 WSD(N, 80N, 2N) 的退火比例为 2/80 = 2.5%， 而 WSD(N, 80,N, 8N) 的比例为 8/80 = 10%。我们发现在我们测试的 4 个训练阶段（20N，40N， 60N，80N），2.5% 的退火长度都不够达到最优的性能。而 10% 的退火长度则足够达到甚至超越 Cosine LRS。

我们对 Decay 阶段的训练步数需求进行了探索。我们发现在所有训练时长中，总步数 10% 的 Decay 都足够达到最好的效果，而 2.5% 的 Decay 都较为欠缺。因此我们最终将 Decay 的步数定为最后 10%。

![](https://shengdinghu.notion.site/image/https%3A%2F%2Fprod-files-secure.s3.us-west-2.amazonaws.com%2F30c36155-a603-469f-957f-b0854b6e2372%2Fe88c58ab-438f-4f7f-998d-723c28f9a6f0%2FUntitled.png?table=block&id=cec1ddac-687d-4cec-8733-4133cebcc393&spaceId=30c36155-a603-469f-957f-b0854b6e2372&width=1060&userId=&cache=v2)

图中粉色线是 0.17B 的模型，绿色线（系列）是使用 WSD 的 0.036B 的模型，黄色线是使用 Cosine 调度的 0.036B 的模型，其中 Cosine 周期设置在训练数据为 80 倍模型参数量的时候。可以看到退火终点的连线大致可以外插到和 0.17B 的终点（具体看本节的第 6 部分）

根据 Batchsize 随损失变化的实验结果，不难猜想，用更大的 Batchsize 可能可以达到更低的 loss。我们在 0.036B、2.4B 的模型实验中都发现了类似的现象，即在扩大 Batchsize 的时候损失会有一次较大幅度的下降（我们猜想 Batchsize 扩大和 Learning Rate 降低可能有相似的动力学效果）。如下图所示，我们进行了 Batch size 扩大，Loss 降低约 0.2，并且在后续的退火阶段，仍然能形成一样的下降效果。但是遗憾的是，在我们正式实验中，我们进行的 Batchsize 扩大后退火阶段的降低有所减少，因此我们最终没有采用 Batchsize 扩大的训练方法。这个问题留作后续研究。

![](https://shengdinghu.notion.site/image/https%3A%2F%2Fprod-files-secure.s3.us-west-2.amazonaws.com%2F30c36155-a603-469f-957f-b0854b6e2372%2F1d4c30cb-ef95-4812-8aa2-2c797768f509%2FUntitled.png?table=block&id=c1ed0e7a-2f08-4da2-bc5f-39172ce16685&spaceId=30c36155-a603-469f-957f-b0854b6e2372&width=1060&userId=&cache=v2)

0.036B 的模型在 Batchsize 扩大以后 loss 有降低

由于我们的 WSD 学习率优化器可以在任何阶段退火，取得该阶段最优的模型，因此我们有机会探索，如果持续训练一个大小为 N 的模型，最优情况下能超过多大参数量的 Chinchilla-Optimal 模型。

首先我们估计了持续训练过程中，模型性能随计算量的变化。由于不确定函数形式，我们尝试了两种拟合公式。1）指数形式：$L(C) = \alpha e^{-\beta C} + L_0$﻿ 和 2）幂律形式：$L(C) = \beta C^{-\alpha} + L_0$﻿ 。 两种函数的拟合结果如图：

![](https://shengdinghu.notion.site/image/https%3A%2F%2Fprod-files-secure.s3.us-west-2.amazonaws.com%2F30c36155-a603-469f-957f-b0854b6e2372%2F771a8ea8-a219-41eb-b66f-6a76bad70fda%2FUntitled.png?table=block&id=8721843d-04d0-4b97-b568-174238ac722b&spaceId=30c36155-a603-469f-957f-b0854b6e2372&width=1060&userId=&cache=v2)

因此我们认为幂律形式的拟合效果更好。通过拟合我们得到 0.036B 持续训练，最终理论上可以达到 3.27 的 C4 Loss。为了从直观上估计和感受，在可接受的训练时长内，0.036B 模型可以达到多大的 Chinchilla Optimal 模型的效果，我们同样以最优配置训练了一个 0.17B 的模型。0.17B 模型在 2 倍 Chinchilla Optimal 数据量下训练，消耗的计算量为 $6.6\times 10^{18}$﻿ Flops。在这个计算量下，0.036B 的模型可以获得 3.37 的 C4 Loss，与 0.17B 的模型的 3.34 Loss 接近。因此我们认为一个模型用我们的 WSD 调度器训练，在消耗等量计算量时，可以达到约 5 倍模型参数量的模型。而持续训练下去，有可能超越更大的模型。

我们在 MiniCPM 上进行了验证。我们以完全相同的数据配方训练了 0.036B、0.1B、0.2B、0.5B、0.8B，1.2B 六个小模型，分别至其 Chinchilla Optimal 的数据量。绘制得到 Scaling 图像如下，根据这个图像，我们可以预测 9B 模型的 Chinchilla Optimal 的终态 C4 Loss 约为 2.40，7B 模型约为 2.45。MiniCPM 的最终 C4 Loss 为 2.41，接近于 9B 的 Chinchilla Optimal 模型。

![](https://shengdinghu.notion.site/image/https%3A%2F%2Fprod-files-secure.s3.us-west-2.amazonaws.com%2F30c36155-a603-469f-957f-b0854b6e2372%2Fe4d54434-f379-405c-b2d2-ec8fde2cc22c%2F%25E4%25BC%2581%25E4%25B8%259A%25E5%25BE%25AE%25E4%25BF%25A1%25E6%2588%25AA%25E5%259B%25BE_005962f2-f0db-4285-bf8b-5231af98b03f.png?table=block&id=2178499c-1453-4be8-ab12-5cf08cb2a442&spaceId=30c36155-a603-469f-957f-b0854b6e2372&width=830&userId=&cache=v2)

由于 WSD LRS 的退火阶段模型会有较大幅度的损失下降，我们猜想在这个阶段加入高质量数据，会有如下两个优点：

相对于在 sft 阶段加入高质量数据，在退火阶段加入数据，模型学习更充分。

相对于在 pretrain 一开始阶段加入高质量数据，更能支持小数据的训练，否则在一个未预先定好训练步数的持续预训练过程中，小数据会重复过多次数，造成负面影响。

基于这两点猜想，我们提出：在预训练阶段只使用通用、量大的预训练粗质量数据，而在退火阶段，使用非常广泛的高质量知识和能力数据以及 SFT 的高质量数据，混合入预训练数据进行退火。

为了验证我们的方法与直接 SFT 相比的优势，我们从一个中间检查点开始进行了两组实验。

实验 A：仅使用预训练数据进行退火，接着进行 4B token 的 SFT。

实验 B：使用如上的高质量数据 +SFT 数据混入预训练数据进行退火，同样进行 4B token 的 SFT。

实验结果表明在退火开始时加入高质量数据的收益远高于在退火完成后的 sft 阶段加入。因此我们建议模型能力的特化和增强应从退火阶段开始进行。

MiniCPM 模型作为通用模型，具备英文、中文、中国古文、代码、表情符号，其他语言等多方面能力，因此词表相对较大，大小为 122753。该词表构建于大量综合语料上，使用 sentencepiece 库进行 BPE，添加了包括繁体中文、罕见字、emoji、希腊字母、俄文字母等等特殊符号。

我们在中文、英文、代码、论文各 30 万篇非训练文档上进行了压缩率测量，MiniCPM 的 tokenizer 取得了最高的压缩率（Bytes/Tokens）

词表的大小主要决定了模型覆盖面的宽窄，而不决定模型本身的能力深浅。在大模型参数规模下，词嵌入参数量可以忽略不计，但是小模型下却不可忽视，因此我们使用了 tie\_word\_embedding 的方案进一步减少参数量，即输出层和输入层共享参数，在预实验中，我们发现这几乎不会牺牲性能。在使用 tie\_word\_embedding 后，MiniCPM 的词嵌入为 $2304\times 122753\sim0.28B$﻿ 的参数量。

我们使用了 1T 的去重后的数据，其中大部分数据从开源数据中收集来，比例如下图。

![](https://shengdinghu.notion.site/image/https%3A%2F%2Fprod-files-secure.s3.us-west-2.amazonaws.com%2F30c36155-a603-469f-957f-b0854b6e2372%2F421772f6-b3e5-49bc-b3ad-3e0aedcdea17%2Fstable_mixture.png?table=block&id=5e1bf0f5-6f4c-4a56-b936-774db553d918&spaceId=30c36155-a603-469f-957f-b0854b6e2372&width=960&userId=&cache=v2)

我们使用了模型沙盒实验中探索出的最优配置，WSD LRS，batchsize 为 3.93M，Max Learning Rate 为 0.01。

![](https://shengdinghu.notion.site/image/https%3A%2F%2Fprod-files-secure.s3.us-west-2.amazonaws.com%2F30c36155-a603-469f-957f-b0854b6e2372%2Fb32a04ad-8e78-46e4-9bdc-036e496d9927%2Fdecay_data_mixture.png?table=block&id=e60a4355-4654-4f95-9bae-9caca8095a32&spaceId=30c36155-a603-469f-957f-b0854b6e2372&width=960&userId=&cache=v2)

最终我们两阶段预训练的过程中，C4 Loss 的变化如下图，在 263000 步（约 1T 数据）时，开始进行退火，退火过程也出现了损失函数急剧下降的现象，同时在各种任务数据、SFT 数据上的 Loss 也有显著下降。

具体在 WSD 调度器的退火形式上，我们采用了指数退火，即 $f(s-S）= \eta \times 0.5^{(s-S)/T}$﻿。其中 T 为退火的半衰期，我们设置为 T=8000 步。

![](https://shengdinghu.notion.site/image/https%3A%2F%2Fprod-files-secure.s3.us-west-2.amazonaws.com%2F30c36155-a603-469f-957f-b0854b6e2372%2Ff70abacd-85ca-423c-b107-875cf6c96707%2FUntitled.png?table=block&id=ddb7a6fd-594e-4110-9e9a-491586d113f4&spaceId=30c36155-a603-469f-957f-b0854b6e2372&width=770&userId=&cache=v2)

整个训练过程中，C4 训练集上 Loss，由于数据不重复，所以训练集 loss 可以当作验证集 Loss。

在上述获得的基础模型之上，我们进行了对齐（alignment）。尽管 sft 的数据已经加入退火阶段，但是我们发现仍然有必要进行 SFT 阶段，换言之，退火阶段和 SFT 阶段缺一不可。我们使用了和退火阶段类似的 SFT 数据（不同的是将预训练数据、高质量的无标注数据例如 wiki 去除，只留高质量有标标注数据，即 sft 数据）。我们进行了约 6B token 的 SFT 训练。SFT 的学习率衔接上退火结束的学习率，为 1e-3，同样使用了 WSD Scheduler。

在 SFT 之后，我们采用 DPO 对模型进行进一步的人类偏好对齐。在这一阶段，我们采用 UltraFeedback 作为主要的对齐数据集，并内部构建了一个用于增强模型代码和数学能力的偏好数据集。我们进行了一个 Epoch 的 DPO 训练，学习率为 1e-5 且使用 Cosine Scheduler。更多 DPO 和数据设置细节可以可见我们 UltraFeedback 论文。

整体评测使用了我们的开源工具 [UltraEval](https://github.com/OpenBMB/UltraEval)。UltraEval 是一个开源的基础模型能力评测框架，提供了一套轻量级、易于使用的评测体系，支持主流大模型的性能评估，服务模型训练团队的快速评测需求。底层使用开源框架 vLLM 进行推理和加速，数据集选取了常用的权威数据集，包括：

问答，选取了 HellaSwag、ARC-E、ARC-C

由于大模型评测比较难以统一，且大量评测也没有公开的 prompt 和测试代码，对于具体评测方式，我们只能尽量做到适合各类模型。整体而言，我们测试时采用统一的输入 prompt，并会按照各自模型适合的模板进行调整。模型评测脚本及 prompt 已开源在我们的 Github 仓库中，也欢迎更多开发者来不断改进我们的评测方式。

整体评测结果如下。总体而言，MiniCPM 在上述数据集上，英文均分与 Mistral-7B-v0.1 相近，中文均分显著优于 Mistral-7B-v0.1。

与大模型相比：超过或持平大部分 7B 规模模型，超越部分 10B 以上的模型。

与小模型对比，除部分英文评测集外，其他测试集均超过现有模型。

MiniCPM 的评测时模型推理使用的是 vllm=0.2.2，这是一个我们在两个月前 fork 的稳定版本，我们计划不久后将和 vllm 官方沟通，将推理代码增加至 vllm 官方仓库。而 Mistral-7B-v0.1 则用的是 vllm 最新的 vllm=0.2.7。

我们对 QA 任务进行测试时，通常可能采用两种方式，第一种是 PPL：将选项作为题目延续时的 PPL 作为选择指标（以上表格中\* 表示使用 PPL），第二种是直接生成，直接让模型输出答案选项。我们发现，这两种方式得到的结果差异较大。事实上，MiniCPM 在直接生成和 PPL 的测试结果接近，直接生成的表现性能较好，而 Mistral-7B-v0.1 则在 PPL 上表现较好，直接生成上效果较差。为了应对这种现象，我们汇报每个模型的分数时，采纳两种评测方式得分最高方式的得分，以此保证对比的公平性。

我们观察到，虽然 Phi-2 的评测结果超过 Mistral-7B，但是实用体感并没有达到同等水平。

在使用 DPO 完成偏好对齐后，模型在 MT-Bench 上的分数从 SFT 后的 6.89 上涨至 7.25，甚至超过了包括 Llama2-70B-Chat 在内的大模型。

我们挑选了一些展示 MiniCPM-2B 通用能力的例子。这些例子说明了 MiniCPM 的能力多样性。

![](https://shengdinghu.notion.site/image/https%3A%2F%2Fprod-files-secure.s3.us-west-2.amazonaws.com%2F30c36155-a603-469f-957f-b0854b6e2372%2F82310bd8-d650-48e3-a012-e17cea9d577c%2Fknowledge.case1.png?table=block&id=922cc019-fd4d-4eb1-9cdb-0f12749140c1&spaceId=30c36155-a603-469f-957f-b0854b6e2372&width=1420&userId=&cache=v2)

![](https://file.notion.so/f/f/30c36155-a603-469f-957f-b0854b6e2372/ad4f17ad-fa46-48e1-b6f6-0ea0a78ffca8/code.case1.gif?table=block&id=658ed53d-c781-4eab-8f05-ece9240f2c30&spaceId=30c36155-a603-469f-957f-b0854b6e2372&expirationTimestamp=1733385600000&signature=bLn9Au9YIR5IN99Peu_xayKPfCNDjgNgSXrwy60Cq5k)

![](https://file.notion.so/f/f/30c36155-a603-469f-957f-b0854b6e2372/805742f9-4b6b-4702-bbbe-5292665a0798/code.case2.gif?table=block&id=8467c56d-65f3-4324-8609-38c10048d5fc&spaceId=30c36155-a603-469f-957f-b0854b6e2372&expirationTimestamp=1733385600000&signature=uGkt0rEmR4UIzT7kruaQpA1inCaxe1zndXSSZo9Z7QI)

![](https://shengdinghu.notion.site/image/https%3A%2F%2Fprod-files-secure.s3.us-west-2.amazonaws.com%2F30c36155-a603-469f-957f-b0854b6e2372%2F6a33bf55-6a17-425d-8023-59d17f67536e%2Fmath.case2.png?table=block&id=1891db3c-a526-45c7-b565-c0768ed6c587&spaceId=30c36155-a603-469f-957f-b0854b6e2372&width=1420&userId=&cache=v2)

![](https://shengdinghu.notion.site/image/https%3A%2F%2Fprod-files-secure.s3.us-west-2.amazonaws.com%2F30c36155-a603-469f-957f-b0854b6e2372%2Fb67008cc-fefc-4d91-acf7-935903ab499d%2Ftranslation.case2.png?table=block&id=fe58412c-db37-4855-803f-b14d8218155e&spaceId=30c36155-a603-469f-957f-b0854b6e2372&width=1420&userId=&cache=v2)

![](https://shengdinghu.notion.site/image/https%3A%2F%2Fprod-files-secure.s3.us-west-2.amazonaws.com%2F30c36155-a603-469f-957f-b0854b6e2372%2Fc92bba8f-cc01-4503-9b88-c25a0b2ce749%2Fspecial_char.case1.png?table=block&id=30e9dd36-4f44-4f75-af82-02cee22f177e&spaceId=30c36155-a603-469f-957f-b0854b6e2372&width=1420&userId=&cache=v2)

为进一步降低 MiniCPM 的计算开销，使用 GPT-Q 方法将 MiniCPM 量化为 int4 版本。相较于 bfloat16 版本与 float32 版本，使用 int4 版本时，模型存储开销更少，推理速度更快。量化模型时，我们量化 Embedding 层、LayerNorm 层外的模型参数。

对于参数矩阵 $\bold{W}\in \mathbb{R}^{d_{out}\times d_{in}}$﻿, 我们将连续 $G$﻿列聚合为一组，形成 $d_{in} / G$﻿ 个参数组，而后对每个参数组分别进行量化。对于参数组中的具体参数 $\bold{w}$﻿，其量化放缩系数 $scale$﻿和零点 $zero$﻿以如下方式计算：

$$$
scale=\frac{\max(\bold{w} ) - \min(\bold{w})}{2^4-1}, zero = -\frac{\min(\bold{w})}{scale} - 2^3
$$$

依照上述放缩系数和零点，$\bold{w}$﻿量化后为

$$$
\bold{\hat w} = quant(\bold{w}) = round(\frac{\bold{w}}{scale} +zero)
$$$

其中取整函数 $round()$﻿为向最近整数取整。反量化时，操作方式如下：

$$$
dequant(\bold{\hat w}) = scale \cdot (\bold{\hat w} - zero)
$$$

使用 GPT-Q 方法进行量化时，在标注数据 $\bold{X}$﻿上最小化量化误差 $||\bold{WX} - dequant(\bold{\hat W}\bold{X})||^2_2$﻿, 并循环对矩阵的未量化权重进行如下更新，其中 $q$﻿是当前量化的参数位置，$\bold{F}$﻿为未量化权重，$\bold{H}_\bold{F}$﻿是量化误差的 Hessian 矩阵。

$$$
\delta_F=-\frac{\bold{w}_q - dequant(quant(\bold{w}_q))}{[\bold{H}_\bold{F}]^{-1}_{qq}}\cdot (\bold{H}_\bold{F}^{-1})_{:,q}
$$$

对于 MiniCPM-2B-SFT 与 MiniCPM-2B-DPO，我们均进行了 int4 量化，导出模型 MiniCPM-2B-SFT-Int4 与 MiniCPM-2B-DPO-Int4。

基于 MiniCPM，我们构建了一个支持中英双语对话的端侧多模态模型 MiniCPM-V。该模型可以接受图像和文本输入，并输出文本内容。MiniCPM-V 的视觉模型部分由 SigLIP-400M 进行初始化，语言模型部分由 MiniCPM 进行初始化，两者通过 perceiver resampler 进行连接。

高效推理：MiniCPM-V 可以高效部署在大多数 GPU 显卡和个人电脑，甚至手机等边缘设备上。在图像编码表示方面，我们基于 perciever resampler 将每个图像压缩表示为 64 个 token，显著少于其他基于 MLP 架构的多模态模型的 token 数量（通常大于 512）。这使得 MiniCPM-V 可以在推理过程中以更低的存储成本和更高的运算速度进行推理。

性能强劲：在多个基准测试（包括 MMMU、MME 和 MMbech 等）中，MiniCPM-V 在同规模模型中实现了最佳性能，超越了基于 Phi-2 构建的现有多模态大模型。MiniCPM-V 在部分数据集上达到了与 9.6B Qwen-VL-Chat 相当甚至更好的性能。

双语支持：MiniCPM-V 是首个支持中英文双语能力的可边缘部署的多模态端侧大模型。该能力是通过跨语言泛化多模态能力高效实现的，这项技术来自我们的 ICLR 2024 splotlight 论文。

MiniCPM-V 模型的训练分为 2 个基本阶段：

预训练阶段: 我们使用 300M 中英图文对数据进行视觉和语言基本概念的对齐，并学习大规模多模态世界知识。

指令微调阶段: 我们一共使用了 6M 多任务问答数据、1M 纯文本数据、1M 多模态对话数据进行指令微调，进一步对齐并激发多模态基础能力。

详细训练与微调过程也可见我们 ICLR 2024 splotlight 论文。

进行 Int4 量化后，MiniCPM 只占 2GB 空间，具备了在端侧手机进行模型部署的条件。对此，我们针对 Android 和 Harmony 系统使用开源框架 MLC-LLM 进行模型适配，针对 iPhone 系统使用开源框架 LLMFarm 进行模型适配，并分别选取了部分端侧手机设备进行了测试。此外，我们首次验证了在端侧手机运行多模态大模型的可行性，并成功在手机上运行。

值得注意的是，我们并未针对手机部署进行优化，仅验证 MiniCPM 在手机侧进行推理的可行性，我们也欢迎更多开发者进一步调优更新下面的测试列表，不断提升大模型在手机侧的推理性能。