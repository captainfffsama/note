---
title: "2万字详解MoE：模型结构与工程实现"
source: "https://mp.weixin.qq.com/s/devmnLbP8PKUlZ1480SCkQ"
author:
  - "[[魔法学院的Chilia]]"
published:
created: 2026-09-08
description: "2万字详解MoE：模型结构与工程实现"
tags:
  - "clippings"
---
魔法学院的Chilia 吃果冻不吐果冻皮 *2026年9月2日 22:43*

为了内容的完整性，本文和上一篇文章的 MoE 部分有一定程度的重合。上一篇是综述性的文章，这一篇会讲得更细一些，关注 MoE 的细节。

上一篇回顾： [1.5万字速通LLM主流模型结构（Llama/Qwen/GLM）](https://mp.weixin.qq.com/s?__biz=MzU3Mzg5ODgxMg==&mid=2247497856&idx=1&sn=5a9d79e623c37fbcd249df0bfe834e18&scene=21#wechat_redirect)

01

**MoE 结构介绍**

1.1 整体结构

Scaling Law 告诉我们，LLM 的表现与算力、数据、参数量息息相关。但是，对于稠密模型来说，我们提高模型参数量时必须同时提高所使用的算力。

而 MoE 模型的解决方法则是在计算时只激活部分参数，这样我们就可以在同等算力下训练更大参数量的模型，从而达到更好地表现。

如下图所示，MoE 就是将稠密模型的 FFN 部分变成多个，每次激活一部分 FFN 参数。这些 FFN 就被称作“专家”。

![图片](https://mmbiz.qpic.cn/mmbiz_png/WFTLymtmNia2NZOXgVB2exibZyj6S0uzSF0y1GrfQ4IOhPL3qnttfsI7PO9FgLaicaQicorgqzvO5oXmvUUkK7NFkgFUWeEshZyE1p2HdgibCY6I/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=1)

图源：https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-mixture-of-experts

1.2 路由器模块（Router）的设计

路由器（Router）是 MoE 的核心。Router 的作用是将每个 token 分配给不同的专家。

路由器最终会输出两个张量：probs（路由权重）和 routing\_map（布尔掩码，表示 token-专家的分配关系）。

下面这张图直观地展示了关于路由器的几个模块：

![图片](https://mmbiz.qpic.cn/mmbiz_png/WFTLymtmNia2NRJscErFfvqAlM82cuEgmNF7WRURwGNZ24EbYOb9kPUCpe9F1sRrcNfqyNZdYV8VLCgUo3eHXnu74HFDroBRNCPaTFM85KLU/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=2)

首先，隐藏状态（hidden state）经过一个线性层（Gating Layer）得到分配给不同专家的 logits；之后 logits 经过打分函数（Score Function）和 Top-K 得到选择的专家 index 和分配给选中专家的权重。

在这个过程中，因为涉及到专家负载不均衡问题，所以需要各种负载均衡的策略（Load Balancing Mechanisms）。

我们在这节会对上述模块进行介绍。

1.2.1 可学习路由与哈希路由

（1）可学习路由器（Learnable Router）

可学习路由器是目前绝大多数主流 MoE 模型的标配。

路由器其实本质上就是一个可训练的线性层 W\_r ∈ R^h\*E（其中 E 为专家数、h 为隐藏层的维度），它将每个 token 的隐藏状态（hidden state）映射为 E 个 logits，再通过评分函数（Score Function）+ top-k 选择专家。路由器的权重是随训练更新的。

如下图所示，得到路由器分配给每个专家的权重之后，会首先进行专家的 Top-K 选择，之后对这 Top-K 个专家的输出结果按照权重进行加权求和，得到输出结果。

![图片](https://mmbiz.qpic.cn/sz_mmbiz_png/WFTLymtmNia3vnKo53ibpeC2RaoNNpnzQfA6pnsZnBTKfhG1jAlJGgzpGmazLksZazqZL0FvXMeBkRVPe9FgmaOWcAL2sYl6W4TvBPXjMC4Hs/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=3)

图源：https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-mixture-of-experts

（2）哈希路由器（Hash Router）

“自由人已经了解比勒的秘密，自动参加神圣的抽签仪式。抽签仪式每隔六十夜在神的迷宫里举行，决定人在下一次抽签之前的命运......”——《巴比伦彩票》，博尔赫斯

可学习路由器可以能学到 token 语义与专家能力的匹配，但是缺点就是路由经常不平衡，因此需要负载均衡辅助损失（load balancing loss）。

相比之下，哈希路由（Hash Router）用确定性的哈希函数将 token 分配给专家。

哈希函数在训练前是固定好的，没有可训练参数。而且可以保证它的分配是均衡的，所以也不需要负载均衡辅助损失。

哈希路由随机分配专家。当专家数量比较少的时候，即使分配是随机的，模型仍能学到有用的专家分化。但当模型和专家数量扩大后，可学习路由的优势就变得不可忽视了。

因此哈希路由在主流 MoE 模型中其实一直很少使用。不过，最近的 DeepSeek-V4 使用了一部分哈希路由，说明哈希路由并没有完全过时，所以我们在这一节也介绍一下这种方式。

哈希路由出自 2021 年的文章 Hash Layers For Large Sparse Models。哈希路由决策取决于输入 token 的词表 ID x\_t，而非隐藏状态 h\_t。

一个 token 在每一层都会使用哈希函数“抽签”决定自己这一层选择什么专家，然后在下一层继续“抽签”选择。

由于哈希函数是确定的，所以在训练之前，每个 token 在每层该选择哪个专家就已经固定了。（这颇有一些宿命论的感觉）

![图片](https://mmbiz.qpic.cn/sz_mmbiz_png/WFTLymtmNia3rBSznrlXnyw8z0csVz6F20pAIlTwiaHbWM3FcmNth5hpkchkKbkrXTbjZjWMPfAECv8AWiaXGaBibCApAGI98dL4pJ5eBnAx4rw/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=4)

每层都直接根据输入的 token ID 进行路由选择

我们再来看看 DeepSeek-V4 是如何使用哈希路由的。

DeepSeek-V4 的技术报告是这样写的：

...compared with DeepSeek-V3, we replace the dense FFN layers in the initial several Transformer blocks with MoE layers that employ Hash routing (Roller et al., 2021). The Hash routing strategy determines the target experts of each token according to a predefined hash function with regard to the input token ID.

在上一篇文章的最后我们提到，DeepSeek-V3 的前三个 Transformer Block 是稠密的，后面每个都是 MoE 层。

这种交替使用 dense 层和 moe 层的坏处就是会带来一些训练/推理调优的困难，因为每个层的参数量不一样、计算量不一样，会给调优带来一些麻烦。

可能是因为这个原因，DeepSeek-V4 就把前三个稠密层替换成了带哈希路由的 MoE 层，后面的层仍保持为标准的 MoE 层。

前几层特征比较粗糙，所以用哈希路由也没有什么大问题。哈希路由足够稳定且避免了早期训练的路由不稳定问题。

不过 DeepSeek-V4 的路由方式并不是完全无学习的，它是一个混合设计。去哪个专家是固定的（专家选择用查表，不用 topk）；但是权重多少是可学习的，用来加权被选中专家的输出。

具体代码如下：

```python
class DeepseekV4HashRouter(nn.Module):      def __init__(self, config):          self.weight = nn.Parameter(torch.empty(num_experts, hidden_dim))  # 可学习的门控权重          self.score_fn = ACT2FN[config.scoring_func]                       # sqrt(softplus)          self.tid2eid = nn.Buffer(                                         # 固定的 token_id → expert_id 映射表              torch.zeros(config.vocab_size, self.top_k, dtype=torch.long),              persistent=True          )      def forward(self, hidden_states, input_ids):          logits = F.linear(flat, self.weight)          # 1. 仍然计算所有专家的 logits          scores = self.score_fn(logits)                # 2. sqrt(softplus) 评分          indices = self.tid2eid[input_ids.reshape(-1)]  # 3. ★ 专家选择用查表，不用 topk# 4. 从 scores 中提取被选中专家的权重
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)  # 5. 归一化          return logits, weights * self.routed_scaling_factor, indices
```

1.2.2 评分函数（Score Function）

MoE 的路由器收到一个 token，要先给 E 个专家分别打分、得到 logits，然后用评分函数（Score Function）把这些原始分数变成权重或者选中的概率。

这里介绍三种打分函数：Softmax、Sigmoid、Sqrt（Softplus）。

（1）Softmax 的两种模式

我们知道，经过 Softmax 之后，所有专家分数会被归一化成一个总和为 1 的概率分布。

这意味着一个专家得分的提高就必然会导致其他专家的降低，因此 Softmax 天然具有竞争性。

按照 Softmax 和 Top-K 的先后关系，可以分为 Post-softmax 和 Pre-softmax 两种：

(a) Post-softmax（先 Top-K 后 Softmax，默认模式）

```makefile
scores, top_indices = topk(logits, k)       # 在原始 logits 上选出前 k 个probs = softmax(scores, dim=-1)             # 仅对这 k 个做 softmax → 和为 1
```

这样，梯度只传回被选中的 K 个专家，未被选中的专家完全没有梯度信号。

(b) Pre-softmax

```apache
scores = softmax(logits, dim=-1)                 # 1. 对全部 E 个专家做 softmaxprobs, top_indices = topk(scores, k)             # 2. 选概率最高的 k 个probs = probs / probs.sum(dim=-1, keepdim=True)  # 3. 二次归一化（因为 top-k 后和<1）
```

因为 Softmax 计算涉及全部专家,所以计算量更大了，但是梯度会传回所有专家。

由于 Top-K 只截取了概率总和的一部分（小于 1），需要用归一化重新让权重之和为 1。

（2）Sigmoid

Softmax 那种"你死我活"的竞争关系会带来一个问题，就是所有专家分配的概率权重是耦合的，我们无法单独调整某一个专家的概率而不影响其它专家。

相比之下，Sigmoid 各自算每个专家的 logits 再归一化，每个专家的分数是独立的、完全不受其他专家的影响。

![图片](https://mmbiz.qpic.cn/sz_mmbiz_png/WFTLymtmNia1BVqVddU83tHxvoPB9icjtfyULic2Zwblqiaf9FNOG94GLnkt1796BYM0V7FFicgsEiapuZq1qjRw06YTwKPJ441gICNryWjiaTe8Gg/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=5)

Sigmoid Gate 的 Router 计算公式

```nginx
scores = sigmoid(logits)                          # 或 sqrt(softplus(logits))  _, top_indices = topk(scores, k)                  # 选概率最高的 k 个# 用原始分数做权重
probs = probs / (probs.sum(dim=-1) + 1e-20)       #  归一化
```

注：其实在sigmoid的情况下，“先 sigmoid 再 top-k”和“先 top-k 再 sigmoid”在数学上完全等价。

但在实际使用中，因为 Aux-Loss-Free 中的 expert bias 必须加在 sigmoid 之后的得分上（见 1.2.4.2），所以顺序一般是“先 sigmoid 再 top-k”。

在 Loss-Free Balancing 中，作者通过实验发现使用 Sigmoid 作为评分函数的效果比 Softmax 更好。

其对应的实验结果如下图所示，可以看到 Sigmoid Gate 相比 Softmax Gate 在同样文本上的困惑度（Perplexity）更低：

![图片](https://mmbiz.qpic.cn/mmbiz_png/WFTLymtmNia3UI9LYcOkfQKtPyXZ7fNicicYwSkU5bCuHVBUJARnhFjQf8cWTMAwzaaib8sRE7wv2Eq7OFnKaaCyhdPWatZmWXbJVB75xHp5m3Y/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=6)

\\alpha 是论文中辅助损失的系数，横轴 MaxVio 是论文中提的一种衡量专家不平衡程度的一个指标，纵轴是困惑度（越低越好）

（3）Sqrt（Softplus(·)）

DeepSeek-V4 将评分函数从 Sigmoid 换成了 Sqrt（Softplus(·)），所以我们在这里也看一下这个评分函数。

Sqrt（Softplus(·)）的公式如下：

![图片](https://mmbiz.qpic.cn/mmbiz_png/WFTLymtmNia0xc93hZEEFhDvgqtttvQW8bd7PUNETKNrgeBFOSiaRibS7E2awPNGwNDib76bPpyibOV8Saujr9NibPCW17mqib84pe3XklJkteogoI/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=7)

1.2.3 关于专家分化

一个很符合直觉的猜测是：MoE 中的每个专家都像一个领域专家，有的擅长哲学，有的擅长生物学，路由器会把不同学科的 token 精准分发给对应专家。

实际上果然如此吗？

Mixtral 的研究表明，专家们其实并不会按主题/学科形成清晰的分工，而是更偏向于捕捉语法结构和表层格式特征，并没有出现“这些专家只处理哲学，那些只处理生物学”这样明显的模式。

![图片](https://mmbiz.qpic.cn/mmbiz_png/WFTLymtmNia3vczdWUzLWvCUUibtW3iaia662ibnTwJiaHnJpic6EVSlzViaH6uRZDWhy58xiaYCq8SKPVMibNjsib1Y74vGnyH2ciavCYgqQ8D0RXVK5FA/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=8)

比如某个专家更擅长处理标点，某个专家更擅长处理连接词

这种语法结构的专业化具体体现在下面的一些现象：

- Python 代码中的 self（通常由多个 token 构成）会反复被路由到同一个专家；
- 英文中的 Question 等词表现出稳定的专家偏好；
- 代码缩进 token 几乎总是分配给特定专家
- ......
![图片](https://mmbiz.qpic.cn/mmbiz_png/WFTLymtmNia306UEltNh4IywhBu6x2AHMzeR0Kxu7VXcgnbqwmGf9yhKBNfh9EYU3MTWKNk6LV4roLdVFJd2eQm6tAxgB87sWfG9VLyB1YAY/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=9)

https://arxiv.org/pdf/2401.04088

OpenMoE 的研究则提供了一些更深入的分析：

第一个观察到的现象是，同一个 Token ID 无论出现在什么上下文中，都更倾向于路由到同一个专家。

"MoE tends to simply cluster tokens based on similar token-level semantics, implying that, regardless of context, a certain token is more likely to be routed to a certain expert."

此外，实际上路由在训练极早期就已经固化了。这可能是因为一旦某个 token 被固定分配到某个专家，如果此时突然改变分配，loss 就会大幅上升，梯度会把它推回原来的专家。

"The model has started to fix its routing at the very early stage of training. Even if we change the training data mixture (from 52.25% code to 20% code) and training objective (from UL2 to CasualLM), the routing decision is still fixed."

1.2.4 负载均衡，辅助损失（Auxiliary Loss）与 token drop

如果不加干预，router 很容易出现负载不均衡的情况。即训练初期某些专家学得更好，router 就更频繁地把 token 分配给它们，导致这些专家训练数据更多、越学越好，而其他专家则越来越被冷落。

这就是路由坍塌，被闲置的专家形同虚设，模型实际上退化成了一个更小的稠密模型，浪费了 MoE 的参数量优势。

1.2.4.1 辅助损失（Auxiliary Loss）

辅助损失的核心思想是，在不影响模型主任务（下一个 token 预测）性能的前提下，通过添加额外的损失项，软性地引导 Router 做出更均衡的专家选择。

①专家级均衡损失（Expert-Level Balance Loss）

这是最基础的负载均衡损失，旨在防止路由坍塌。确保每个专家都能被充分训练，而不是只有少数几个专家被频繁使用。

![图片](https://mmbiz.qpic.cn/sz_mmbiz_png/WFTLymtmNia0MT61vmLwozVAKytTdM0jGA4ut0iaeyzhQs8NLvLCgiabdcMmrPy40fBMc9ROlOt41JWDkymnEiceB0HkEXAs8M0JoicWGDq7EWgM/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=10)

Switch Transformers (https://arxiv.org/pdf/2101.03961)

其中：

![图片](https://mmbiz.qpic.cn/sz_mmbiz_png/WFTLymtmNia3ZNtLxj6xPQaO4Eia5QmOicOtzy50PJRA15kpf9HSqqpsd2KBaqicZrCc2tgVugNJZCMHj5lBvcicdOEzxaGcDibBiaCJsC6xw8TmHw/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=11)

说明完全不均匀的极端情况下，Loss 会是均匀情况下的 N 倍。

此外，除了这种 Batch-wise 的粒度，还有一种全局（Global-wise）粒度的负载均衡损失函数。

在 Batch-wise 粒度中，f\_i 是当前 micro-batch 中专家 i 被分配的 token 比例；而在 Global-wise 粒度中，f\_i 则替换为历史累积的平均专家 i 被分配 token 的比例。

使用 Global-wise 粒度的考虑主要是：当 micro-batch size 很小时，单 batch 的统计信息噪声很大。

如果当前 micro-batch 的 token 分布恰好很极端（比如恰好全是代码 token），那么统计信息会剧烈波动。

这种波动会导致辅助损失的梯度方向不稳定：模型可能在两个连续 micro-batch 之间被迫朝相反方向调整路由，造成训练震荡。而全局平均则可以有效降低这种噪声。

②设备级均衡损失（Device-Level Balance Loss）

在大规模的 MoE 模型中，专家数量通常远大于单张 GPU 的承载能力（例如 DeepSeek-V3 有 256 个 routed experts，而单卡只能放下几个专家）。

因此，专家会被分组部署在多台设备（GPU）上，这也叫“专家并行”：

![图片](https://mmbiz.qpic.cn/sz_mmbiz_png/WFTLymtmNia1VXqAz0nicByic7SABykJ5G3rNcGA1iafpanpeQovec4qSN47DODYVgfC6wm4QkZGaCQEePUhe0OHED3YQKJg3EOHlG5Q3rIvFn0/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=12)

在同步训练中，所有设备必须等待最慢的设备完成计算才能继续（比如，设备 1 上的 token 堆积时，其它设备会处于空闲等待状态），这种设备间的不均衡会造成计算瓶颈，拖慢整个系统的效率，造成算力的浪费。

设备级平衡损失就是为了解决这个层面的负载不均，它鼓励每个 device 接收到的总 token 数量和路由到这个 device 的概率都保持均衡，因此各 device 的计算量相当，避免某些 device 过载而其他闲置。

![图片](https://mmbiz.qpic.cn/mmbiz_png/WFTLymtmNia3gpBXPK0QLAH2p6dHiaJS4ziculahiaT5hgicCRr9iciaMAiciaNOUXgrRiaKvRRVgVlbibXCMcx8qibk0pq54y1w7NRJSxwZlp2JyAOADFg/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=13)

③通信均衡损失（Communication Balance Loss）

因为不同专家分布在不同设备上，所以 token 的分发和结果聚合会引入额外通信开销。

要理解这个 balance loss，首先我们要区分网络通信的两个方向：

- 发送（Send）：一个设备上的 token，需要被路由到其他设备上的专家去计算。这是设备向外发送数据。
- 接收（Receive）：一个设备上的专家，收到了来自其他设备发来的 token 进行计算。这是设备从外部接收数据。

每个设备既发送 token 也接收 token，这就是All-to-All通信。

在此前，Deepseek-V2 中为了均衡通信开销，已经提了一个方法：设备限制路由（Device-Limited Routing）。

即，对于当前 token 先计算它与每个设备的总体亲和度。这个总体亲和度通常取该设备上所有专家的路由分数的最大值（或总和）。

然后从所有设备中选出亲和度最高的 M 个设备，然后只在这 M 个设备包含的专家集合中执行 Top‑K 选择。

这样，无论 K 有多大，每个 token 至多只与 M 个设备发生通信，最坏情况下的通信开销被严格控制。

但是这个 Device-Limited Routing 只解决了发送侧的问题，接收侧是完全没有约束的。

一种典型的不均衡场景是：某个设备上恰好存放了当前非常热门的专家，导致大量 token 都被路由器选中，需要路由到这个设备来计算。这就会导致接收端的网络带宽被打满。

这里你可能会有一个疑问：之前不是有设备级均衡损失了吗？它不也是让每个设备上的计算量差不多吗？那么为什么还需要通信辅助损失呢？

区别在于统计口径不同：

- 设备级均衡损失统计的是，每个设备上的专家总共被选了多少次。
- 通信均衡损失统计的是，每个设备作为接收方，总共收到了多少个外部来的token。

一个典型的冲突场景是：假设所有 token 都采用一种策略，即选一个本地专家 + 一个远程设备A的专家。

那么，从专家计算量看：每个设备上的专家都被均匀地选中，设备级均衡损失为 0，完美！

但从接收 token 数看：设备 1 收到了所有 token 的远程请求，而其他设备接收的远程请求为 0，通信严重失衡。

所以，通信均衡损失通过惩罚接收 token 过多的热门设备，从而避免接收端出现通信和计算瓶颈。

计算公式如下：

![图片](https://mmbiz.qpic.cn/mmbiz_png/WFTLymtmNia0lgGCCldPRRqtelsiauN1rdDq6JjlicwgbdQ24SOnn70rRRXjylGiaNZz9dPt3LicKBr0VYjKnv0B5UXwUMUhrGwRUwgBApMRdibfI/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=15)

【综上所述】，上面的三个 balance loss 构成了完整的、分工明确的负载均衡体系：

- 专家级均衡损失：在微观层面，确保每个专家个体能够收到数量差不多的token，防止路由坍塌。
- 设备级均衡损失：在宏观层面，确保每个设备计算量均衡，解决计算瓶颈。
- 通信均衡损失：在网络层面，确保每个设备收发数据量均衡，解决通信瓶颈。

1.2.4.2 Auxiliary-Loss-Free：

Auxiliary Loss 的一个问题就是，加多了会损害模型性能，因为这种强制均匀分配可能与数据本身的自然分布相悖，模型被迫把一些本不该由某专家处理的 token 硬塞给它；调小辅助损失又怕负载失衡。这是一个艰难的权衡。

所以 Aux-Loss-free 的方法引入了一个为每个专家设置的偏置项 b\_i，并对其进行动态调整。

![图片](https://mmbiz.qpic.cn/sz_mmbiz_png/WFTLymtmNia0IvlO1vaNkqsBoHia0WJmlGGZWVJHuFMYAv0OBUcHZl6qSAkBhUNfmqn4Q3QYRqnQBqx5ndSYHUDt8QFZgfTta8xBtg4XddNOo/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=17)

这里有一个关键的细节是，bias 只影响路由选择（即 token 分给谁），而不影响分过去之后的权重。

这就是说，一旦决定了被选中的专家，真正与该专家 FFN 输出相乘的权重仍然使用原始的权重 s\_i，t。

这意味着 bias 项可以为了负载均衡而影响该选择哪些专家，但它不会直接影响模型实际“信任”这个专家的程度，从而保护了模型的上限性能不被损害。

尽管 Aux-Loss-Free 是主要的均衡手段，DeepSeek-V3 仍然引入了一个极小权重的序列级辅助损失（Complementary Sequence-Wise Auxiliary Loss），目的是防止单个序列内部出现极端不均衡。

1.2.4.3 Token Drop

辅助损失能鼓励负载均衡，但无法保证严格的负载均衡。

为了固定计算/内存开销并保证硬件效率，许多早期 MoE 模型会为每个专家设定一个最大处理 token 数（即容量，capacity）。

容量由 capacity\_factor（容量因子）决定：

![图片](https://mmbiz.qpic.cn/sz_mmbiz_png/bnRkHqDib9pCqLUnIHTGJ8d5L0OHeVPYRGhzHZvwDYLp8nbxgG35vAZjYuykcjibnPEfkdKS7YRoxfibWNJsibgUJQOuGltz9nfA3SucQ0pPnyY/640?wx_fmt=png&from=appmsg#imgIndex=18)

例如 capacity\_factor = 1.25 表示每个专家最多处理 1.25 倍于平均负载的 token。

实际训练中可能会出现某个专家/设备接收的 token 超出其容量capacity的情况，所以就要用到 Token Drop。

多余的 token 被直接丢弃，不经过任何 FFN 专家计算，仅依靠残差连接向前传递。

比如 DeepSeek-V2 就引入了 device 级的 Token-Drop 策略：如果某个设备接收到的 token 超过了预算，丢弃那些亲和度得分最低的 token，直到满足预算为止。

亲和度得分低意味着该 token 本就不太“适合”这个专家，丢弃它们对最终输出的影响相对最小。这些 token 只走残差连接，而不会参与 MoE 的计算。

但这会带来若干问题：

- 信息丢失：被丢弃的 token 在当前层没有获得任何专家知识，可能损害模型性能
- 训练‑推理不一致：若训练时有 token drop 而推理时没有 token drop（或使用不同的 capacity\_factor），会导致训推不一致。

所以现在的主流模型大多转向无 token drop 的设计，依靠精心设计的辅助损失来实现自然负载均衡。

1.2.5 Sinkhorn，Z-loss

现在我们再回头看这张图：

![图片](https://mmbiz.qpic.cn/mmbiz_png/WFTLymtmNia2ehlSOeWtdcLm4P1lkJMlZGnFN00pvTn1nW39pkw3yBSI7GgtUuoicdpYXG8u1pXnqCt55DHQLZQzpkZia4buuDticH2ZrFndBPo/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=19)

你可以发现，我们之前的讲解已经覆盖了这个图中的大部分内容，但是在"Load Balancing Mechanisms"里面还有两个东西我们没有讲：Z-Loss 和 Sinkhorn。

这两个“技巧”在主流的 MoE 模型中使用已经不多，不过为了内容的完整性，在这里我们也略微介绍一下。

（1）Sinkhorn

Sinkhorn 本质上是一个数学算法，它解决的问题是“ 给定任意一个非负矩阵，如何把它调整成一个双随机矩阵”。

名词解释：双随机矩阵（Doubly Stochastic Matrix）

一个 n×n 的非负实数矩阵 A=(a\_ij) 被称为双随机矩阵，如果它满足每一行的元素之和与每一列的元素之和都等于 1。

Sinkhorn 算法很简单，就是交替地归一化行和列，矩阵会来回震荡、最终稳定到一个双随机矩阵的状态。数学上可以证明它一定会收敛。

不过双随机矩阵、以及 Sinkhorn 算法和 MoE 有什么关系呢？

如果我们把每个 token 路由给哪个专家用一个矩阵来表示：路由矩阵 = \[num\_tokens, num\_experts\]。

那么这个矩阵的：

- 每一行表示一个 token 对所有专家的亲和力；
- 每一列表示一个专家收到的所有 token

Softmax 只能保证行和为 1（每个 token 分配到所有专家的概率和为 1），但是这样可不会保证列和为 1，所以才会出现某些专家负载不均衡情况。

Sinkhorn 算法使得这个路由矩阵变成双随机矩阵，让列和也均衡，这样天然保证了专家负载均衡，不需要 aux loss。

![图片](https://mmbiz.qpic.cn/sz_mmbiz_png/WFTLymtmNia01Oxd79fzSBgDKyVIeoN1SvjLdEruJpDl20kTqpzCZCe6ic4g1A7ZwgEFHmKkjqad35tibXI1jGU2nYqlZiaE1YtgAp2gxDSHpUk/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=20)

左：Sinkhorn 之前，专家 0 负载很重；右：Sinkhorn 之后，专家负载均衡

和 Aux-Loss-Free 一样，Sinkhorn 只用来决定选哪个专家，最终的权重还是用原始 logits。

Sinkhorn 算法有缺点也很明显，首先它额外的迭代计算开销很大；而且它只在训练时使用，推理时无效，这会引入训练和推理的路由行为不一致。实际测试下来，在性能和均衡性上都不如 Aux-Loss-Free 方案。

（2）Z-Loss

Z-loss（全称 router z-loss）是 MoE（Mixture-of-Experts 训练中用于数值稳定的正则项，用来防止路由器 logits 变得过大。

这是因为路由器的 logits 有时会不受控制地增大。因为 softmax/sigmoid 的输出只依赖 logits 的相对差异，所以模型可以在不改变路由结果的前提下，把所有 logits 整体放大（比如从 \[2, 1, 0\] 放大到 \[20, 10, 0\]）。

而当 logits 变得很大的时候，e^logits 的计算会出现溢出问题。比如 logit 到 ~88 就会在 bf16 精度上出现 exp 溢出。

Z-loss 是一个加在主 Loss 上的辅助 loss，用来惩罚 logits 的绝对大小，公式如下：

![图片](https://mmbiz.qpic.cn/sz_mmbiz_png/WFTLymtmNia0YRvWkTkPKxUAUl0j9dkQrbU22tyZRYGafXXfF2n9TWFpeq4btx5oGdib4C3iaakQKhEKCoJH6EWw6nQib3lUvnI9K1zeDmXYSE8/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=21)

B 代表 Token 数，N 代表专家数， x ∈ R^{B×N} 是进入 router 的 logits

在上面的公式中，我们看到了一个平方项，这说明惩罚是随 logit 增大而超线性增长的。

logit 小的时候，惩罚微不足道，几乎不影响训练；而 logit 大的时候，惩罚急剧上升。

不过，现在明确表示自己使用 Z-loss 训练的团队并不多。我也曾经试过使用 Z-loss，并没有观察到明显的效果增益。

可能是因为 Z-loss 主要是防止训练不稳定，而使用好的初始化、warmup、梯度裁剪等策略，不用 Z-loss 也能稳定训练。

1.3 专家模块的设计

1.3.1 专家的粒度

早期的 MoE 专家数量通常很少（比如 8 个或 16 个）。每个专家被分配到的 token 会覆盖多种不同类型的知识。

这会导致知识混杂的问题：一个专家不得不把五花八门的知识都塞进自己的参数里，结果就是什么都会一点，但什么都学不精。模型很难在推理时高效调用这些混杂在一起的知识。

细粒度专家，就是把每个标准 FFN 专家的中间隐藏维度缩小为原来的 1/m，也就是把一个“大专家”切成 m 个“小专家”。

总专家数从 N 变为 mN，但总参数量不变。相应地，把每个 token 激活的专家数量从 K 增加到 mK，所以总计算量也不变。

（当然这只是理论上的啊，实际上因为要考虑到路由和 all-to-all 通信，细粒度专家会导致训练和推理速度大幅下降，需要在 infra 层面做很多的优化才可以）

拆分后，不同的小专家可以分别学习被拆解得更细的知识，每个专家更专精。从组合爆炸的角度看，可选的专家组合数量暴增。

比如，从 16 个专家选 2 个只有 120 种组合；但拆成 m=4 后，变成从 64 个小专家选 8 个，组合数飙升至 44 亿种。这让每个 token 可以更灵活、更精准地组合所需的知识模块。

![图片](https://mmbiz.qpic.cn/mmbiz_png/WFTLymtmNia17RtBsXZwvYZSrjtSkr8dryK1QnVmRY3GzMpIRDpySkDhSlibaEZkmHUdSy0Nbx50t36icsn6OwbSEd32A9BXasyJyYDOrun08o/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=22)

从粗粒度专家，到细粒度专家

1.3.2 共享专家

不同专家处理的 token 可能都需要某些共性知识。如果没有共享专家，那么多个专家会在各自的参数中分别学会同一套知识，造成参数空间的浪费，也就是冗余。

所以，可以隔离开一部分专家，指定为共享专家，每个 token 都会无条件地、确定性地经过这些共享专家。那么为了保持总计算量不变，需要相应地减少可路由专家中激活的数量。

这样的好处是，共享专家可以专门负责学习共性知识，其他路由专家就不再需要各自学习一遍这些冗余知识，它们就可以更加聚焦于自己那部分独特的、非共享的知识。这提升了参数效率，让每个路由专家的特化程度更高。

![图片](https://mmbiz.qpic.cn/mmbiz_png/WFTLymtmNia1jibN5I7fzx7Dz4DpAT24vUib08LArMTnV76Qiadh2tNrrJYLua4IVQ3icEdP4yyCxpESibjSZte304ATZAOl0jqWicYjjuLMibAUiaM4/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=23)

shared Expert 一直被激活；Routed Expert 激活 top K

结合细粒度专家和共享专家，一个完整的 MoE 层的输出 = 所有共享专家的输出之和 + 选中的若干细粒度路由专家的加权输出之和 + 残差连接。

共享专家 K\_s 个，始终激活。细粒度路由专家 mN - K\_s 个，通过路由稀疏激活。

细粒度专家+共享专家的计算公式如下：

![图片](https://mmbiz.qpic.cn/mmbiz_png/WFTLymtmNia3AavTTINeXzoibjXvIfGomibNXgvvwdXELMuw7ialeNZ5ssxU20KEeqcJanw8fQ9Nr2Lkg5vUVZibG7OILEfJtIibCZsjcHTmwj4VQ/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=24)

1.3.3 Upcycling

Upcycling 指的是把一个已经训练好的 dense 模型转换成 MoE 模型，复用已有权重继续训练，而不是从零开始。

这样做的动机是，dense checkpoint 的训练算力既然已经属于沉没成本，那么如果能把这部分成本转化为 MoE 的初始化优势，就可以在有限预算下获得更好的起点。

朴素的做法很简单，就是把 dense 模型里的 FFN 权重复制 N 份，作为 N 个专家的初始化权重；路由器则随机初始化。

这样 MoE 模型一开始就继承了 dense 模型的知识，起始 loss 应该是保值的，即和 dense 模型一模一样。

但 Upcycling 方案有一个明显缺陷：复制出来的 N 个专家在初始化时完全一样。

它们缺乏多样性，后续训练中很容易出现梯度同质化，导致专家难以自然分化。

即使路由器随机初始化，专家之间也需要经历漫长的过程，才能真正形成不同功能，因此 Upcycling 的上限是远低于 From Scratch 训练的。

Skywork 的研究分析了不同 MoE 训练 Budget 下的 Upcycling 和 From Scratch 训练效果对比，如下图所示：

![图片](https://mmbiz.qpic.cn/sz_mmbiz_png/WFTLymtmNia3kFiaDDDW6SZNIO7RR8cCUMfG9TpyGAKcKWicQN9CBmiagDib9HYGFluJCK4JNg44et30FrRAbc2NfetjT6cbBWrXmwd1CWh1TYhM/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=25)

在 MoE 有 100B tokens 的训练预算下（左图）：from-scratch 模型虽然起始 loss 远高于 upcycled 版本，但最终追上了 upcycle-100B（训练 100B 的 dense 再 upcycle），并且反超了 upcycle-300B（训练 300B 的 dense 再 upcycle）。

在 MoE 有 100B tokens 的训练预算下（中图）：from-scratch 要优于所有 upcycled 版本。

upcycle 版本的上限比 From scratch 低，主要是因为专家相似度太高，而 from scratch 由于是随机初始化，所以专家相似度一直是 0（右图）。

论文还给出了一个 Rules of Thumb：

- 设 C\_dense = 训 dense 模型的成本，C\_MoE = MoE 的训练预算，那么：
- 当C\_MoE ≪ C\_dense时，建议用 upcycling，来最大化利用 dense 的沉没成本
- 当 C\_MoE ≥ 2 × C\_dense时，建议用 from-scratch，因为 from-scratch的上限更高。

如果你本来就没有一个 dense checkpoint（C\_dense = 0），那么总是 from-scratch 训 MoE。不要为了 upcycling 而专门去训一个 dense 模型。

02

**Megatron-LM 中的 MoE 实现**

2.1 MoE 的工程挑战

随着 MoE 模型向数百个专家、且每个专家更细粒度的方向发展，MoE 的稀疏性给工程优化造成了很大的困难。

MoE 的稀疏性主要表现为两种不匹配：

参数-计算不匹配（Parameter-Compute Mismatch）。指的是模型的总参数量远大于实际激活的计算量，比如 DeepSeek-V3 有 685B 总参数，但每个 token 只激活 37B。

所以内存里要装下 685B 参数的所有优化器状态和梯度，但每个 token 实际只计算 37B 参数的量。计算太少了，没法隐藏住通信，导致 GPU 的算力利用率很低。

稠密-稀疏不匹配（Dense-Sparse Mismatch）。指的是 Attention 层是“稠密”的，而 MoE 层是“稀疏”的，所以这两个模块的最优并行配置是冲突的，不能用同一种配置。

在 MoE 的工程优化中，通常会遇到三个相互耦合的挑战（“三堵墙”），限制了训练效率的提升：

（1）内存墙（The Memory Wall）

训练时，优化器状态（比如 Adam 的一阶矩、二阶矩）通常比模型参数本身还大好几倍。

因为“参数-计算不匹配”，所以全部 E 个专家的参数+梯度+优化器状态都必须常驻内存，哪怕每个 token 只用其中 K 个专家。

有一些解决内存不足的方式，但是它们都会引入新的问题：

- 把参数分布到更多 GPU → 通信量上升，通信墙更严重
- 重算激活值而不是存储 → 计算量上升，计算墙更严重
- 卸载到 CPU 内存 → PCIe 带宽远低于 GPU 间通信，通信墙更严重

（2）通信墙（The Communication Wall）

当专家并行的规模很大时，all-to-all 的代价不可忽视。（后文会介绍专家并行与 All-to-All 通信）

（3）计算效率墙（The Compute Efficiency Wall）

小 GEMM 的问题：细粒度专家意味着每个专家矩阵很小，GPU 的 Tensor Core 需要大矩阵才能满负荷运转。几十个小矩阵乘法的效率远低于一个大矩阵乘法。

路由（route）和分发（dispatch）开销：token 需要被重新排列、分组、发送到对应专家。这些操作需要大量内存搬运。

更雪上加霜的是，如果有负载不均衡，空闲专家的 GPU 还会闲着等，浪费算力。

Megatron-Core（NVIDIA 的 MoE 训练框架）把三堵墙当成一个整体来优化，而不是分开处理。

下面是一些典型的优化方式，我们在这一节会对其中大部分做简要介绍：

多维并行：把 EP、TP、PP、DP、CP 灵活组合。比如注意力层用 TP/CP、MoE 层用 EP。

通信优化：用专门的 token 分发器（DeepEP、HybridEP）来加速 all-to-all；同时把通信和计算重叠，当 token 还在传输时，GPU 先算其他已经到达的 token。

计算优化：用 Grouped GEMM 把多个小矩阵乘法打包成一个大 kernel，减少发射次数；用 kernel 融合减少中间结果的搬运；用 CUDA Graphs 消除发射开销（在有容量控制、静态形状的模式下使用）。

内存优化：选择性重计算，只重计算那些重算成本低于存储成本的部分；把优化器状态用更低精度存储；必要时卸载到主机内存但和计算重叠。

2.2 MoE 层的计算：Route，Dispatch，Compute，Combine

![图片](https://mmbiz.qpic.cn/mmbiz_png/WFTLymtmNia2TE0QFgXRWG4f0dR5xcR51Ztz90uSicnOsxtQIYCcd5TwqeYQFccrT02pnUibg0MHNCKXicOWTQ1KQKs4iaAknFObWOqxsBs6f51I/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=26)

（1）Route（路由）

在上文中，我们已经详细介绍了 Router 的设计。首先，一个线性投影将每个 token 的隐藏状态映射为 E 个 logits（每个专家一个），然后评分函数将 logits 转为概率，再 Top-K 选择找出每个 token 得分最高的专家。

路由器会输出两个张量：probs（路由权重）和 routing\_map（布尔掩码，表示 token-专家的分配关系）。

为了在专家数量较多时保证数值稳定性，路由器可通过 --moe-router-dtype fp32 在 FP32 精度下运算。

（2）Token Dispatcher（分发）

“他戴着破旧白手套的右手指向塑料椅子这边，意思是让我去那里等候，我的眼睛看着沙发那边。他提醒我沙发那边是贵宾区域，我的身份属于塑料椅子这边的普通区域。我手里拿着A64号走向塑料椅子......”——《第七天》，余华

接下来，每个 token 需要被送到它被路由到的专家所在的 GPU 上。

Token Dispatcher 的流程分为三个阶段：

permute：在本地 GPU 上，根据 routing map 对 token 进行重新排列。比如在某个 GPU 上有 100 个 token，其中 30 个要去专家 5（在 GPU 1 上）、40 个留在本地、30 个要去专家 9（在 GPU 2 上）。

这一步就是把这些 token 按目标 GPU 分组排好，使去往同一专家的 token 在内存中连续。这是因为连续内存访问和传输是大批量通信的基础，能显著减少开销。

token\_dispatch（token 分发）：执行实际的通信。每个 GPU 把属于其他 GPU 的 token 发出去，同时接收其他 GPU 发来的、属于本地专家的 token。这一步就是 All-to-All 通信。

dispatch\_postprocess（分发后处理）：每个目标 GPU 还需要再次重新排列，按专家分组。

假如某 GPU 上有两个专家：专家 1 和专家 2，但是现在 token 分发之后，从不同源 GPU 发来的 token 混在了一起。现在需要把它们整理成“专家 1 的 token 放一起、专家 2 的 token 放一起”的形式，然后送入专家计算。

这是因为 Grouped GEMM 要求每个专家的输入 token 在内存中连续，这样可以把多个专家的小 GEMM 打包成一个大 kernel 高效执行。

在这里顺便介绍一下 All-to-All 通信。

名词解释——All-to-All 通信：

All-to-All 通信是一种全局集合通信操作，每个设备都会向其他每个设备发送一份专属的数据块，同时也会从其他每个设备接收一份专属的数据块。

All-to-All 的通信复杂度是 O(N²) 的（N 为设备数），通信量会随着设备数量的增加而爆炸式增长。它往往是限制 MoE 大规模扩展的主要瓶颈。

我们知道，GPU 之间的连接分两种：

- 节点内（intra-node）：同一个服务器机箱里的 GPU 用 NVLink 连接，带宽极高（几百 GB/s 到 TB/s 级别）。
- 节点间（inter-node）：不同服务器之间走 InfiniBand 或以太网，带宽低。

所以，如果专家都放在同一个节点里，那么 all-to-all 走的是 NVLink，速度还行。

但当专家数到几百个、EP 跨多个节点时，通信就会很慢了。在类似 DeepSeek-V3 的结构中，未经优化的 all-to-all 通信会占 60% 的总训练时间

Megatron 中的 Flex（灵活）模式内部支持两种高性能内核，来加速 All-to-All：

DeepEP：核心思路是将 all-to-all 通信与专家计算重叠。在标准的 all-to-all 流程中，通信和计算是串行的，需要等所有 token 送到再开始计算专家 。

DeepEP 的做法则是把 token 分成多个小块（chunk），然后边传边算。当一批 token 的还在传输时，GPU 已经开始计算另一批已经到达的 token了。DeepSeek-V3 的通信优化思路就是这类。

HybridEP：是一种针对某些特定 NVLink 拓扑（如 NVL72，72 张卡通过 NVLink 全互连）而优化的高带宽通信内核。

在 NVL72 这种拓扑下，大量 GPU 之间有高速直连，可以比传统 all-to-all 更灵活地路由。

（3）Expert Computation（专家计算）

每个 GPU 在接收到的 token 上执行本地专家计算。不过，如果逐个专家计算，会出现大量非常小的矩阵乘法，GPU 利用率低。

所以要使用 Grouped GEMM，把属于同一个专家的所有 token 分组，然后一次性对多个专家的矩阵做批量矩阵乘法。

这样虽然每个专家的矩阵还是小的，但打包后的 GEMM 足够大。这是解决前面提到的“小 GEMM 问题”的关键技术。

Megatron-Core 提供了基于 NVIDIA Transformer Engine 优化的 TEGroupedMLP，支持 FP8/FP4 量化，内部使用高效的 Grouped GEMM kernel，作为专家计算的生产级实现。

（4）Token Combiner（合并）

Combine 阶段就是 Dispatch 阶段的逆向，也分为三个阶段：

token\_combine（token 合并）：专家计算完成后，执行 all-to-all 通信，把结果送回各 token 的原始 GPU。

unpermute（逆排列）：各 GPU 收到返回的 token 结果后，需要把 token 逆排列（unpermute）恢复原始序列顺序，这样才能和之前的 hidden states 对齐，继续后续的残差连接和下一层计算。

weighted combine by routing probs（加权求和）：unpermute 之后，每个 token 有 K 个专家输出，每个输出都带着对应的权重。

weighted combine 就是把这 K 个输出按权重相加，得到该 token 最终的 MoE 层输出。

如果配置了共享专家，共享专家的输出会在 Token Combiner 之后的这个阶段加入。

值得注意的是，共享专家的计算可以与 dispatch-compute-combine 流水线并行执行。

这样共享专家不增加关键路径的延迟，它的计算时间被隐藏在了通信和路由专家计算的时间里。

2.3 专家并行（Expert Parallel）：第五个维度

在稠密模型中，一般有下面这四种并行方式：

- 张量并行（Tensor Parallel，TP）：把一个大的权重矩阵沿隐藏维度切分到多张 GPU。
- 流水线并行（Pipeline Parallel，PP）：按层切分模型，不同 GPU 负责模型的不同层。
- 数据并行（Data Parallel，DP）：每张卡有一份完整模型副本，处理不同 micro-batch，梯度在组内做 all-reduce。
- 上下文并行（Context Parallel，CP）：把长序列沿序列维度切分到多张卡。每张卡处理序列的一段，只在注意力计算时需要跨卡通信。

稠密模型的每一层结构相同，所以整个模型可以使用同一套并行方式划分。比如所有层都用 TP=4、DP=8，那么所有参数都遵循相同的切分和通信规则。

但是，MoE 有着“稠密-稀疏不匹配”的特点，Attention 层是稠密的，MoE 层却是稀疏的。如果强制所有层用同一套并行配置，就会出现很多矛盾。

首先，TP 是不适合用于专家的，因为专家的隐藏维度通常很小，尤其是现在的细粒度专家。对专家应用高 TP 等于把本来就不大的矩阵切得更碎，GEMM 效率恶化。

另外，只依赖 PP 切分 MoE 模型也不合适。因为 MoE 通常参数量巨大，如果只用 PP 切分会产生大量流水线气泡。

所以，必须允许 Attention 层和 MoE 层使用不同的并行组。这就是并行折叠（Parallel Folding）的核心思想。

注意力层用 TP/CP 切分稠密计算，MoE 层用 EP 分布专家，两者互不干扰。

这就引入了几个 MoE 专用的并行维度：

EP（Expert Parallel）：专家并行，即把不同的专家放到不同的 GPU 上。比如当专家个数为 8，EP=4 的时候，每个 GPU 上会放两个专家，如下图所示：

![图片](https://mmbiz.qpic.cn/mmbiz_png/WFTLymtmNia0lIhubpdib6XDJ67mqKmCj81tzMtrzDFHgOnHVuiaNpJFic7zXUAQJElRhicDP3gFTh6v6G96fPV2cjJkva609jHVtcKuiaGlUcRzM/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=27)

ETP（Expert Tensor Parallel）：专家张量并行。在 EP 的基础上，如果单个专家仍然太大、单卡放不下，再对单个专家内部做 TP 切分。实际中很少用，因为专家通常已经很小。

EDP（Expert Data Parallel）：专家数据并行。指的是同一个专家会在多个 GPU 上重复存在，这些副本处理不同的数据，梯度在 EDP 组内做 all-reduce。

比如 64 张 GPU 时，模型只有 16 个专家，EP=16 就够把每个专家放到不同的卡上了，这时每个专家可以有 4 个副本，即 EDP=4。

值得注意的是，梯度只在同一个专家的副本之间做 all-reduce，而不是在整个 DP 组里做 all-reduce。

这也是为什么 Megatron-Core 要为稠密参数和专家参数分别设置梯度归约组。

这样，我们可以将 Attention 层和 MoE 层的并行策略解耦开。Attention 层在 TP × CP × DP × PP 上组建通信组，针对序列级稠密计算优化；MoE 层在 ETP × EP × EDP × PP上组建通信组，其中 ETP（专家张量并行）和 EDP（专家数据并行）都是 MoE 的专用维度。

唯一的约束就是 PP 必须在两种布局中保持一致，确保梯度能正确流过整个模型。

2.4 CUDA Graph

CPU 向 GPU 发送指令的过程叫做 kernel 发射（kernel launch），它有固定的时间开销。

因为稀疏性和路由，MoE 特别受这个问题困扰。MoE 需要大量小 kernel，这些开销累积起来，GPU 就在 kernel 之间“空转”。

而 CUDA Graphs 就发生在 CPU 调度 GPU 工作的层面。它会先通过录制阶段，把整个 MoE 层的所有 kernel（GEMM、激活、重排、all-to-all 等）的执行顺序录制下来，形成一个“图”。

之后每次执行 MoE 层，只需要一条指令：“回放这张图”。GPU 按照图里的顺序自己连续执行所有 kernel，不再需要 CPU 逐个发射了，这就是回放阶段。

这样，发射开销从每个 kernel 一次变成整张图一次，对于 kernel 数量多的 MoE 层，可以节省大量时间。

不过，CUDA Graphs 有一个关键约束：录制时每个 kernel 的输入输出形状必须固定。

因为图是提前录好的，内存地址和 kernel 配置都写死了。这和 MoE 的 dropless 路由有冲突。因为 Dropless 路由中，每个专家收到的 token 数是不确定的。

假如某专家这个 batch 收到 50 个 token，下个 batch 收到 73 个，那 GEMM 的形状就变了，预录的图就不能用。

所以现在我们就明白了，为什么我们需要之前说的容量控制（capacity factor）。

容量控制给每个专家设定一个 token 上限（比如平均值\*1.25）、超过上限的 token 被丢弃，这样每个专家的输入形状就变成固定的了：最多 capacity 个 token，不足则 padding。

所以说 CUDA Graphs 只能在有容量控制、静态形状的模式下使用。

2.5 选择性重计算

反向传播计算到某一层的梯度时，需要用到那一层前向传播时的输入和中间结果。

这些中间结果就叫激活值（activations）。标准做法是前向时把每一层的激活值都存下来，反向时取用。

但是激活值会占用大量的内存，所以就有了重计算（recompute）这种方式：

不保存任何激活值，反向每次需要某一层的激活值时就重新计算。本质上是一种时间换空间的做法。

而选择性重计算（selective recompute）的核心在于区分哪些层重计算更加划算：

- 如果重算这层的激活值很计算量不大，但存它们很占内存，那么则重计算
- 如果重算这层的激活值很计算量很大，但存它们不占多少内存，那么则存着，不重计算

具体来说，Attention 层的激活值巨大，但计算量相对较小。主要是因为注意力分数矩阵的大小是 SeqLen × SeqLen 的，存储成本极高。

所以 Attention 部分适合重计算。相反，FFN 层不适合重计算，因为它的计算量巨大，但激活值相对较小。

当然了，Megatron-Core 在层内部也做了精细化选择，比如重算注意力分数矩阵、但保留 QKV 投影的结果。

本文参考：Scalable Training of Mixture-of-Experts Models with Megatron Core

作者：魔法学院的Chilia

来源：https://zhuanlan.zhihu.com/p/2065153726868877489