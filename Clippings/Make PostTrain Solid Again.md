---
title: "Make PostTrain Solid Again"
source: "https://zhuanlan.zhihu.com/p/1995265459285694156?share_code=h4Dsgfii6oJu&utm_psn=1996952003692216623"
author:
  - "[[ybq​中国科学院大学 信号与信息处理硕士]]"
published:
created: 2026-01-20
description: "LLM 论文千千万，有用的工作却没几篇。这篇文章，我想简单讨论下到底该如何把后训练工作做的 solid。文章并没什么技术细节，大家随便看看。 敲定正确的 Baseline有太多论文工作不置信的根因就是没有选对 baseline…"
tags:
  - "LLM"
---
目录

[LLM](https://zhida.zhihu.com/search?content_id=269084433&content_type=Article&match_order=1&q=LLM&zhida_source=entity) 论文千千万，有用的工作却没几篇。这篇文章，我想简单讨论下到底该如何把后训练工作做的 solid。文章并没什么技术细节，大家随便看看。

---

### 敲定正确的 Baseline

有太多论文工作不置信的根因就是没有选对 baseline。

以 [length penalty](https://zhida.zhihu.com/search?content_id=269084433&content_type=Article&match_order=1&q=length+penalty&zhida_source=entity) 为例，选择一个有着 30% 截断率的 [sft 模型](https://zhida.zhihu.com/search?content_id=269084433&content_type=Article&match_order=1&q=sft+%E6%A8%A1%E5%9E%8B&zhida_source=entity) ，不加任何控长策略的 [rl](https://zhida.zhihu.com/search?content_id=269084433&content_type=Article&match_order=1&q=rl&zhida_source=entity) 作为 baseline，然后顶着 30% 的截断率去做带有控长策略的实验。在这种 setting 下，实验组里 30% 的数据既有一个 max\_response\_length 的推理 buffer 策略，又有我们设计的 length penalty 策略。模型的长度变短了，我们下结论说是 length penalty 策略有效了，这 solid 吗？

因为模型的输出长度变短了，所以 32K 的指标变得更高，但这个模型的推理长度从 32K 扩展到 64K 后却毫无提升，而原本没有控长策略的 baseline 实验，虽然在 32K 的时候，由于截断很高导致指标很低，但当推理长度从 32K 扩展到 64K 后指标却有大幅度提升，64K 指标甚至明显高于实验组，这种牺牲上限换取的指标提升，真能说明控长策略有效吗？

还有一类典型的不太 solid 的工作就是对 token clip 进行精雕细琢，这些工作往往都没去分析在 [on-policy](https://zhida.zhihu.com/search?content_id=269084433&content_type=Article&match_order=1&q=on-policy&zhida_source=entity) 的情况下的训练会是什么样的情况。如果用自己提出的方法去和 \[1 - 0.2, 1 + 0.2\] 比较，那和与 random 选阈值策略进行对比又有什么区别呢？

token clip 的工作应该是如何让 [off-policy](https://zhida.zhihu.com/search?content_id=269084433&content_type=Article&match_order=1&q=off-policy&zhida_source=entity) 策略无限接近于 on-policy 的效果，并且去证明为什么这个策略好于固定的阈值。此外，如果 on-policy 都会出现的崩溃现象，被 off-policy + clip\_token 给解决了，那情况更加糟糕，只能说明这个策略是为这个模型、这份数据、这个 topic 量身制定的，毫无泛化意义。

综上，我们需要有一个好的 baseline 去支撑后续的实验结论： **答案正确且易验证结果的数据，接近百分之百准确的判分模型，完全 on-policy，尽最大可能保证训推一致，给较大的 max\_response\_length 保证没有长度打压，不引入任何帮助稳定训练的正则项** 。先看一看，这个理论上不会出错的实验，它能跑多远，它跑起来是什么样子的。如果它跑不起来，或者跑不了太远，就说明框架是存在 Bug 的，或者我们的冷启动数据、RL 数据存在明显的缺陷。那就需要先把基础工作搞定，再深入研究策略问题。

有了一个稳定能跑起来的 baseline 后，就可以一点点的去添加点缀了。mini-batch，partial-rollout，async\_infer，这些会导致 off-policy 的策略挨个去尝试，看看一个优秀的 baseline 是怎么随着 **staleness** 逐渐变大而变得崩溃，再去想办法修复好因为 staleness 增大引起的训练崩溃。概括下来就是三步走：跑出完美的实验，一步步破坏它，再一步步修好它 —— 这个过程中得到的认知成长与实验结论，可称 solid。

### 少用 sense 挑战 math

大多数的算法程序员都是一个半道出家的 RLer，第一次接触的 RL 算法就是 [PPO](https://zhida.zhihu.com/search?content_id=269084433&content_type=Article&match_order=1&q=PPO&zhida_source=entity) 或 [GRPO](https://zhida.zhihu.com/search?content_id=269084433&content_type=Article&match_order=1&q=GRPO&zhida_source=entity) ，并没有认真学习过 Policy Gradient 算法的发展史。

1. 难题就应该得到更大的 loss / reward；
2. However、But 这种“思考” token 应该得到更多的关注，重点训或只训它们；
3. pass@K 算 reward 比 pass@1 reward 更合理；
4. 在某个指标变化幅度过大（如 entropy，kl）的时候，调整某个参数来强行将该指标拉回正轨；
5. ……

我相信这些方法一定能最快解决当前实验的痛点，但它们完全不具有可迁移性。换个数据，换个底座，所有的结论全崩塌了，就算不崩塌也需要重新摸索阈值。

换言之，math 驱动的实验是 solid 的，sense 驱动的实验则是救火的：

- sense 驱动：观察到了一个现象，设计了一个比较 make sense 的改动点，做实验。效果有提升则总结成新算法，效果无提升就算了，或者思考一下不提升的原因，重新设计改动；
- math 驱动：观察到了一个现象，设计了一个比较 make sense 的改动点，推导公式，通过公式去预估自己的改动会影响哪些训练指标，做实验观察是否符合预期。符合预期，则根据公式去修改一些变量继续做实验，验证公式的鲁棒性；不符合预期，则去重新建模理论公式，分析问题出在哪里。

指标压力不那么大的时候，做那些不可迁移的 make sense 的工作，远不如花点时间去深究下 math 原理。LLM 产生一个 sentence 的过程是一个自回归语言建模，entropy 的计算公式是 $−P(x)log⁡[P(x)]-P(x)\log[P(x)]-P(x)\log[P(x)]$ ，kl 的计算公式是 $P(x)log⁡P(x)Q(x)P(x)\log\frac{P(x)}{Q(x)}$ ，policy gradient 算法的公式在那里放着，grad 的计算人人都会，adamw 的公式里也清楚写着当前 sentence 的梯度是如何在影响模型参数更新的。

把这一堆的公式串联起来，在适当的地方进行数学建模，实在搞不明白原理的地方就引入一些传说中的“核函数”，我们完全可以给出一个公式去证明“某个改动是如何影响某个训练指标的”。就像 qwen 在 [MiniRL](https://zhida.zhihu.com/search?content_id=269084433&content_type=Article&match_order=1&q=MiniRL&zhida_source=entity) 论文里给出“用 sentence level reward 逼近 token level reward 的关键条件就是训推一致”的建模，证明过程并不复杂，但却为整篇论文添彩不少。

> Gemini-3 与 GPT-5 已经足够强大， 只要 prompt 写的明确，它们的数学建模能力与公式推导能力，足够帮助大家完成这种简单的理论推导。我们只需要再推导一遍看看是否正确，把自己看不懂或觉着有问题的地方指出来，它们是可以自我修正的。

乘着 deepseek 的东风，GRPO 几乎成为了大家默认的 RL 算法，但随着时间演变，deepseek 在最新的技术报告里也选择将 GRPO 算法退化成了 [RLOO](https://zhida.zhihu.com/search?content_id=269084433&content_type=Article&match_order=1&q=RLOO&zhida_source=entity) 算法（略有区别），在估计 reward 的时候不再除以“标准差”。这种演变几乎是必然的，因为 RLOO 的论文里明确指出过， **RLOO 的估计方式是“an unbiased estimate of the expected return”** ，而 GRPO 更像是一种比较 make sense 的算法设计。

另外一个 math > sense 的例子便是 [KL loss](https://zhida.zhihu.com/search?content_id=269084433&content_type=Article&match_order=1&q=KL+loss&zhida_source=entity) 了，关于该不该在 RL 中引入 KL loss 是一个讨论比较多的话题。归其根源，是 PPO 论文中压根就没有明说这个 KL loss 有必须存在的意义：PPO 的前身 TRPO 论文里满篇都是数学公式，去证明了TRPO 算法中的 KL 是优化过程中的 trust region；与之相对，PPO 算法中的trust region 来自于 ratio clip，

### 大小模型的结论谨慎迁移

实践中，一个比较让人绝望的现象在于：dense 模型的结论无法迁移到 moe 模型上，小模型的结论无法迁移到大模型上。目前的论文工作大多都围绕着 [qwen-4B](https://zhida.zhihu.com/search?content_id=269084433&content_type=Article&match_order=1&q=qwen-4B&zhida_source=entity) 、 [qwen-7B](https://zhida.zhihu.com/search?content_id=269084433&content_type=Article&match_order=1&q=qwen-7B&zhida_source=entity) 开展，一旦放到 [qwen-A3B](https://zhida.zhihu.com/search?content_id=269084433&content_type=Article&match_order=1&q=qwen-A3B&zhida_source=entity) 上便无法复现，qwen-A3B 的结论同样很难迁移到 qwen-A22B 上。

这个现象很诡异，但似乎也合理，用同样的教学理念教一个小学生和教一个大学生的确会得到不同的反馈。类似的，蒸馏强有力模型的思维链往往都能有不错的指标，但 gemini-3 似乎是个例外，对于这种 T 级别（据传）的学霸模型来说，它极高的 token 效率似乎不太合适 B 级别的模型来学习。

qwen 的 MiniRL 论文里曾说过：不同的冷启动数据去做强化，最终指标都会收敛到一个几乎相同的高度。实验是在 Qwen3Next MoE 上做的，很可惜这个观点在大模型上完全不可复现。我也在 qwen-A3B 上跑过很多实验，用过很多 cold start 数据、rl 数据、乱七八糟的算法，只要实验不崩，它的 AIME 永远收敛到 85 分左右。但是在大模型上，无论是切换 cold start 数据还是 rl 数据，收敛后的指标差距都是非常明显的。

也许，小模型的上限就是更好触达一些。又也许， AIME 这种随便训训就能 80+ 的测试集，早已不适合衡量推理模型的能力了。找个 HLE 难度的测试集，A3B 模型的表现大概只有 10 分左右，不同数据 setting 下的实验应该还是能看出一些差距的。

话说回来，虽然实验现象往往大相径庭，但并不是说小模型的实验没有意义，前面提到的“故意训崩一个模型，再拯救一个训崩的模型”是适用于任何尺寸的模型的，这个过程可以培养我们的 debug 能力与对 RL 算法的灵敏嗅觉。所以，小模型就是一个实验场，围绕着小模型开展的实验就是公司在花钱去锻炼我们的算法素养，为的是让我们的脑子变得灵光，从而在训大模型的时候少走一点弯路、节省一些算力。

盲目的迁移或迷信小模型的结论，某种意义上也是一种实验不 solid 的体现。至少，在某个理论工作证明出大小模型后训练能力的迁移遵循哪种 scaling law 之前，是这样的。

### simple yet effective

过去一年在纯语言模型领域，几乎只有两个工作是得到了业界所有同行的认可：上半年的利用 ORM 提升模型推理能力，下半年的利用 TIS / IcePop 保证训推一致性，都是 simple yet effective 的完美代言。

这里，我们重点回顾一下训推不一致：2024 年所有同行就都知道 vllm、model.generate、megatron 前向算子，这之间的结果有较大差异；2025 年从 TIS 提出到 ICEpop 的这段时间内，几乎所有同行者都能想到 IcePop 的方案。大家都曾有机会提出这两个算法，但把握住机会的就是那两篇 Notion 分享，行动力强、实验严谨、理论扎实，两个团队配得上大家的赞扬。话说回来，连 TIS 这种 simple 的 idea 都埋没了一年才被广而告之，围绕着 LLM 的 policy gradient 算法必有宝藏等着大家去挖掘。

从经验上来说，如果某个工作的核心步骤不是两句话能概括出来的，那这个工作似乎离雕花标签也不远了。目前的 LLM，找不到什么 solid 的工作是不 simple 的。

---

文章的大多观点依然是与我的富哥朋友

[@真中合欢](https://www.zhihu.com/people/bf1764dccc55b8f831b89c9103f41564)

讨论交流所得。LLM 领域不仅技术迭代的快，对互联网的人员流动冲击也是真的大。过去几年的饭搭子，今年已经是竞对关系啦，真是时过境迁啊！😂

旧爱不去，新欢不来，文章同样感谢我新的技术搭子同事：零零后小天才

[@Moyu](https://www.zhihu.com/people/54d136871b0478611e790ce8f7ef7fb6)

，我会鞭策他也多多分享大模型算法日常的。

发布于 2026-01-19 23:42・北京[LLM（大型语言模型）](https://www.zhihu.com/topic/26797383)[RLVR](https://www.zhihu.com/topic/1915337254311789283)[大模型训练](https://www.zhihu.com/topic/29083863)