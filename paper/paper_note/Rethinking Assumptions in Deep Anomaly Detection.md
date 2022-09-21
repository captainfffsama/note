#异常检测 
#单分类

# Rethinking Assumptions in Deep Anomaly Detection
- 文章: <http://arxiv.org/abs/2006.00339>
- 代码: <https://github.com/lukasruff/Classification-AD>
- 会议: ICML 2021

## 摘要

## 1. 引言
现有的 AD 方法大致分三类:
- Unsupervised: 仅仅使用正常数据,这是最经典且普遍的做法
- Unsupervised OE: 使用来额外辅助数据来训练,也可以称之为半监督的方法.比如各种离群暴露的方法
- Supervised OE: 直接监督学习来处理 AD 问题.即使用一个标准分类器,将正常的视为正样本,其余所有样本视为父类

## 2. Deep One-Class Classification
Deep One-Class 分类将正常数据表示映射到特征空间中使之尽量聚集,比如 Deep SVDD 的优化目标为:
$$
\underset{\theta}{min} \frac{1}{n} \sum_{i=1}^{n} ||\phi_{\theta}(x_i)-c||^2  \tag{1}
$$

这里 $\phi$ 代表了网络,$\theta$ 代表参数.

而 Deep Semi-supervised Anomaly Detection (Deep SAD),则不仅仅将正常数据聚集到中心 $c$,还将异常样本映射到远离 $c$.这是一个典型的半监督方法.其方法被称为 [Hypersphere classification,HSC](Explainable%20Deep%20One-Class%20Classification.md#^a639b1). 这个方法是提升 Deep SAD 的关键,也被作为本文试验中无监督 OE AD方法的代表.

下面一大段都在介绍这个 HSC,就是链接中的那个公式.

## 实验
作者使用的数据集有:
- MNIST: OE使用的是[EMNIST](https://zhuanlan.zhihu.com/p/55045479)
- CIFAR-10: OE 使用 80 Million Tiny Images
- ImageNet-1K: OE 使用 ImageNet-22K 

在进行 one vs. rest classes 实验时,发现 BCE 取得了最好效果.

然后作者验证OE 数量对实验的影响,发现当OE数量很少时, HSC 比 BCE要好,但是随着 OE增加 BCE 要比HSE 好,但是当OE数量到 $2^{11}$之后, 二者性能相当了.

作者猜测 OE 能起作用的原因是因为 OE 样本包含了丰富的自然图像多尺度信息.为此他采用不同大小的的高斯核对 OE 图片进行平滑,以此抹去小尺度信息,看看对实验影响.实验表明,随着 OE 信息被平滑的越多, HSC 和 BCE 的性能都会出现下滑,但是 HSC 比 BCE 要鲁棒得多.

## 结论
作者总结说与经典 AD 相比,少量的 OE 是可以给 AD 基准提供丰富的信息的,并且这个现象和自然图片丰富的多尺度信息有关.但是这并不意味着监督式 OE 就是 AD 的最优解,或者说无监督的 OE 没用.

