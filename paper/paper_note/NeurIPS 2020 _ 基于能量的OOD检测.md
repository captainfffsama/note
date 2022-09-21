#待处理 

©PaperWeekly 原创 · 作者｜张一帆  

学校｜华南理工大学本科生

研究方向｜CV，Causality

![](https://mmbiz.qpic.cn/mmbiz_png/VBcD02jFhglKWSjgMlbibg1kkicpJt6MsnFW7jKDicYMZRndVHrw5u4Ucfku0I0TKxAO5gD2IIcgicxsRBqqHVl32w/640?wx_fmt=png)

**论文标题：** 

Energy-based Out-of-distribution Detection

**论文链接：** 

https://arxiv.org/abs/2010.03759

**代码链接：** 

https://github.com/wetliu/energy\_ood

![](https://mmbiz.qpic.cn/mmbiz_png/Psho9dm7oDHKVtfYDubjKdZRUjAfBQQicF9W4GratnjUc2u6C9rYob1Hv7ebEE1y0fPnNXHJLqK0r0MF65IVvzg/640?wx_fmt=png)

**动机 & 相关工作**

当机器学习模型看到与其训练数据不同的输入时，就会出现 out-of-distribution （OOD）uncertainty，因此模型很难对他们进行正确预测（也即在与训练数据分布差距较大的数据点上表现极差）。对于将 ML 应用于安全关键的应用（如罕见疾病鉴定）而言，确定输入是否超出了分布范围是一个基本问题。

OOD（Out-of-distribution）检测的传统方法之一是基于 softmax confidence。直觉上来看，对于 in distribution 的数据点，我们有高可信度给他们一个结果（就分类问题而言即将一张猫的图片分类为“猫”的概率很高），**那么可信度低的就是 OOD inputs**。但是因为 DNN 在样本空间的过拟合，经常会对OOD的样本（比如对抗样本）一个很高的可信度。

另一种检测方法是基于生成模型的，这类方法从生成建模的角度推导出似然分数 ，主要利用 Variational Autoencoder 的 reconstruction error 或者其他度量方式来判断一个样本是否属于 ID 或 OOD 样本。主要的假设是，Autoencoder 的隐含空间（latent space）能够学习出 ID 数据的明显特征 (silent vector)，而对于 OOD 样本则不行，因此 OOD 样本会产生较高的  reconstruction error。这类方法的缺点在于生成模型难以优化而且不稳定，因为它需要对样本空间的归一化密度进行估计。

![](https://mmbiz.qpic.cn/mmbiz_png/Psho9dm7oDHKVtfYDubjKdZRUjAfBQQicGHwySqq1pHs7jzBRW57YbSic1bxn8EKpyWsPaNoUBdheeibFauUE8Aag/640?wx_fmt=png)

**贡献**

在本文中，作者使用 energy score 来检测 OOD 输入，ID 的数据 energy score 低，OOD 的数据 energy score 高。作者详尽证明了 energy score 优于基于 softmax 的得分和基于生成模型的方法。相比于基于 softmax 可信度得分的方法，energy score 不太受到 NN 在样本空间过拟合的影响。相比于基于生成模型的方法，energy score 又不需要进行显式的密度估计。

![](https://mmbiz.qpic.cn/mmbiz_png/Psho9dm7oDHKVtfYDubjKdZRUjAfBQQicE509QKTUnsPWrev9RQslwXA0xCBxpgLAntdcjOsqcm1s9yPFj1QGpQ/640?wx_fmt=png)

**背景：基于能量的模型**

基于能量的模型（EBM）的本质是构建一个函数 ，它将输入空间中的每个点 映射到一个称为能量的单个 non-probabilistic scalar。通过 Gibbs 分布我们可以将能量转化为概率密度：

![](https://mmbiz.qpic.cn/mmbiz_png/Psho9dm7oDGmzVW3iciaEaeyTu0FOKdNrf36IInj5Bgy8ibuJkLkicy6DAOlribLQfiaTtmZermQyVJq7bM7wxegqnHw/640?wx_fmt=png)

分母被称为配分函数， 是温度参数。此时我们可以得到任意点的自由能 为：

![](https://mmbiz.qpic.cn/mmbiz_png/Psho9dm7oDGmzVW3iciaEaeyTu0FOKdNrfvfUAswfRO2mhr9WPTOwGcHO1F8dFmoVxCqGYBmtwVTg59wJAHNhbqw/640?wx_fmt=png)

我们可以轻易的联系分类模型与能量模型，考虑一个 类的 NN 分类器 将输入映射到 个对数值，通过 softmax 归一化得到属于某一类别的概率，分类分布如下：

![](https://mmbiz.qpic.cn/mmbiz_png/Psho9dm7oDGmzVW3iciaEaeyTu0FOKdNrfMU3t0kKpD8ySlYHuGHa20waY1mB1mONJxEZ19Fia3t61k7GSQevZwrw/640?wx_fmt=png)

这里的 即 的第 个值，而此时我们可以定义能量为 即负对数。同时我们可以得到关于 的自由能：

![](https://mmbiz.qpic.cn/mmbiz_png/Psho9dm7oDGmzVW3iciaEaeyTu0FOKdNrfRPOYEgHtiaS8q5Pibngf8kMluH82zlEWic4BXvbdbYYz0dCoBSdRYavLw/640?wx_fmt=png)

这里需要强调一下，这个能量已经与数据本身的标签无关了，可以看作是输出向量 的一种范数。

![](https://mmbiz.qpic.cn/mmbiz_png/Psho9dm7oDHKVtfYDubjKdZRUjAfBQQicWok9zL9F1SkDEB5Yxl0m5G6NFHYp4FTgfHKU0MzibmiaYONMcNicwpKQg/640?wx_fmt=png)

**基于能量的OOD检测**

我们知道 OOD detection 实际上就是一个二分类问题，判别模型的密度函数可以写作：

![](https://mmbiz.qpic.cn/mmbiz_png/Psho9dm7oDGmzVW3iciaEaeyTu0FOKdNrfLNTNibyOaRicMe3utS0UWLIGXxEn5Mp4eGg3MyunGu9MUL2vqp8pZI0A/640?wx_fmt=png)

其中配分函数 是未知的归一化常数，是 intractable 的。幸运的是：

![](https://mmbiz.qpic.cn/mmbiz_png/Psho9dm7oDGmzVW3iciaEaeyTu0FOKdNrfOt2Mulic6bia4RUSkYXVoZJXqn0cfia7n5XxibibPYgyFCpFibbBpPKtHdtQ/640?wx_fmt=png)

因为 是样本独立的，不影响总体能量得分分布，所以说 和数据点的负对数似然是线性对齐的，低能量意味着高似然，即更有可能是 ID 数据，反之更有可能是 OOD 数据。这涉及到一个阈值 ，比较 empirical，这里不多说。

此时我们可能会想到，这比 softmax 函数好在哪里呢？不妨写出 softmax 分类的形式：

![](https://mmbiz.qpic.cn/mmbiz_png/Psho9dm7oDGmzVW3iciaEaeyTu0FOKdNrfTEqrMbmoBHvvbtBbMico8dNYRsgd7p9t2FS39SNEeR0tSY9VdcQNX8w/640?wx_fmt=png)

当 的时候，这其实就是：

![](https://mmbiz.qpic.cn/mmbiz_png/Psho9dm7oDGmzVW3iciaEaeyTu0FOKdNrfJFmzp9VuEGJqLedqAn2V5glN1HzpGSoe2LlF9gxIs8DdzCVbrTSIxw/640?wx_fmt=png)

如果再进行一步化简我们可以得到：

![](https://mmbiz.qpic.cn/mmbiz_png/Psho9dm7oDGmzVW3iciaEaeyTu0FOKdNrfu9qjjbZ5yFf0HbGckahI06clXopWZ4DzDOGFtREZljsdygRstBe8ibw/640?wx_fmt=png)

后两项 并不是一个常数，相反对于一个 ID 的数据，其负对数似然期望是更小的，但是 这个分类置信度却是越大越好，二者冲突。这一定程度上解释了基于 softmax confidence 方法的问题。

那么能量模型如何进行训练呢——通过分配较低的能量给 ID 数据，和更高的能量给 OOD 数据，明确地创造一个能量差距。总体的损失函数如下：

![](https://mmbiz.qpic.cn/mmbiz_png/Psho9dm7oDGmzVW3iciaEaeyTu0FOKdNrfnQgJekXfBY8LJ9pWpgnHq7pbiaIm47JFfO8X6MnE6IP3cFnUEwpxicZA/640?wx_fmt=png)

其中 是分类模型的 softmax 输出，即标准的交叉熵分布加上一个能量约束项。

即用了两个平方的 hinge loss 来分别惩罚能量高于 和能量低于 的 ID/OOD 数据。这里也即该方法的另一个好处，可以利用没有标签的 OOD 数据帮助训练。一旦模型训练完成，我们就可以按照按照能量进行 OOD 检测。

![](https://mmbiz.qpic.cn/mmbiz_png/Psho9dm7oDHKVtfYDubjKdZRUjAfBQQic2ibc4lzvUPBOBcoLTliatnqRJWOiazYYxu54ycn4u6MXOnHdb61OAh5OQ/640?wx_fmt=png)

**实验结果**

实验中有一点需要注意，作者采用了两个 setting：

*   No fine-tune: 使用 backbone 的输出，只是将 softmax confidence 换成能量得分。注意样本的能量我们定义为 ，其中 即 backbone 的第 维输出。
    
*   Fine-tune：使用上述的损失函数对 backbone 进行 fine-tune，然后使用 energy score 进行 OOD 检测。
    

实验统一使用 WideResNet 作为预训练分类模型，在六种 OOD 数据集上的表现如下，可以看到在不进行 fine-tune 的情况下基本碾压了基于 softmax confidence 的方法。有 fine-tune 的情况下，也比目前的 sota-OE 好很多。不过这里需要指出表格中标注的 应该指的是训练集，因为作者也提到了下表是“We use WideResNet to train on the in-distribution dataset CIFAR-10.”。

作者进一步比较了各种方法之间的差距，可以看到即使不使用 fine-tune，只是将 softmax confidence 换成 energy score 效果就已经很不错了，进行 fine-tune 之后更是惊为天人。

![](https://mmbiz.qpic.cn/mmbiz_png/6TIzll6EQmhibW7Dkqk4q92mcMuouL0eDnC04kYticibyk09jJLjDXxspwCibOoxE1hiamlOqo7wsJpEWyBbnib9lT6Q/640?wx_fmt=png)

同样还有与生成模型的比较，metric 是 the area under the receiver operating characteristic curve（AUROC）——越高越好。

![](https://mmbiz.qpic.cn/mmbiz_png/U9CnFePkcNVHhEojGb4eHjDvdZTo96hntPEH53TJoKVx1JI2xTicGYH7LViasxOIibFIcl4fpiazXgxQ5uFz1fqA2A/640?wx_fmt=png)

  
