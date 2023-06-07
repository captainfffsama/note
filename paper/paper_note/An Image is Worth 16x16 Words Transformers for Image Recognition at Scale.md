[原博客](https://aitechtogether.com/article/20375.html)

- 论文: <https://readpaper.com/paper/3094502228>
- 代码: <https://github.com/google-research/vision_transformer>
- 会议: ICLR 2021
- 讲解视频: [【ViT 论文逐段精读【论文精读】】 ]( https://www.bilibili.com/video/BV15P4y137jb/?share_source=copy_web&vd_source=ec533c1ebc465970d3af0e7c410a05c5 )


# Abstact

虽然 Transformer 架构已经成为自然语言处理任务的事实标准，但它在计算机视觉中的应用仍然有限。在视觉中，注意力要么与卷积网络结合使用，要么用于替换卷积网络的某些组件（替换 CNN 模型的一部分），同时保持其整体结构。我们表明，这种对 CNN 的依赖是不必要的，直接应用于图像块序列的 pure Transformer 可以很好地执行图像分类任务。当对大量数据进行预训练并将其传输到多个中小型图像识别基准（ImageNet、CIFAR-100、VTAB 等）时，与最先进的卷积网络相比，Vision Transformer（ViT）可以获得优异的效果，同时训练所需的计算资源大大减少（这里指的需求资源小是相对的，它本身需要使用 2500 天 TPU v3 的天数）。

# 1 、Introduction

基于 Self-attention-based 的体系结构，尤其是 Transformers（V aswani 等人，2017），已经成为自然语言处理（NLP）的首选模型。主要的方法是在大型文本语料库上进行预训练，然后在较小的任务特定数据集上进行微调（Devlin 等人，2019 年）。由于 Transformers 的计算效率和可扩展性，它已经能够训练具有超过 100 B 参数的空前规模的模型（Brown 等人，2020 年；Lepikhin 等人，2020 年）。随着模型和数据集的增长，性能仍然没有饱和的迹象。

Transformer 应用在 CV 领域的难点：

​ 1）在 nlp 领域一般使用 Transformer 的序列大小为几百，Bert 的大小为 512，由于 Transformer 的计算特性，模型的时间复杂度为 $O(n^2)$ , 这已经是比较复杂的模型了。使用在 CV 领域，一般训练使用的样本集是 224×224，将其转换成 1 d 的数据传入 Transformer，它的序列大小就变成了 224×224，相较 Bert 的 100 倍，计算量非常大。

​ 2）应用在 CV 的其他任务中计算量会进一步增大，例如在视频分类中的输入图片时 800×800，任务量又进一步增大。

然而，在计算机视觉中，卷积结构仍然占主导地位（LeCun 等人，1989 年；Krizhevsky 等人，2012 年；He 等人，2016 年）。受 NLP 成功的启发，许多作品尝试将 CNN 式架构与自我关注相结合（Wang 等人，2018 年；Carion 等人，2020 年），有些作品完全取代了卷积（轴注意力机制）（Ramachandran 等人，2019 年；Wang 等人，2020 a）。后一种模型虽然在理论上有效，但由于使用了专门的注意力模式，尚未在现代硬件加速器上有效地扩展。因此，在大规模图像识别中，经典的 ResNet-like 架构仍然是最先进的（Mahajan et al.，2018；Xie et al.，2020；Kolesnikov et al.，2020）。

受 NLP 中 Transformer 缩放成功的启发，我们尝试将标准 Transformer 直接应用于图像，并进行尽可能少的修改。为此，我们将一幅图像分割为多个 patches，并将这些 patches 的线性 embeddings 序列作为 Transformer 的输入。图像 patches 的处理方式与 NLP 应用程序中的 tokens（单词）相同。我们以有监督的方式对模型进行图像分类训练。（将图片裁剪成 16×16 的 patch 传入，这样之前 224×224 大小的图片的序列就变成了 14×14=196 大小的序列）

![](https://aitechtogether.com/wp-content/uploads/2022/04/27b8be5096bbf70a5f2f0753116739db.webp)

在中等规模的数据集（如 ImageNet）上进行训练时，如果没有很强的正则化，这些模型产生的精确度会比同等规模的 RESNET 低几个百分点。这种看似令人沮丧的结果可能是意料之中的： Transformer 缺少一些 CNN 固有的 inductive biases（特指一种先验知识或者是提前做好的假设, CNN 中有两个假设，第一个是 Locality 是指相邻的区域会有相邻的特征，第二个是 translation equians（平移不变性），公式上就是 f (g (x)) = g (f (x)) ）。例如翻译的等变性和局部性，因此在数据量不足的情况下不能很好地概括。

然而，如果模型在更大的数据集（14 M-300 M 图像）上进行训练，情况就会发生变化。我们发现，大规模培训胜过归纳偏见。我们的视觉转换器（ViT）在足够规模的预先培训和转移到数据点较少的任务中时，会取得优异的效果。当在公共 ImageNet-21 k 数据集或内部 JFT-300 M 数据集上进行预训练时，ViT 在多个图像识别基准上接近或超过了最新水平。特别是，最佳模型在 ImageNet 上的精度达到 88.55%，在 ImageNet ReaL 上的精度达到 90.72%，在 CIFAR-100 上的精度达到 94.55%，在 VTAB-suite 19 项任务上的精度达到 77.63%。

# 2 、Related Work

Transformers 由 V aswani 等人（2017）提出用于机器翻译，并已成为许多 NLP 任务中最先进的方法。基于 Transformers 的大型模型通常在大型语料库上进行预训练，然后针对手头的任务进行微调：BERT（Devlin 等人，2019）使用 denosing self-supervised pre-training 任务，而 GPT 工作线使用 language modeling 作为其预训练任务 (Radford 等人，2018；2019；Brown 等人，2020). （BERT 是类似完型填空的方式进行预测，GPT 是已经有一个句子预测下一个词是什么，或者叫 next word prediction。两种方式都是自监督的方式）

单纯地将自我关注应用于图像需要每个像素关注其他像素。由于像素数量是二次成本，因此无法按实际输入大小进行缩放。因此，为了在图像处理中应用变压器，过去曾尝试过几种近似方法。Parmar 等人（2018 年）仅在每个查询像素的 in local neighborhoods 应用自我关注，而不是全局应用。这种局部多头点积自我注意块可以完全取代卷积（胡等人，2019 年；拉马钱德兰等人，2019 年；赵等人，2020 年）。在另一种工作中，稀疏变换器（Child 等人，2019 年）采用可伸缩的全局自我关注近似，以便适用于图像。衡量注意力的另一种方法是将其应用于不同大小的区块（Weissenborn 等人，2019 年），在极端情况下，仅沿着各个轴（轴注意力，先 x 后 Y）（Ho 等人，2019 年；Wang 等人，2020 a）。许多这种专门的注意力体系结构在计算机视觉任务上展示了有希望的结果，但需要在硬件加速器上高效地实施复杂的工程。

与我们最相关的是 Cordonnier 等人（2020）的模型，该模型从输入图像中提取大小为 2×2 的 patches，并在顶部应用完全的自我关注。该模型与 ViT 非常相似，但我们的工作进一步证明，大规模的 pre-training 使 vanilla transformers 与最先进的 CNN 竞争（甚至优于）。此外，Cordonnier 等人（2020 年）使用了 2×2 像素的小 patch 大小，这使得该模型仅适用于小分辨率图像，而我们也处理中分辨率图像。

人们还对将卷积神经网络（CNN）与自我注意的形式相结合感兴趣，例如通过增强图像分类的特征图（Bello 等人，2019），或通过使用自我注意进一步处理 CNN 的输出，例如用于目标检测（Hu 等人，2018；Carion 等人，2020），视频处理（Wang 等人，2018；Sun 等人，2019），图像分类（Wu et al.，2020）、无监督目标发现（Locatello et al.，2020）或统一文本视觉任务（Chen et al.，2020 c；Lu et al.，2019；Li et al.，2019）。

另一个最近的相关模型是图像 GPT（iGPT）（Chen 等人，2020 a），该模型在降低图像分辨率和颜色空间后，将变换器应用于图像像素。该模型以无监督的方式作为生成模型进行训练，然后可以对结果表示进行微调或线性探测，以提高分类性能，在 ImageNet 上实现\*\*72%\*\* 的最大精度。

我们的工作增加了在比标准 ImageNet 数据集更大范围内探索图像识别的论文的数量。使用额外的数据源可以实现标准基准的最新结果（Mahajan 等人，2018 年；Touvron 等人，2019 年；Xie 等人，2020 年）。此外，Sun 等人（2017 年）和 Kolesnikov 等人（2020 年）研究了 CNN 的性能如何随数据集大小而变化；Djolonga 等人（2020 年）从 ImageNet-21 k 和 JFT-300 M 等大规模数据集对 CNN 迁移学习进行了实证探索。我们也关注后两个数据集，但培训变压器，而不是之前工作中使用的基于 ResNet 的模型。

# 3 、Method

在模型设计中，我们尽可能遵循原始变压器（V aswani 等人，2017 年）。这种有意设计的简单设置的一个优点是，几乎可以开箱即用地使用可扩展的 NLP Transformer 体系结构及其高效实现。

![](https://aitechtogether.com/wp-content/uploads/2022/04/ad93b158d826b122fc638590750285a9.webp)

注：这里很多内容借鉴了 Bert 模型，例如添加了位置信息 Position Embedding 记录图片的相对位置，使用类似（Extra learnable embedding–cls）【class】的标记这里使用的是位置信息为 0 的【】，这个的作用是经过 transformer encoder 会有多个输出，使用带【\*\*\*\*】的输出去做最后的分类（本文认为这个标记可以从其他的 embedding 中学习到有用的信息）。MLP Head 就是一个简单的分类头，Transformer 结构没有改变，只有 encode，没有 decode

粗略的参数推导：![](https://aitechtogether.com/wp-content/uploads/2022/04/bc926fc86c3c09866d4f2989065bbee6.webp)

![](https://aitechtogether.com/wp-content/uploads/2022/04/400d6a1f902a76e7acee9aa6ca8f7caf.webp)

图片预处理步骤以及其中的维度计算请参考 B 站视频 34 分：

<https://www.bilibili.com/video/BV15P4y137jb?spm\_id\_from=333.1007.top\_right\_bar\_window\_history.content.click>

附录还有几个消融实验，具体讲了对 transformer encoder 的输出使用 cls 标签和不使用直接接一个 Global Average Pooling 的效果是一样的，同时还介绍了他们相对于原始 transformer 位置信息改变，使用 2 D 标签进行记录更适合图像，其中 1 D 使用的维度为 d，换成 2 D 之后使用 x，y 矩阵表示，x 和 y 分别取 d/2, d/2（因为维度是进行 concat，不是 sum 所以不是使用根方 d)，同样的从实验中可以看到使用 1 D 和 2 D 标签是影响不大的，但是作者尽可能的保持和原 Transformer 相同的结构，才做了适配图像的改动。

## 3.1、Vision Transformer (ViT)

为了说明计算过程，这里使用原文：

与 BERT 的\[class\] 标记类似，我们为嵌入 patch 序列（z 00=xclass）预先准备了一个可学习的嵌入，其在变压器编码器（z 0 L）输出端的状态用作图像表示 y（等式 4）。在预培训和微调过程中，一个分类头连接到 z 0 L。分类头在预训练时由一个带有一个隐藏层的 MLP 实现，在微调时由一个线性层实现

Position embeddings 将添加到 patch 嵌入以保留位置信息。我们使用标准的可学习一维 Position embeddings，因为我们没有观察到使用更先进的 2 D-aware position embeddings 带来的显著性能提升（附录 D.4）。生成的嵌入向量序列用作编码器的输入。

The Transformer encoder（Vaswani 等人，2017 年）由交替的多头自我注意（MSA，见附录 A）和 MLP 块（等式 2、3）层组成。在每个区块之前应用分层模型（LN），在每个区块之后应用 residual connections （Wang 等人，2019 年；Baevski&Auli，2019 年）。

整个过程的公式表示如下：

![](https://aitechtogether.com/wp-content/uploads/2022/04/2d17be74dc8ab1a5d1f6d32635b46d7e.webp)

Inductive bias：我们注意到，视觉变压器比 CNN 具有更少的图像特定感应偏压。在 CNN 中， locality, two-dimensional neighborhood structure, and translation equivariance 被烘焙到整个模型的每一层中。在 ViT 中，只有 MLP 层是局部的和翻译等变的，而自我注意层是全局的。二维邻域结构的使用非常少：在模型开始时，通过将图像切割成小块，并在微调时间调整不同分辨率图像的位置嵌入（如下所述）。除此之外，初始化时的位置嵌入没有关于 patch 二维位置的信息，并且 patch 之间的所有空间关系都必须从头开始学习

Hybrid Architecture：作为原始图像块的替代，输入序列可以由 CNN 的特征图形成（LeCun 等人，1989）。在这个混合模型中，将 patch embedding 投影 E（等式 1）应用于从 CNN 特征图中提取的 patch 。作为一种特殊情况，patch 可以具有空间大小 1×1，这意味着通过简单地展平特征地图的空间维度并投影到变换维度来获得输入序列。如上所述，添加了分类输入嵌入和位置嵌入。 （不做预处理，而是使用 CNN 将图片降成 14×14 大小的小图）

## 3.2、Fine-Tuning and Higher Resolution (更大的图片)

通常，我们在大型数据集上预先训练 ViT，并对（较小的）下游任务进行微调。为此，我们移除预先训练好的预测头，并附加一个初始化为零的 D×K 前馈层，其中 K 是下游类的数量。以比训练前更高的分辨率进行微调通常是有益的（Touvron 等人，2019 年；Kolesnikov 等人，2020 年）。当馈送更高分辨率的图像时，我们保持面片大小不变，这会导致更大的有效序列长度。Vision Transformer 可以处理任意序列长度（最多可达内存限制），但是，预先训练的位置嵌入可能不再有意义。因此，我们根据预训练位置嵌入在原始图像中的位置，对其执行 2 D 插值。请注意，这种 resolution adjustment and patch extraction 是将图像 2 D 结构的感应偏差手动注入视觉转换器的唯一点。

# 4 、Experiments

我们评估了 ResNet、Vision Transformer（ViT）和 hybrid 的表征学习能力。为了理解每个模型的数据需求，我们对不同大小的数据集进行预训练，并评估许多基准任务。当考虑到预训练模型的计算成本时，ViT 表现非常好（计算成本相对小），以较低的预训练成本在大多数识别基准上达到了最先进的水平。最后，我们使用自监督进行了一个小实验，并表明自我监督的 ViT 对未来是有希望的。

## 4.1、Setup

Datasets：为了探索模型的可扩展性，我们使用了 ILSVRC-2012 ImageNet 数据集，该数据集包含 1 k 类和 130 万幅图像（我们在下文中将其称为 ImageNet），其超集 ImageNet-21 k 包含 21 k 类和 14 M 图像（邓等人，2009 年），JFT（孙等人，2017 年）包含 18 k 类和 303 M 高分辨率图像。我们按照 Kolesnikov 等人（2020 年）的要求，对训练前数据集和下游任务的测试集进行了重复数据消除。我们将在这些数据集上训练的模型转移到几个基准任务中：原始验证标签上的 ImageNet 和清理后的真实标签（Beyer 等人，2020 年）、CIFAR-10/100（Krizhevsky，2009 年）、Oxford IIIT Pets（Parkhi 等人，2012 年）和 Oxford Flowers-102（Nilsback&Zisserman，2008 年）。对于这些数据集，预处理遵循 Kolesnikov 等人（2020 年）。

网络的详细参数如下：

![](https://aitechtogether.com/wp-content/uploads/2022/04/48d43415e5713b3f65744741d4da3541.webp)

我们还评估了 19 项任务 VTAB 分类套件（翟等人，2019 b）。VTAB 评估不同任务的低数据传输，每个任务使用 1000 个培训示例。这些任务分为三组：自然任务——如上述任务、Pets、CIFAR 等。专业任务——医学和卫星图像，以及结构化任务——需要几何理解的任务，如定位。

\*\*Model Variants：\*\* 我们将 ViT 配置基于用于 BERT 的配置（Devlin 等人，2019 年），如表 1 所示。“基本”和“大型”模型直接采用了 BERT 模型，我们添加了更大的“大型”模型。在下面的内容中，我们使用简短的符号来表示模型大小和输入 patch 大小：例如，ViT-L/16 表示具有 16×16 输入 patch 大小的“大”变体。请注意，Transformer 的序列长度与 patch 大小的平方成反比，因此 patch 大小越小的模型计算成本越高。

对于基线 CNN，我们使用 ResNet（He 等人，2016 年），但将批量标准化层（Ioffe&Szegedy，2015 年）替换为组标准化层（Wu&He，2018 年），并使用标准化卷积（Qiao 等人，2019 年）。这些修改改善了传输（Kolesnikov 等人，2020 年），我们将修改后的模型称为“ResNet（BiT）”。对于混合体，我们将中间特征映射以一个“像素”的面片大小提供给 ViT。为了对不同的序列长度进行实验，我们要么（i）获取常规 ResNet 50 的第 4 阶段输出，要么（ii）移除第 4 阶段，在第 3 阶段放置相同数量的层（保持总层数），然后获取扩展的第 3 阶段输出。选项（ii）导致序列长度延长 4 倍，ViT 型号更昂贵。

Training & Fine-tuning：我们使用β1=0.9、β2=0.999、批量为 4096 的 Adam（Kingma&Ba，2015）对所有模型（包括 Resnet）进行训练，并应用 0.1 的高重量衰减，我们发现这对所有模型的传输非常有用（附录 D.1 显示，与常规做法相比，Adam 在我们的设置中对 Resnet 的效果略好于 SGD）。我们使用线性学习率预热和衰减，详情见附录 B.1。对于微调，我们使用带有动量的 SGD，批量为 512，对于所有型号，请参见附录 B.1.1。对于表 2 中的 ImageNet 结果，我们以更高的分辨率进行了微调：ViT-L/16 为 512，ViT-H/14 为 518，还使用了 Polyak&Juditsky（1992）的平均值，系数为 0.9999（Ramachandran 等人，2019 年；Wang 等人，2020b）。

\*\*Metrics：\*\* 我们报告了下游数据集的结果，是通过 few-shot or fine-tuning accuracy。 Fine-tuning 捕获在各自数据集上微调后每个模型的性能。通过解决一个正则化最小二乘回归问题，将训练图像子集的（冻结）表示映射到{−1,1}K 目标向量。这个公式允许我们以封闭形式恢复精确解。虽然我们主要关注微调性能，但我们有时会使用线性少镜头精度进行快速动态评估，因为微调成本太高。

## 4.2 COMPARISON TO STATE OF THE ART（对比）

![](https://aitechtogether.com/wp-content/uploads/2022/04/f1a0d12ae7c638cd7db1f096adca6ad6.webp)

可以看到 VIT-H 的效果在每一项实验中都是效果最好的（有一项不是最好，但是也差不多），使用了而较大、较昂贵的模型结构。虽然比之前的 sota 模型效果好，但是没有很明显的提升，因此作者从计算时间上下分析，相对之前的模型，ViT 的训练消耗时间大大降低（表中最后一行，只能说是相对少，财大气粗）

### 使用 ViT 应该具备多大的数据量

![](https://aitechtogether.com/wp-content/uploads/2022/04/2a574ad4b19fe9b5406a1ffb700e8307.webp)

其中灰色方块所标识的区域是指 BiT 的训练效果范围，其他彩色圆点是指 ViT 不同大小模型的效果。

从图中可以看出在少量数据集 ImageNet 中 ViT 的效果整体不如 BiT, 随着数据量增大，ViT 的效果逐渐超越 BiT，因此可以的得出结论，ViT 模型在大型数据集上的效果好，最少是要在 ImageNet-21 K 这样的数据中，如果是小数据集，还是使用使用传统的卷积神经网络更好

### 小样本结果比较

![](https://aitechtogether.com/wp-content/uploads/2022/04/631620fc8e6db9f11f8e0ec5064a1a31.webp)

同上一节的图一样，样本量很少时 ResNet 表现好，数据量相对变大 ViT 的效果变好，最后一种情况 ViT 的效果稍微比 Res 152 好，作者提出小样本学习在 ViT 上的应用是以后的一个研究方向（又挖了一个坑）

### 为什么 ViT 比卷积神经网络的预训练便宜？

![](https://aitechtogether.com/wp-content/uploads/2022/04/d1e9340ef74166aab40121bc222ce24d.webp)

预训练的数据集都是在 JFT 数据集上进行的

测试：

​ Average-5: 在五个数据集上做 evaluation 在做平均五个数据集 imageNet real、 pets、 flowers、cifar 10、cifar 100

​ 另外一个图的 ImageNet 比较重要单独做一个

1）如果将不同计算时间复杂度的 ViT 和 Bit 做比较，相同时间复杂度情况下，ViT 的准确率高

2）在时间复杂度较少的情况下，混合模型（Hybrid）的效果是最好的，这说明混合模型吸收的 ViT 和 ResNet 各自的优点。但是随着时间复杂度增大，混合模型的效果趋近于 ViT 模型结果，甚至不如 ViT 的效果，所以数据预训练对 ViT 的表现有很大影响。

3）BiT 和 ViT 的效果都还没有达到顶峰，增长趋势还是线性增加

## 4.5、Inspecting Vision Transformer（可视化）

为了开始理解视觉转换器如何处理图像数据，我们分析了它的内部表示。视觉变换器的第一层线性地将平坦的面片投影到较低维度的空间中（等式 1）。图 7（左）显示了学习的嵌入过滤器的顶部主要组件。这些成分类似于每个斑块内精细结构的低维表示（线条、颜色、斑块等）的合理基函数。

Patch embedding：

![](https://aitechtogether.com/wp-content/uploads/2022/04/872d9c86687b8e7c1a8b8f4af7eccb05.webp)

Position embedding

1）已经学习到一些位置信息，比如中心点的小图中可以看到越靠近中心的点激活越高（越接近 1），越远离中心点的位置激活越低（越接近 0）

2）学习到了行列信息，每一个图的位置对应的同行同列相对激活更高（1 D 编码学习到了 2 D 的位置信息）

![](https://aitechtogether.com/wp-content/uploads/2022/04/bc928746b37b378048e52e3c3f998a55.webp)

不同网络深度中不同 head 的 Mean attention adistance（pixels）：

可以看到随着网络深度加大，多头之间的距离也在变大。刚开始有的 head 之间的距离很近，有的很远，说明 attention 从刚开始就可以关注全局信息，。到后面的层数 head 之间的 pixels 都变得很大，这说明网络以及不再通过临近像素点进行判断，已经学习到了高层的语义信息（学习到了语义性的概念）

![](https://aitechtogether.com/wp-content/uploads/2022/04/eff03c1a3c6adbfd2bf83d709d1737a2.webp)

为了验证上面的观点，作者将输出的 token 做了原图的映射，如下图：

![](https://aitechtogether.com/wp-content/uploads/2022/04/04defe4e16f42d6487cbd655147a7f60.webp)

我们发现，一些 head 关注的是已经位于最底层的大部分图像，这表明该模型确实使用了 globally 集成信息的能力。其他注意头在低层的注意距离一直很小。这种 highly localized attention 在 Transformer 之前应用 ResNet 的混合模型中不太明显（图 7，右图），这表明它可能与 CNN 中的早期卷积层具有类似的功能。此外，the attention distance 随着网络深度的增加而增加。在 Globally 内，我们发现该模型关注与分类语义相关的图像区域（图 6）（head 之间的 the attention distance 变大说明网络判断分类不再使用相邻像素关系，变成了更高级特征的语义信息）

![](https://aitechtogether.com/wp-content/uploads/2022/04/3099fad3b4fc8c1bf5afe52ae2390cbd.webp)

## 4.6、Self-supervision（自监督）

Transformers 在 NLP 任务中表现出色。然而，他们的成功不仅源于出色的可扩展性，还源于大规模的自我监督预训练。我们还模拟了 BERT 中使用的 masked 语言建模任务，对用于自我监督的 masked patch prediction 进行了初步探索。通过自我监督预训练，我们较小的 ViT-B/16 模型在 ImageNet 上实现了 79.9% 的准确率，与从头开始的训练相比，显著提高了 2%，但仍落后于有监督预训练 4%。附录 B.1.2 包含更多详细信息。我们将 contrastive pre-training（对比学习）的探索（Chen 等人，2020 b；He 等人，2020；Bachman 等人，2019；Hénaff 等人，2020）留给未来的工作。 （对比学习、迁移学习、横向迁移其他领域等大坑）

# 5 、Conclusion

我们探索了变压器在图像识别中的直接应用。与以前在计算机视觉中使用自我注意的工作不同，除了最初的 patch 提取步骤外，我们没有在体系结构中引入特定于图像的感应偏差。相反，我们将图像解释为一系列补丁，并使用 NLP 中使用的标准 Transformer 编码器对其进行处理。这种简单但可扩展的策略在与大型数据集的预培训相结合时出人意料地有效。因此，Vision Transformer 在许多图像分类数据集上都达到或超过了最先进的水平，同时预训练成本相对较低。 （总结我们的效果很好）

虽然这些初步结果令人鼓舞，但仍存在许多挑战。一种是将 ViT 应用于其他计算机视觉任务，如检测和分割。我们的结果，加上 Carion 等人（2020 年）的结果，表明了这种方法的前景。另一个挑战是继续探索自我监督的训练前方法。我们的初步实验表明，自我监督预训练效果有所改善，但自我监督预训练与大规模监督预训练之间仍存在较大差距。最后，进一步扩展 ViT 可能会提高性能。 （挖坑，说明 ViT 在其他领域也可以做很多事情）

# 其他参考
- <https://zhuanlan.zhihu.com/p/356155277>