---
title: "MobileCLIP来袭 | 如果CLIP可以通过重参加速，你会选择用它作为Backbone预训练吗"
source: "https://developer.volcengine.com/articles/7382370140857303091"
author:
  - "[[集智书童]]"
published: 2024-06-20
created: 2024-12-13
description: "点击下方卡片，关注「集智书童」公众号点击加入👉「集智书童」交流群对比预训练的图像文本基础模型，如CLIP，在零样本性能上表现出优异的表现，并在各种下游任务上提高了鲁棒性。然而，这些模型使用了具有大量变压器基编码器的模型，这导致了在移动设备上的部署存在显著的内存和延迟开销。在这项工作中，作者引入了MobileCLIP - 一个为运行时性能优化的高效图像文本模型的新一代家族，以及一种新颖且高效的训练"
tags:
  - "clippings"
---
点击下方卡片，关注「集智书童」公众号

[点击加入👉「集智书童」交流群](http://mp.weixin.qq.com/s?__biz=MzU5OTA2Mjk5Mw==&mid=2247499235&idx=2&sn=8927bb7201b93fc0fd6e7273b31cfa1e&chksm=feb81d5dc9cf944b0b014e4408c6f832f5ea6d84129c3a6c849e5bff393c48b3c178ab1b43c1&scene=21#wechat_redirect)

![picture.image](https://p6-volc-community-sign.byteimg.com/tos-cn-i-tlddhu82om/fe925b3b90c9435db731a1a0635cb62a~tplv-tlddhu82om-image.image?=&rk3s=8031ce6d&x-expires=1734167216&x-signature=bVMRh%2BN%2Fhkvt3t8cv6ifHtdAeSc%3D)

> 对比预训练的图像文本基础模型，如CLIP，在零样本性能上表现出优异的表现，并在各种下游任务上提高了鲁棒性。然而，这些模型使用了具有大量变压器基编码器的模型，这导致了在移动设备上的部署存在显著的内存和延迟开销。
> 
> 在这项工作中，作者引入了MobileCLIP - 一个为运行时性能优化的高效图像文本模型的新一代家族，以及一种新颖且高效的训练方法，即多模态强化训练。所提出的训练方法利用了图像描述模型的知识迁移和一组强大的CLIP编码器的集成来提高高效模型的准确性。作者的方法通过将额外的知识存储在强化数据集中，避免了训练时的计算开销。MobileCLIP 在多个数据集上为零样本分类和检索任务设置了一个新的最先进延迟-准确性权衡。作者的MobileCLIP-S2变体比基于VT-B/16的前一代最佳CLIP模型快2.3倍，同时更加准确。
> 
> 作者进一步展示了多模态强化训练的有效性，通过训练基于ViT-B/16图像背书的CLIP模型，与之前的最佳结果相比，在38个评估基准上实现了+2.9%的平均性能提升。此外，作者证明了与非强化CLIP训练相比，所提出的方法在10倍至1000倍之间实现了改进的学习效率。

## 1 Introduction

大型图像文本基础模型，如CLIP，在零样本性能上表现出优异的表现，并在各种下游任务上提高了鲁棒性。然而，由于这些模型的大小和延迟，在移动设备上部署它们具有挑战性。作者的目标是设计一个新的对齐图像文本编码器家族，使其适合移动设备。实现这一目标的主要挑战有两个：

1. 首先，运行时性能（例如延迟）与不同架构的准确性之间存在权衡，因此作者应该能够快速而全面地分析不同的架构设计。大型CLIP模型的训练在计算上是昂贵的，这阻碍了高效架构设计的快速开发和探索。另一方面，标准的小规模多模态对比学习会导致较差的准确性，这不能为架构设计选择提供有用的信号。
2. 第二，较小的架构的容量降低导致精度不佳，可以通过更好的训练方法来提高。

为了克服这些挑战，作者开发了一种新颖的训练方法，基于数据集强化方法：

1. 用附加信息强化一次数据集
2. 使用强化后的数据集多次进行实验

对于给定的计算预算，使用强化后的数据集进行训练相对于原始数据集可以获得更高的准确性。作者提出了一种多模态的数据集强化变体用于训练高效的CLIP模型。具体而言，作者通过添加来自一组预训练CLIP模型的合成描述符和嵌入（如图3所示），强化了图像文本数据集DataComp，得到了DataCompDR。

![picture.image](https://p6-volc-community-sign.byteimg.com/tos-cn-i-tlddhu82om/cdd16dfe95914f71a121262daa6c6e80~tplv-tlddhu82om-image.image?=&rk3s=8031ce6d&x-expires=1734167216&x-signature=5pQQFciBxzZZkAAfrER5BilghQ0%3D)

作者提出了两种强化数据集的变体，其中DataCompDR-12M适用于快速迭代高效模型设计，而DataCompDR-1B则适用于最佳大规模训练性能。在DataCompDR上进行训练与标准CLIP训练相比具有显著的学习效率提升。例如，在单节点8A100 GPU上，当在DataCompDR-12M上从零开始训练基于ViT-B/16的CLIP时，大约一天就能实现ImageNet-val上的61.7%的零样本分类。

![picture.image](https://p6-volc-community-sign.byteimg.com/tos-cn-i-tlddhu82om/b92dbb0469ba499b99a74c5f205230cf~tplv-tlddhu82om-image.image?=&rk3s=8031ce6d&x-expires=1734167216&x-signature=qHP673BTmvT4dTiD55lP9X6Lzr0%3D)

在使用比以前的工作更少的训练计算预算的情况下，使用DataCompDR-1B进行训练在多个指标上取得了新的最先进性能（见图2）。利用DataCompDR，作者探索了设计空间，并获得了比前人更好的时延-准确性权衡的移动友好对齐图像文本编码器家族，称为MobileCLIP（见图1）。

![picture.image](https://p6-volc-community-sign.byteimg.com/tos-cn-i-tlddhu82om/c7066e474c6148fa9ad28b187e02860c~tplv-tlddhu82om-image.image?=&rk3s=8031ce6d&x-expires=1734167216&x-signature=aSQDBmw85QX%2Bo1IDBtP4LSU8x6Q%3D)

作者利用几种架构设计技术来获得高效的图像和文本编码器，包括结构重参数化和卷积Token混合。MobileCLIP包括S0、S1、S2和B变体，覆盖了不同的尺寸和延迟，以适应不同的移动应用。作者的最快变体MobileCLIP-S0比标准OpenAI ViT-B/16 CLIP模型快约5倍，小约3倍，但平均准确性相同。

作者的贡献如下：

1. 设计了一个新的移动友好CLIP模型家族，称为MobileCLIP。MobileCLIP的变体使用混合CNN-transformer架构，并在图像和文本编码器中进行结构重参数化，以减少大小和延迟。
2. 介绍了多模态强化训练，这是一种新颖的训练策略，它将来自一个预训练图像描述模型和一组强大的CLIP模型的知识迁移相结合，以提高学习效率。
3. 提出了两种强化数据集的变体：DataCompDR-12M和DataCompDR-1B。使用DataCompDR，作者在与DataComp的比较中展示了10倍至1000倍的学习效率。
4. MobileCLIP家族在零样本任务上获得了最先进的时延-准确性权衡，包括创纪录地成为基于ViT-B/16的CLIP模型的新最佳。

## 2 Related Work

CLIP的高效学习。可以通过利用增强的训练目标来提高学习效率。例如图像掩码、单模态自监督、细粒度图像文本对齐、图像文本标签空间的对比学习以及成对Sigmoid损失。最近，CLIPa提出了在多个分辨率上进行CLIP训练以实现经济高效训练的方法。这些方法与作者的提议方法互补，可以用于进一步提高性能。

CLIP训练数据集通常包含在网络尺度上获得的噪声图像文本对。自CLIP模型以来，许多工作已经在大规模和过滤数据集上展示了改进的结果。与数据收集和过滤互补，最近的工作表明，使用由预训练描述模型生成的视觉丰富的合成描述符，并结合真实描述符，可以提高CLIP模型的质量。

作者提出的强化多模态数据集也受益于由合成生成的描述符，作者证明这对于提高学习效率至关重要。先前的研究探索了将单模态知识蒸馏扩展到视觉语言模型。DIME-FM提出使用领域内的单模态数据进行蒸馏，重点关注零样本分类。

TinyCLIP通过跨模态亲和性模拟和权重继承训练紧凑的CLIP模型。在学生是一个特定任务的融合视觉语言模型的设置中，也探索了多模态蒸馏。作者提出的多模态强化训练也包括了跨模态亲和性模拟，该模拟针对强化数据集中的目标。此外，作者将单模态模型集成扩展到多模态设置，并将来自CLIP模型集成的目标存储起来。

最近，由于运行大型教师模型产生的训练时间开销成本，提出了离线知识蒸馏方法。作者将\[14\]中的\_数据集强化\_策略扩展到CLIP的多模态设置。作者提出的强化多模态数据集在不需要增加训练时间计算开销的情况下，实现了显著的准确性改进。用于CLIP的高效架构。最近有许多架构在资源受限的设备上完成视觉任务方面展现出巨大的潜力。

这些架构可以广泛地分类为纯粹卷积的，基于Transformer的和卷积-Transformer混合的。同样，对于文本编码，也有基于Transformer的和卷积-Transformer混合的。有一些工作，如\[67\]，将ViT架构剪枝以获得更小、更快的CLIP模型，或者像\[3\]这样的工作，将图像文本Token减少以实现视觉语言模型的更快推理。这些模型仍然可能相当大且低效，无法在移动设备上部署。在作者的工作中，作者引入了一种改进的卷积-Transformer混合架构，用于视觉和文本模态，该架构优于最近的一些最先进的状态。\[3, 67\]中引入的优化可以用于进一步改进作者模型的效率。

## 3 Multi-Modal Reinforced Training

作者的多模态强化训练利用来自图像描述模型的知识迁移和一组预训练的CLIP模型的强大集成来训练目标模型。

它由两个主要组成部分构成：

1. 通过合成描述符利用图像描述模型的知识
2. 从一组强大的预训练CLIP模型的集成中进行图像文本对齐的知识蒸馏

作者遵循\[14\]中的数据集强化策略，并将额外的知识（合成描述符和教师嵌入）存储在数据集中（见图3），从而避免了评估描述符模型或集成教师所带来的任何额外的训练时间计算开销。所提出的训练策略在提高学习效率方面取得了显著改进，即在较少的训练预算和更少的样本数量下达到特定目标性能。

### Dataset Reinforcement

合成描述符。用于训练CLIP模型的图像文本数据集大多来源于网络，这是固有的噪声。最近的努力，如DataComp和数据过滤网络，通过使用广泛的过滤机制改进了从网络获取的数据集的质量。虽然这些过滤的数据集噪声较低，但描述符可能仍然不够充分。为了提高描述符的视觉描述性，作者使用了流行的CoCa模型\[73\]，并为每个图像生成多个合成描述符（见图3a）。

在第5.1节中提供了关于每个图像生成的合成描述符数量的ablation。图5展示了由CoCa模型生成的合成描述符的示例。与合成描述符相比，真实描述符通常更具体，但噪声更大。

![picture.image](https://p6-volc-community-sign.byteimg.com/tos-cn-i-tlddhu82om/64f3effd9ccb46609e38f27204cf0803~tplv-tlddhu82om-image.image?=&rk3s=8031ce6d&x-expires=1734167216&x-signature=6%2FK2aLNT30Z3sm0XBcy3Gurz8%2FU%3D)

作者在表2a中展示了将真实和合成描述符相结合对于获得最佳的零样本检索和分类性能至关重要。

**图像增强。** 对于每个图像，作者使用一个参数化的增强函数生成多个增强图像。

其中是足以从生成的增强参数（见图3a）。每张图像使用的增强数量和不同类型的增强的ablation分别在Tabs. 3a和11中提供。

**集成教师。** 模型集成是一种广泛使用的技术，用于从一组独立训练的模型中创建更强的模型。作者将这种技术扩展到多模态设置，并使用个CLIP模型作为强大的教师（请参阅第5.1节以了解作者的教师ablation）。作者计算这些模型的特征嵌入，为增强的图像和合成描述符计算维向量和，其中是第个教师模型。作者还计算了真实描述符的教师嵌入（见图3b）。

请注意，数据集强化是一次性的成本，通过多次高效的模型训练和实验进行摊销。

### Training

**损失函数。** 直观上，作者的损失函数将多个图像文本教师编码器之间的图像文本对之间的相似性矩阵蒸馏为学生图像文本编码器。令表示一个批次的（图像，文本）对，分别表示教师集成体中第个模型的第个图像和文本的维向量矩阵，其中表示批次的数量。相应地，作者用表示目标模型的图像和文本嵌入矩阵。

对于给定的和矩阵，令表示通过应用行向量Softmax操作得到的结果，其中是一个温度参数。作者的训练损失由两个组成部分组成，即标准的CLIP损失和知识蒸馏损失。其中KL表示Kullback-Leibler散度，是通过交换中的文本和图像嵌入项得到的结果，是一个折衷参数。

**高效的训练。** 在强化数据集上进行训练就像修改数据加载器和损失函数以利用存储在数据集中的额外知识一样简单，并且具有与标准CLIP训练相同的训练成本（见Tab. (d)d）。对于每个样本，作者从数据集中读取图像和相应的真实描述符。然后，作者随机加载一个存储的增强参数并复制增强后的图像。

此外，作者还随机加载一个合成描述符。最后，作者读取存储的嵌入，, ，和，对应于个教师模型。使用加载的数据，作者构造了两个数据批次，对应于（增强的图像，真实的描述符）对，对应于（增强的图像，合成描述符）对，并在和上分别计算作者的训练损失，方程（2）。

作者最终的损失函数定义为：

请注意，作者可以在学生模型进行前向传播后，不需要额外的教师相关计算就可以计算总损失，因为计算蒸馏损失所需的教师嵌入是数据集中的组成部分。

## 4 Architecture

### Text Encoder

CLIP模型将视觉Transformer与包含自注意力层的传统Transformer配对，用于文本编码。虽然这个模型很有效，但对于移动部署，更喜欢更小、更高效的模型。最近，像\[66\]这样的工作表明卷积对于文本编码也可以同样有效。与使用全卷积结构显著不如Transformer对应模型相比，作者发现使用纯卷积结构在文本编码方面表现不佳。因此，作者提出了一种混合文本编码器，该编码器使用一维卷积和自注意力层。作者设计了使用Transformer和混合文本编码器的MobileCLIP变体。

对于混合文本编码器，作者引入了\_Text-RepMixer\_，这是一个卷积Token混合器，将训练时间和推理时间架构解耦。Text-RepMixer受到可参数化的卷积Token混合（RepMixer）的启发。在推理时，跳跃连接被重新参数化。该架构如图4所示。

![picture.image](https://p6-volc-community-sign.byteimg.com/tos-cn-i-tlddhu82om/7a094eef5d8d4862af548edc49993d86~tplv-tlddhu82om-image.image?=&rk3s=8031ce6d&x-expires=1734167216&x-signature=aFkZFjAanjRqvRjVH03s30RTdO0%3D)

对于Feed-Forward网络（FFN）块，作者将线性层与类似Token混合器的同维深度卷积进行扩展，以获得\_ConvFFN\_块。这种结构类似于\[19\]中使用的卷积块，主要区别在于使用了批量归一化和能够与接下来的深度卷积层折叠的能力，以实现高效的推理。对于所有深度卷积，作者使用核大小为11。

![picture.image](https://p6-volc-community-sign.byteimg.com/tos-cn-i-tlddhu82om/5d4ec1b03139469d89a0492228ce6240~tplv-tlddhu82om-image.image?=&rk3s=8031ce6d&x-expires=1734167216&x-signature=NfXCZdZYwY7tco0lLTvFggNvP0s%3D)

这种选择基于准确率-延迟权衡，完整的ablation可以在附录E中找到。为了找到作者混合文本编码器的最佳设计，作者从纯粹卷积文本编码器开始，并逐步用自注意力层替换卷积块（见Tab. 4(b)）。

### Image Encoder

近年来，一些工作已经展示了混合视觉Transformer在学习良好视觉表示方面的有效性。对于MobileCLIP，作者引入了一种改进的混合视觉Transformer，称为MCi，它基于最近发布的FastViT架构，但与FastViT有一些关键差异，如下所述。在FastViT中，FFN块使用了4.0的MLP扩展比。一些最近的工作，如\[38, 67\]，揭示了FFN块线性层的显著冗余。为了提高参数效率，作者简单地将扩展比降低到3.0，并增加架构的深度。

![picture.image](https://p6-volc-community-sign.byteimg.com/tos-cn-i-tlddhu82om/6ee63197c8a1407398f5e7fac0ed225a~tplv-tlddhu82om-image.image?=&rk3s=8031ce6d&x-expires=1734167216&x-signature=AN3DB%2BK1PWKLMWjauY59nWzFCmY%3D)

这样，作者在保持图像编码器参数数量相同的同时，将FastViT中的MLP扩展比从4.0降低到3.0，并增加架构深度。这样做，作者保持了三个变体的阶段配置相似。Tab. 3(a)描述了三个变体的阶段配置。MCi0与\[60\]类似。MCi1是MCi0的更深版本，MCi2是MCi1的更宽版本。作者变体的阶段计算比例与\[60\]相似。作者发现这种设计对延迟的影响最小，但模型容量的好转（反映在下游任务性能）却很好。

在从ImageNet数据集对图像进行分类任务进行从零开始训练时，MCi2达到了与FastViT（之前的混合视觉Transformer状态最先进）相同的Top-1准确率84.5%，同时比FastViT小15%且比FastViT快14.3%。有关更多细节，请参阅附录A。

## 5 Experiments

在这个部分，作者介绍了作者的实验设置，对所提出的作者的方法和快速MobileCLIP架构进行了ablation，并提供了结果。

**评估。** 作者使用DataComp的评估基准来评估图像文本模型。具体来说，作者在ImageNet验证集上报告零样本分类，以及包括ImageNet-V2，ImageNet-A，ImageNet-O，ImageNet-R和ObjectNet在内的分布转移，作者将其平均值报告为IN-Shift。对于零样本图像文本检索，作者在MSCOCO和Flickr30k数据集上报告了recall@1。此外，作者在DataComp评估中报告了所有38个数据集的平均性能。作者还在视觉基因组关系、视觉基因组属性、Flickr30k-Order和COCO-Order数据集上评估了作者的模型，这些数据集是最近属性、关系和顺序（ARO）基准的一部分。

在剩余部分，IN-val指的是在ImageNet验证集上的零样本准确性，而Flickr30k指的是图像文本和文本图像检索的平均零样本recall@1。所有报告的指标都是在没有进行任何微调的情况下获得的。

**训练设置。** 作者有两个设置用于ablation和大规模实验。对于ablation，作者使用具有1280万图像文本对的数据集，全局批量为8192，使用8NVIDIA-A100-80GB GPUs进行30-45k迭代。对于大规模训练，作者使用全局批量为65536，使用256A100 GPUs进行200k迭代。所有模型都是从零开始训练的（有关详细信息请参阅附录A）。

**数据集。** 作者在DataComp数据集的图像文本数据集上进行训练。作者使用了Bestpool过滤的1.28亿样本子集，该子集在最大数据集规模上提供了最佳性能。作者将这个集合称为DataComp-1B。为了快速实验，作者创建了一个固定子集，该子集是从12.8百万个均匀采样的对中选择的，作者称之为DataComp-12M。

DataComp-12M在\[18\]中未被研究过，但在作者的实验中，作者观察到DataComp-12M在相比DataComp-medium Bestpool子集具有可比样本的情况下，始终实现了更好的性能。

DataCompDR：强化数据集。作者使用作者的多模态数据集强化策略强化DataComp数据集。具体而言，作者通过强化DataComp-1B和DataCompDR-12M来创建DataCompDR-1B和DataCompDR-12M。作者有一个一次生成的过程，其成本在多个架构和广泛的ablation中摊销。

作者使用OpenCLIP中的coca\_ViT-L-14模型为每张图像生成5个合成描述符，并使用10个数据CompDR-1B和30个数据CompDR-12M的强随机图像增强。作者计算两个强大教师（ViT-L-14带有预训练权重datacomp\_xl\_s13b\_b90k和OpenAI在OpenCLIP中的）在增强图像以及真实和合成描述符上的嵌入。嵌入是由2768-D向量的1536-D串联组成的。作者将所有的强化都使用无损压缩和BFloat16进行存储。作者在Sec. 5.1中分析了作者所有的选择。

**MobileCLIP架构。** 作者的移动CLIP架构由MCi:MCt架构对组成。特别地，作者创建了3个小变体MobileCLIP-S0（MCi0:MCt），MobileCLIP-S1（MCi1:Base）和MobileCLIP-S2（MCi2:Base），其中Base是一个类似于基于ViT-B/16的CLIP中的文本编码器的12层Transformer。此外，作者还训练了一对标准的ViT-B/16:Base，并将其训练好的模型称为MobileCLIP-B。

为了测量延迟，作者使用对应的方法的输入大小。对于iPhone延迟测量，作者使用Core ML Tools（v7.0）导出模型，并在带有iOS 17.0.3的iPhone12 Pro Max上运行。批量大小设置为所有模型均为1。作者遵循与\[60\]中描述相同的协议。

### Ablation Studies

在本节中，作者分析了训练和架构中每个组件的影响。除 otherwise 外，作者使用在DataComp-12M上训练30k迭代，全局批量为8k（约20个epoch）的ViT-B/16:Base编码器。Table 1总结了作者的训练分析。

![picture.image](https://p6-volc-community-sign.byteimg.com/tos-cn-i-tlddhu82om/8a974f37b1ea44c79dc8a5302be294a3~tplv-tlddhu82om-image.image?=&rk3s=8031ce6d&x-expires=1734167216&x-signature=Dv7z6XbsEN7VThUJpfu9O9%2Fo2cY%3D)

**强烈的图像增强。** 与用于单模态视觉监督和自监督方法中的强烈增强不同\[13, 59\]，CLIP训练配置通常使用轻图像增强来避免图像文本对齐错误。然而，一些工作\[2, 14, 45\]表明，在蒸馏设置中，强烈图像增强是有效的。在Table 1中，作者表明强烈图像增强可以提高蒸馏性能（在IN-val上+4.8%和+4.4%在Flickr30k）。作者在附录B中提供了关于图像增强对蒸馏效果影响的详细ablation。

**合成描述符。** 与图像增强类似，合成描述符（或描述符增强）可以进一步提高CLIP模型的性能，尤其是在图像文本检索方面。对于常规CLIP训练（），作者在Table 1中观察到，包括既包含合成描述符又包含真实描述符的批次，在IN-val上的性能提高了+7.4%，在Flickr30k上的性能提高了+27.5%。在Table 1(a)中，作者观察到仅使用蒸馏损失的CLIP训练（）的趋势类似。

在Table 1(b)中，作者分析了的影响，并观察到在IN-val上，是最佳选择，而在Flickr30k上，是最佳选择。以前利用合成描述符的工作主要关注改进检索，而蒸馏工作主要关注零样本分类。在作者的大规模实验中，作者使用来平衡MobileCLIP-B的权衡，并使用来处理作者的小变体。

**存储大小。** 作者报告了与原始DataComp数据集相比，作者的强化数据集的存储要求。一般来说，数据集的存储大小取决于文件格式和加载时间与压缩率之间的权衡。作者报告了每个图像文本对一个文件的存储大小。

如果存在，作者将所有相应的强化存储在同一个文件中。作者将文件存储在Pickle格式中，并使用Gzip压缩每个文件。图像文本嵌入以BFloat16格式保存。作者在Table 2(c)中报告了12.8M样本的DataCompDR-12M和1.28B样本的DataCompDR-1B的总存储大小。

![picture.image](https://p6-volc-community-sign.byteimg.com/tos-cn-i-tlddhu82om/37e27a774e2e45058b99cbfb8e63f54f~tplv-tlddhu82om-image.image?=&rk3s=8031ce6d&x-expires=1734167216&x-signature=NcerVjt1pNOjz7nJx7RckVzK8qU%3D)

作者在附录D中提供了关于附加大小减少的分析，并验证了使用BFloat16不会影响准确性。为了最小化存储开销，作者推荐在DataCompDR-12M上使用5个增强/合成描述符进行30个epoch的训练，并在DataCompDR-1B上使用2个增强/合成描述符进行10个epoch的训练，这些建议基于作者在Tabs. 2(a)和2(b)中的ablation。

**混合文本编码器。** 作者对可以有效替代自注意力层而不会对零样本性能产生明显影响的自注意层数量进行ablation。在这个ablation中，作者选择了一个6层纯粹卷积的文本编码器，并在中间系统地引入自注意力层。

从Table 3(b)中，作者可以发现，即使引入一个自注意力层，也会显著提高零样本性能。最佳的权衡是使用2个TextRepMixer块和4个自注意力层。这个变体MCt获得了与纯Transformer变体相似的性能，同时比纯Transformer变体小5%且比纯Transformer变体快15.8%。

### Small Scale Regime

在Table 5中，作者比较了在具有12-20M样本的数据集上训练的方法，这是一个相对较小的范围用于快速探索（例如，架构搜索）。在具有不到370M样本的DataCompDR-12M上训练的MobileCLIP-B显著优于所有具有多达4倍训练时间的其他方法。

![picture.image](https://p6-volc-community-sign.byteimg.com/tos-cn-i-tlddhu82om/13780b768cba465baf919d673851355b~tplv-tlddhu82om-image.image?=&rk3s=8031ce6d&x-expires=1734167216&x-signature=H7G%2FGIpXKtJpros220ZciYX%2BYM8%3D)

此外，与之前的SLIP工作相比（在65.3%到71.7%的可见样本数量之间），MobileCLIP-B表现出巨大的可伸缩性（42.8%到45.0%）。与使用多分辨率训练进行效率的CLIPA\[33\]相比，使用DataCompDR-12M进行训练更加高效：CLIPA在2.69亿多分辨率可见样本（相当于0.5亿224^2可见样本）上获得了63.2%，而MobileCLIP-B仅使用0.37亿可见样本获得了65.3%。

此外，与MobileCLIP-S2相比，TinyCLIP-39M/16具有更高的延迟和更少的准确性，而TinyCLIP-8M/16的准确性显著低于MobileCLIP-S0（41.1% vs 59.1%），尽管它们的延迟非常接近（2.6 ms vs 3.1 ms）。

### Learning Efficiency

更长的知识蒸馏训练已被证明可以一致性地提高分类模型的性能。在图6(a)中，作者展示了作者的强化训练也受益于更长的训练，在仅使用DataComp-1B的12M子集上，在120个epoch后达到了71.7%的ImageNet-val零样本准确率。与非强化训练相比，最好的结果是达到55.7%的准确性。

![picture.image](https://p6-volc-community-sign.byteimg.com/tos-cn-i-tlddhu82om/b634c15a95904008891b6d54c1043723~tplv-tlddhu82om-image.image?=&rk3s=8031ce6d&x-expires=1734167216&x-signature=o%2FaUQodI3xPuCN2sw1UmsOMiMnc%3D)

作者还在图6(b)中展示了与数据集大小相关的扩展性，其中作者使用从1.28M到所有1.28B样本的DataComp-1B子集。对于所有实验，作者在20k迭代后训练，全局批量为65k（相当于在1.28B子集上进行一次epoch训练）。

在这个设置下，作者在使用1.28M样本的DataCompDR上训练时，可以达到55.2%以上的准确率，而使用DataComp-1B的准确率只有约6%。在这个设置下，作者观察到使用DataCompDR的数据效率比使用DataComp-1B的数据效率高100倍以上。此外，作者还观察到在Flickr30k上的数据效率比使用DataComp-1B的数据效率高1000倍。

### Comparison with State-of-the-art

在Table 6中，作者与具有大规模训练的相比进行了比较。在DataCompDR-1B上训练的MobileCLIP-S0显著优于最近的工作，如TinyCLIP，并具有与在DataComp上训练的ViT-B/32模型相似的性能，同时比它们小2.8倍且快3倍。

![picture.image](https://p6-volc-community-sign.byteimg.com/tos-cn-i-tlddhu82om/9c3bfced53954474854b81e64f79276f~tplv-tlddhu82om-image.image?=&rk3s=8031ce6d&x-expires=1734167216&x-signature=uRD6BS6jb9j7PtcjPMdxaz2NpbI%3D)

MobileCLIP-S2在38个数据集上获得了2.8%的平均性能提升，并且在与在DataComp上训练2.6倍更长的ViT-B/32-256模型相比，在检索性能上显著更好。MobileCLIP-S2比ViT-B/32-256模型小1.5倍且快1.4倍。

MobileCLIP-B在38个数据集上获得了2.9%的平均性能提升，并且在检索性能上优于SigLIP-B/16模型，该模型是在WebLI数据集上训练约3倍 longer。MobileCLIP-B比SigLIP-B/16模型小26.3%。

### Retrieval Performance Analysis

作者在\[74\]最近发布的属性、关系和顺序（ARO）基准上评估作者的模型。作者在Table 7中比较了在DataCompDR-1B上训练的作者的MobileCLIP-B与所有公开可用的ViT-B/16:Base模型。

![picture.image](https://p6-volc-community-sign.byteimg.com/tos-cn-i-tlddhu82om/485d681ce4da4b7abc1c9e707fade521~tplv-tlddhu82om-image.image?=&rk3s=8031ce6d&x-expires=1734167216&x-signature=sPoTnXD7OC5ps56usX4IXY%2BigzA%3D)

仅优化零样本分类或检索任务，使用噪声的webscale数据集可能会降低对自然场景的组成理解。DataCompDR在提高模型在ARO基准上的性能的同时，在零样本分类和检索任务上获得了良好的性能。

与最近的方法SigLIP相比，MobileCLIP-B在Visual Genome Relation和Attributes数据集上分别获得了19.5%和12.4%的更高准确率，并在Flickr30k-Order和COCO-Order数据集上分别实现了69.7%和50.3%的提高召回率@1。

## 6 Conclusion

在这项工作中，作者引入了MobileCLIP对齐图像文本Backbone，旨在用于设备上的CLIP推理（低延迟和大小）。作者还引入了DataCompDR，这是对DataComp的强化，并利用来自预训练图像描述模型的知识以及一组强大的CLIP模型。

作者演示了10倍-1000倍的学习效率，作者的强化数据集。MobileCLIP模型在DataCompDR上训练时，与先前的作品相比，获得了最先进的时延-准确性权衡。MobileCLIP模型还表现出更好的鲁棒性和在属性、关系和顺序（ARO）基准上的改进性能。

## Appendix A Experimental Setup

有关作者训练和评估的更多详细信息，请参阅这里。作者在224分辨率上训练所有模型。

Table 10总结了作者在训练MobileCLIP-B时使用的超参数。对于MobileCLIP的其他变体（S0、S1和S2），作者使用相同的超参数，除了使用外。对于在DataCompDR-12M上的实验，作者使用全局批量大小为8192。

![picture.image](https://p6-volc-community-sign.byteimg.com/tos-cn-i-tlddhu82om/b0f7f16486894e41b0f0fb592f5956ff~tplv-tlddhu82om-image.image?=&rk3s=8031ce6d&x-expires=1734167216&x-signature=6XfbMA%2FIymP332z%2B2bA9bKtlPhE%3D)

在附录C中的集成蒸馏ablation中，作者使用了32个总的A100 GPU，但使用了与其他ablation相同的全局批量大小8192。作者还使用了一个较小的均匀采样的DataComp-8M用于附录B和C中的ablation，这导致性能比使用DataCompDR-12M进行其余ablation时的性能略低。

在Sec. 4.2中的ImageNet-1k实验中，作者遵循\[37, 58\]中规定的训练配方，即使用AdamW优化器训练300个epoch，权重衰减为0.05，峰值学习率为，总批量大小为1024。热身轮数为5，使用余弦退火率来衰减学习率。

![picture.image](https://p6-volc-community-sign.byteimg.com/tos-cn-i-tlddhu82om/94209dd81acc478d87fa21ad6bbe41ce~tplv-tlddhu82om-image.image?=&rk3s=8031ce6d&x-expires=1734167216&x-signature=g%2FN8tc8tcpHcE2Al81OJK9AZo%2FU%3D)

用于蒸馏的老师模型是RegNetY-16GF。作者的实现使用了Timm库，所有模型都在单机上使用8NVIDIA A100 GPUs进行训练。Tab. 8中详细列出了MCi的三个变体的超参数。

![picture.image](https://p6-volc-community-sign.byteimg.com/tos-cn-i-tlddhu82om/8ec8703a917842c9a6e7c0d1665935e4~tplv-tlddhu82om-image.image?=&rk3s=8031ce6d&x-expires=1734167216&x-signature=qbbJnZwmMc55dbQLPGfEGKBBvGk%3D)

MCi变体的性能详细列在Tab. 9中，并与其他最近的最先进的有效架构进行了比较。如Fig. 7所示，MCi在最近的有效架构中取得了最佳权衡。

![picture.image](https://p6-volc-community-sign.byteimg.com/tos-cn-i-tlddhu82om/6eb50b1fa78041d19ea51a4115f5505a~tplv-tlddhu82om-image.image?=&rk3s=8031ce6d&x-expires=1734167216&x-signature=mG8fZkil6YPIwuogk0C5PNZiOKg%3D)

### B. Image Augmentation

作者进一步利用RangeAugment自动调整亮度、对比度和中性噪点。作者使用PSNR指标，目标范围为\[20, 40\]，并采用余弦机制。由于在RangeAugment中，单个增强量的调整在训练过程中是动态的，因此它们不能作为数据集强化过程的一部分进行存储。

![picture.image](https://p6-volc-community-sign.byteimg.com/tos-cn-i-tlddhu82om/95885ba15a014e4e8ebc1a3c95d722f6~tplv-tlddhu82om-image.image?=&rk3s=8031ce6d&x-expires=1734167216&x-signature=I3Y%2Bqz3hLcsctcdsMKxPKI%2Bd1Lk%3D)

因此，作者只将其应用于输入到学生模型的图像。作者发现，如果将相同的增强应用于学生和老师（对于作者 的数据集强化方法来说是不切实际的），可以进一步获得改进（在ImageNet-val上的56.6% vs 55.9%）。

最后，作者考虑了RandomHorizontalFlip、RandomErasing和RandAugment，并发现只有RandAugment在作者的设置中是有利的。作者的强化数据集包括RRC和RandAugment的参数，并在训练时间内将RangeAugment应用于输入到学生模型的图像。

## Appendix C CLIP Ensembles

在这个部分，作者详细分析了CLIP集成。首先，作者展示了通过集成预训练的单个CLIP模型，可以构建出更准确的零样本模型。对于推理，作者将每个模态的归一化嵌入拼接在一起，然后进行重新归一化。

![picture.image](https://p6-volc-community-sign.byteimg.com/tos-cn-i-tlddhu82om/21f394b1f82945269511622582b322ba~tplv-tlddhu82om-image.image?=&rk3s=8031ce6d&x-expires=1734167216&x-signature=s0tjl1QUvxaY8RJSQZRCnBUJlIQ%3D)

在Table 12中，作者展示了作者从OpenCLIP\[28\]中挑选的一些CLIP集成模型（包括单个模型）的性能。显然，集成可以提高性能。

例如，来自datacomp\_x1\_s13b\_b90k和openai的ViT-L-14预训练CLIP模型集成的平均性能为67.3%，而每个个体模型的性能分别为66.3%和61.7%。此外，集成可以是一种更参数高效的途径来获得更强的模型。例如，两个ViT-L-14预训练CLIP模型的集成比具有ViT-bigG-14图像编码器的模型具有更少的参数，但具有相同的ImageNet-val性能（80.1%）。

总的来说，如果作者有一组预训练的CLIP模型（例如，像OpenCLIP中的那样），作者可以使用这种方法将状态与效果推向最先进，并获得更强的零样本性能。在这里，作者展示了一个由四个CLIP模型组成的集成可以达到81.7%的ImageNet-val零样本分类性能，而单个模型的性能并未超过80.1%。随着更强个体的模型变得公开，作者可以使用这种方法创建更强的集成。

![picture.image](https://p6-volc-community-sign.byteimg.com/tos-cn-i-tlddhu82om/294fde50b6da49b78589a147359c6670~tplv-tlddhu82om-image.image?=&rk3s=8031ce6d&x-expires=1734167216&x-signature=9lIw5B2bYkruAw2%2FcuuidCBNekE%3D)

在这项工作中，作者感兴趣的是创建一个强大的集成模型，在蒸馏的背景下作为老师使用。在Tab. 13中，作者展示了使用不同的CLIP模型作为教师的ViT-B/16 CLIP模型的性能。训练设置与Sec. 5.2相同，除了作者使用一个均匀采样的8M子集。类似于分类任务的常规蒸馏，作者观察到更准确的CLIP模型不一定是更好的教师。作者在数据集强化过程中选择了两个基于ViT-L-14的CLIP模型作为教师模型（用蓝色突出显示）。

## Appendix D Ablations on Lossy Compressions

在Table 2(c)中，作者展示了使用BFloat16压缩的DataCompDR-12M和DataCompDR-1B的存储大小。在本节中，作者进一步分析了通过减少增强数量（i）和损失压缩嵌入（ii）实现的存储减少。

![picture.image](https://p6-volc-community-sign.byteimg.com/tos-cn-i-tlddhu82om/1b792c19f64e4912909c632867375ce9~tplv-tlddhu82om-image.image?=&rk3s=8031ce6d&x-expires=1734167216&x-signature=0nPh%2BrwrJQM8%2BLTLp1b5Jly8jDU%3D)

作者在Table 14中报告了12.8k样本的DataCompDR的存储大小。DataCompDR-12M的存储大小可以通过将数字乘以1000（TBs而不是GBs）和10的5次方（DataCompDR-1B）得出。

![picture.image](https://p6-volc-community-sign.byteimg.com/tos-cn-i-tlddhu82om/1de62092afa34935ab746c5fc3844f6b~tplv-tlddhu82om-image.image?=&rk3s=8031ce6d&x-expires=1734167216&x-signature=glpKzSAEMQxV6rLUsWmjFTX4rLo%3D)

Table 15表明，使用BFloat16嵌入训练时，训练准确率在DataComp-12M训练的标准偏差范围内。

## Appendix E Hybrid Text Encoder

在这个ablation中，作者研究了混合文本编码器的内核尺寸。为此，作者使用了一个6层的完全卷积文本编码器，并系统地增加内核大小。作者使用ViT-B/16作为图像编码器。这些模型在DataCompDR-12M上进行了30k迭代训练。

![picture.image](https://p6-volc-community-sign.byteimg.com/tos-cn-i-tlddhu82om/f7a833d06e584bfe8103a98ca522394c~tplv-tlddhu82om-image.image?=&rk3s=8031ce6d&x-expires=1734167216&x-signature=ksoBn%2F6xUb%2FrBlNqDxngLAN5d%2B0%3D)

从Tab. 16中，作者可以看到，随着内核尺寸的增加，零样本IN-val性能确实有所提高，但运行模型在移动设备上的成本显著增加。对于零样本IN-val性能提高1.1%，模型运行速度将慢4.5倍。从Tab. 16中，内核大小为11时，在准确性和延迟之间取得了最佳权衡。

## Appendix F Extended Results

在这个部分，作者提供了作者提出的CLIP模型家族MobileCLIP-S0、MobileCLIP-S1、MobileCLIP-S2和MobileCLIP-B的扩展零样本结果。零样本分类和检索结果在Tab. 17中提供。作者还包括了一些相关工作的附加结果，其中只有部分评估可用。

![picture.image](https://p6-volc-community-sign.byteimg.com/tos-cn-i-tlddhu82om/f6a350e81cf24f1e93f2298caaadc4cd~tplv-tlddhu82om-image.image?=&rk3s=8031ce6d&x-expires=1734167216&x-signature=SR6sv1NmNxXSSUiNkvsqPd0S7sY%3D)

![picture.image](https://p6-volc-community-sign.byteimg.com/tos-cn-i-tlddhu82om/9d1b3e34fa2f4f5db16cec1503a5c3d3~tplv-tlddhu82om-image.image?=&rk3s=8031ce6d&x-expires=1734167216&x-signature=4SfOXmE7wBM2Bh%2BLeZtVpYJhzvs%3D)

扫码加入👉「集智书童」交流群

（备注： 方向+学校/公司+昵称 ）

![picture.image](https://p6-volc-community-sign.byteimg.com/tos-cn-i-tlddhu82om/e7a20b372b2d4c9f83558377cfa050ab~tplv-tlddhu82om-image.image?=&rk3s=8031ce6d&x-expires=1734167216&x-signature=q3gddyuaHabshHHGdz96cixRwFs%3D)

![picture.image](https://p6-volc-community-sign.byteimg.com/tos-cn-i-tlddhu82om/479b8b020d2c469b9e61f58d2fbd6516~tplv-tlddhu82om-image.image?=&rk3s=8031ce6d&x-expires=1734167216&x-signature=y0l0Z7eOUCTdsBrARsGIZohFkPU%3D)

![picture.image](https://p6-volc-community-sign.byteimg.com/tos-cn-i-tlddhu82om/5f6aacf84982468d8547c048bf379269~tplv-tlddhu82om-image.image?=&rk3s=8031ce6d&x-expires=1734167216&x-signature=E8909BRl54Mo7i%2B2H17TNrXwII0%3D)

![picture.image](https://p6-volc-community-sign.byteimg.com/tos-cn-i-tlddhu82om/1cf1f0725f8d4602958d739185d3d800~tplv-tlddhu82om-image.image?=&rk3s=8031ce6d&x-expires=1734167216&x-signature=YmFpLkUP6dyTO4fjJC9XrxBipXo%3D)

![picture.image](https://p6-volc-community-sign.byteimg.com/tos-cn-i-tlddhu82om/1728c0cd83cc44eeba90708a98dc8a8c~tplv-tlddhu82om-image.image?=&rk3s=8031ce6d&x-expires=1734167216&x-signature=KawGzyL0UnlmR%2FxYIf47ZWhOrpw%3D)

![picture.image](https://p6-volc-community-sign.byteimg.com/tos-cn-i-tlddhu82om/10f54cd3bf9a450bbf2219dcb0f1eb04~tplv-tlddhu82om-image.image?=&rk3s=8031ce6d&x-expires=1734167216&x-signature=teDmL%2FRR%2B2oTNoti%2BSiF16%2FQm%2FI%3D)

想要了解更多：

前沿AI视觉感知全栈知识👉「分类、检测、分割、关键点、车道线检测、3D视觉（分割、检测）、多模态、目标跟踪、NerF」

**行业技术方案** 👉「AI安防、AI医疗、AI自动驾驶」

**AI模型部署落地实战** 👉「CUDA、TensorRT、NCNN、OpenVINO、MNN、ONNXRuntime以及地平线框架」

欢迎扫描上方二维码，加入「 **集智书童-知识星球** 」，日常分享论文、学习笔记、问题解决方案、部署方案以及全栈式答疑，期待交流！

免责声明

凡本公众号注明“来源：XXX（非集智书童）”的作品，均转载自其它媒体，版权归原作者所有，如有侵权请联系我们删除，谢谢。

点击下方“ **阅读原文** ”，

了解更多AI学习路上的 「武功秘籍」