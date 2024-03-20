![](https://mmbiz.qpic.cn/mmbiz_jpg/3zd5t92QHVWLeDfKEEhACuCXiakuasnKWicpjkbB3b77fiaJkmEPYdF6ftrpNh3S6icIvStM4jbAfdjQ8UgSicHkfwg/640?wx_fmt=jpeg&from=appmsg)

> 尽管视觉Transformer（ViT）在计算机视觉中已经取得了显著的成就，但由于缺乏内部斑块信息交互和特征尺度多样性有限，它在密集预测任务中的表现并不理想。大多数现有研究致力于设计视觉特定的Transformer来解决上述问题，这引入了额外的预训练成本。因此，我们提出了一种简单的、无需预训练的、特征增强的ViT主干网络，具有卷积多尺度特征交互功能，名为ViT-CoMer，它促进了CNN与Transformer之间的双向交互。
> 
> 与最先进的方法相比，ViT-CoMer具有以下优点：
> 
> 1.  将空间金字塔多接收域卷积特征注入到ViT架构中，有效缓解了ViT中局部信息交互有限和单一特征表示的问题。
>     
> 2.  提出了一种简单高效的CNN-Transformer双向融合交互模块，它能够在不同层次的特征之间进行多尺度融合，有利于处理密集预测任务。
>     
> 3.  在各种密集预测任务上评估了ViT-CoMer的性能，包括不同的框架和多种先进的预训练方法。
>     
> 
> 值得注意的是，ViT-CoMer-L在COCO val2017上达到了64.3% AP（无需额外训练数据），在ADE20K val上达到了62.1% mIoU，这两个结果都与最先进的方法相当。我们希望ViT-CoMer可以作为密集预测任务的新主干网络，以促进未来的研究。
> 
> 代码：https://github.com/Traffic-X/ViT-CoMer

1 Introduction
--------------

近年来，由于大规模数据集的发布和深度学习的发展，在诸如目标检测、实例分割和语义分割等密集预测任务上已取得显著进展（例如YOLO系列、RCNN系列、DETR）。这一进展催生了许多经典的卷积神经网络（CNNs），包括ResNet、ConvNeXt等。这些模型利用了卷积神经网络的局部连续性和多尺度能力，使它们能有效应用于密集预测任务。

同时，受到Transformer在自然语言处理中成功的启发，视觉Transformer（ViT）作为将Transformer应用于视觉任务的开创性方法，受到了广泛关注。目前，专为密集预测任务设计的基于Transformer的网络架构主要分为三种范式：普通Backbone网、视觉特定Backbone网和适应性Backbone网，如图2所示。

![](https://mmbiz.qpic.cn/mmbiz_jpg/3zd5t92QHVWLeDfKEEhACuCXiakuasnKWKXf79GulJEQKF1u2LZdHM1AeXcdI8ZotHS7P0pF3rbFunKJZfsQKHA/640?wx_fmt=jpeg&from=appmsg)

普通Backbone网在不改变ViT框架的情况下优化使用ViT特征，如图2(a)所示的ViTDet。视觉特定Backbone网（例如Swin、CMT、MPViT、PVT系列）结合了CNN和Transformer的优势，重新设计网络结构，这帮助它们在密集预测任务中实现更好的性能，如图2(b)所示。图2(c)中的适应性Backbone网基于普通ViT，仅通过增加额外分支引入CNN特征，并可直接加载各种开源和强大的ViT预训练权重，以提高在密集预测任务上的ViT性能。

在本研究中提出了一种普通、无需预训练且功能增强的ViTBackbone网，命名为ViT-CoMer，它可以直接加载各种开源和先进的预训练权重。具体而言，设计了两个核心模块：多感受野特征金字塔模块（MRFP）和CNN-Transformer双向融合交互模块（CTI）。

MRFP可以为ViT补充更丰富的多尺度空间信息；CTI可以融合来自CNN和Transformer的多尺度特征，使模型具备更强大的特征表示能力。在ViT-CoMer的权重初始化过程中，ViT模块直接使用开源预训练权重，其余部分使用随机初始化。

![](https://mmbiz.qpic.cn/mmbiz_jpg/3zd5t92QHVWLeDfKEEhACuCXiakuasnKWrlyIiciaGvTIYw4lvCS7Svu6NghfBuicWTfF5oodFhb81UVDWibPIRib7pw/640?wx_fmt=jpeg&from=appmsg)

如图1所示，当使用ViT的高级预训练权重时，ViT-CoMer表现更佳。

主要贡献如下：

1.  提出了一种新颖的密集预测Backbone网络，该网络通过将普通ViT与CNN特征相结合。它有效地利用了各种开源预训练的ViT权重，并融入了空间金字塔卷积特征，以解决局部ViT特征之间缺乏交互作用以及单一尺度表示的挑战。
    
2.  设计了一个多接收域特征金字塔模块和一个CNN-Transformer双向融合交互模块。前者能够捕捉各种空间特征，后者在层次化特征之间进行多尺度融合以获得更丰富的语义信息，这有利于处理密集预测任务。
    
3.  ViT-CoMer在多个具有挑战性的密集预测基准测试上进行评估，包括目标检测、实例分割和语义分割。实验结果证明了ViT-CoMer显著提升了普通ViT的能力。特别是，当利用先进的开源预训练方法如DINOv2时，在公平的比较条件下，ViT-CoMer可以持续超越现有最佳（SOTA）方法。
    
4.  值得注意的是，采用先进预训练的ViT-CoMer-L模型，在COCO val2017上达到了64.3% AP，这是在不使用额外检测数据（例如Objects365）情况下的最佳记录。此外，ViT-CoMer-L在ADE20K val上的mIoU达到了\*\*62.1%\*\*，与现有最佳方法相当。
    

2 Related Work
--------------

### Plain backbones

ViT 是首次将Transformer（transformer）结构引入到图像分类任务中并取得显著成果的工作。ViTDet 是基于 ViT 的一个简单、非层次化的检测器，通过融入一个简单的特征金字塔模块。然而，ViTDet 与最先进方法相比，性能上存在差距。一个可能的原因是 ViT 的特征表示可能不够丰富。

尽管如此，密集预测模型需要对多尺度感知具有强大的能力。ViT-CoMer将多尺度增强的卷积特征与 ViT 特征结合，使模型在处理密集预测任务时能够提取丰富的多尺度特征。

### Vision-specific backbones

视觉特定的Backbone网络主要是为了缓解ViT（视觉Transformer）中的挑战，比如非层次化特征以及局部特征间缺乏交互。Swin Transformer 采用移位窗口来缓解ViT中局部信息交互的不足。

同时，它构建了多尺度特征以适应密集预测。PVT 构建了一个特征金字塔结构，以解决ViT中单尺度特征的局限性，简化了Transformer的结构，有效地降低了计算复杂性。MixFormer 利用双向特征交互算子结合卷积和自注意力来增强特征表示。iFormer 分析了CNN和Transformer架构在高频和低频上的优势。MetaFormer 引入了一种通用的层次化网络架构，该架构使用池化而不是注意力，在各种视觉任务中取得了良好的结果。

UniFormer 在一个块内级联CNN和注意力，集成了CNN和Transformer的优点。视觉特定的Backbone网络改变了ViT的结构，这使得它们不能直接使用现有的强大预训练权重，例如BEiT系列。ViT-CoMer保留了原始的ViT，允许它直接加载基于ViT的开源预训练权重。这使得ViT-CoMer能够快速获得增强的泛化性能。

### Adapted backbones

ViT-Adapter 提出了一种集成空间先验信息的ViT框架。它利用了ViT预训练权重的优势。在训练过程中，ViT-adapter需要进行完全微调，从而在密集预测任务中取得令人印象深刻的性能。同时，它缺乏空间先验信息之间的特征交互。

VPT 引入了一种方法，在训练过程中冻结ViT的预训练权重，只更新适配器模块的参数。尽管这种方法在某些任务中可以产生与完全微调方法相当的结果，但在语义分割方面并不如完全微调表现得好。LoRand 也是一种保留ViT权重、只训练适配器模块的算法，它只需要训练总体训练参数的1%-3%。然而，其性能并不如完全微调方法有效。ViT-CoMer通过特征融合增强空间层次特征，并在训练期间采用完全微调方法来优化模型，这有效地提升了模型的性能。

3 The ViT-CoMer Method
----------------------

### Overall Architecture

ViT-CoMer的总体架构如图3所示，包括三个部分：

1.  纯粹的ViT
    
2.  多感受野特征金字塔模块（MRFP）
    
3.  CNN-Transformer双向融合交互模块（CTI）
    

![](https://mmbiz.qpic.cn/mmbiz_jpg/3zd5t92QHVWLeDfKEEhACuCXiakuasnKWvibrbw3V4NyR88vNDqFNg1Bwv8JIdUXEzEGZskRqGqWdSFxFAcjAibdw/640?wx_fmt=jpeg&from=appmsg)

*   首先，对于ViT分支（见图3(a)），将形状为的输入图像送入块嵌入以获得原图像分辨率的特征。同时，对于另一个分支，此图像通过一系列卷积层以获得具有、和分辨率的特征金字塔、和，每个都包含D维特征图。
    
*   其次，两个分支的特征都通过N阶段的特征交互。在每一阶段，特征金字塔首先通过MRFP模块进行增强，然后通过CTI模块与ViT的特征进行双向交互，这可以获得具有丰富语义信息的多尺度特征。CTI在每个阶段的开始和结束时进行操作。经过N阶段的特征交互后，两个分支的特征在每个尺度上相加，用于密集预测任务。
    

### Multi-Receptive Field Feature Pyramid

多接收域特征金字塔模块包括一个特征金字塔和多接收域卷积层。特征金字塔可以提供丰富的多尺度信息，而后者通过不同的卷积核扩展接收域，增强了CNN特征的长距离建模能力。该模块如图4所示。

![](https://mmbiz.qpic.cn/mmbiz_jpg/3zd5t92QHVWLeDfKEEhACuCXiakuasnKWo37ea08p3XmBRRjMcRumx827Yb3ia79ZgrmicHdFqhwLjLErZnS0WicDA/640?wx_fmt=jpeg&from=appmsg)

MRFP由两个线性投影层和一组具有多接收域的深度可分卷积组成。具体来说，模块的输入是一组多尺度特征 ，将这些特征图展平并拼接成特征标记 ，这些特征首先通过一个线性投影层以获得维度降低的特征，然后特征在通道维度上被分为 组。不同的特征组对应于具有不同接收域的卷积层（例如，）。最后，经过处理后的特征通过线性投影层进行拼接并维度增加。这个过程可以表示为：

其中 代表线性投影， 是一系列具有不同核大小的深度可分卷积。

### CNN-Transformer Bidirectional Fusion Interaction

我们提出了一种名为CTI的跨架构特征融合方法，如图5所示。

![](https://mmbiz.qpic.cn/mmbiz_jpg/3zd5t92QHVWLeDfKEEhACuCXiakuasnKWhj25aDpwia3TxXoUNDia8DNu3ibCQdv5zbrtapYQUlZscKZqxImH4Va7Q/640?wx_fmt=jpeg&from=appmsg)

该方法在不改变ViT结构的前提下引入了CNN的多尺度特征。同时，通过双向互动，我们缓解了ViT中内斑块信息交互不足和非层次化特征的问题，同时进一步增强了CNN的长距离建模能力和语义表示。为了融合ViT特征和通过MRFP模块获得的多尺度特征，可以表示为。

我们直接将特征和相加，得到集合，表示为，它从不同的架构中聚合了多尺度特征。然而，由于架构差异，它们在模态表示上存在偏差（例如，高低频语义，以及全局-局部信息）。为了解决这个问题，采用自注意力机制来统一CNN和Transformer特征，加强了对模态差异的表示不变性。这个过程可以描述为：

在文中， 表示的是层归一化， 是多尺度可变形注意力，而 是前馈网络。最后，我们通过双线性插值将 和 的特征图大小对齐到 ，并将 作为下一个ViT层的输入。此外，由于 包含了 和 分辨率的多尺度特征，自注意力可以促进多尺度特征之间的互动，并使模型能够更好地捕捉图像中的多尺度信息。这不同于传统的Transformer架构，后者仅在单一尺度特征上使用自注意力。通过有效地融合多尺度CNN和Transformer特征，模型获得了增强的建模能力。

关于在不同架构间融合特征，采用了双向互动的方式来更新ViT和CNN分支的特征。具体来说，对于第阶段，在阶段的开始，两个分支的特征被融合，然后将融合后的特征注入到ViT分支中。这个过程可以表述为：

其中 是 ViT 分支的更新特征， 是一个初始化为零的可学习变量，它最小化了在早期训练过程中随机初始化的 CNN 架构对 ViT 的影响。在第 阶段结束时，重复该过程将特征注入到 CNN 分支，表示为：

其中 是卷积神经网络分支的更新特征，阶段的数量 是根据 ViT 的深度来确定的。跨架构特征融合和双向互动使得能够利用来自多尺度和多级别的特征，增强了模型的表达性和泛化能力。同时，所提出的组件可以轻松地整合到其他高级模型中，并且在密集预测任务中表现更佳。

4 Experiment
------------

我们选择了密集预测中的典型任务：目标检测、实例分割和语义分割，并在COCO和ADE20K数据集上进行了大量实验（涉及不同的模型大小、算法框架和配置），以验证ViT-CoMer的有效性。

同时，使用了各种预训练的ViT，包括在ImageNet-1K、ImageNet-22K和多模态数据上预训练的权重。ViT-CoMer取得了优于现有基于SOTA ViT的方法（例如，ViTDet，ViT-Adapter）且与视觉特定的高级方法相当的结果。

此外，对设计的模块进行了消融实验，并对密集预测任务进行了定性实验。这些结果表明，ViT-CoMer能够提升普通ViT的性能，达到优越的表现，并且可以作为健壮的Backbone网络迁移到各种密集预测任务框架中。

### Object Detection and Instance Segmentation

设置. 采用MMDetection框架来实现ViT-CoMer，并在COCO数据集上进行目标检测和实例分割实验。

目标检测和实例分割框架包括Mask R-CNN、Cascade Mask R-CNN、ATSS和GFL。参考PVT，进行了1x（12个周期）或3x（36个周期）的训练计划实验。我们使用的总批处理大小为16，采用AdamW优化器，学习率为，权重衰减为0.05。

与不同基础架构和框架的比较。 表1展示了在Mask R-CNN 1x和3x计划下，ViT-CoMer与不同规模的普通ViT、视觉特定和适应型基础架构之间的比较。

![](https://mmbiz.qpic.cn/mmbiz_jpg/3zd5t92QHVWLeDfKEEhACuCXiakuasnKWaZRxT7Ae1va1Arib1VjonU248vsbxfKUO7qwn2jTKEyUuQxavos6pgQ/640?wx_fmt=jpeg&from=appmsg)

可以看出，在类似模型尺寸下，ViT-CoMer在COCO目标检测和实例分割这两个典型密集预测任务中，表现优于其他基础架构。例如，与1x（3x）计划下的普通ViT-S相比，ViT-CoMer-S在box mAP上显著提高了+5.6% (+4.8%)，在mask mAP上提高了+3.4% (+3.1%)。在仅使用的参数情况下，ViT-CoMer-S相比于ViT-L取得了更优的检测结果。此外，ViT-CoMer与视觉特定和适应型基础架构，如InternImage和ViT-Adapter，相比仍然显示出显著的改进。

我们进一步使用不同的检测框架评估了ViT-CoMer，结果如表2所示。可以看出，ViT-CoMer在各个框架、模型大小和配置中一致地优于其他基础网络。

![](https://mmbiz.qpic.cn/mmbiz_jpg/3zd5t92QHVWLeDfKEEhACuCXiakuasnKW1EWU8JHVJicvndicQ8cUIz2l3HlS14obMA9YiaXG27t2maPFQJcfNibFuw/640?wx_fmt=jpeg&from=appmsg)

不同预训练权重的结果。 在不同的预训练权重下对Mask R-CNN（3倍计划）进行了实验，结果如表3所示。

![](https://mmbiz.qpic.cn/mmbiz_jpg/3zd5t92QHVWLeDfKEEhACuCXiakuasnKWbXkYMFd58FqpVWOhRl4vqr4yXZw1u9q0B9M5SWjbVt6Psx1ibF2gbDw/640?wx_fmt=jpeg&from=appmsg)

具体来说，具有多模态预训练的ViT-CoMer-B，与ImageNet-1K相比，可以达到 +1.7% 和 +1.7% 的提升。此外，我们在ViT-CoMer-L上比较了更多的预训练，其中自监督预训练取得了显著的结果。与ImageNet-22K预训练相比，它实现了 +3.0% 和 +2.7% 的提升。这些结果表明，ViT-CoMer可以轻松利用各种开源大规模预训练来提高下游任务的性能。

与现有技术水平比较。 为了进一步提高性能基于Co-DETR 进行了实验，使用ViT-CoMer作为基础模型，并以多模态预训练的BEiTv2初始化模型。

![](https://mmbiz.qpic.cn/mmbiz_jpg/3zd5t92QHVWLeDfKEEhACuCXiakuasnKWcTRwXYiaIAaMeRaPibBqXVRaPkJOzicHicqXyhNaoibIVcL3OWlZ5Ynic45g/640?wx_fmt=jpeg&from=appmsg)

如表4所示，在COCO val2017上，ViT-CoMer在不使用额外训练数据的情况下，超越了现有的SOTA算法，这充分证明了ViT-CoMer的有效性。

### Semantic Segmentation

设置。 我们的语义分割实验是基于ADE20K数据集，并使用MMSegmentation。选择UperNet作为基本框架。训练配置与Swin保持一致，包括进行160,000次迭代训练。批量大小设置为16，并使用AdamW优化器。学习率和权重衰减参数分别调至和0.05。

与不同Backbone网络的比较。 表5展示了在单尺度与多尺度mIoU方面，ViT-CoMer与各种Backbone网络的对比，包括普通ViT、针对视觉特化的Backbone网络，以及在语义分割任务中适配的Backbone网络。

![](https://mmbiz.qpic.cn/mmbiz_jpg/3zd5t92QHVWLeDfKEEhACuCXiakuasnKWKyjC7iaAqicuFansBWspANncLibtHqw7EeHLWEsuH13958exT0vIS6eAw/640?wx_fmt=jpeg&from=appmsg)

结果显示，在可比的模型大小下，ViT-CoMer超过了ViT和许多视觉特化的Backbone网络。例如，ViT-CoMer-S实现了47.7%的多尺度mIoU，优于许多强大的竞争对手，如Swin-T（高出1.9%）和ViT-Adapter-S（高出0.6%）。同样，ViT-CoMer-L报告了具有竞争力的55.6%的多尺度mIoU，这比Swin-L高出2.1%，比ViT-Adapter-L高出1.2%。这些公平的比较验证了ViT-CoMer在语义分割任务中的有效性和普遍性。

不同预训练权重的比较。 表6展示了在UperNet上使用不同预训练权重的结果。

![](https://mmbiz.qpic.cn/mmbiz_jpg/3zd5t92QHVWLeDfKEEhACuCXiakuasnKWHFibsC5I00BmAaKv7Hks3uaticEUv3fwJQIgZ0LZQt8jicbQchw3ic7W1g/640?wx_fmt=jpeg&from=appmsg)

当使用ImageNet-22K预训练权重时，ViT-CoMer-L达到了55.6%的MS mIoU，比ViT-Adapter-L高出1.2%的mIoU。接着，使用多模态预训练来初始化ViT-CoMer-L，这使ViT-CoMer获得了令人印象深刻的2.0%的mIoU提升，比ViT-Adapter-L高出1.4%。这些重大而一致的改进表明，ViT-CoMer可以有效提高普通的ViT，并充分利用各种基于ViT的开源预训练权重，使模型在语义分割方面表现更佳。

与最先进技术的比较。 为了进一步提升性能，基于Mask2Former 进行了实验，使用ViT-CoMer作为基础模型，并以多模态预训练的BEiTv2初始化模型。

![](https://mmbiz.qpic.cn/mmbiz_jpg/3zd5t92QHVWLeDfKEEhACuCXiakuasnKWDdchMNaGib7Fr1JYvd9w7EBN30TdwEs6RGn0A2UsXuPOpXmdmtlNOXw/640?wx_fmt=jpeg&from=appmsg)

如表7所示，在ADE20K数据集上，ViT-CoMer以更少的参数达到了与SOTA方法相当的性能。具体的实现细节在补充材料中提供。

### Ablation Study

设置。 在ViT-CoMer-S上进行了消融实验，使用Mask R-CNN（1x计划）进行目标检测和实例分割任务。在训练过程中使用的总批处理大小为16，所采用的优化器是AdamW，学习率参数和权重衰减分别设置为和0.05。

组件消融研究。 我们逐步将提出的子模块添加到ViT-S中，最终将模型发展成ViT-CoMer。消融实验的结果展示在表8中。当使用MRFP为普通ViT提供CNN的多尺度和多感受野特征（特征直接相加）时，可以带来1.3% 和1.1% 的提升。

![](https://mmbiz.qpic.cn/mmbiz_jpg/3zd5t92QHVWLeDfKEEhACuCXiakuasnKWv7T9RTPGtCF2lpvibVuuAezXe6Agia4icKH1A46j6rHEDjrFZNHL8YLTA/640?wx_fmt=jpeg&from=appmsg)

此外，我们将“直接相加”的操作替换为本工作中提出的CTI。当仅使用CTI（至V）时，模型提升了1.8% 和1.1% ；当同时使用CTI（至V）和CTI（至C）时，性能进一步显著提升，分别提高了2.5% 和1.2% 。总的来说，相比于普通ViT，ViT-CoMer取得了显著的提升，分别达到5.6% 和3.4% 。实验结果证明，我们提出的MRFP和CTI模块可以显著增强普通ViT的能力，使其更好地适应密集预测任务。

双向融合交互的数量。 在表9中分析了双向融合交互模块数量对模型的影响。观察到随着N的增加，模型准确性达到了一个平台期，引入更多的交互作用并不始终能提升性能。因此，我们默认将N设置为4。

![](https://mmbiz.qpic.cn/mmbiz_jpg/3zd5t92QHVWLeDfKEEhACuCXiakuasnKWspxGoibIYmTyHbyeXam2PCfDnt50Tk5ohQC6QFAWcGJxzRXL9K46N5g/640?wx_fmt=jpeg&from=appmsg)

MRFP中不同核尺寸的影响。 表10展示了不同核尺寸对MRFP的影响。结果显示，随着核尺寸的增加，参数数量也随之增加。同时，我们观察到在使用3和5尺寸的核时， 和 达到峰值，因此我们采用这些尺寸作为默认设置。

![](https://mmbiz.qpic.cn/mmbiz_jpg/3zd5t92QHVWLeDfKEEhACuCXiakuasnKWOiaXXXq5xSz5cOIXibqQ2UooP9KUmLA5ibhBEJzAuO0H4NovOwAgjTWicg/640?wx_fmt=jpeg&from=appmsg)

### Scalability

ViT-CoMer也可以应用于具有层次结构的视觉变换器，如Swin。我们将该方法应用于带有Mask R-CNN（1x计划）的Swin-T进行目标检测。

![](https://mmbiz.qpic.cn/mmbiz_jpg/3zd5t92QHVWLeDfKEEhACuCXiakuasnKWx8BAYibxyBZsW7Npw1xuyDbyF8CwwSLz6XibXuiaIZGSHGDjk5nYFVNoQ/640?wx_fmt=jpeg&from=appmsg)

如表11所示，ViT-CoMer将Swin-T的box AP提高了+2.1%，mask AP提高了+1.2%。由于Swin架构已经引入了归纳偏置，与普通的ViT相比，改进幅度相对较小。尽管如此，这些结果仍然证实了ViT-CoMer的可扩展性。

### Qualitative Results

根据iFormer的研究，普通的ViT（视觉Transformer）由于自注意力操作倾向于捕捉图像中的全局和低频特征，而CNN（卷积神经网络）则由于卷积操作倾向于捕捉图像中的局部和高频特征。然而，在密集预测任务中，图像中会出现不同大小和密度各种物体，这要求模型具备同时提取和捕捉局部和全局、高频和低频特征的能力。我们通过在不同层次（下采样、、和）上对实例分割和目标检测任务进行特征图的视觉化，定性地评估了普通ViT和所提出的ViT-CoMer之间的差异。

![](https://mmbiz.qpic.cn/mmbiz_jpg/3zd5t92QHVWLeDfKEEhACuCXiakuasnKWF4Dej6Md6hiaQfruOtqJFia5FRwDJLTFgqibXibLwcYj4QDz24iaCF3EAOA/640?wx_fmt=jpeg&from=appmsg)

定性可视化结果如图6所示。可以看出，与普通ViT相比，ViT-CoMer生成了更加细致的多尺度特征，从而增强了模型的物体定位能力。

5 Conclusion
------------

在这项工作中，我们提出了ViT-CoMer，一个简单、非层次化且特征增强的ViT主干网络，它有效地结合了CNN和Transformer的优势。在不改变ViT架构的情况下，我们集成了一个多尺度卷积特征交互模块以重建细粒度的层次语义特征。我们对ViT-CoMer在密集预测任务上进行了验证，包括目标检测、实例分割和语义分割。

大量实验表明，与纯和适配的主干网络相比，ViT-CoMer可以取得更优的性能。此外，ViT-CoMer可以轻松获得先进的ViT预训练权重，并且与最先进的主干网络相比，取得可比较甚至超越的性能。

参考
--

\[1\].ViT-CoMer: Vision Transformer with Convolutional Multi-scale Feature Interaction for Dense Predictions.