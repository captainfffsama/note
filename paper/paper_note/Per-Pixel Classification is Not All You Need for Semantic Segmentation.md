#图像分割

# Per-Pixel Classification is Not All You Need for Semantic Segmentation

- 作者:
	1. Bowen Cheng 
	2. Alexander G. Schwing  
	3. Alexander Kirillov
- 机构:
	1. Facebook AI Research (FAIR)   
	2. University of Illinois at Urbana-Champaign (UIUC)
- 代码: <https://github.com/facebookresearch/MaskFormer>
- 文章:<https://link.zhihu.com/?target=http%3A//arxiv.org/abs/2107.06278>
- 相关解读:
	- <https://zhuanlan.zhihu.com/p/389457610>
	- <https://www.zhihu.com/question/472122951>
	- <https://medium.com/@HannaMergui/maskformer-per-pixel-classification-is-not-all-you-need-for-semantic-segmentation-1e2fe3bf31cb>

## 摘要

Modern approaches typically formulate semantic segmentation as a _per-pixel classification_ task, while instance-level segmentation is handled with an alternative _mask classification_. Our key insight: mask classification is sufficiently general to solve both semantic- and instance-level segmentation tasks in a unified manner using the exact same model, loss, and training procedure. Following this observation, we propose MaskFormer, a simple mask classification model which predicts a set of binary masks, each associated with a _single_ global class label prediction. Overall, the proposed mask classification-based method simplifies the landscape of effective approaches to semantic and panoptic segmentation tasks and shows excellent empirical results. In particular, we observe that MaskFormer outperforms per-pixel classification baselines when the number of classes is large. Our mask classification-based method outperforms both current state-of-the-art semantic (55.6 mIoU on ADE20K) and panoptic segmentation (52.7 PQ on COCO) models.

Project page: [https://bowenc0221.github.io/maskformer](https://bowenc0221.github.io/maskformer)  
现代方法通常将语义分割表述为每像素分类任务，而实例级分割则使用替代掩码分类进行处理。我们的主要见解是：掩码分类足够通用，可以使用完全相同的模型、损失和训练过程以统一的方式解决语义级和实例级分割任务。根据这一观察结果，我们提出了 MaskFormer，这是一个简单的掩码分类模型，它预测一组二进制掩码，每个掩码都与单个全局类标签预测相关联。总体而言，所提出的基于掩模分类的方法简化了语义和全景分割任务的有效方法的格局，并显示出优异的实证结果。特别是，我们观察到，当类数量较大时，MaskFormer 的性能优于每像素分类基线。我们基于掩模分类的方法优于当前最先进的语义（ADE20K 上的 55.6 mIoU）和全景分割（COCO 上的 52.7 PQ）模型。 

## 1 Introduction1 引言

The goal of semantic segmentation is to partition an image into regions with different semantic categories. Starting from Fully Convolutional Networks (FCNs) work of Long _et al_. \[[30](#bib.bib30)\], most _deep learning-based_ semantic segmentation approaches formulate semantic segmentation as _per-pixel classification_ (Figure [1](#S1.F1 "Figure 1 ‣ 1 Introduction ‣ Per-Pixel Classification is Not All You Need for Semantic Segmentation") left), applying a classification loss to each output pixel \[[9](#bib.bib9), [52](#bib.bib52)\]. Per-pixel predictions in this formulation naturally partition an image into regions of different classes.  
语义分割的目标是将图像划分为具有不同语义类别的区域。从 Long 等人\[30\] 的全卷积网络（FCN）工作开始，大多数基于深度学习的语义分割方法将语义分割表述为每像素分类（左图 1），对每个输出像素应用分类损失\[9,52\]。此公式中的每像素预测自然地将图像划分为不同类别的区域。

Mask classification is an alternative paradigm that disentangles the image partitioning and classification aspects of segmentation. Instead of classifying each pixel, mask classification-based methods predict a set of binary masks, each associated with a _single_ class prediction (Figure [1](#S1.F1 "Figure 1 ‣ 1 Introduction ‣ Per-Pixel Classification is Not All You Need for Semantic Segmentation") right). The more flexible mask classification dominates the field of instance-level segmentation. Both Mask R-CNN \[[21](#bib.bib21)\] and DETR \[[4](#bib.bib4)\] yield a single class prediction per segment for instance and panoptic segmentation. In contrast, per-pixel classification assumes a static number of outputs and cannot return a variable number of predicted regions/segments, which is required for instance-level tasks.  
掩码分类是一种替代范式，它解开了分割的图像分区和分类方面。基于掩码分类的方法不是对每个像素进行分类，而是预测一组二进制掩码，每个掩码都与单个类预测相关联（右图 1）。更灵活的掩码分类在实例级分割领域占主导地位。Mask R-CNN \[21\] 和 DETR \[4\] 都对每个段产生一个类别预测，例如全景分割。相比之下，每像素分类假定输出数量为静态数，并且无法返回可变数量的预测区域/段，而这是实例级任务所必需的。

![](https://ar5iv.labs.arxiv.org/html/2107.06278/assets/x1.png)

Figure 1: Per-pixel classification _vs_. mask classification. (left) Semantic segmentation with per-pixel classification applies the same classification loss to each location. (right) Mask classification predicts a set of binary masks and assigns a single class to each mask. Each prediction is supervised with a per-pixel binary mask loss and a classification loss. Matching between the set of predictions and ground truth segments can be done either via _bipartite matching_ similarly to DETR \[[4](#bib.bib4)\] or by _fixed matching_ via direct indexing if the number of predictions and classes match, _i.e_., if N=K𝑁𝐾N=K.  
图 1：每像素分类与蒙版分类。（左）使用每像素分类的语义分割将相同的分类损失应用于每个位置。（右）掩码分类预测一组二进制掩码，并为每个掩码分配一个类。每个预测都通过每像素二进制掩码损失和分类损失进行监督。预测集和真实线段之间的匹配可以通过类似于 DETR \[4\] 的二分匹配来完成，或者如果预测和类的数量匹配，即如果 N=K𝑁𝐾N=K .

Our key observation: mask classification is sufficiently general to solve both semantic- and instance-level segmentation tasks. In fact, before FCN \[[30](#bib.bib30)\], the best performing semantic segmentation methods like O2P \[[5](#bib.bib5)\] and SDS \[[20](#bib.bib20)\] used a mask classification formulation. Given this perspective, a natural question emerges: _can a single mask classification model simplify the landscape of effective approaches to semantic- and instance-level segmentation tasks? And can such a mask classification model outperform existing per-pixel classification methods for semantic segmentation?_  
我们的主要观察结果是：掩码分类足够通用，可以解决语义级和实例级的分割任务。事实上，在 FCN \[30\] 之前，性能最好的语义分割方法，如 O2P \[5\] 和 SDS \[20\]，都使用了掩码分类公式。从这个角度来看，一个自然而然的问题出现了：单个掩码分类模型能否简化语义级和实例级分割任务的有效方法的格局？这样的掩码分类模型能否优于现有的每像素分类方法进行语义分割？

To address both questions we propose a simple MaskFormer approach that seamlessly converts any existing per-pixel classification model into a mask classification. Using the set prediction mechanism proposed in DETR \[[4](#bib.bib4)\], MaskFormer employs a Transformer decoder \[[41](#bib.bib41)\] to compute a set of pairs, each consisting of a class prediction and a mask embedding vector. The mask embedding vector is used to get the binary mask prediction via a dot product with the per-pixel embedding obtained from an underlying fully-convolutional network. The new model solves both semantic- and instance-level segmentation tasks in a unified manner: no changes to the model, losses, and training procedure are required. Specifically, for semantic and panoptic segmentation tasks alike, MaskFormer is supervised with the same per-pixel binary mask loss and a single classification loss per mask. Finally, we design a simple inference strategy to blend MaskFormer outputs into a task-dependent prediction format.  
为了解决这两个问题，我们提出了一种简单的 MaskFormer 方法，该方法可以无缝地将任何现有的每像素分类模型转换为掩码分类。使用 DETR \[4\] 中提出的集合预测机制，MaskFormer 使用 Transformer 解码器\[41\] 来计算一组对，每个对由一个类预测和一个掩码嵌入向量组成。掩模嵌入向量用于通过点积获得二进制掩码预测，每像素嵌入从底层全卷积网络获得。新模型以统一的方式解决了语义级和实例级分割任务：无需更改模型、损失和训练过程。具体来说，对于语义和全景分割任务，MaskFormer 的监督作用与每个像素的二进制二进制掩码损失和每个掩码的单个分类损失相同。最后，我们设计了一个简单的推理策略，将 MaskFormer 输出混合到与任务相关的预测格式中。

We evaluate MaskFormer on five semantic segmentation datasets with various numbers of categories: Cityscapes \[[15](#bib.bib15)\] (19 classes), Mapillary Vistas \[[34](#bib.bib34)\] (65 classes), ADE20K \[[55](#bib.bib55)\] (150 classes), COCO-Stuff-10K \[[3](#bib.bib3)\] (171 classes), and ADE20K-Full \[[55](#bib.bib55)\] (847 classes). While MaskFormer performs on par with per-pixel classification models for Cityscapes, which has a few diverse classes, the new model demonstrates superior performance for datasets with larger vocabulary. We hypothesize that a single class prediction per mask models fine-grained recognition better than per-pixel class predictions. MaskFormer achieves the new state-of-the-art on ADE20K (55.6 mIoU) with Swin-Transformer \[[29](#bib.bib29)\] backbone, outperforming a per-pixel classification model \[[29](#bib.bib29)\] with the same backbone by 2.1 mIoU, while being more efficient (10% reduction in parameters and 40% reduction in FLOPs).  
我们在五个具有不同类别的语义分割数据集上评估了 MaskFormer：Cityscapes \[15\]（19 个类）、Mapillary Vistas \[34\]（65 个类）、ADE20K \[55\]（150 个类）、COCO-Stuff-10K \[3\]（171 个类）和 ADE20K-Full \[55\]（847 个类）。虽然 MaskFormer 的性能与具有几个不同类别的 Cityscapes 的每像素分类模型相当，但新模型在具有较大词汇量的数据集中表现出卓越的性能。我们假设每个掩码的单个类预测比每个像素类预测更好地模拟细粒度识别。MaskFormer 在采用 Swin-Transformer \[29\] 主干的 ADE20K （55.6 mIoU）上实现了最先进的技术，比具有相同主干的每像素分类模型\[29\] 高出 2.1 mIoU，同时效率更高（参数减少 10%，FLOP 减少 40%）。

Finally, we study MaskFormer’s ability to solve instance-level tasks using two panoptic segmentation datasets: COCO \[[28](#bib.bib28), [24](#bib.bib24)\] and ADE20K \[[55](#bib.bib55)\]. MaskFormer outperforms a more complex DETR model \[[4](#bib.bib4)\] with the same backbone and the same post-processing. Moreover, MaskFormer achieves the new state-of-the-art on COCO (52.7 PQ), outperforming prior state-of-the-art \[[42](#bib.bib42)\] by 1.6 PQ. Our experiments highlight MaskFormer’s ability to unify instance- and semantic-level segmentation.  
最后，我们研究了 MaskFormer 使用两个全景分割数据集（COCO \[28， 24\] 和 ADE20K \[55\] 解决实例级任务的能力。MaskFormer 优于具有相同主干和相同后处理的更复杂的 DETR 模型\[4\]。此外，MaskFormer 在 COCO（52.7 PQ）上实现了新的最先进水平，比之前最先进的\[42\] 高出 1.6 PQ。我们的实验突出了 MaskFormer 统一实例级和语义级分割的能力。

## 2 Related Works 相关著作

Both per-pixel classification and mask classification have been extensively studied for semantic segmentation. In early work, Konishi and Yuille \[[25](#bib.bib25)\] apply per-pixel Bayesian classifiers based on local image statistics. Then, inspired by early works on non-semantic groupings \[[13](#bib.bib13), [36](#bib.bib36)\], mask classification-based methods became popular demonstrating the best performance in PASCAL VOC challenges \[[18](#bib.bib18)\]. Methods like O2P \[[5](#bib.bib5)\] and CFM \[[16](#bib.bib16)\] have achieved state-of-the-art results by classifying mask proposals \[[6](#bib.bib6), [40](#bib.bib40), [2](#bib.bib2)\]. In 2015, FCN \[[30](#bib.bib30)\] extended the idea of per-pixel classification to deep nets, significantly outperforming all prior methods on mIoU (a per-pixel evaluation metric which particularly suits the per-pixel classification formulation of segmentation).  
对于语义分割，每像素分类和掩码分类都进行了广泛的研究。在早期的工作中，Konishi 和 Yuille\[25\] 应用了基于局部图像统计的每像素贝叶斯分类器。然后，受到早期非语义分组工作\[13,36\] 的启发，基于掩码分类的方法开始流行，在 PASCAL VOC 挑战中表现出最佳性能\[18\]。O2P \[5\] 和 CFM \[16\] 等方法通过对掩模建议进行分类 \[6， 40， 2\] 取得了最先进的结果。2015 年，FCN \[30\] 将每像素分类的思想扩展到深度网络，显著优于 mIoU（一种特别适合分割的每像素分类公式）的所有先前方法。

Per-pixel classification became the dominant way for _deep-net-based_ semantic segmentation since the seminal work of Fully Convolutional Networks (FCNs) \[[30](#bib.bib30)\]. Modern semantic segmentation models focus on aggregating long-range context in the final feature map: ASPP \[[7](#bib.bib7), [8](#bib.bib8)\] uses atrous convolutions with different atrous rates; PPM \[[52](#bib.bib52)\] uses pooling operators with different kernel sizes; DANet \[[19](#bib.bib19)\], OCNet \[[51](#bib.bib51)\], and CCNet \[[23](#bib.bib23)\] use different variants of non-local blocks \[[43](#bib.bib43)\]. Recently, SETR \[[53](#bib.bib53)\] and Segmenter \[[37](#bib.bib37)\] replace traditional convolutional backbones with Vision Transformers (ViT) \[[17](#bib.bib17)\] that capture long-range context starting from the very first layer. However, these concurrent Transformer-based \[[41](#bib.bib41)\] semantic segmentation approaches still use a per-pixel classification formulation. Note, that our MaskFormer module can convert any per-pixel classification model to the mask classification setting, allowing seamless adoption of advances in per-pixel classification.  
自全卷积网络（FCN）的开创性工作以来，每像素分类成为基于深度网络的语义分割的主要方式\[30\]。现代语义分割模型侧重于在最终特征图中聚合长程上下文：ASPP \[7， 8\] 使用具有不同特征率的弹性卷积;PPM \[52\] 使用具有不同内核大小的池化运算符;DANet \[19\]、OCNet \[51\] 和 CCNet \[23\] 使用非本地块的不同变体 \[43\]。最近，SETR \[53\] 和 Segmenter \[37\] 用 Vision Transformer （ViT） \[17\] 取代了传统的卷积主干，从第一层开始捕获远程上下文。然而，这些基于 Transformer 的并发\[41\] 语义分割方法仍然使用每像素分类公式。请注意，我们的 MaskFormer 模块可以将任何每像素分类模型转换为掩码分类设置，从而无缝采用每像素分类的进步。

Mask classification is commonly used for instance-level segmentation tasks \[[20](#bib.bib20), [24](#bib.bib24)\]. These tasks require a dynamic number of predictions, making application of per-pixel classification challenging as it assumes a static number of outputs. Omnipresent Mask R-CNN \[[21](#bib.bib21)\] uses a global classifier to classify mask proposals for instance segmentation. DETR \[[4](#bib.bib4)\] further incorporates a Transformer \[[41](#bib.bib41)\] design to handle thing and stuff segmentation simultaneously for panoptic segmentation \[[24](#bib.bib24)\]. However, these mask classification methods require predictions of bounding boxes, which may limit their usage in semantic segmentation. The recently proposed Max-DeepLab \[[42](#bib.bib42)\] removes the dependence on box predictions for panoptic segmentation with conditional convolutions \[[39](#bib.bib39), [44](#bib.bib44)\]. However, in addition to the main mask classification losses it requires multiple auxiliary losses (_i.e_., instance discrimination loss, mask-ID cross entropy loss, and the standard per-pixel classification loss).  
掩码分类通常用于实例级分割任务\[20,24\]。这些任务需要动态数量的预测，这使得每像素分类的应用具有挑战性，因为它假定了静态数量的输出。Omnipresent Mask R-CNN \[21\] 使用全局分类器对掩码建议进行分类，以便进行实例分割。DETR \[4\] 进一步采用了 Transformer \[41\] 设计，以同时处理物和物的分割，以实现全景分割 \[24\]。但是，这些掩码分类方法需要对边界框进行预测，这可能会限制它们在语义分割中的使用。最近提出的 Max-DeepLab\[42\] 消除了对条件卷积全景分割的箱式预测的依赖\[39,44\]。但是，除了主要的掩码分类损失外，它还需要多个辅助损失（即实例辨别损失、掩码 ID 交叉熵损失和标准每像素分类损失）。

## 3 From Per-Pixel to Mask Classification  

In this section, we first describe how semantic segmentation can be formulated as either a per-pixel classification or a mask classification problem. Then, we introduce our instantiation of the mask classification model with the help of a Transformer decoder \[[41](#bib.bib41)\]. Finally, we describe simple inference strategies to transform mask classification outputs into task-dependent prediction formats.  
在本节中，我们首先介绍如何将语义分割表述为每像素分类或掩码分类问题。然后，我们介绍了在 Transformer 解码器\[41\] 的帮助下对掩码分类模型的实例化。最后，我们描述了将掩码分类输出转换为与任务相关的预测格式的简单推理策略。

### 3.1 Per-pixel classification formulation  

For per-pixel classification, a segmentation model aims to predict the probability distribution over all possible K𝐾K categories for every pixel of an H×W𝐻𝑊H\\times W image: y={pi|pi∈ΔK}i=1H⋅W𝑦superscriptsubscriptconditional-setsubscript𝑝𝑖subscript𝑝𝑖superscriptΔ𝐾𝑖1⋅𝐻𝑊y=\\{p_{i}|p_{i}\\in\\Delta^{K}\\}_{i=1}^{H\\cdot W}. Here ΔKsuperscriptΔ𝐾\\Delta^{K} is the K𝐾K-dimensional probability simplex. Training a per-pixel classification model is straight-forward: given ground truth category labels ygt={yigt|yigt∈{1,…,K}}i=1H⋅Wsuperscript𝑦gtsuperscriptsubscriptconditional-setsuperscriptsubscript𝑦𝑖gtsuperscriptsubscript𝑦𝑖gt1…𝐾𝑖1⋅𝐻𝑊y^{\\text{gt}}=\\{y_{i}^{\\text{gt}}|y_{i}^{\\text{gt}}\\in\\{1,\\dots,K\\}\\}_{i=1}^{H\\cdot W} for every pixel, a per-pixel cross-entropy (negative log-likelihood) loss is usually applied, _i.e_., ℒpixel-cls​(y,ygt)=∑i=1H⋅W−log⁡pi​(yigt)subscriptℒpixel-cls𝑦superscript𝑦gtsuperscriptsubscript𝑖1⋅𝐻𝑊subscript𝑝𝑖superscriptsubscript𝑦𝑖gt\\mathcal{L}_{\\text{pixel-cls}}(y,y^{\\text{gt}})=\\sum\\nolimits_{i=1}^{H\\cdot W}-\\log p_{i}(y_{i}^{\\text{gt}}).  
对于每像素分类，分割模型旨在预测 H×W𝐻𝑊H\\times W 图像中每个像素在所有可能 K𝐾K 类别上的概率分布： y={pi|pi∈ΔK}i=1H⋅W𝑦superscriptsubscriptconditional-setsubscript𝑝𝑖subscript𝑝𝑖superscriptΔ𝐾𝑖1⋅𝐻𝑊y=\\{p_{i}|p_{i}\\in\\Delta^{K}\\}_{i=1}^{H\\cdot W} 。这是 ΔKsuperscriptΔ𝐾\\Delta^{K} K𝐾K - 维概率单纯形。训练每像素分类模型很简单：给定每个像素的 ygt={yigt|yigt∈{1,…,K}}i=1H⋅Wsuperscript𝑦gtsuperscriptsubscriptconditional-setsuperscriptsubscript𝑦𝑖gtsuperscriptsubscript𝑦𝑖gt1…𝐾𝑖1⋅𝐻𝑊y^{\\text{gt}}=\\{y_{i}^{\\text{gt}}|y_{i}^{\\text{gt}}\\in\\{1,\\dots,K\\}\\}_{i=1}^{H\\cdot W} 真值类别标签，通常会应用每像素交叉熵（负对数似然）损失，即 ℒpixel-cls​(y,ygt)=∑i=1H⋅W−log⁡pi​(yigt)subscriptℒpixel-cls𝑦superscript𝑦gtsuperscriptsubscript𝑖1⋅𝐻𝑊subscript𝑝𝑖superscriptsubscript𝑦𝑖gt\\mathcal{L}_{\\text{pixel-cls}}(y,y^{\\text{gt}})=\\sum\\nolimits_{i=1}^{H\\cdot W}-\\log p_{i}(y_{i}^{\\text{gt}}) 。

### 3.2 Mask classification formulation  

Mask classification splits the segmentation task into 1) partitioning/grouping the image into N𝑁N regions (N𝑁N does not need to equal K𝐾K), represented with binary masks {mi|mi∈\[0,1\]H×W}i=1Nsuperscriptsubscriptconditional-setsubscript𝑚𝑖subscript𝑚𝑖superscript01𝐻𝑊𝑖1𝑁\\{m_{i}|m_{i}\\in\[0,1\]^{H\\times W}\\}_{i=1}^{N}; and 2) associating each region as a whole with some distribution over K𝐾K categories. To jointly group and classify a segment, _i.e_., to perform mask classification, we define the desired output z𝑧z as a set of N𝑁N probability-mask pairs, _i.e_., z={(pi,mi)}i=1N.𝑧superscriptsubscriptsubscript𝑝𝑖subscript𝑚𝑖𝑖1𝑁z=\\{(p_{i},m_{i})\\}_{i=1}^{N}. In contrast to per-pixel class probability prediction, for mask classification the probability distribution pi∈ΔK+1subscript𝑝𝑖superscriptΔ𝐾1p_{i}\\in\\Delta^{K+1} contains an auxiliary “no object” label (∅\\varnothing) in addition to the K𝐾K category labels. The ∅\\varnothing label is predicted for masks that do not correspond to any of the K𝐾K categories. Note, mask classification allows multiple mask predictions with the same associated class, making it applicable to both semantic- and instance-level segmentation tasks.  
掩码分类将分割任务拆分为 1） 将图像划分/分组为 N𝑁N 区域（ N𝑁N 不需要相等 K𝐾K ），用二进制掩码 {mi|mi∈\[0,1\]H×W}i=1Nsuperscriptsubscriptconditional-setsubscript𝑚𝑖subscript𝑚𝑖superscript01𝐻𝑊𝑖1𝑁\\{m_{i}|m_{i}\\in\[0,1\]^{H\\times W}\\}_{i=1}^{N} 表示;2）将每个区域作为一个整体与类别的 K𝐾K 某种分布相关联。为了对一个片段进行联合分组和分类，即执行掩码分类，我们将所需的输出 z𝑧z 定义为一组 N𝑁N 概率掩码对，即， z={(pi,mi)}i=1N.𝑧superscriptsubscriptsubscript𝑝𝑖subscript𝑚𝑖𝑖1𝑁z=\\{(p_{i},m_{i})\\}_{i=1}^{N}. 与每像素类概率预测相反，对于掩码分类，概率分布 pi∈ΔK+1subscript𝑝𝑖superscriptΔ𝐾1p_{i}\\in\\Delta^{K+1} 除了 K𝐾K 类别标签外，还包含一个辅助的“无对象”标签（ ∅\\varnothing ）。对于与任何 K𝐾K 类别不对应的口罩，预测标签 ∅\\varnothing 。请注意，掩码分类允许使用相同的关联类进行多个掩码预测，使其适用于语义级和实例级分段任务。

To train a mask classification model, a matching σ𝜎\\sigma between the set of predictions z𝑧z and the set of Ngtsuperscript𝑁gtN^{\\text{gt}} ground truth segments zgt={(cigt,migt)|cigt∈{1,…,K},migt∈{0,1}H×W}i=1Ngtsuperscript𝑧gtsuperscriptsubscriptconditional-setsuperscriptsubscript𝑐𝑖gtsuperscriptsubscript𝑚𝑖gtformulae-sequencesuperscriptsubscript𝑐𝑖gt1…𝐾superscriptsubscript𝑚𝑖gtsuperscript01𝐻𝑊𝑖1superscript𝑁gtz^{\\text{gt}}=\\{(c_{i}^{\\text{gt}},m_{i}^{\\text{gt}})|c_{i}^{\\text{gt}}\\in\\{1,\\dots,K\\},m_{i}^{\\text{gt}}\\in\\{0,1\\}^{H\\times W}\\}_{i=1}^{N^{\\text{gt}}} is required.222Different mask classification methods utilize various matching rules. For instance, Mask R-CNN \[[21](#bib.bib21)\] uses a heuristic procedure based on anchor boxes and DETR \[[4](#bib.bib4)\] optimizes a bipartite matching between z𝑧z and zgtsuperscript𝑧gtz^{\\text{gt}}.  Here cigtsuperscriptsubscript𝑐𝑖gtc_{i}^{\\text{gt}} is the ground truth class of the ithsuperscript𝑖thi^{\\text{th}} ground truth segment. Since the size of prediction set |z|=N𝑧𝑁|z|=N and ground truth set |zgt|=Ngtsuperscript𝑧gtsuperscript𝑁gt|z^{\\text{gt}}|=N^{\\text{gt}} generally differ, we assume N≥Ngt𝑁superscript𝑁gtN\\geq N^{\\text{gt}} and pad the set of ground truth labels with “no object” tokens ∅\\varnothing to allow one-to-one matching.  
若要训练掩码分类模型，需要在预测集 z𝑧z 和 Ngtsuperscript𝑁gtN^{\\text{gt}} 真值段 zgt={(cigt,migt)|cigt∈{1,…,K},migt∈{0,1}H×W}i=1Ngtsuperscript𝑧gtsuperscriptsubscriptconditional-setsuperscriptsubscript𝑐𝑖gtsuperscriptsubscript𝑚𝑖gtformulae-sequencesuperscriptsubscript𝑐𝑖gt1…𝐾superscriptsubscript𝑚𝑖gtsuperscript01𝐻𝑊𝑖1superscript𝑁gtz^{\\text{gt}}=\\{(c_{i}^{\\text{gt}},m_{i}^{\\text{gt}})|c_{i}^{\\text{gt}}\\in\\{1,\\dots,K\\},m_{i}^{\\text{gt}}\\in\\{0,1\\}^{H\\times W}\\}_{i=1}^{N^{\\text{gt}}} 集之间进行匹配 σ𝜎\\sigma 。 2 这是 cigtsuperscriptsubscript𝑐𝑖gtc_{i}^{\\text{gt}} ithsuperscript𝑖thi^{\\text{th}} 真值段的真值类。由于预测集 |z|=N𝑧𝑁|z|=N 和真值集 |zgt|=Ngtsuperscript𝑧gtsuperscript𝑁gt|z^{\\text{gt}}|=N^{\\text{gt}} 的大小通常不同，因此我们假设 N≥Ngt𝑁superscript𝑁gtN\\geq N^{\\text{gt}} 并用“无对象”标记填充真值标签集 ∅\\varnothing ，以允许一对一匹配。

For semantic segmentation, a trivial _fixed matching_ is possible if the number of predictions N𝑁N matches the number of category labels K𝐾K. In this case, the ithsuperscript𝑖thi^{\\text{th}} prediction is matched to a ground truth region with class label i𝑖i and to ∅\\varnothing if a region with class label i𝑖i is not present in the ground truth. In our experiments, we found that a _bipartite matching_-based assignment demonstrates better results than the fixed matching. Unlike DETR \[[4](#bib.bib4)\] that uses bounding boxes to compute the assignment costs between prediction zisubscript𝑧𝑖z_{i} and ground truth zjgtsuperscriptsubscript𝑧𝑗gtz_{j}^{\\text{gt}} for the matching problem, we directly use class and mask predictions, _i.e_., −pi​(cjgt)+ℒmask​(mi,mjgt)subscript𝑝𝑖superscriptsubscript𝑐𝑗gtsubscriptℒmasksubscript𝑚𝑖superscriptsubscript𝑚𝑗gt-p_{i}(c_{j}^{\\text{gt}})+\\mathcal{L}_{\\text{mask}}(m_{i},m_{j}^{\\text{gt}}), where ℒmasksubscriptℒmask\\mathcal{L}_{\\text{mask}} is a binary mask loss.  
对于语义分割，如果预测的数量与类别标签的数量 N𝑁N 匹配，则可以进行微不足道的 K𝐾K 固定匹配。在这种情况下， ithsuperscript𝑖thi^{\\text{th}} 预测将与具有类标签 i𝑖i 的真值区域匹配， ∅\\varnothing 并且如果真值中不存在具有类标签 i𝑖i 的区域，则与预测匹配。在我们的实验中，我们发现基于二分匹配的赋值比固定匹配显示出更好的结果。与 DETR \[4\] 不同，DETR \[4\] 使用边界框来计算匹配问题的预测 zisubscript𝑧𝑖z_{i} 和地面实况 zjgtsuperscriptsubscript𝑧𝑗gtz_{j}^{\\text{gt}} 之间的分配成本，我们直接使用类和掩码预测，即 −pi​(cjgt)+ℒmask​(mi,mjgt)subscript𝑝𝑖superscriptsubscript𝑐𝑗gtsubscriptℒmasksubscript𝑚𝑖superscriptsubscript𝑚𝑗gt-p_{i}(c_{j}^{\\text{gt}})+\\mathcal{L}_{\\text{mask}}(m_{i},m_{j}^{\\text{gt}}) ，其中 ℒmasksubscriptℒmask\\mathcal{L}_{\\text{mask}} 是二进制掩码损失。

To train model parameters, given a matching, the main mask classification loss ℒmask-clssubscriptℒmask-cls\\mathcal{L}_{\\text{mask-cls}} is composed of a cross-entropy classification loss and a binary mask loss ℒmasksubscriptℒmask\\mathcal{L}_{\\text{mask}} for each predicted segment:  
为了训练模型参数，给定匹配，主掩码分类损失 ℒmask-clssubscriptℒmask-cls\\mathcal{L}_{\\text{mask-cls}} 由每个预测段的交叉熵分类损失和二元掩码损失 ℒmasksubscriptℒmask\\mathcal{L}_{\\text{mask}} 组成：

|  | ℒmask-cls​(z,zgt)=∑j=1N\[−log⁡pσ​(j)​(cjgt)+𝟙cjgt≠∅​ℒmask​(mσ​(j),mjgt)\].subscriptℒmask-cls𝑧superscript𝑧gtsuperscriptsubscript𝑗1𝑁delimited-\[\]subscript𝑝𝜎𝑗superscriptsubscript𝑐𝑗gtsubscript1superscriptsubscript𝑐𝑗gtsubscriptℒmasksubscript𝑚𝜎𝑗superscriptsubscript𝑚𝑗gt\\mathcal{L}_{\\text{mask-cls}}(z,z^{\\text{gt}})=\\sum\\nolimits_{j=1}^{N}\\left\[-\\log p_{\\sigma(j)}(c_{j}^{\\text{gt}})+\\mathds{1}_{c_{j}^{\\text{gt}}\\neq\\varnothing}\\mathcal{L}_{\\text{mask}}(m_{\\sigma(j)},m_{j}^{\\text{gt}})\\right\]. |  | (1) |

Note, that most existing mask classification models use auxiliary losses (_e.g_., a bounding box loss \[[21](#bib.bib21), [4](#bib.bib4)\] or an instance discrimination loss \[[42](#bib.bib42)\]) in addition to ℒmask-clssubscriptℒmask-cls\\mathcal{L}_{\\text{mask-cls}}. In the next section we present a simple mask classification model that allows end-to-end training with ℒmask-clssubscriptℒmask-cls\\mathcal{L}_{\\text{mask-cls}} alone.  
请注意，大多数现有的掩码分类模型除了使用辅助损失（例如，边界框损失 \[21， 4\] 或实例判别损失 \[42\]）之外 ℒmask-clssubscriptℒmask-cls\\mathcal{L}_{\\text{mask-cls}} ，还使用辅助损失。在下一节中，我们将介绍一个简单的掩码分类模型，该模型允许 ℒmask-clssubscriptℒmask-cls\\mathcal{L}_{\\text{mask-cls}} 单独进行端到端训练。

### 3.3 MaskFormer3.3 蒙版成型

![](https://ar5iv.labs.arxiv.org/html/2107.06278/assets/x2.png)

Figure 2: MaskFormer overview. We use a backbone to extract image features ℱℱ\\mathcal{F}. A pixel decoder gradually upsamples image features to extract per-pixel embeddings ℰpixelsubscriptℰpixel\\mathcal{E}_{\\text{pixel}}. A transformer decoder attends to image features and produces N𝑁N per-segment embeddings 𝒬𝒬\\mathcal{Q}. The embeddings independently generate N𝑁N class predictions with N𝑁N corresponding mask embeddings ℰmasksubscriptℰmask\\mathcal{E}_{\\text{mask}}. Then, the model predicts N𝑁N possibly overlapping binary mask predictions via a dot product between pixel embeddings ℰpixelsubscriptℰpixel\\mathcal{E}_{\\text{pixel}} and mask embeddings ℰmasksubscriptℰmask\\mathcal{E}_{\\text{mask}} followed by a sigmoid activation. For semantic segmentation task we can get the final prediction by combining N𝑁N binary masks with their class predictions using a simple matrix multiplication (see Section [3.4](#S3.SS4 "3.4 Mask-classification inference ‣ 3 From Per-Pixel to Mask Classification ‣ Per-Pixel Classification is Not All You Need for Semantic Segmentation")). Note, the dimensions for multiplication ⨂tensor-product\\bigotimes are shown in gray.  
图 2：MaskFormer 概述。我们使用主干来提取图像特征 ℱℱ\\mathcal{F} 。像素解码器逐渐对图像特征进行上采样，以提取每个像素的嵌入 ℰpixelsubscriptℰpixel\\mathcal{E}_{\\text{pixel}} 。Transformer 解码器处理图像特征并生成 N𝑁N 每个段的嵌入 𝒬𝒬\\mathcal{Q} 。嵌入独立生成 N𝑁N 具有 N𝑁N 相应掩码嵌入的类预测 ℰmasksubscriptℰmask\\mathcal{E}_{\\text{mask}} 。然后，该模型通过像素嵌入 ℰpixelsubscriptℰpixel\\mathcal{E}_{\\text{pixel}} 和掩码嵌入之间的点积 ℰmasksubscriptℰmask\\mathcal{E}_{\\text{mask}} ，然后进行 S 形激活，预测 N𝑁N 可能重叠的二元掩模预测。对于语义分割任务，我们可以通过使用简单的矩阵乘法将二进制掩码与其类预测相结合 N𝑁N 来获得最终预测（参见第 3.4 节）。请注意，乘法 ⨂tensor-product\\bigotimes 的维度以灰色显示。

We now introduce MaskFormer, the new mask classification model, which computes N𝑁N probability-mask pairs z={(pi,mi)}i=1N𝑧superscriptsubscriptsubscript𝑝𝑖subscript𝑚𝑖𝑖1𝑁z=\\{(p_{i},m_{i})\\}_{i=1}^{N}. The model contains three modules (see Fig. [2](#S3.F2 "Figure 2 ‣ 3.3 MaskFormer ‣ 3 From Per-Pixel to Mask Classification ‣ Per-Pixel Classification is Not All You Need for Semantic Segmentation")): 1) a pixel-level module that extracts per-pixel embeddings used to generate binary mask predictions; 2) a transformer module, where a stack of Transformer decoder layers \[[41](#bib.bib41)\] computes N𝑁N per-segment embeddings; and 3) a segmentation module, which generates predictions {(pi,mi)}i=1Nsuperscriptsubscriptsubscript𝑝𝑖subscript𝑚𝑖𝑖1𝑁\\{(p_{i},m_{i})\\}_{i=1}^{N} from these embeddings. During inference, discussed in Sec. [3.4](#S3.SS4 "3.4 Mask-classification inference ‣ 3 From Per-Pixel to Mask Classification ‣ Per-Pixel Classification is Not All You Need for Semantic Segmentation"), pisubscript𝑝𝑖p_{i} and misubscript𝑚𝑖m_{i} are assembled into the final prediction.

Pixel-level module takes an image of size H×W𝐻𝑊H\\times W as input. A backbone generates a (typically) low-resolution image feature map ℱ∈ℝCℱ×HS×WSℱsuperscriptℝsubscript𝐶ℱ𝐻𝑆𝑊𝑆\\mathcal{F}\\in\\mathbb{R}^{C_{\\mathcal{F}}\\times\\frac{H}{S}\\times\\frac{W}{S}}, where Cℱsubscript𝐶ℱC_{\\mathcal{F}} is the number of channels and S𝑆S is the stride of the feature map (Cℱsubscript𝐶ℱC_{\\mathcal{F}} depends on the specific backbone and we use S=32𝑆32S=32 in this work). Then, a pixel decoder gradually upsamples the features to generate per-pixel embeddings ℰpixel∈ℝCℰ×H×Wsubscriptℰpixelsuperscriptℝsubscript𝐶ℰ𝐻𝑊\\mathcal{E}_{\\text{pixel}}\\in\\mathbb{R}^{C_{\\mathcal{E}}\\times H\\times W}, where Cℰsubscript𝐶ℰC_{\\mathcal{E}} is the embedding dimension. Note, that any per-pixel classification-based segmentation model fits the pixel-level module design including recent Transformer-based models \[[37](#bib.bib37), [53](#bib.bib53), [29](#bib.bib29)\]. MaskFormer seamlessly converts such a model to mask classification.

Transformer module uses the standard Transformer decoder \[[41](#bib.bib41)\] to compute from image features ℱℱ\\mathcal{F} and N𝑁N learnable positional embeddings (_i.e_., queries) its output, _i.e_., N𝑁N per-segment embeddings 𝒬∈ℝC𝒬×N𝒬superscriptℝsubscript𝐶𝒬𝑁\\mathcal{Q}\\in\\mathbb{R}^{C_{\\mathcal{Q}}\\times N} of dimension C𝒬subscript𝐶𝒬C_{\\mathcal{Q}} that encode global information about each segment MaskFormer predicts. Similarly to \[[4](#bib.bib4)\], the decoder yields all predictions in parallel.

Segmentation module applies a linear classifier, followed by a softmax activation, on top of the per-segment embeddings 𝒬𝒬\\mathcal{Q} to yield class probability predictions {pi∈ΔK+1}i=1Nsuperscriptsubscriptsubscript𝑝𝑖superscriptΔ𝐾1𝑖1𝑁\\{p_{i}\\in\\Delta^{K+1}\\}_{i=1}^{N} for each segment. Note, that the classifier predicts an additional “no object” category (∅\\varnothing) in case the embedding does not correspond to any region. For mask prediction, a Multi-Layer Perceptron (MLP) with 2 hidden layers converts the per-segment embeddings 𝒬𝒬\\mathcal{Q} to N𝑁N mask embeddings ℰmask∈ℝCℰ×Nsubscriptℰmasksuperscriptℝsubscript𝐶ℰ𝑁\\mathcal{E}_{\\text{mask}}\\in\\mathbb{R}^{C_{\\mathcal{E}}\\times N} of dimension Cℰsubscript𝐶ℰC_{\\mathcal{E}}. Finally, we obtain each binary mask prediction mi∈\[0,1\]H×Wsubscript𝑚𝑖superscript01𝐻𝑊m_{i}\\in\[0,1\]^{H\\times W} via a dot product between the ithsuperscript𝑖thi^{\\text{th}} mask embedding and per-pixel embeddings ℰpixelsubscriptℰpixel\\mathcal{E}_{\\text{pixel}} computed by the pixel-level module. The dot product is followed by a sigmoid activation, _i.e_., mi​\[h,w\]=sigmoid​(ℰmask​\[:,i\]T⋅ℰpixel​\[:,h,w\])subscript𝑚𝑖ℎ𝑤sigmoid⋅subscriptℰmasksuperscript:𝑖Tsubscriptℰpixel:ℎ𝑤m_{i}\[h,w\]=\\text{sigmoid}(\\mathcal{E}_{\\text{mask}}\[:,i\]^{\\text{T}}\\cdot\\mathcal{E}_{\\text{pixel}}\[:,h,w\]).  
分割模块在每段嵌入的基础上应用线性分类器，然后激活 softmax， 𝒬𝒬\\mathcal{Q} 以生成每个段 {pi∈ΔK+1}i=1Nsuperscriptsubscriptsubscript𝑝𝑖superscriptΔ𝐾1𝑖1𝑁\\{p_{i}\\in\\Delta^{K+1}\\}_{i=1}^{N} 的类概率预测。请注意，如果嵌入不对应于任何区域，分类器会预测额外的“无对象”类别 （ ∅\\varnothing ）。对于掩码预测，具有 2 个隐藏层的多层感知器 （MLP） 将每个段的嵌入 𝒬𝒬\\mathcal{Q} 转换为维度 ℰmask∈ℝCℰ×Nsubscriptℰmasksuperscriptℝsubscript𝐶ℰ𝑁\\mathcal{E}_{\\text{mask}}\\in\\mathbb{R}^{C_{\\mathcal{E}}\\times N} 的 N𝑁N 掩码嵌入 Cℰsubscript𝐶ℰC_{\\mathcal{E}} 。最后，我们通过 ithsuperscript𝑖thi^{\\text{th}} 掩码嵌入和像素级模块 ℰpixelsubscriptℰpixel\\mathcal{E}_{\\text{pixel}} 计算的每像素嵌入之间的点积获得每个二进制掩码预测 mi∈\[0,1\]H×Wsubscript𝑚𝑖superscript01𝐻𝑊m_{i}\\in\[0,1\]^{H\\times W} 。点积之后是 S 形激活，即 mi​\[h,w\]=sigmoid​(ℰmask​\[:,i\]T⋅ℰpixel​\[:,h,w\])subscript𝑚𝑖ℎ𝑤sigmoid⋅subscriptℰmasksuperscript:𝑖Tsubscriptℰpixel:ℎ𝑤m_{i}\[h,w\]=\\text{sigmoid}(\\mathcal{E}_{\\text{mask}}\[:,i\]^{\\text{T}}\\cdot\\mathcal{E}_{\\text{pixel}}\[:,h,w\]) .

Note, we empirically find it is beneficial to _not_ enforce mask predictions to be mutually exclusive to each other by using a softmax activation. During training, the ℒmask-clssubscriptℒmask-cls\\mathcal{L}_{\\text{mask-cls}} loss combines a cross entropy classification loss and a binary mask loss ℒmasksubscriptℒmask\\mathcal{L}_{\\text{mask}} for each predicted segment. For simplicity we use the same ℒmasksubscriptℒmask\\mathcal{L}_{\\text{mask}} as DETR \[[4](#bib.bib4)\], _i.e_., a linear combination of a focal loss \[[27](#bib.bib27)\] and a dice loss \[[33](#bib.bib33)\] multiplied by hyper-parameters λfocalsubscript𝜆focal\\lambda_{\\text{focal}} and λdicesubscript𝜆dice\\lambda_{\\text{dice}} respectively.  
请注意，我们根据经验发现，通过使用 softmax 激活，不强制将掩码预测强制为相互排斥是有益的。在训练期间， ℒmask-clssubscriptℒmask-cls\\mathcal{L}_{\\text{mask-cls}} 损失结合了每个预测段的交叉熵分类损失和二元掩码损失 ℒmasksubscriptℒmask\\mathcal{L}_{\\text{mask}} 。为简单起见，我们使用与 DETR \[4\] 相同的 ℒmasksubscriptℒmask\\mathcal{L}_{\\text{mask}} 方法，即焦点损失 \[27\] 和骰子损失 \[33\] 分别乘以超参数 λfocalsubscript𝜆focal\\lambda_{\\text{focal}} 和 λdicesubscript𝜆dice\\lambda_{\\text{dice}} 的线性组合。

### 3.4 Mask-classification inference  
3.4 

First, we present a simple _general inference_ procedure that converts mask classification outputs {(pi,mi)}i=1Nsuperscriptsubscriptsubscript𝑝𝑖subscript𝑚𝑖𝑖1𝑁\\{(p_{i},m_{i})\\}_{i=1}^{N} to either panoptic or semantic segmentation output formats. Then, we describe a _semantic inference_ procedure specifically designed for semantic segmentation. We note, that the specific choice of inference strategy largely depends on the evaluation metric rather than the task.  
首先，我们提出了一个简单的通用推理过程，该过程将掩码分类输出 {(pi,mi)}i=1Nsuperscriptsubscriptsubscript𝑝𝑖subscript𝑚𝑖𝑖1𝑁\\{(p_{i},m_{i})\\}_{i=1}^{N} 转换为全景或语义分割输出格式。然后，我们描述了一个专门为语义分割设计的语义推理过程。我们注意到，推理策略的具体选择很大程度上取决于评估指标而不是任务。

General inference partitions an image into segments by assigning each pixel \[h,w\]ℎ𝑤\[h,w\] to one of the N𝑁N predicted probability-mask pairs via arg​maxi:ci≠∅⁡pi​(ci)⋅mi​\[h,w\]⋅subscriptargmax:𝑖subscript𝑐𝑖subscript𝑝𝑖subscript𝑐𝑖subscript𝑚𝑖ℎ𝑤\\operatorname*{arg\\,max}_{i:c_{i}\\neq\\varnothing}p_{i}(c_{i})\\cdot m_{i}\[h,w\]. Here cisubscript𝑐𝑖c_{i} is the most likely class label ci=arg​maxc∈{1,…,K,∅}⁡pi​(c)subscript𝑐𝑖subscriptargmax𝑐1…𝐾subscript𝑝𝑖𝑐c_{i}=\\operatorname*{arg\\,max}_{c\\in\\{1,\\dots,K,\\varnothing\\}}p_{i}(c) for each probability-mask pair i𝑖i. Intuitively, this procedure assigns a pixel at location \[h,w\]ℎ𝑤\[h,w\] to probability-mask pair i𝑖i only if both the _most likely_ class probability pi​(ci)subscript𝑝𝑖subscript𝑐𝑖p_{i}(c_{i}) and the mask prediction probability mi​\[h,w\]subscript𝑚𝑖ℎ𝑤m_{i}\[h,w\] are high. Pixels assigned to the same probability-mask pair i𝑖i form a segment where each pixel is labelled with cisubscript𝑐𝑖c_{i}. For semantic segmentation, segments sharing the same category label are merged; whereas for instance-level segmentation tasks, the index i𝑖i of the probability-mask pair helps to distinguish different instances of the same class. Finally, to reduce false positive rates in panoptic segmentation we follow previous inference strategies \[[4](#bib.bib4), [24](#bib.bib24)\]. Specifically, we filter out low-confidence predictions prior to inference and remove predicted segments that have large parts of their binary masks (mi>0.5subscript𝑚𝑖0.5m_{i}>0.5) occluded by other predictions.  
一般推理通过 将每个像素 \[h,w\]ℎ𝑤\[h,w\] 分配给预测 arg​maxi:ci≠∅⁡pi​(ci)⋅mi​\[h,w\]⋅subscriptargmax:𝑖subscript𝑐𝑖subscript𝑝𝑖subscript𝑐𝑖subscript𝑚𝑖ℎ𝑤\\operatorname*{arg\\,max}_{i:c_{i}\\neq\\varnothing}p_{i}(c_{i})\\cdot m_{i}\[h,w\] 的概率掩码对之一 N𝑁N ，将图像划分为多个段。以下是 cisubscript𝑐𝑖c_{i} 每个概率掩码对 i𝑖i 最有可能的类标签 ci=arg​maxc∈{1,…,K,∅}⁡pi​(c)subscript𝑐𝑖subscriptargmax𝑐1…𝐾subscript𝑝𝑖𝑐c_{i}=\\operatorname*{arg\\,max}_{c\\in\\{1,\\dots,K,\\varnothing\\}}p_{i}(c) 。直观地说，仅当最可能的类概率 pi​(ci)subscript𝑝𝑖subscript𝑐𝑖p_{i}(c_{i}) 和掩码预测概率 mi​\[h,w\]subscript𝑚𝑖ℎ𝑤m_{i}\[h,w\] 都很高时，此过程才会将位置 \[h,w\]ℎ𝑤\[h,w\] 的像素分配给概率掩码对 i𝑖i 。分配给同一概率掩码对 i𝑖i 的像素形成一个线段，其中每个像素都用 cisubscript𝑐𝑖c_{i} 标记。对于语义分割，将合并共享同一类别标签的区段; 而对于实例级分段任务，概率掩码对的索引 i𝑖i 有助于区分同一类的不同实例。最后，为了降低全景分割中的误报率，我们遵循先前的推理策略\[4,24\]。具体来说，我们在推理之前过滤掉低置信度的预测，并删除其二进制掩码 （ mi>0.5subscript𝑚𝑖0.5m_{i}>0.5 ） 的大部分被其他预测遮挡的预测段。

Semantic inference is designed specifically for semantic segmentation and is done via a simple matrix multiplication. We empirically find that marginalization over probability-mask pairs, _i.e_., arg​maxc∈{1,…,K}​∑i=1Npi​(c)⋅mi​\[h,w\]subscriptargmax𝑐1…𝐾superscriptsubscript𝑖1𝑁⋅subscript𝑝𝑖𝑐subscript𝑚𝑖ℎ𝑤\\operatorname*{arg\\,max}_{c\\in\\{1,\\dots,K\\}}\\sum_{i=1}^{N}p_{i}(c)\\cdot m_{i}\[h,w\], yields better results than the hard assignment of each pixel to a probability-mask pair i𝑖i used in the general inference strategy. The argmax does not include the “no object” category (∅\\varnothing) as standard semantic segmentation requires each output pixel to take a label. Note, this strategy returns a per-pixel class probability ∑i=1Npi​(c)⋅mi​\[h,w\]superscriptsubscript𝑖1𝑁⋅subscript𝑝𝑖𝑐subscript𝑚𝑖ℎ𝑤\\sum_{i=1}^{N}p_{i}(c)\\cdot m_{i}\[h,w\]. However, we observe that directly maximizing per-pixel class likelihood leads to poor performance. We hypothesize, that gradients are evenly distributed to every query, which complicates training.  
语义推理是专门为语义分割而设计的，通过简单的矩阵乘法完成。我们根据经验发现，对概率掩码对的边缘化，即 arg​maxc∈{1,…,K}​∑i=1Npi​(c)⋅mi​\[h,w\]subscriptargmax𝑐1…𝐾superscriptsubscript𝑖1𝑁⋅subscript𝑝𝑖𝑐subscript𝑚𝑖ℎ𝑤\\operatorname*{arg\\,max}_{c\\in\\{1,\\dots,K\\}}\\sum_{i=1}^{N}p_{i}(c)\\cdot m_{i}\[h,w\] ，比将每个像素硬分配给一般推理策略中使用的概率掩码对 i𝑖i 产生更好的结果。argmax 不包括“无对象”类别 （ ∅\\varnothing ），因为标准语义分割要求每个输出像素都采用标签。请注意，此策略返回每个像素类的概率 ∑i=1Npi​(c)⋅mi​\[h,w\]superscriptsubscript𝑖1𝑁⋅subscript𝑝𝑖𝑐subscript𝑚𝑖ℎ𝑤\\sum_{i=1}^{N}p_{i}(c)\\cdot m_{i}\[h,w\] 。但是，我们观察到，直接最大化每个像素类的可能性会导致性能不佳。我们假设，梯度均匀地分布到每个查询，这使得训练变得复杂。

## 4 Experiments 实验

We demonstrate that MaskFormer seamlessly unifies semantic- and instance-level segmentation tasks by showing state-of-the-art results on both semantic segmentation and panoptic segmentation datasets. Then, we ablate the MaskFormer design confirming that observed improvements in semantic segmentation indeed stem from the shift from per-pixel classification to mask classification.  
我们证明了 MaskFormer 通过在语义分割和全景分割数据集上显示最先进的结果，无缝统一了语义级和实例级分割任务。然后，我们消融了 MaskFormer 设计，确认观察到的语义分割改进确实源于从每像素分类到掩码分类的转变。

Datasets. We study MaskFormer using four widely used semantic segmentation datasets: ADE20K \[[55](#bib.bib55)\] (150 classes) from the SceneParse150 challenge \[[54](#bib.bib54)\], COCO-Stuff-10K \[[3](#bib.bib3)\] (171 classes), Cityscapes \[[15](#bib.bib15)\] (19 classes), and Mapillary Vistas \[[34](#bib.bib34)\] (65 classes). In addition, we use the ADE20K-Full \[[55](#bib.bib55)\] dataset annotated in an open vocabulary setting (we keep 874 classes that are present in both train and validation sets). For panotic segmenation evaluation we use COCO \[[28](#bib.bib28), [3](#bib.bib3), [24](#bib.bib24)\] (80 “things” and 53 “stuff” categories) and ADE20K-Panoptic \[[55](#bib.bib55), [24](#bib.bib24)\] (100 “things” and 50 “stuff” categories). Please see the appendix for detailed descriptions of all used datasets.  
数据。我们使用四个广泛使用的语义分割数据集来研究 MaskFormer：来自 SceneParse150 挑战 \[54\] 的 ADE20K \[55\]（150 个类）、COCO-Stuff-10K \[3\]（171 个类）、Cityscapes \[15\]（19 个类）和 Mapillary Vistas \[34\]（65 个类）。此外，我们使用在开放词汇表设置中注释的 ADE20K-Full \[55\] 数据集（我们保留了训练集和验证集中都存在的 874 个类）。对于全景分类评估，我们使用 COCO \[28， 3， 24\]（80 个“事物”和 53 个“东西”类别）和 ADE20K-Panoptic \[55， 24\]（100 个“事物”和 50 个“东西”类别）。有关所有已用数据集的详细说明，请参阅附录。

Evaluation metrics. For _semantic segmentation_ the standard metric is mIoU (mean Intersection-over-Union) \[[18](#bib.bib18)\], a per-pixel metric that directly corresponds to the per-pixel classification formulation. To better illustrate the difference between segmentation approaches, in our ablations we supplement mIoU with PQStSt{}^{\\text{St}} (PQ stuff) \[[24](#bib.bib24)\], a per-region metric that treats all classes as “stuff” and evaluates each segment equally, irrespective of its size. We report the median of 3 runs for all datasets, except for Cityscapes where we report the median of 5 runs. For _panoptic segmentation_, we use the standard PQ (panoptic quality) metric \[[24](#bib.bib24)\] and report single run results due to prohibitive training costs.  
评估指标。对于语义分割，标准指标是 mIoU（平均交集并集）\[18\]，这是一个直接对应于每像素分类公式的每像素指标。为了更好地说明分割方法之间的差异，在我们的消融中，我们用 PQ StSt{}^{\\text{St}} （PQ 内容）\[24\] 补充了 mIoU，这是一个每个区域的指标，将所有类别视为“内容”，并平等地评估每个片段，无论其大小如何。我们报告所有数据集的 3 次运行中位数，但 Cityscapes 除外，其中我们报告的中位数为 5 次运行。对于全景分割，我们使用标准的 PQ（全景质量）指标\[24\]，并报告单次运行结果，因为训练成本过高。

![](https://ar5iv.labs.arxiv.org/html/2107.06278/assets/x3.png)

Baseline models. On the right we sketch the used per-pixel classification baselines. The PerPixelBaseline uses the pixel-level module of MaskFormer and directly outputs per-pixel class scores. For a fair comparison, we design PerPixelBaseline+ which adds the transformer module and mask embedding MLP to the PerPixelBaseline. Thus, PerPixelBaseline+ and MaskFormer differ only in the formulation: per-pixel _vs_. mask classification. Note that these baselines are for ablation and we compare MaskFormer with state-of-the-art per-pixel classification models as well.  
基线模型。在右边，我们勾勒了使用的每像素分类基线。PerPixelBaseline 使用 MaskFormer 的像素级模块，直接输出每个像素的类分数。为了公平地进行比较，我们设计了 PerPixelBaseline+，它将转换器模块和掩模嵌入 MLP 添加到 PerPixelBaseline 中。因此，PerPixelBaseline+ 和 MaskFormer 仅在公式上有所不同：每像素与蒙版分类。请注意，这些基线是用于消融的，我们将 MaskFormer 与最先进的每像素分类模型进行了比较。

### 4.1 Implementation details  
4.1 实现细节

Backbone. MaskFormer is compatible with any backbone architecture. In our work we use the standard convolution-based ResNet \[[22](#bib.bib22)\] backbones (R50 and R101 with 50 and 101 layers respectively) and recently proposed Transformer-based Swin-Transformer \[[29](#bib.bib29)\] backbones. In addition, we use the R101c model \[[7](#bib.bib7)\] which replaces the first 7×7777\\times 7 convolution layer of R101 with 3 consecutive 3×3333\\times 3 convolutions and which is popular in the semantic segmentation community \[[52](#bib.bib52), [8](#bib.bib8), [9](#bib.bib9), [23](#bib.bib23), [50](#bib.bib50), [11](#bib.bib11)\].  
骨干。MaskFormer 与任何骨干架构兼容。在我们的工作中，我们使用了基于卷积的标准 ResNet \[22\] 主干（R50 和 R101 分别有 50 层和 101 层）和最近提出的基于 Transformer 的 Swin-Transformer \[29\] 主干。此外，我们使用 R101c 模型 \[7\]，该模型将 R101 的第一个 7×7777\\times 7 卷积层替换为 3 个连续 3×3333\\times 3 卷积，该模型在语义分割社区中很流行 \[52， 8， 9， 23， 50， 11\]。

Pixel decoder. The pixel decoder in Figure [2](#S3.F2 "Figure 2 ‣ 3.3 MaskFormer ‣ 3 From Per-Pixel to Mask Classification ‣ Per-Pixel Classification is Not All You Need for Semantic Segmentation") can be implemented using any semantic segmentation decoder (_e.g_., \[[9](#bib.bib9), [10](#bib.bib10), [11](#bib.bib11)\]). Many per-pixel classification methods use modules like ASPP \[[7](#bib.bib7)\] or PSP \[[52](#bib.bib52)\] to collect and distribute context across locations. The Transformer module attends to all image features, collecting global information to generate class predictions. This setup reduces the need of the per-pixel module for heavy context aggregation. Therefore, for MaskFormer, we design a light-weight pixel decoder based on the popular FPN \[[26](#bib.bib26)\] architecture.  
像素解码器。图 2 中的像素解码器可以使用任何语义分割解码器（例如，\[9,10,11\]）来实现。许多每像素分类方法使用 ASPP \[7\] 或 PSP \[52\] 等模块来收集和分发跨位置的上下文。Transformer 模块关注所有图像特征，收集全局信息以生成类预测。此设置减少了对每像素模块进行大量上下文聚合的需求。因此，对于 MaskFormer，我们设计了一个基于流行的 FPN \[26\] 架构的轻量级像素解码器。

Following FPN, we 2×2\\times upsample the low-resolution feature map in the decoder and sum it with the projected feature map of corresponding resolution from the backbone; Projection is done to match channel dimensions of the feature maps with a 1×1111\\times 1 convolution layer followed by GroupNorm (GN) \[[45](#bib.bib45)\]. Next, we fuse the summed features with an additional 3×3333\\times 3 convolution layer followed by GN and ReLU activation. We repeat this process starting with the stride 32 feature map until we obtain a final feature map of stride 4. Finally, we apply a single 1×1111\\times 1 convolution layer to get the per-pixel embeddings. All feature maps in the pixel decoder have a dimension of 256 channels.  
在 FPN 之后，我们对 2×2\\times 解码器中的低分辨率特征图进行上采样，并将其与来自主干的相应分辨率的投影特征图求和; 进行投影是为了将特征图的通道维度与 1×1111\\times 1 卷积层匹配，然后是 GroupNorm（GN）\[45\]。接下来，我们将求和特征与额外的 3×3333\\times 3 卷积层融合，然后进行 GN 和 ReLU 激活。我们从步幅 32 特征图开始重复此过程，直到获得步幅 4 的最终特征图。最后，我们应用单个 1×1111\\times 1 卷积层来获得每个像素的嵌入。像素解码器中的所有特征图的尺寸为 256 个通道。

Transformer decoder. We use the same Transformer decoder design as DETR \[[4](#bib.bib4)\]. The N𝑁N query embeddings are initialized as zero vectors, and we associate each query with a learnable positional encoding. We use 6 Transformer decoder layers with 100 queries by default, and, following DETR, we apply the same loss after each decoder. In our experiments we observe that MaskFormer is competitive for semantic segmentation with a single decoder layer too, whereas for instance-level segmentation multiple layers are necessary to remove duplicates from the final predictions.  
变压器解码器。我们使用与 DETR \[4\] 相同的 Transformer 解码器设计。 N𝑁N 查询嵌入初始化为零向量，我们将每个查询与可学习的位置编码相关联。默认情况下，我们使用 6 个 Transformer 解码器层和 100 个查询，并且在 DETR 之后，我们在每个解码器后应用相同的损失。在我们的实验中，我们观察到 MaskFormer 在使用单个解码器层进行语义分割方面也具有竞争力，而对于实例级分割，需要多个层才能从最终预测中删除重复项。

Segmentation module. The multi-layer perceptron (MLP) in Figure [2](#S3.F2 "Figure 2 ‣ 3.3 MaskFormer ‣ 3 From Per-Pixel to Mask Classification ‣ Per-Pixel Classification is Not All You Need for Semantic Segmentation") has 2 hidden layers of 256 channels to predict the mask embeddings ℰmasksubscriptℰmask\\mathcal{E}_{\\text{mask}}, analogously to the box head in DETR. Both per-pixel ℰpixelsubscriptℰpixel\\mathcal{E}_{\\text{pixel}} and mask ℰmasksubscriptℰmask\\mathcal{E}_{\\text{mask}} embeddings have 256 channels.  
分段模块。图 2 中的多层感知器 （MLP） 有 2 个隐藏层，每层 256 个通道用于预测掩模嵌入 ℰmasksubscriptℰmask\\mathcal{E}_{\\text{mask}} ，类似于 DETR 中的盒头。每像素 ℰpixelsubscriptℰpixel\\mathcal{E}_{\\text{pixel}} 和蒙版 ℰmasksubscriptℰmask\\mathcal{E}_{\\text{mask}} 嵌入都有 256 个通道。

Loss weights. We use focal loss \[[27](#bib.bib27)\] and dice loss \[[33](#bib.bib33)\] for our mask loss: ℒmask​(m,mgt)=λfocal​ℒfocal​(m,mgt)+λdice​ℒdice​(m,mgt)subscriptℒmask𝑚superscript𝑚gtsubscript𝜆focalsubscriptℒfocal𝑚superscript𝑚gtsubscript𝜆dicesubscriptℒdice𝑚superscript𝑚gt\\mathcal{L}_{\\text{mask}}(m,m^{\\text{gt}})=\\lambda_{\\text{focal}}\\mathcal{L}_{\\text{focal}}(m,m^{\\text{gt}})+\\lambda_{\\text{dice}}\\mathcal{L}_{\\text{dice}}(m,m^{\\text{gt}}), and set the hyper-parameters to λfocal=20.0subscript𝜆focal20.0\\lambda_{\\text{focal}}=20.0 and λdice=1.0subscript𝜆dice1.0\\lambda_{\\text{dice}}=1.0. Following DETR \[[4](#bib.bib4)\], the weight for the “no object” (∅\\varnothing) in the classification loss is set to 0.1.  
减肥重量。我们使用焦点损失 \[27\] 和骰子损失 \[33\] 来表示掩模损失： ℒmask​(m,mgt)=λfocal​ℒfocal​(m,mgt)+λdice​ℒdice​(m,mgt)subscriptℒmask𝑚superscript𝑚gtsubscript𝜆focalsubscriptℒfocal𝑚superscript𝑚gtsubscript𝜆dicesubscriptℒdice𝑚superscript𝑚gt\\mathcal{L}_{\\text{mask}}(m,m^{\\text{gt}})=\\lambda_{\\text{focal}}\\mathcal{L}_{\\text{focal}}(m,m^{\\text{gt}})+\\lambda_{\\text{dice}}\\mathcal{L}_{\\text{dice}}(m,m^{\\text{gt}}) ，并将超参数设置为 λfocal=20.0subscript𝜆focal20.0\\lambda_{\\text{focal}}=20.0 和 λdice=1.0subscript𝜆dice1.0\\lambda_{\\text{dice}}=1.0 。根据 DETR \[4\]，分类损失中“无对象”（ ∅\\varnothing ）的权重设置为 0.1。

### 4.2 Training settings4.2 培训设置

Semantic segmentation. We use Detectron2 \[[46](#bib.bib46)\] and follow the commonly used training settings for each dataset. More specifically, we use AdamW \[[31](#bib.bib31)\] and the _poly_ \[[7](#bib.bib7)\] learning rate schedule with an initial learning rate of 10−4superscript10410^{-4} and a weight decay of 10−4superscript10410^{-4} for ResNet \[[22](#bib.bib22)\] backbones, and an initial learning rate of 6⋅10−5⋅6superscript1056\\cdot 10^{-5} and a weight decay of 10−2superscript10210^{-2} for Swin-Transformer \[[29](#bib.bib29)\] backbones. Backbones are pre-trained on ImageNet-1K \[[35](#bib.bib35)\] if not stated otherwise. A learning rate multiplier of 0.10.10.1 is applied to CNN backbones and 1.01.01.0 is applied to Transformer backbones. The standard random scale jittering between 0.50.50.5 and 2.02.02.0, random horizontal flipping, random cropping as well as random color jittering are used as data augmentation \[[14](#bib.bib14)\]. For the ADE20K dataset, if not stated otherwise, we use a crop size of 512×512512512512\\times 512, a batch size of 161616 and train all models for 160k iterations. For the ADE20K-Full dataset, we use the same setting as ADE20K except that we train all models for 200k iterations. For the COCO-Stuff-10k dataset, we use a crop size of 640×640640640640\\times 640, a batch size of 32 and train all models for 60k iterations. All models are trained with 8 V100 GPUs. We report both performance of single scale (s.s.) inference and multi-scale (m.s.) inference with horizontal flip and scales of 0.50.50.5, 0.750.750.75, 1.01.01.0, 1.251.251.25, 1.51.51.5, 1.751.751.75. See appendix for Cityscapes and Mapillary Vistas settings.  
语义分割。我们使用 Detectron2 \[46\] 并遵循每个数据集的常用训练设置。更具体地说，我们使用 AdamW \[31\] 和 poly \[7\] 学习率表，ResNet \[22\] 主干的初始学习率 10−4superscript10410^{-4} 和权重衰减为 Swin-Transformer 10−4superscript10410^{-4} \[29\] 主干的初始学习率 6⋅10−5⋅6superscript1056\\cdot 10^{-5} 和权重衰减 10−2superscript10210^{-2} 。如果没有特别说明，主干网会在 ImageNet-1K \[35\] 上进行预训练。学习率乘数应用于 0.10.10.1 CNN 主干网，并 1.01.01.0 应用于 Transformer 主干网。使用标准随机尺度在和 2.02.02.0 之间 0.50.50.5 抖动、随机水平翻转、随机裁剪以及随机颜色抖动作为数据增强\[14\]。对于 ADE20K 数据集，如果没有另行说明，我们使用 的裁剪大小 512×512512512512\\times 512 ，批量大小 ， 161616 并训练所有模型进行 160k 迭代。对于 ADE20K-Full 数据集，我们使用与 ADE20K 相同的设置，只是我们训练所有模型进行 200k 迭代。对于 COCO-Stuff-10k 数据集，我们使用 640×640640640640\\times 640 裁剪大小 ，批处理大小为 32，并训练所有模型进行 60k 迭代。所有型号均使用 8 个 V100 GPU 进行训练。我们报告了单尺度 （s.s.） 推理和多尺度 （m.s.） 推理的性能，水平翻转和尺度为 0.50.50.5 、 0.750.750.75 、 1.01.01.0 、 1.251.251.25 1.51.51.5 1.751.751.75 、 。有关 Cityscapes 和 Mapillary Vistas 设置，请参阅附录。

Panoptic segmentation. We follow exactly the same architecture, loss, and training procedure as we use for semantic segmentation. The only difference is supervision: _i.e_., category region masks in semantic segmentation _vs_. object instance masks in panoptic segmentation. We strictly follow the DETR \[[4](#bib.bib4)\] setting to train our model on the COCO panoptic segmentation dataset \[[24](#bib.bib24)\] for a fair comparison. On the ADE20K panoptic segmentation dataset, we follow the semantic segmentation setting but train for longer (720k iterations) and use a larger crop size (640×640640640640\\times 640). COCO models are trained using 64 V100 GPUs and ADE20K experiments are trained with 8 V100 GPUs. We use the general inference (Section [3.4](#S3.SS4 "3.4 Mask-classification inference ‣ 3 From Per-Pixel to Mask Classification ‣ Per-Pixel Classification is Not All You Need for Semantic Segmentation")) with the following parameters: we filter out masks with class confidence below 0.8 and set masks whose contribution to the final panoptic segmentation is less than 80% of its mask area to VOID. We report performance of single scale inference.  
全景分割。我们遵循与语义分割完全相同的架构、损失和训练过程。唯一的区别是监督：即语义分割中的类别区域掩码与全景分割中的对象实例掩码。我们严格遵循 DETR \[4\] 设置，在 COCO 全景分割数据集 \[24\] 上训练我们的模型，以便进行公平的比较。在 ADE20K 全景分割数据集上，我们遵循语义分割设置，但训练时间更长（720k 迭代）并使用更大的裁剪大小 （ 640×640640640640\\times 640 ）。COCO 模型使用 64 个 V100 GPU 进行训练，ADE20K 实验使用 8 个 V100 GPU 进行训练。我们使用具有以下参数的一般推理（第 3.4 节）：我们过滤掉类置信度低于 0.8 的掩模，并将对最终全景分割的贡献小于其遮罩面积 80% 的遮罩设置为 VOID。我们报告了单尺度推理的性能。

Table 1: Semantic segmentation on ADE20K val with 150 categories. Mask classification-based MaskFormer outperforms the best per-pixel classification approaches while using fewer parameters and less computation. We report both single-scale (s.s.) and multi-scale (m.s.) inference results with ±plus-or-minus\\pmstd. FLOPs are computed for the given crop size. Frames-per-second (fps) is measured on a V100 GPU with a batch size of 1.444It isn’t recommended to compare fps from different papers: speed is measured in different environments. DeepLabV3+ fps are from MMSegmentation \[[14](#bib.bib14)\], and Swin-UperNet fps are from the original paper \[[29](#bib.bib29)\]. Backbones pre-trained on ImageNet-22K are marked with ††{}^{\\text{\\textdagger}}.  
表 1：ADE20K val 的语义分割，包含 150 个类别。基于掩模分类的 MaskFormer 优于最佳的每像素分类方法，同时使用更少的参数和更少的计算。我们报告了单尺度 （s.s.） 和多尺度 （m.s.） 推理结果，并针对 ±plus-or-minus\\pm 给定的作物大小计算了 FLOP。每秒帧数 （fps） 是在批处理大小为 1 的 V100 GPU 上测量的。 4 在 ImageNet-22K 上预训练的主干网标有 ††{}^{\\text{\\textdagger}} 。

|  | method | backbone | crop size 作物大小 | mIoU (s.s.)mIoU （s.s.） | mIoU (m.s.)mIoU （硕士） | #params. | FLOPs | fps |

| CNN backbonesCNN 骨干网 | OCRNet \[[50](#bib.bib50)\] | R101c | 520×520520520520\\times 520 | \- ±plus-or-minus\\pm0.5 | 45.3 ±plus-or-minus\\pm0.5 | - | - | - |

| DeepLabV3+ \[[9](#bib.bib9)\]DeepLabV3+（深实验室 V3+） \[9\] | 0R50c | 512×512512512512\\times 512 | 44.0 ±plus-or-minus\\pm0.5 | 44.9 ±plus-or-minus\\pm0.5 | 044M | 177G | 21.0 |

| R101c | 512×512512512512\\times 512 | 45.5 ±plus-or-minus\\pm0.5 | 46.4 ±plus-or-minus\\pm0.5 | 063M | 255G | 14.2 |

| MaskFormer (ours)MaskFormer（我们的） | 0R50c | 512×512512512512\\times 512 | 44.5 ±plus-or-minus\\pm0.5 | 46.7 ±plus-or-minus\\pm0.6 | 041M | 053G | 24.5 |

| R101c | 512×512512512512\\times 512 | 45.5 ±plus-or-minus\\pm0.5 | 47.2 ±plus-or-minus\\pm0.2 | 060M | 073G | 19.5 |

| R101c | 512×512512512512\\times 512 | 46.0  ±plus-or-minus\\pm0.1 | 48.1  ±plus-or-minus\\pm0.2 | 060M | 080G | 19.0 |

| Transformer backbones 变压器主干网 | SETR \[[53](#bib.bib53)\] 塞特 53 | ViT-L††{}^{\\text{\\textdagger}}ViT-L ††{}^{\\text{\\textdagger}} 型 | 512×512512512512\\times 512 | \- ±plus-or-minus\\pm0.5 | 50.3 ±plus-or-minus\\pm0.5 | 308M | - | - |

| Swin-UperNet \[[29](#bib.bib29), [49](#bib.bib49)\] 斯温 - 乌珀网 \[29， 49\] | Swin-T††{}^{\\text{\\textdagger}}Swin-T ††{}^{\\text{\\textdagger}} 型 | 512×512512512512\\times 512 | \- ±plus-or-minus\\pm0.5 | 46.1 ±plus-or-minus\\pm0.5 | 060M | 236G | 18.5 |

| Swin-S††{}^{\\text{\\textdagger}}斯温 -S ††{}^{\\text{\\textdagger}} | 512×512512512512\\times 512 | \- ±plus-or-minus\\pm0.5 | 49.3 ±plus-or-minus\\pm0.5 | 081M | 259G | 15.2 |

| Swin-B††{}^{\\text{\\textdagger}}Swin-B ††{}^{\\text{\\textdagger}} 型 | 640×640640640640\\times 640 | \- ±plus-or-minus\\pm0.5 | 51.6 ±plus-or-minus\\pm0.5 | 121M | 471G | 08.7 |

| Swin-L††{}^{\\text{\\textdagger}}斯温 -L ††{}^{\\text{\\textdagger}} | 640×640640640640\\times 640 | \- ±plus-or-minus\\pm0.5 | 53.5 ±plus-or-minus\\pm0.5 | 234M | 647G | 06.2 |

| MaskFormer (ours)MaskFormer（我们的） | Swin-T††{}^{\\text{\\textdagger}}Swin-T ††{}^{\\text{\\textdagger}} 型 | 512×512512512512\\times 512 | 46.7 ±plus-or-minus\\pm0.7 | 48.8 ±plus-or-minus\\pm0.6 | 042M | 055G | 22.1 |

| Swin-S††{}^{\\text{\\textdagger}}斯温 -S ††{}^{\\text{\\textdagger}} | 512×512512512512\\times 512 | 49.8 ±plus-or-minus\\pm0.4 | 51.0 ±plus-or-minus\\pm0.4 | 063M | 079G | 19.6 |

| Swin-B††{}^{\\text{\\textdagger}}Swin-B ††{}^{\\text{\\textdagger}} 型 | 640×640640640640\\times 640 | 51.1 ±plus-or-minus\\pm0.2 | 52.3 ±plus-or-minus\\pm0.4 | 102M | 195G | 12.6 |

| Swin-B††{}^{\\text{\\textdagger}}Swin-B ††{}^{\\text{\\textdagger}} 型 | 640×640640640640\\times 640 | 52.7 ±plus-or-minus\\pm0.4 | 53.9 ±plus-or-minus\\pm0.2 | 102M | 195G | 12.6 |

| Swin-L††{}^{\\text{\\textdagger}}斯温 -L ††{}^{\\text{\\textdagger}} | 640×640640640640\\times 640 | 54.1  ±plus-or-minus\\pm0.2 | 55.6  ±plus-or-minus\\pm0.1 | 212M | 375G | 07.9 |

Table 2: MaskFormer _vs_. per-pixel classification baselines on 4 semantic segmentation datasets. MaskFormer improvement is larger when the number of classes is larger. We use a ResNet-50 backbone and report single scale mIoU and PQStSt{}^{\\text{St}} for ADE20K, COCO-Stuff and ADE20K-Full, whereas for higher-resolution Cityscapes we use a deeper ResNet-101 backbone following \[[8](#bib.bib8), [9](#bib.bib9)\].

|  | Cityscapes (19 classes) | ADE20K (150 classes) | COCO-Stuff (171 classes) | ADE20K-Full (847 classes) |

|  | mIoU | PQStSt{}^{\\text{St}} | mIoU | PQStSt{}^{\\text{St}} | mIoU | PQStSt{}^{\\text{St}} | mIoU | PQStSt{}^{\\text{St}} |

| PerPixelBaseline | 77.4  (+0.0) | 58.9  (+0.0) | 39.2  (+0.0) | 21.6  (+0.0) | 32.4  (+0.0) | 15.5  (+0.0) | 12.4  (+0.0) | 05.8  (+0.0) |

| PerPixelBaseline+ | 78.5  (+0.0) | 60.2 (+0.0) | 41.9 (+0.0) | 28.3 (+0.0) | 34.2 (+0.0) | 24.6 (+0.0) | 13.9 (+0.0) | 09.0 (+0.0) |

| MaskFormer (ours) | 78.5  (+0.0) | 63.1  (+2.9) | 44.5  (+2.6) | 33.4  (+5.1) | 37.1  (+2.9) | 28.9  (+4.3) | 17.4  (+3.5) | 11.9  (+2.9) |

### 4.3 Main results

Semantic segmentation. In Table [1](#S4.T1 "Table 1 ‣ 4.2 Training settings ‣ 4 Experiments ‣ Per-Pixel Classification is Not All You Need for Semantic Segmentation"), we compare MaskFormer with state-of-the-art per-pixel classification models for semantic segmentation on the ADE20K val set. With the same standard CNN backbones (_e.g_., ResNet \[[22](#bib.bib22)\]), MaskFormer outperforms DeepLabV3+ \[[9](#bib.bib9)\] by 1.7 mIoU. MaskFormer is also compatible with recent Vision Transformer \[[17](#bib.bib17)\] backbones (_e.g_., the Swin Transformer \[[29](#bib.bib29)\]), achieving a new state-of-the-art of 55.6 mIoU, which is 2.1 mIoU better than the prior state-of-the-art \[[29](#bib.bib29)\]. Observe that MaskFormer outperforms the best per-pixel classification-based models while having fewer parameters and faster inference time. This result suggests that the mask classification formulation has significant potential for semantic segmentation. See appendix for results on test set.

Beyond ADE20K, we further compare MaskFormer with our baselines on COCO-Stuff-10K, ADE20K-Full as well as Cityscapes in Table [2](#S4.T2 "Table 2 ‣ 4.2 Training settings ‣ 4 Experiments ‣ Per-Pixel Classification is Not All You Need for Semantic Segmentation") and we refer to the appendix for comparison with state-of-the-art methods on these datasets. The improvement of MaskFormer over PerPixelBaseline+ is larger when the number of classes is larger: For Cityscapes, which has only 19 categories, MaskFormer performs similarly well as PerPixelBaseline+; While for ADE20K-Full, which has 847 classes, MaskFormer outperforms PerPixelBaseline+ by 3.5 mIoU.  
除了 ADE20K 之外，我们还进一步将 MaskFormer 与表 2 中的 COCO-Stuff-10K、ADE20K-Full 以及 Cityscapes 的基线进行了比较，并参考附录与这些数据集上的最新方法进行了比较。当类数量较多时，MaskFormer 相对于 PerPixelBaseline+ 的改进更大：对于只有 19 个类别的 Cityscapes，MaskFormer 的表现与 PerPixelBaseline+ 类似; 而对于具有 847 个类别的 ADE20K-Full，MaskFormer 的性能比 PerPixelBaseline+ 高出 3.5 mIoU。

Although MaskFormer shows no improvement in mIoU for Cityscapes, the PQStSt{}^{\\text{St}} metric increases by 2.9 PQStSt{}^{\\text{St}}. We find MaskFormer performs better in terms of recognition quality (RQStSt{}^{\\text{St}}) while lagging in per-pixel segmentation quality (SQStSt{}^{\\text{St}}) (we refer to the appendix for detailed numbers). This observation suggests that on datasets where class recognition is relatively easy to solve, the main challenge for mask classification-based approaches is pixel-level accuracy (_i.e_., mask quality).  
尽管 MaskFormer 在城市景观的 mIoU 中没有显示任何改进，但 PQ StSt{}^{\\text{St}} 指标增加了 2.9 PQ StSt{}^{\\text{St}} 。我们发现 MaskFormer 在识别质量（RQ StSt{}^{\\text{St}} ）方面表现更好，而在每像素分割质量（SQ StSt{}^{\\text{St}} ）方面表现较差（详见附录）。这一观察结果表明，在类识别相对容易解决的数据集上，基于掩码分类的方法的主要挑战是像素级精度（即掩码质量）。

Table 3: Panoptic segmentation on COCO panoptic val with 133 categories. MaskFormer seamlessly unifies semantic- and instance-level segmentation without modifying the model architecture or loss. Our model, which achieves better results, can be regarded as a box-free simplification of DETR \[[4](#bib.bib4)\]. The major improvement comes from “stuff” classes (PQStSt{}^{\\text{St}}) which are ambiguous to represent with bounding boxes. For MaskFormer (DETR) we use the exact same post-processing as DETR. Note, that in this setting MaskFormer performance is still better than DETR (+2.2 PQ). Our model also outperforms recently proposed Max-DeepLab \[[42](#bib.bib42)\] without the need of sophisticated auxiliary losses, while being more efficient. FLOPs are computed as the average FLOPs over 100 validation images (COCO images have varying sizes). Frames-per-second (fps) is measured on a V100 GPU with a batch size of 1 by taking the average runtime on the entire val set _including post-processing time_. Backbones pre-trained on ImageNet-22K are marked with ††{}^{\\text{\\textdagger}}.  
表 3：COCO 全景值的 133 个类别的全景细分。MaskFormer 无缝统一了语义级和实例级分段，而不会修改模型架构或丢失。我们的模型取得了更好的结果，可以看作是 DETR 的无框简化\[4\]。主要的改进来自“东西”类（PQ StSt{}^{\\text{St}} ），这些类用边界框表示是模棱两可的。对于 MaskFormer （DETR），我们使用与 DETR 完全相同的后处理。请注意，在此设置中，MaskFormer 的性能仍然优于 DETR （+2.2 PQ）。我们的模型也优于最近提出的 Max-DeepLab\[42\]，不需要复杂的辅助损耗，同时效率更高。FLOP 计算为 100 个验证图像的平均 FLOP（COCO 图像的大小各不相同）。每秒帧数 （fps） 是在批处理大小为 1 的 V100 GPU 上通过获取整个 val 集的平均运行时间（包括后处理时间）来测量的。在 ImageNet-22K 上预训练的主干网标有 ††{}^{\\text{\\textdagger}} 。

|  | method | backbone | PQ | PQThTh{}^{\\text{Th}}餐前 ThTh{}^{\\text{Th}} | PQStSt{}^{\\text{St}}餐前 StSt{}^{\\text{St}} | SQ | RQ | #params. | FLOPs | fps |

| CNN backbonesCNN 骨干网 | DETR \[[4](#bib.bib4)\] 德特 \[4\] | 0R50 + 6 Enc | 43.4 | 48.2 (+0.2) | 36.3 (+2.4) | 79.3 | 53.8 | - | - | - |

| MaskFormer (DETR) 掩模成型 （DETR） | 0R50 + 6 Enc | 45.6 | 50.0 (+1.8) | 39.0 (+2.7) | 80.2 | 55.8 | - | - | - |

| MaskFormer (ours)MaskFormer（我们的） | 0R50 + 6 Enc | 46.5 | 51.0  (+2.8) | 39.8  (+3.5) | 80.4 | 56.8 | 045M | 0181G | 17.6 |

| DETR \[[4](#bib.bib4)\] 德特 \[4\] | R101 + 6 EncR101 + 6 恩加 | 45.1 | 50.5 (+0.2) | 37.0 (+2.4) | 79.9 | 55.5 | - | - | - |

| MaskFormer (ours)MaskFormer（我们的） | R101 + 6 EncR101 + 6 恩加 | 47.6 | 52.5  (+2.0) | 40.3  (+3.3) | 80.7 | 58.0 | 064M | 0248G | 14.0 |

| Transformer backbones 变压器主干网 | Max-DeepLab \[[42](#bib.bib42)\] 马克斯深度实验室 \[42\] | Max-S | 48.4 | 53.0 (+0.2) | 41.5 (+0.2) | - | - | 062M | 0324G | 07.6 |

| Max-L | 51.1 | 57.0 (+0.2) | 42.2 (+0.2) | - | - | 451M | 3692G | - |

| MaskFormer (ours)MaskFormer（我们的） | Swin-T††{}^{\\text{\\textdagger}}Swin-T ††{}^{\\text{\\textdagger}} 型 | 47.7 | 51.7 (+0.2) | 41.7 (+0.2) | 80.4 | 58.3 | 042M | 0179G | 17.0 |

| Swin-S††{}^{\\text{\\textdagger}}斯温 -S ††{}^{\\text{\\textdagger}} | 49.7 | 54.4 (+0.2) | 42.6 (+0.2) | 80.9 | 60.4 | 063M | 0259G | 12.4 |

| Swin-B††{}^{\\text{\\textdagger}}Swin-B ††{}^{\\text{\\textdagger}} 型 | 51.1 | 56.3 (+0.2) | 43.2 (+0.2) | 81.4 | 61.8 | 102M | 0411G | 08.4 |

| Swin-B††{}^{\\text{\\textdagger}}Swin-B ††{}^{\\text{\\textdagger}} 型 | 51.8 | 56.9 (+0.2) | 44.1  (+0.2) | 81.4 | 62.6 | 102M | 0411G | 08.4 |

| Swin-L††{}^{\\text{\\textdagger}}斯温 -L ††{}^{\\text{\\textdagger}} | 52.7 | 58.5  (+0.2) | 44.0  (+0.2) | 81.8 | 63.5 | 212M | 0792G | 05.2 |

Panoptic segmentation. In Table [3](#S4.T3 "Table 3 ‣ 4.3 Main results ‣ 4 Experiments ‣ Per-Pixel Classification is Not All You Need for Semantic Segmentation"), we compare the same exact MaskFormer model with DETR \[[4](#bib.bib4)\] on the COCO panoptic val set. To match the standard DETR design, we add 6 additional Transformer encoder layers after the CNN backbone. Unlike DETR, our model does not predict bounding boxes but instead predicts masks directly. MaskFormer achieves better results while being simpler than DETR. To disentangle the improvements from the model itself and our post-processing inference strategy we run our model following DETR post-processing (MaskFormer (DETR)) and observe that this setup outperforms DETR by 2.2 PQ. Overall, we observe a larger improvement in PQStSt{}^{\\text{St}} compared to PQThTh{}^{\\text{Th}}. This suggests that detecting “stuff” with bounding boxes is suboptimal, and therefore, box-based segmentation models (_e.g_., Mask R-CNN \[[21](#bib.bib21)\]) do not suit semantic segmentation. MaskFormer also outperforms recently proposed Max-DeepLab \[[42](#bib.bib42)\] without the need of special network design as well as sophisticated auxiliary losses (_i.e_., instance discrimination loss, mask-ID cross entropy loss, and per-pixel classification loss in \[[42](#bib.bib42)\]). _MaskFormer, for the first time, unifies semantic- and instance-level segmentation with the exact same model, loss, and training pipeline._  
全景分割。在表 3 中，我们在 COCO 全景值集上比较了相同的 MaskFormer 模型和 DETR \[4\]。为了匹配标准的 DETR 设计，我们在 CNN 主干网之后增加了 6 个额外的 Transformer 编码器层。与 DETR 不同，我们的模型不预测边界框，而是直接预测掩码。MaskFormer 比 DETR 更简单，但效果更好。为了将改进与模型本身和我们的后处理推理策略区分开来，我们按照 DETR 后处理 （MaskFormer （DETR）） 运行模型，并观察到此设置的性能比 DETR 高出 2.2 PQ。总体而言，我们观察到与 PQ ThTh{}^{\\text{Th}} 相比，PQ StSt{}^{\\text{St}} 的改善更大。这表明，用边界框检测“东西”是次优的，因此，基于框的分割模型（例如，Mask R-CNN \[21\]）不适合语义分割。MaskFormer 的性能也优于最近提出的 Max-DeepLab \[42\]，无需特殊的网络设计以及复杂的辅助损失（即\[42\] 中的实例区分损失、掩码 -ID 交叉熵损失和每像素分类损失）。MaskFormer 首次将语义级和实例级分割与完全相同的模型、损失和训练管道统一起来。

We further evaluate our model on the panoptic segmentation version of the ADE20K dataset. Our model also achieves state-of-the-art performance. We refer to the appendix for detailed results.  
我们在 ADE20K 数据集的全景分割版本上进一步评估了我们的模型。我们的模型还实现了最先进的性能。有关详细结果，请参阅附录。

### 4.4 Ablation studies4.4 消融研究

We perform a series of ablation studies of MaskFormer using a single ResNet-50 backbone \[[22](#bib.bib22)\].  
我们使用单个 ResNet-50 骨架对 MaskFormer 进行了一系列消融研究\[22\]。

Per-pixel _vs_. mask classification. In Table [4b](#S4.T4.sf2 "In Table 4 ‣ 4.4 Ablation studies ‣ 4 Experiments ‣ Per-Pixel Classification is Not All You Need for Semantic Segmentation"), we verify that the gains demonstrated by MaskFromer come from shifting the paradigm to mask classification. We start by comparing PerPixelBaseline+ and MaskFormer. The models are very similar and there are only 3 differences: 1) per-pixel _vs_. mask classification used by the models, 2) MaskFormer uses bipartite matching, and 3) the new model uses a combination of focal and dice losses as a mask loss, whereas PerPixelBaseline+ utilizes per-pixel cross entropy loss. First, we rule out the influence of loss differences by training PerPixelBaseline+ with exactly the same losses and observing no improvement. Next, in Table [4a](#S4.T4.sf1 "In Table 4 ‣ 4.4 Ablation studies ‣ 4 Experiments ‣ Per-Pixel Classification is Not All You Need for Semantic Segmentation"), we compare PerPixelBaseline+ with MaskFormer trained using a fixed matching (MaskFormer-fixed), _i.e_., N=K𝑁𝐾N=K and assignment done based on category label indices identically to the per-pixel classification setup. We observe that MaskFormer-fixed is 1.8 mIoU better than the baseline, suggesting that shifting from per-pixel classification to mask classification is indeed the main reason for the gains of MaskFormer. In Table [4b](#S4.T4.sf2 "In Table 4 ‣ 4.4 Ablation studies ‣ 4 Experiments ‣ Per-Pixel Classification is Not All You Need for Semantic Segmentation"), we further compare MaskFormer-fixed with MaskFormer trained with bipartite matching (MaskFormer-bipartite) and find bipartite matching is not only more flexible (allowing to predict less masks than the total number of categories) but also produces better results.  
每像素与蒙版分类。在表 4b 中，我们验证了 MaskFromer 所展示的收益来自将范式转变为掩模分类。我们首先比较 PerPixelBaseline+ 和 MaskFormer。这些模型非常相似，只有 3 个差异：1） 模型使用的每像素与掩码分类，2） MaskFormer 使用二分匹配，以及 3） 新模型使用焦点和骰子损失的组合作为掩码损失，而 PerPixelBaseline+ 利用每像素交叉熵损失。首先，我们通过训练具有完全相同损失的 PerPixelBaseline+ 并且没有观察到任何改进来排除损失差异的影响。接下来，在表 4a 中，我们将 PerPixelBaseline+ 与使用固定匹配（MaskFormer-fixed）训练的 MaskFormer 进行比较， N=K𝑁𝐾N=K 即基于类别标签索引完成的赋值与每像素分类设置相同。我们观察到 MaskFormer-fixed 比基线高 1.8 mIoU，这表明从每像素分类转向掩模分类确实是 MaskFormer 收益的主要原因。在表 4b 中，我们进一步比较了 MaskFormer-fixed 和用二分匹配训练的 MaskFormer（MaskFormer-bipartite），发现二分匹配不仅更灵活（允许预测的掩码少于类别总数），而且产生更好的结果。

Table 4: Per-pixel _vs_. mask classification for semantic segmentation. All models use 150 queries for a fair comparison. We evaluate the models on ADE20K val with 150 categories. [4a](#S4.T4.sf1 "In Table 4 ‣ 4.4 Ablation studies ‣ 4 Experiments ‣ Per-Pixel Classification is Not All You Need for Semantic Segmentation"): PerPixelBaseline+ and MaskFormer-fixed use similar fixed matching (_i.e_., matching by category index), this result confirms that the shift from per-pixel to mask classification is the key. [4b](#S4.T4.sf2 "In Table 4 ‣ 4.4 Ablation studies ‣ 4 Experiments ‣ Per-Pixel Classification is Not All You Need for Semantic Segmentation"): bipartite matching is not only more flexible (can make less prediction than total class count) but also gives better results.  
表 4：语义分割的每像素与掩码分类。所有模型都使用 150 个查询进行公平比较。我们在 ADE20K val 上评估了 150 个类别的模型。图 4a：PerPixelBaseline+ 和 MaskFormer-fixed 使用类似的固定匹配（即按类别索引匹配），此结果证实了从每像素到蒙版分类的转变是关键。4B：二分匹配不仅更灵活（可以做出的预测比总类数少），而且提供更好的结果。

|  | mIoU | PQStSt{}^{\\text{St}}餐前 StSt{}^{\\text{St}} |

| PerPixelBaseline+ | 41.9 (+0.0) | 28.3 (+0.0) |

| MaskFormer-fixed | 43.7  (+1.8) | 30.3  (+2.0) |

(a) Per-pixel _vs_. mask classification.  
（一）每像素与蒙版分类。

|  | mIoU | PQStSt{}^{\\text{St}}餐前 StSt{}^{\\text{St}} |

| MaskFormer-fixed | 43.7 (+0.0) | 30.3 (+0.0) |

| MaskFormer-bipartite (ours)  
MaskFormer-bipartite （我们的） | 44.2  (+0.5) | 33.4  (+3.1) |

(b) Fixed _vs_. bipartite matching assignment.  
（二）固定匹配分配与二分匹配分配。

|  | ADE20K | COCO-Stuff | ADE20K-Full |

| \# of queries\# 个查询 | mIoU | PQStSt{}^{\\text{St}}餐前 StSt{}^{\\text{St}} | mIoU | PQStSt{}^{\\text{St}}餐前 StSt{}^{\\text{St}} | mIoU | PQStSt{}^{\\text{St}}餐前 StSt{}^{\\text{St}} |

| PerPixelBaseline+ | 41.9 | 28.3 | 34.2 | 24.6 | 13.9 | 9.0 |

| 20 | 42.9 | 32.6 | 35.0 | 27.6 | 14.1 | 10.8 |

| 50 | 43.9 | 32.7 | 35.5 | 27.9 | 15.4 | 11.1 |

| 100 | 44.5 | 33.4 | 37.1 | 28.9 | 16.0 | 11.9 |

| 150 | 44.2 | 33.4 | 37.0 | 28.9 | 15.5 | 11.5 |

| 300 | 43.5 | 32.3 | 36.1 | 29.1 | 14.2 | 10.3 |

| 1000 | 35.4 | 26.7 | 34.4 | 27.6 | 08.0 | 05.8 |

Number of queries. The table to the right shows results of MaskFormer trained with a varying number of queries on datasets with different number of categories. The model with 100 queries consistently performs the best across the studied datasets. This suggest we may not need to adjust the number of queries w.r.t. the number of categories or datasets much. Interestingly, even with 20 queries MaskFormer outperforms our per-pixel classification baseline.  
查询数。右表显示了 MaskFormer 的结果，这些结果使用不同数量的查询对具有不同类别数的数据集进行训练。具有 100 个查询的模型在所研究的数据集中始终表现最佳。这表明我们可能不需要过多地调整查询的数量，而不是类别或数据集的数量。有趣的是，即使有 20 个查询，MaskFormer 的性能也优于我们的每像素分类基线。

We further calculate the number of classes which are on average present in a _training set_ image. We find these statistics to be similar across datasets despite the fact that the datasets have different number of total categories: 8.2 classes per image for ADE20K (150 classes), 6.6 classes per image for COCO-Stuff-10K (171 classes) and 9.1 classes per image for ADE20K-Full (847 classes). We hypothesize that each query is able to capture masks from multiple categories.  
我们进一步计算训练集图像中平均存在的类数。我们发现这些统计数据在数据集中是相似的，尽管数据集的总类别数量不同：ADE20K（150 个类）每个图像 8.2 个类，COCO-Stuff-10K（171 个类）每个图像 6.6 个类，ADE20K-Full（847 个类）每个图像 9.1 个类。我们假设每个查询都能够捕获来自多个类别的掩码。

| Number of unique classes predicted by each query on validation set  
验证集上每个查询预测的唯一类数 | ![](https://ar5iv.labs.arxiv.org/html/2107.06278/assets/x4.png)

 |

| (a) ADE20K (150 classes)（a） ADE20K（150 类） |

| ![](https://ar5iv.labs.arxiv.org/html/2107.06278/assets/x5.png)

 |

| (b) COCO-Stuff-10K (171 classes)  
（b） COCO-Stuff-10K（171 类） |

| ![](https://ar5iv.labs.arxiv.org/html/2107.06278/assets/x6.png)

 |

| (c) ADE20K-Full (847 classes)  
（c） ADE20K-Full （847 类） |

The figure to the right shows the number of unique categories predicted by each query (sorted in descending order) of our MaskFormer model on the validation sets of the corresponding datasets. Interestingly, the number of unique categories per query does not follow a uniform distribution: some queries capture more classes than others. We try to analyze how MaskFormer queries group categories, but we do not observe any obvious pattern: there are queries capturing categories with similar semantics or shapes (_e.g_., “house” and “building”), but there are also queries capturing completely different categories (_e.g_., “water” and “sofa”).  
右图显示了 MaskFormer 模型的每个查询（按降序排序）在相应数据集的验证集上预测的唯一类别数。有趣的是，每个查询的唯一类别数量并不遵循均匀分布：某些查询捕获的类比其他查询更多。我们试图分析 MaskFormer 查询如何对类别进行分组，但我们没有观察到任何明显的模式：有些查询捕获具有相似语义或形状的类别（例如，“房子”和“建筑物”），但也有一些查询捕获完全不同的类别（例如，“水”和“沙发”）。

Number of Transformer decoder layers. Interestingly, MaskFormer with even a single Transformer decoder layer already performs well for semantic segmentation and achieves better performance than our 6-layer-decoder PerPixelBaseline+. For panoptic segmentation, however, multiple decoder layers are required to achieve competitive performance. Please see the appendix for a detailed discussion.  
Transformer 解码器层数。有趣的是，即使是单个 Transformer 解码器层的 MaskFormer 在语义分割方面也表现出色，并且比我们的 6 层解码器 PerPixelBaseline+ 具有更好的性能。然而，对于全景分割，需要多个解码器层才能实现有竞争力的性能。详细讨论见附录。

## 5 Discussion 讨论

Our main goal is to show that mask classification is a general segmentation paradigm that could be a competitive alternative to per-pixel classification for semantic segmentation. To better understand its potential for segmentation tasks, we focus on exploring mask classification independently of other factors like architecture, loss design, or augmentation strategy. We pick the DETR \[[4](#bib.bib4)\] architecture as our baseline for its simplicity and deliberately make as few architectural changes as possible. Therefore, MaskFormer can be viewed as a “box-free” version of DETR.  
我们的主要目标是表明掩码分类是一种通用的分割范式，可以成为语义分割的每像素分类的竞争性替代方案。为了更好地了解其在分割任务中的潜力，我们专注于探索独立于其他因素（如架构、损失设计或增强策略）的掩码分类。我们选择 DETR \[4\] 架构作为我们的基准，因为它很简单，并特意进行尽可能少的架构更改。因此，MaskFormer 可以看作是 DETR 的“无盒”版本。

Table 5: Matching with masks _vs_. boxes. We compare DETR \[[4](#bib.bib4)\] which uses box-based matching with two MaskFormer models trained with box- and mask-based matching respectively. To use box-based matching in MaskFormer we add to the model an additional box prediction head as in DETR. Note, that with box-based matching MaskFormer performs on par with DETR, whereas with mask-based matching it shows better results. The evaluation is done on COCO panoptic val set.  
表 5：与蒙版与盒子的匹配。我们将使用基于盒的匹配的 DETR \[4\] 与分别使用基于盒和基于盒的匹配训练的两个 MaskFormer 模型进行了比较。为了在 MaskFormer 中使用基于框的匹配，我们在模型中添加了一个额外的框预测头，就像在 DETR 中一样。请注意，使用基于框的匹配时，MaskFormer 的性能与 DETR 相当，而使用基于掩码的匹配时，它显示的结果更好。评估是在 COCO 全景值集上完成的。

| method | backbone | matching | PQ | PQThTh{}^{\\text{Th}}餐前 ThTh{}^{\\text{Th}} | PQStSt{}^{\\text{St}}餐前 StSt{}^{\\text{St}} |

| DETR \[[4](#bib.bib4)\] 德特 \[4\] | R50 + 6 EncR50 + 6 恩加 | by box 按包装盒 | 43.4 | 48.2 | 36.3 |

| MaskFormer (ours)MaskFormer（我们的） | R50 + 6 EncR50 + 6 恩加 | by box 按包装盒 | 43.7 | 49.2 | 35.3 |

| R50 + 6 EncR50 + 6 恩加 | by mask 按面具 | 46.5 | 51.0 | 39.8 |

In this section, we discuss in detail the differences between MaskFormer and DETR and show how these changes are required to ensure that mask classification performs well. First, to achieve a pure mask classification setting we remove the box prediction head and perform matching between prediction and ground truth segments with masks instead of boxes. Secondly, we replace the compute-heavy _per-query_ mask head used in DETR with a more efficient _per-image_ FPN-based head to make end-to-end training without box supervision feasible.  
在本节中，我们将详细讨论 MaskFormer 和 DETR 之间的区别，并展示如何进行这些更改以确保掩码分类性能良好。首先，为了实现纯掩码分类设置，我们移除了框预测头，并使用掩码而不是框在预测和真实线段之间执行匹配。其次，我们将 DETR 中使用的计算量大的每查询掩码头替换为更高效的基于每图像 FPN 的头，使没有框监督的端到端训练变得可行。

Matching with masks is superior to matching with boxes. We compare MaskFormer models trained using matching with boxes or masks in Table [5](#S5.T5 "Table 5 ‣ 5 Discussion ‣ Per-Pixel Classification is Not All You Need for Semantic Segmentation"). To do box-based matching, we add to MaskFormer an additional box prediction head as in DETR \[[4](#bib.bib4)\]. Observe that MaskFormer, which directly matches with mask predictions, has a clear advantage. We hypothesize that matching with boxes is more ambiguous than matching with masks, especially for stuff categories where completely different masks can have similar boxes as stuff regions often spread over a large area in an image.  
与口罩搭配优于与盒子搭配。我们比较了使用表 5 中的框或掩码匹配训练的 MaskFormer 模型。为了进行基于框的匹配，我们在 MaskFormer 中添加了一个额外的框预测头，如 DETR \[4\] 所示。观察一下，与掩码预测直接匹配的 MaskFormer 具有明显的优势。我们假设与框匹配比与蒙版匹配更模糊，特别是对于完全不同的蒙版可能具有相似框的素材类别，因为素材区域通常分布在图像中的大面积区域。

MaskFormer mask head reduces computation. Results in Table [5](#S5.T5 "Table 5 ‣ 5 Discussion ‣ Per-Pixel Classification is Not All You Need for Semantic Segmentation") also show that MaskFormer performs on par with DETR when the same matching strategy is used. This suggests that the difference in mask head designs between the models does not significantly influence the prediction quality. The new head, however, has significantly lower computational and memory costs in comparison with the original mask head used in DETR. In MaskFormer, we first upsample image features to get high-resolution per-pixel embeddings and directly generate binary mask predictions at a high-resolution. Note, that the per-pixel embeddings from the upsampling module (_i.e_., pixel decoder) are shared among all queries. In contrast, DETR first generates low-resolution attention maps and applies an independent upsampling module to each query. Thus, the mask head in DETR is N𝑁N times more computationally expensive than the mask head in MaskFormer (where N𝑁N is the number of queries).  
MaskFormer 掩模头可减少计算量。表 5 中的结果还表明，当使用相同的匹配策略时，MaskFormer 的性能与 DETR 相当。这表明模型之间掩模头设计的差异不会显着影响预测质量。然而，与 DETR 中使用的原始掩模磁头相比，新磁头的计算和内存成本要低得多。在 MaskFormer 中，我们首先对图像特征进行上采样以获得高分辨率的每像素嵌入，并直接生成高分辨率的二进制掩码预测。请注意，来自上采样模块（即像素解码器）的每像素嵌入在所有查询之间共享。相比之下，DETR 首先生成低分辨率的注意力图，并对每个查询应用独立的上采样模块。因此，DETR 中的掩码头比 MaskFormer 中的掩码头计算成本 N𝑁N 高出几倍（其中 N𝑁N 是查询数）。

## 6 Conclusion 结论

The paradigm discrepancy between semantic- and instance-level segmentation results in entirely different models for each task, hindering development of image segmentation as a whole. We show that a simple mask classification model can outperform state-of-the-art per-pixel classification models, especially in the presence of large number of categories. Our model also remains competitive for panoptic segmentation, without a need to change model architecture, losses, or training procedure. We hope this unification spurs a joint effort across semantic- and instance-level segmentation tasks.  
语义级和实例级分割之间的范式差异导致每个任务的模型完全不同，阻碍了整个图像分割的发展。我们表明，简单的掩码分类模型可以优于最先进的每像素分类模型，尤其是在存在大量类别的情况下。我们的模型在全景分割方面也保持竞争力，无需更改模型架构、损失或训练程序。我们希望这种统一能够促进语义级和实例级分段任务的共同努力。

Acknowledgments and Disclosure of Funding  
资金的确认和披露
----------------------------------------------------

We thank Ross Girshick for insightful comments and suggestions. Work of UIUC authors Bowen Cheng and Alexander G. Schwing was supported in part by NSF under Grant #1718221, 2008387, 2045586, 2106825, MRI #1725729, NIFA award 2020-67021-32799 and Cisco Systems Inc. (Gift Award CG 1377144 - thanks for access to Arcetri).  
我们感谢 Ross Girshick 的有见地的评论和建议。UIUC 作者 Bowen Cheng 和 Alexander G. Schwing 的工作得到了 NSF 的部分支持，包括 Grant #1718221、2008387、2045586、2106825、MRI #1725729、NIFA 奖 2020-67021-32799 和 Cisco Systems Inc.（礼品奖 CG 1377144 - 感谢您访问 Arcetri）。

Appendix 附录

We first provide more information regarding the datasets used in our experimental evaluation of MaskFormer (Appendix [A](#A1 "Appendix A Datasets description ‣ Per-Pixel Classification is Not All You Need for Semantic Segmentation")). Then, we provide detailed results of our model on more semantic (Appendix [B](#A2 "Appendix B Semantic segmentation results ‣ Per-Pixel Classification is Not All You Need for Semantic Segmentation")) and panoptic (Appendix [C](#A3 "Appendix C Panoptic segmentation results ‣ Per-Pixel Classification is Not All You Need for Semantic Segmentation")) segmentation datasets. Finally, we provide additional ablation studies (Appendix [D](#A4 "Appendix D Additional ablation studies ‣ Per-Pixel Classification is Not All You Need for Semantic Segmentation")) and visualization (Appendix [E](#A5 "Appendix E Visualization ‣ Per-Pixel Classification is Not All You Need for Semantic Segmentation")).  
我们首先提供有关 MaskFormer 实验评估中使用的数据集的更多信息（附录 A）。然后，我们在更多语义（附录 B）和全景（附录 C）分割数据集上提供了我们的模型的详细结果。最后，我们提供了额外的消融研究（附录 D）和可视化（附录 E）。

Appendix A Datasets description  
附录 ADatasets 说明
-------------------------------------------------

We study MaskFormer using five semantic segmentation datasets and two panoptic segmentation datasets. Here, we provide more detailed information about these datasets.  
我们使用五个语义分割数据集和两个全景分割数据集来研究 MaskFormer。在这里，我们提供了有关这些数据集的更多详细信息。

### A.1 Semantic segmentation datasets  

A.1 语义分割数据集

ADE20K \[[55](#bib.bib55)\] contains 20k images for training and 2k images for validation. The data comes from the ADE20K-Full dataset where 150 semantic categories are selected to be included in evaluation from the SceneParse150 challenge \[[54](#bib.bib54)\]. The images are resized such that the shortest side is no greater than 512 pixels. During inference, we resize the shorter side of the image to the corresponding crop size.  
ADE20K \[55\] 包含用于训练的 20k 图像和用于验证的 2k 图像。数据来自 ADE20K-Full 数据集，其中选择了 150 个语义类别，以包含在 SceneParse150 挑战\[54\] 的评估中。调整图像大小，使最短边不大于 512 像素。在推理过程中，我们将图像的较短边调整为相应的裁剪大小。

COCO-Stuff-10K \[[3](#bib.bib3)\] has 171 semantic-level categories. There are 9k images for training and 1k images for testing. Images in the COCO-Stuff-10K datasets are a subset of the COCO dataset \[[28](#bib.bib28)\]. During inference, we resize the shorter side of the image to the corresponding crop size.  
COCO-Stuff-10K \[3\] 有 171 个语义级类别。有 9k 图像用于训练，1k 图像用于测试。COCO-Stuff-10K 数据集中的图像是 COCO 数据集的一个子集\[28\]。在推理过程中，我们将图像的较短边调整为相应的裁剪大小。

ADE20K-Full \[[55](#bib.bib55)\] contains 25k images for training and 2k images for validation. The ADE20K-Full dataset is annotated in an open-vocabulary setting with more than 3000 semantic categories. We filter these categories by selecting those that are present in both training and validation sets, resulting in a total of 847 categories. We follow the same process as ADE20K-SceneParse150 to resize images such that the shortest side is no greater than 512 pixels. During inference, we resize the shorter side of the image to the corresponding crop size.  
ADE20K-Full \[55\] 包含用于训练的 25k 图像和用于验证的 2k 图像。ADE20K-Full 数据集在具有 3000 多个语义类别的开放词汇设置中进行注释。我们通过选择训练集和验证集中存在的类别来过滤这些类别，从而得出总共 847 个类别。我们遵循与 ADE20K-SceneParse150 相同的过程来调整图像大小，使最短边不大于 512 像素。在推理过程中，我们将图像的较短边调整为相应的裁剪大小。

Cityscapes \[[15](#bib.bib15)\] is an urban egocentric street-view dataset with high-resolution images (1024×2048102420481024\\times 2048 pixels). It contains 2975 images for training, 500 images for validation, and 1525 images for testing with a total of 19 classes. During training, we use a crop size of 512×10245121024512\\times 1024, a batch size of 16 and train all models for 90k iterations. During inference, we operate on the whole image (1024×2048102420481024\\times 2048).  
Cityscapes \[15\] 是一个以城市自我为中心的街景数据集，具有高分辨率图像（ 1024×2048102420481024\\times 2048 像素）。它包含 2975 张用于训练的图像、500 张用于验证的图像和 1525 张用于测试的图像，共 19 个类。在训练期间，我们使用 512×10245121024512\\times 1024 裁剪大小 ，批处理大小为 16，并训练所有模型进行 90k 迭代。在推理过程中，我们对整个图像 （ 1024×2048102420481024\\times 2048 ） 进行操作。

Mapillary Vistas \[[34](#bib.bib34)\] is a large-scale urban street-view dataset with 65 categories. It contains 18k, 2k, and 5k images for training, validation and testing with a variety of image resolutions, ranging from 1024×76810247681024\\times 768 to 4000×6000400060004000\\times 6000. During training, we resize the short side of images to 2048 before applying scale augmentation. We use a crop size of 1280×1280128012801280\\times 1280, a batch size of 161616 and train all models for 300k iterations. During inference, we resize the longer side of the image to 2048 and only use three scales (0.5, 1.0 and 1.5) for multi-scale testing due to GPU memory constraints.  
Mapillary Vistas \[34\] 是一个包含 65 个类别的大规模城市街景数据集。它包含 18k、2k 和 5k 图像，用于训练、验证和测试，具有各种图像分辨率，范围 1024×76810247681024\\times 768 4000×6000400060004000\\times 6000 从 .在训练期间，我们将图像的短边调整为 2048，然后再应用比例增强。我们使用 裁 1280×1280128012801280\\times 1280 剪大小 ，批处理大小 ， 161616 并训练所有模型进行 300k 迭代。在推理过程中，由于 GPU 内存限制，我们将图像的较长边调整为 2048，并且仅使用三个比例（0.5、1.0 和 1.5）进行多比例测试。

### A.2 Panoptic segmentation datasets  

A.2 全景分割数据集

COCO panoptic \[[24](#bib.bib24)\] is one of the most commonly used datasets for panoptic segmentation. It has 133 categories (80 “thing” categories with instance-level annotation and 53 “stuff” categories) in 118k images for training and 5k images for validation. All images are from the COCO dataset \[[28](#bib.bib28)\].  
COCO panoptic \[24\] 是最常用的全景分割数据集之一。它有 133 个类别（80 个具有实例级注释的“事物”类别和 53 个“东西”类别），其中 118k 图像用于训练，5k 图像用于验证。所有图像均来自 COCO 数据集\[28\]。

ADE20K panoptic \[[55](#bib.bib55)\] combines the ADE20K semantic segmentation annotation for semantic segmentation from the SceneParse150 challenge \[[54](#bib.bib54)\] and ADE20K instance annotation from the COCO+Places challenge \[[1](#bib.bib1)\]. Among the 150 categories, there are 100 “thing” categories with instance-level annotation. We find filtering masks with a lower threshold (we use 0.7 for ADE20K) than COCO (which uses 0.8) gives slightly better performance.  
ADE20K 全景\[55\] 结合了来自 SceneParse150 挑战\[54\] 的 ADE20K 语义分割注释和来自 COCO+Places 挑战\[1\] 的 ADE20K 实例注释。在 150 个类别中，有 100 个具有实例级注释的“事物”类别。我们发现阈值更低（ADE20K 使用 0.7）的滤波掩码比 COCO（使用 0.8）的性能略好。

Table I: Semantic segmentation on ADE20K test with 150 categories. MaskFormer outperforms previous state-of-the-art methods on all three metrics: pixel accuracy (P.A.), mIoU, as well as the final test score (average of P.A. and mIoU). We train our model on the union of ADE20K train and val set with ImageNet-22K pre-trained checkpoint following \[[29](#bib.bib29)\] and use multi-scale inference.  
表 I：ADE20K 测试的 150 个类别的语义分割 MaskFormer 在所有三个指标上都优于以前最先进的方法：像素精度（PA）、mIoU 以及最终测试分数（PA 和 mIoU 的平均值）。我们在 ADE20K 训练和 val 集的并集上训练我们的模型，并遵循 ImageNet-22K 预训练检查点\[29\]，并使用多尺度推理。

| method | backbone | P.A. | mIoU | score |

| SETR \[[53](#bib.bib53)\] 塞特 \[53\] | ViT-L | 78.35 | 45.03 | 61.69 |

| Swin-UperNet \[[29](#bib.bib29), [49](#bib.bib49)\] 斯温 - 乌珀网 \[29， 49\] | Swin-L | 78.42 | 47.07 | 62.75 |

| MaskFormer (ours)MaskFormer（我们的） | Swin-L | 79.36 | 49.67 | 64.51 |

Table II: Semantic segmentation on COCO-Stuff-10K test with 171 categories and ADE20K-Full val with 847 categories. Table [IIa](#A1.T2.sf1 "In Table II ‣ A.2 Panoptic segmentation datasets ‣ Appendix A Datasets description ‣ Per-Pixel Classification is Not All You Need for Semantic Segmentation"): MaskFormer is competitive on COCO-Stuff-10K, showing the generality of mask-classification. Table [IIb](#A1.T2.sf2 "In Table II ‣ A.2 Panoptic segmentation datasets ‣ Appendix A Datasets description ‣ Per-Pixel Classification is Not All You Need for Semantic Segmentation"): MaskFormer results on the harder large-vocabulary semantic segmentation. MaskFormer performs better than per-pixel classification and requires less memory during training, thanks to decoupling the number of masks from the number of classes. mIoU (s.s.) and mIoU (m.s.) are the mIoU of single-scale and multi-scale inference with ±plus-or-minus\\pmstd.  
表二：COCO-Stuff-10K 检验（171 个类别）和 ADE20K-Full val（847 个类别）的语义分割表 IIa：MaskFormer 在 COCO-Stuff-10K 上具有竞争力，显示了掩模分类的通用性。表 IIb：MaskFormer 对较难的大词汇语义分割的结果。MaskFormer 的性能优于每像素分类，并且由于将掩码数量与类数解耦，因此在训练期间需要更少的内存。mIoU （s.s.） 和 mIoU （m.s.） 是单尺度和多尺度推理的 ±plus-or-minus\\pm mIoU。

| method | backbone | mIoU (s.s.)mIoU （s.s.） | mIoU (m.s.)mIoU （硕士） |

| OCRNet \[[50](#bib.bib50)\] | R101c | \- ±plus-or-minus\\pm0.5 | 39.5 ±plus-or-minus\\pm0.5 |

| PerPixelBaseline | 0R50c | 32.4 ±plus-or-minus\\pm0.2 | 34.4 ±plus-or-minus\\pm0.4 |

| PerPixelBaseline+ | 0R50c | 34.2 ±plus-or-minus\\pm0.2 | 35.8 ±plus-or-minus\\pm0.4 |

| MaskFormer (ours)MaskFormer（我们的） | 0R50c | 37.1 ±plus-or-minus\\pm0.4 | 38.9 ±plus-or-minus\\pm0.2 |

| R101c | 38.1  ±plus-or-minus\\pm0.3 | 39.8  ±plus-or-minus\\pm0.6 |

| R101c | 38.0 ±plus-or-minus\\pm0.3 | 39.3 ±plus-or-minus\\pm0.4 |

(a) COCO-Stuff-10K.（一）COCO- 东西 -10K。

| mIoU (s.s.)mIoU （s.s.） | training memory 训练记忆 |

| \- ±plus-or-minus\\pm0.5 | - |

| 12.4 ±plus-or-minus\\pm0.2 | 08030M |

| 13.9 ±plus-or-minus\\pm0.1 | 26698M |

| 16.0 ±plus-or-minus\\pm0.3 | 06529M |

| 16.8 ±plus-or-minus\\pm0.2 | 06894M |

| 17.4  ±plus-or-minus\\pm0.4 | 06904M |

(b) ADE20K-Full.（二）ADE20K- 全。

Table III: Semantic segmentation on Cityscapes val with 19 categories. [IIIa](#A1.T3.sf1 "In Table III ‣ A.2 Panoptic segmentation datasets ‣ Appendix A Datasets description ‣ Per-Pixel Classification is Not All You Need for Semantic Segmentation"): MaskFormer is on-par with state-of-the-art methods on Cityscapes which has fewer categories than other considered datasets. We report multi-scale (m.s.) inference results with ±plus-or-minus\\pmstd for a fair comparison across methods. [IIIb](#A1.T3.sf2 "In Table III ‣ A.2 Panoptic segmentation datasets ‣ Appendix A Datasets description ‣ Per-Pixel Classification is Not All You Need for Semantic Segmentation"): We analyze MaskFormer with a complimentary PQStSt{}^{\\text{St}} metric, by treating all categories as “stuff.” The breakdown of PQStSt{}^{\\text{St}} suggests mask classification-based MaskFormer is better at recognizing regions (RQStSt{}^{\\text{St}}) while slightly lagging in generation of high-quality masks (SQStSt{}^{\\text{St}}).  
表三：19 个类别的城市景观语义细分 IIIa：MaskFormer 与 Cityscapes 上最先进的方法相当，其类别比其他考虑的数据集少。我们用 ±plus-or-minus\\pm std 报告了多尺度 （m.s.） 推理结果，以便对不同方法进行公平比较。IIIb：我们使用免费的 PQ StSt{}^{\\text{St}} 指标来分析 MaskFormer，将所有类别视为“东西”。PQ StSt{}^{\\text{St}} 的细分表明，基于掩模分类的 MaskFormer 在识别区域（RQ StSt{}^{\\text{St}} ）方面表现更好，而在高质量掩模（SQ StSt{}^{\\text{St}} ）的生成方面略有滞后。

| method | backbone | mIoU (m.s.)mIoU （硕士） |

| Panoptic-DeepLab \[[11](#bib.bib11)\] | X71 \[[12](#bib.bib12)\] | 81.5 ±plus-or-minus\\pm0.2 |

| OCRNet \[[50](#bib.bib50)\] | R101c | 82.0  ±plus-or-minus\\pm0.2 |

| MaskFormer (ours)MaskFormer（我们的） | R101c | 80.3 ±plus-or-minus\\pm0.1 |

| R101c | 81.4 ±plus-or-minus\\pm0.2 |

(a) Cityscapes standard mIoU metric.  
（一）Cityscapes 标准 mIoU 指标。

| PQStSt{}^{\\text{St}} (m.s.)PQ StSt{}^{\\text{St}} （硕士） | SQStSt{}^{\\text{St}} (m.s.)SQ StSt{}^{\\text{St}} （硕士） | RQStSt{}^{\\text{St}} (m.s.)RQ StSt{}^{\\text{St}} （硕士） |

| 66.6 | 82.9 | 79.4 |

| 66.1 | 82.6 | 79.1 |

| 65.9 | 81.5 | 79.7 |

| 66.9 | 82.0 | 80.5 |

(b) Cityscapes analysis with PQStSt{}^{\\text{St}} metric suit.  
（二）使用 PQ StSt{}^{\\text{St}} 公制套装进行城市景观分析。

Table IV: Semantic segmentation on Mapillary Vistas val with 65 categories. MaskFormer outperforms per-pixel classification methods on high-resolution images without the need of multi-scale inference, thanks to global context captured by the Transformer decoder. mIoU (s.s.) and mIoU (m.s.) are the mIoU of single-scale and multi-scale inference.  
表四：65 个类别的 Mapillary Vistas val 的语义细分。MaskFormer 在高分辨率图像上优于每像素分类方法，而无需多尺度推理，这要归功于 Transformer 解码器捕获的全局上下文。mIoU （s.s.） 和 mIoU （m.s.） 是单尺度和多尺度推理的 mIoU。

| method | backbone | mIoU (s.s.)mIoU （s.s.） | mIoU (m.s.)mIoU （硕士） |

| DeepLabV3+ \[[9](#bib.bib9)\]DeepLabV3+（深实验室 V3+） \[9\] | R50 | 47.7 | 49.4 |

| HMSANet \[[38](#bib.bib38)\] | R50 | - | 52.2 |

| MaskFormer (ours)MaskFormer（我们的） | R50 | 53.1 | 55.4 |

Table V: Panoptic segmentation on COCO panoptic test-dev with 133 categories. MaskFormer outperforms previous state-of-the-art Max-DeepLab \[[42](#bib.bib42)\] on the test-dev set as well. We only train our model on the COCO train2017 set with ImageNet-22K pre-trained checkpoint.  
表五：COCO 全景测试开发的全景细分，133 个类别。MaskFormer 在测试开发集上也优于以前最先进的 Max-DeepLab \[42\]。我们只在带有 ImageNet-22K 预训练检查点的 COCO train2017 集上训练我们的模型。

| method | backbone | PQ | PQThTh{}^{\\text{Th}}餐前 ThTh{}^{\\text{Th}} | PQStSt{}^{\\text{St}}餐前 StSt{}^{\\text{St}} | SQ | RQ |

| Max-DeepLab \[[42](#bib.bib42)\] 马克斯深度实验室 \[42\] | Max-L | 51.3 | 57.2 | 42.4 | 82.5 | 61.3 |

| MaskFormer (ours)MaskFormer（我们的） | Swin-L | 53.3 | 59.1 | 44.5 | 82.0 | 64.1 |

Table VI: Panoptic segmentation on ADE20K panoptic val with 150 categories. Following DETR \[[4](#bib.bib4)\], we add 6 additional Transformer encoders when using ResNet \[[22](#bib.bib22)\] (R50 + 6 Enc and R101 + 6 Enc) backbones. MaskFormer achieves competitive results on ADE20K panotic, showing the generality of our model for panoptic segmentation.  
表 VI：ADE20K 全景值的全景分割，150 个类别。继 DETR \[4\] 之后，我们在使用 ResNet \[22\]（R50 + 6 Enc 和 R101 + 6 Enc）主干网时增加了 6 个额外的 Transformer 编码器。MaskFormer 在 ADE20K 全景上取得了具有竞争力的结果，显示了我们的全景分割模型的通用性。

| method | backbone | PQ | PQThTh{}^{\\text{Th}}餐前 ThTh{}^{\\text{Th}} | PQStSt{}^{\\text{St}}餐前 StSt{}^{\\text{St}} | SQ | RQ |

| BGRNet \[[47](#bib.bib47)\]BGRNet 的 \[47\] | R50 | 31.8 | - | - | - | - |

| Auto-Panoptic \[[48](#bib.bib48)\] 自动全景 \[48\] | ShuffleNetV2 \[[32](#bib.bib32)\] 随机网络 V2 \[32\] | 32.4 | - | - | - | - |

| MaskFormer (ours)MaskFormer（我们的） | 0R50 + 6 Enc | 34.7 | 32.2 | 39.7 | 76.7 | 42.8 |

| R101 + 6 EncR101 + 6 恩加 | 35.7 | 34.5 | 38.0 | 77.4 | 43.8 |

Appendix B Semantic segmentation results  
附录 B 语义分割结果
-----------------------------------------------------

ADE20K test. Table [I](#A1.T1 "Table I ‣ A.2 Panoptic segmentation datasets ‣ Appendix A Datasets description ‣ Per-Pixel Classification is Not All You Need for Semantic Segmentation") compares MaskFormer with previous state-of-the-art methods on the ADE20K test set. Following \[[29](#bib.bib29)\], we train MaskFormer on the union of ADE20K train and val set with ImageNet-22K pre-trained checkpoint and use multi-scale inference. MaskFormer outperforms previous state-of-the-art methods on all three metrics with a large margin.  
ADE20K 测试。表 I 将 MaskFormer 与 ADE20K 测试装置上以前最先进的方法进行了比较。按照\[29\]，我们在 ADE20K 训练和 val 集与 ImageNet-22K 预训练检查点的结合上训练 MaskFormer，并使用多尺度推理。MaskFormer 在所有三个指标上都比以前最先进的方法更胜一筹。

COCO-Stuff-10K. Table [IIa](#A1.T2.sf1 "In Table II ‣ A.2 Panoptic segmentation datasets ‣ Appendix A Datasets description ‣ Per-Pixel Classification is Not All You Need for Semantic Segmentation") compares MaskFormer with our baselines as well as the state-of-the-art OCRNet model \[[50](#bib.bib50)\] on the COCO-Stuff-10K \[[3](#bib.bib3)\] dataset. MaskFormer outperforms our per-pixel classification baselines by a large margin and achieves competitive performances compared to OCRNet. These results demonstrate the generality of the MaskFormer model.  
COCO- 东西 -10K。表 IIa 将 MaskFormer 与我们的基线以及 COCO-Stuff-10K \[3\] 数据集上最先进的 OCRNet 模型\[50\] 进行了比较。与 OCRNet 相比，MaskFormer 的性能远远超过我们的每像素分类基准，并具有竞争力的性能。这些结果证明了 MaskFormer 模型的通用性。

ADE20K-Full. We further demonstrate the benefits in large-vocabulary semantic segmentation in Table [IIb](#A1.T2.sf2 "In Table II ‣ A.2 Panoptic segmentation datasets ‣ Appendix A Datasets description ‣ Per-Pixel Classification is Not All You Need for Semantic Segmentation"). Since we are the first to report performance on this dataset, we only compare MaskFormer with our per-pixel classification baselines. MaskFormer not only achieves better performance, but is also more memory efficient on the ADE20K-Full dataset with 847 categories, thanks to decoupling the number of masks from the number of classes. These results show that our MaskFormer has the potential to deal with real-world segmentation problems with thousands of categories.  
ADE20K- 全。我们在表 IIb 中进一步证明了大词汇语义分割的好处。由于我们是第一个报告此数据集性能的公司，因此我们仅将 MaskFormer 与每像素分类基线进行比较。MaskFormer 不仅实现了更好的性能，而且在包含 847 个类别的 ADE20K-Full 数据集上也具有更高的内存效率，这要归功于掩码数量与类数的解耦。这些结果表明，我们的 MaskFormer 具有处理数千个类别的实际细分问题的潜力。

Cityscapes. In Table [IIIa](#A1.T3.sf1 "In Table III ‣ A.2 Panoptic segmentation datasets ‣ Appendix A Datasets description ‣ Per-Pixel Classification is Not All You Need for Semantic Segmentation"), we report MaskFormer performance on Cityscapes, the standard testbed for modern semantic segmentation methods. The dataset has only 19 categories and therefore, the recognition aspect of the dataset is less challenging than in other considered datasets. We observe that MaskFormer performs on par with the best per-pixel classification methods. To better analyze MaskFormer, in Table [IIIb](#A1.T3.sf2 "In Table III ‣ A.2 Panoptic segmentation datasets ‣ Appendix A Datasets description ‣ Per-Pixel Classification is Not All You Need for Semantic Segmentation"), we further report PQStSt{}^{\\text{St}}. We find MaskFormer performs better in terms of recognition quality (RQStSt{}^{\\text{St}}) while lagging in per-pixel segmentation quality (SQStSt{}^{\\text{St}}). This suggests that on datasets, where recognition is relatively easy to solve, the main challenge for mask classification-based approaches is pixel-level accuracy.  
城市景观。在表 IIIa 中，我们报告了 MaskFormer 在 Cityscapes 上的性能，Cityscapes 是现代语义分割方法的标准测试平台。该数据集只有 19 个类别，因此，与其他考虑的数据集相比，数据集的识别方面更具挑战性。我们观察到 MaskFormer 的性能与最佳的每像素分类方法相当。为了更好地分析 MaskFormer，在表 IIIb 中，我们进一步报告了 PQ StSt{}^{\\text{St}} 。我们发现 MaskFormer 在识别质量（RQ StSt{}^{\\text{St}} ）方面表现更好，而在每像素分割质量（SQ StSt{}^{\\text{St}} ）方面落后。这表明，在识别相对容易解决的数据集上，基于掩码分类的方法的主要挑战是像素级精度。

Mapillary Vistas. Table [IV](#A1.T4 "Table IV ‣ A.2 Panoptic segmentation datasets ‣ Appendix A Datasets description ‣ Per-Pixel Classification is Not All You Need for Semantic Segmentation") compares MaskFormer with state-of-the-art per-pixel classification models on the high-resolution Mapillary Vistas dataset which contains images up to 4000×6000400060004000\\times 6000 resolution. We observe: (1) MaskFormer is able to handle high-resolution images, and (2) MaskFormer outperforms mulit-scale per-pixel classification models even without the need of mult-scale inference. We believe the Transformer decoder in MaskFormer is able to capture global context even for high-resolution images.  
Mapillary Vistas（状远景）。表 IV 将 MaskFormer 与高分辨率 Mapillary Vistas 数据集上最先进的每像素分类模型进行了比较，该数据集包含高达 4000×6000400060004000\\times 6000 分辨率的图像。我们观察到：（1） MaskFormer 能够处理高分辨率图像，以及 （2） 即使不需要多尺度推理，MaskFormer 也优于多尺度每像素分类模型。我们相信 MaskFormer 中的 Transformer 解码器能够捕获全局上下文，即使是高分辨率图像也是如此。

Appendix C Panoptic segmentation results  
附录 CPanoptic 分割结果
------------------------------------------------------------

COCO panoptic test-dev. Table [V](#A1.T5 "Table V ‣ A.2 Panoptic segmentation datasets ‣ Appendix A Datasets description ‣ Per-Pixel Classification is Not All You Need for Semantic Segmentation") compares MaskFormer with previous state-of-the-art methods on the COCO panoptic test-dev set. We only train our model on the COCO train2017 set with ImageNet-22K pre-trained checkpoint and outperforms previos state-of-the-art by 2 PQ.  
COCO 全景测试开发表 V 将 MaskFormer 与以前在 COCO 全景测试开发集上最先进的方法进行了比较。我们只在带有 ImageNet-22K 预训练检查点的 COCO train2017 集上训练我们的模型，并且性能优于 previos 最先进的 2 PQ。

ADE20K panoptic. We demonstrate the generality of our model for panoptic segmentation on the ADE20K panoptic dataset in Table [VI](#A1.T6 "Table VI ‣ A.2 Panoptic segmentation datasets ‣ Appendix A Datasets description ‣ Per-Pixel Classification is Not All You Need for Semantic Segmentation"), where MaskFormer is competitive with the state-of-the-art methods.  
ADE20K 全景光学。我们在表 VI 中演示了 ADE20K 全景数据集上全景分割模型的通用性，其中 MaskFormer 与最先进的方法具有竞争力。

Appendix D Additional ablation studies  
附录 Ddditional 消融研究
-----------------------------------------------------------

We perform additional ablation studies of MaskFormer for semantic segmentation using the same setting as that in the main paper: a single ResNet-50 backbone \[[22](#bib.bib22)\], and we report both the mIoU and the PQStSt{}^{\\text{St}}. The default setting of our MaskFormer is: 100 queries and 6 Transformer decoder layers.  
我们使用与主要论文相同的设置对 MaskFormer 进行额外的消融研究，以进行语义分割：单个 ResNet-50 骨干 \[22\]，我们报告了 mIoU 和 PQ StSt{}^{\\text{St}} 。MaskFormer 的默认设置是：100 个查询和 6 个 Transformer 解码器层。

Table VII: Inference strategies for semantic segmentation. _general:_ general inference (Section [3.4](#S3.SS4 "3.4 Mask-classification inference ‣ 3 From Per-Pixel to Mask Classification ‣ Per-Pixel Classification is Not All You Need for Semantic Segmentation")) which first filters low-confidence masks (using a threshold of 0.3) and assigns labels to the remaining ones. _semantic:_ the default semantic inference (Section [3.4](#S3.SS4 "3.4 Mask-classification inference ‣ 3 From Per-Pixel to Mask Classification ‣ Per-Pixel Classification is Not All You Need for Semantic Segmentation")) for semantic segmentation.  
表七：语义分割的推理策略。general：一般推理（第 3.4 节），首先过滤低置信度掩码（使用阈值 0.3）并将标签分配给其余掩码。semantic：语义分割的默认语义推理（第 3.4 节）。

|  | ADE20K (150 classes)ADE20K（150 节课） | COCO-Stuff (171 classes)COCO-Stuff （171 类） | ADE20K-Full (847 classes)  
ADE20K-Full （847 类） |

| inference | mIoU | PQStSt{}^{\\text{St}}餐前 StSt{}^{\\text{St}} | SQStSt{}^{\\text{St}}平方 StSt{}^{\\text{St}} | RQStSt{}^{\\text{St}}RQ（英语：RQ） StSt{}^{\\text{St}} | mIoU | PQStSt{}^{\\text{St}}餐前 StSt{}^{\\text{St}} | SQStSt{}^{\\text{St}}平方 StSt{}^{\\text{St}} | RQStSt{}^{\\text{St}}RQ（英语：RQ） StSt{}^{\\text{St}} | mIoU | PQStSt{}^{\\text{St}}餐前 StSt{}^{\\text{St}} | SQStSt{}^{\\text{St}}平方 StSt{}^{\\text{St}} | RQStSt{}^{\\text{St}}RQ（英语：RQ） StSt{}^{\\text{St}} |

| PerPixelBaseline+ | 41.9 | 28.3 | 71.9 | 36.2 | 34.2 | 24.6 | 62.6 | 31.2 | 13.9 | 09.0 | 24.5 | 12.0 |

| general | 42.4 | 34.2 | 74.4 | 43.5 | 35.5 | 29.7 | 66.3 | 37.0 | 15.1 | 11.6 | 28.3 | 15.3 |

| semantic | 44.5 | 33.4 | 75.4 | 42.4 | 37.1 | 28.9 | 66.3 | 35.9 | 16.0 | 11.9 | 28.6 | 15.7 |

Table VIII: Ablation on number of Transformer decoder layers in MaskFormer. We find that MaskFormer with only one Transformer decoder layer is already able to achieve reasonable semantic segmentation performance. Stacking more decoder layers mainly improves the recognition quality.  
表 VIII：MaskFormer 中 Transformer 解码器层数的烧蚀。我们发现，只有一个 Transformer 解码器层的 MaskFormer 已经能够实现合理的语义分割性能。堆叠更多的解码层主要提高识别质量。

|  | ADE20K-Semantic | ADE20K-Panoptic |

| \# of decoder layers\# 解码器层数 | mIoU | PQStSt{}^{\\text{St}}餐前 StSt{}^{\\text{St}} | SQStSt{}^{\\text{St}}平方 StSt{}^{\\text{St}} | RQStSt{}^{\\text{St}}RQ（英语：RQ） StSt{}^{\\text{St}} | PQ | PQThTh{}^{\\text{Th}}餐前 ThTh{}^{\\text{Th}} | PQStSt{}^{\\text{St}}餐前 StSt{}^{\\text{St}} | SQ | RQ |

| 6 (PerPixelBaseline+)6 （PerPixelBaseline+） | 41.9 | 28.3 | 71.9 | 36.2 | - | - | - | - | - |

| 1 | 43.0 | 31.1 | 74.3 | 39.7 | 31.9 | 29.6 | 36.6 | 76.6 | 39.6 |

| 6 | 44.5 | 33.4 | 75.4 | 42.4 | 34.7 | 32.2 | 39.7 | 76.7 | 42.8 |

| 6 (no self-attention)6（无自我关注） | 44.6 | 32.8 | 74.5 | 41.5 | 32.6 | 29.9 | 38.2 | 75.6 | 40.4 |

| MaskFormer trained for semantic segmentation  
MaskFormer 经过语义分割训练 | MaskFormer trained for panoptic segmentation  
MaskFormer 接受过全景分割训练 |

| ground truth 地面实况 | prediction | ground truth 地面实况 | prediction |

| ![](https://ar5iv.labs.arxiv.org/html/2107.06278/assets/figures/visualization/figure1_jpg/gt.jpg)

 | ![](https://ar5iv.labs.arxiv.org/html/2107.06278/assets/figures/visualization/figure1_jpg/dt.jpg)

 | ![](https://ar5iv.labs.arxiv.org/html/2107.06278/assets/figures/visualization/figure1_jpg/gt_pan_seg.jpg)

 | ![](https://ar5iv.labs.arxiv.org/html/2107.06278/assets/figures/visualization/figure1_jpg/dt_pan_seg.jpg)

 |

| semantic query prediction  
语义查询预测 | panoptic query prediction  
全景查询预测 |

| ![](https://ar5iv.labs.arxiv.org/html/2107.06278/assets/figures/visualization/figure1_jpg/query_62_label_20_score_1.0.jpg)

 | ![](https://ar5iv.labs.arxiv.org/html/2107.06278/assets/figures/visualization/figure1_jpg/pan_seg_query_29_label_20_score_0.96.jpg)

 | ![](https://ar5iv.labs.arxiv.org/html/2107.06278/assets/figures/visualization/figure1_jpg/pan_seg_query_78_label_20_score_1.0.jpg)

 |

Figure I: Visualization of “semantic” queries and “panoptic” queries. Unlike the behavior in a MaskFormer model trained for panoptic segmentation (right), a single query is used to capture multiple instances in a MaskFormer model trained for semantic segmentation (left). Our model has the capacity to adapt to different types of tasks given different ground truth annotations.  
图 I：“语义”查询和“全景”查询的可视化。与为全景分割训练的 MaskFormer 模型中的行为（右图）不同，单个查询用于捕获为语义分割训练的 MaskFormer 模型中的多个实例（左图）。我们的模型能够适应不同类型的任务，给定不同的地面实况注释。

Inference strategies. In Table [VII](#A4.T7 "Table VII ‣ Appendix D Additional ablation studies ‣ Per-Pixel Classification is Not All You Need for Semantic Segmentation"), we ablate inference strategies for mask classification-based models performing semantic segmentation (discussed in Section [3.4](#S3.SS4 "3.4 Mask-classification inference ‣ 3 From Per-Pixel to Mask Classification ‣ Per-Pixel Classification is Not All You Need for Semantic Segmentation")). We compare our default semantic inference strategy and the general inference strategy which first filters out low-confidence masks (a threshold of 0.3 is used) and assigns the class labels to the remaining masks. We observe 1) general inference is only slightly better than the PerPixelBaseline+ in terms of the mIoU metric, and 2) on multiple datasets the general inference strategy performs worse in terms of the mIoU metric than the default semantic inference. However, the general inference has higher PQStSt{}^{\\text{St}}, due to better recognition quality (RQStSt{}^{\\text{St}}). We hypothesize that the filtering step removes false positives which increases the RQStSt{}^{\\text{St}}. In contrast, the semantic inference aggregates mask predictions from multiple queries thus it has better mask quality (SQStSt{}^{\\text{St}}). This observation suggests that semantic and instance-level segmentation can be unified with a single inference strategy (_i.e_., our general inference) and _the choice of inference strategy largely depends on the evaluation metric instead of the task_.  
推理策略。在表 VII 中，我们消融了基于掩码分类的模型的推理策略，这些模型执行语义分割（在第 3.4 节中讨论）。我们比较了默认的语义推理策略和通用推理策略，后者首先过滤掉低置信度掩码（使用阈值 0.3），并将类标签分配给剩余的掩码。我们观察到 1） 就 mIoU 指标而言，一般推理仅略优于 PerPixelBaseline+，以及 2） 在多个数据集上，一般推理策略在 mIoU 指标方面的表现比默认语义推理差。然而，由于更好的识别质量（RQ StSt{}^{\\text{St}} ），一般推理具有更高的 PQ StSt{}^{\\text{St}} 。我们假设过滤步骤消除了误报，从而增加了 RQ StSt{}^{\\text{St}} 。相比之下，语义推理聚合了来自多个查询的掩码预测，因此它具有更好的掩码质量 （SQ StSt{}^{\\text{St}} ）。这一观察结果表明，语义和实例级分割可以与单一的推理策略（即我们的一般推理）统一起来，推理策略的选择在很大程度上取决于评估指标而不是任务。

Number of Transformer decoder layers. In Table [VIII](#A4.T8 "Table VIII ‣ Appendix D Additional ablation studies ‣ Per-Pixel Classification is Not All You Need for Semantic Segmentation"), we ablate the effect of the number of Transformer decoder layers on ADE20K \[[55](#bib.bib55)\] for both semantic and panoptic segmentation. Surprisingly, we find a MaskFormer with even a single Transformer decoder layer already performs reasonably well for semantic segmentation and achieves better performance than our 6-layer-decoder per-pixel classification baseline PerPixelBaseline+. Whereas, for panoptic segmentation, the number of decoder layers is more important. We hypothesize that stacking more decoder layers is helpful to de-duplicate predictions which is required by the panoptic segmentation task.  
Transformer 解码器层数。在表 VIII 中，我们消融了 Transformer 解码器层数对 ADE20K \[55\] 语义和全景分割的影响。令人惊讶的是，我们发现，即使是单个 Transformer 解码器层的 MaskFormer 在语义分割方面的表现也相当不错，并且比我们的 6 层解码器每像素分类基线 PerPixelBaseline+ 具有更好的性能。然而，对于全景分割，解码器层的数量更为重要。我们假设堆叠更多的解码器层有助于消除重复预测，这是全景分割任务所必需的。

To verify this hypothesis, we train MaskFormer models _without_ self-attention in all 6 Transformer decoder layers. On semantic segmentation, we observe MaskFormer without self-attention performs similarly well in terms of the mIoU metric, however, the per-mask metric PQStSt{}^{\\text{St}} is slightly worse. On panoptic segmentation, MaskFormer models without self-attention performs worse across all metrics.  
为了验证这一假设，我们在所有 6 个 Transformer 解码器层中训练 MaskFormer 模型，而无需自我关注。在语义分割方面，我们观察到没有自我注意力的 MaskFormer 在 mIoU 指标方面表现同样出色，但是，每个掩码指标的 PQ StSt{}^{\\text{St}} 略差。在全景分割上，没有自我注意力的 MaskFormer 模型在所有指标上的表现都较差。

“Semantic” queries _vs_. “panoptic” queries. In Figure [I](#A4.F1 "Figure I ‣ Appendix D Additional ablation studies ‣ Per-Pixel Classification is Not All You Need for Semantic Segmentation") we visualize predictions for the “car” category from MaskFormer trained with semantic-level and instance-level ground truth data. In the case of semantic-level data, the matching cost and loss used for mask prediction force a single query to predict one mask that combines all cars together. In contrast, with instance-level ground truth, MaskFormer uses different queries to make mask predictions for each car. This observation suggests that our model has the capacity to adapt to different types of tasks given different ground truth annotations.  
“语义”查询与“全景”查询。在图 I 中，我们可视化了使用语义级和实例级地面实况数据训练的 MaskFormer 对“汽车”类别的预测。对于语义级数据，用于掩码预测的匹配成本和损失会强制单个查询预测一个将所有汽车组合在一起的掩码。相比之下，对于实例级地面事实，MaskFormer 使用不同的查询来对每辆车进行掩码预测。这一观察结果表明，我们的模型有能力适应不同类型的任务，给定不同的地面实况注释。

| ground truth 地面实况 | prediction | ground truth 地面实况 | prediction |

 ![](https://ar5iv.labs.arxiv.org/html/2107.06278/assets/figures/visualization/semseg_predictions_jpg/ADE_val_00000908_gt.jpg)

 ![](https://ar5iv.labs.arxiv.org/html/2107.06278/assets/figures/visualization/semseg_predictions_jpg/ADE_val_00000908_dt.jpg)

 ![](https://ar5iv.labs.arxiv.org/html/2107.06278/assets/figures/visualization/semseg_predictions_jpg/ADE_val_00001785_gt.jpg)

 ![](https://ar5iv.labs.arxiv.org/html/2107.06278/assets/figures/visualization/semseg_predictions_jpg/ADE_val_00001785_dt.jpg) 

 ![](https://ar5iv.labs.arxiv.org/html/2107.06278/assets/figures/visualization/semseg_predictions_jpg/ADE_val_00001827_gt.jpg)

 ![](https://ar5iv.labs.arxiv.org/html/2107.06278/assets/figures/visualization/semseg_predictions_jpg/ADE_val_00001827_dt.jpg)

 ![](https://ar5iv.labs.arxiv.org/html/2107.06278/assets/figures/visualization/semseg_predictions_jpg/ADE_val_00001831_gt.jpg)

 ![](https://ar5iv.labs.arxiv.org/html/2107.06278/assets/figures/visualization/semseg_predictions_jpg/ADE_val_00001831_dt.jpg) 

 ![](https://ar5iv.labs.arxiv.org/html/2107.06278/assets/figures/visualization/semseg_predictions_jpg/ADE_val_00001795_gt.jpg)

 ![](https://ar5iv.labs.arxiv.org/html/2107.06278/assets/figures/visualization/semseg_predictions_jpg/ADE_val_00001795_dt.jpg)

 ![](https://ar5iv.labs.arxiv.org/html/2107.06278/assets/figures/visualization/semseg_predictions_jpg/ADE_val_00001839_gt.jpg)

 ![](https://ar5iv.labs.arxiv.org/html/2107.06278/assets/figures/visualization/semseg_predictions_jpg/ADE_val_00001839_dt.jpg) 

 ![](https://ar5iv.labs.arxiv.org/html/2107.06278/assets/figures/visualization/semseg_predictions_jpg/ADE_val_00000134_gt.jpg)

 ![](https://ar5iv.labs.arxiv.org/html/2107.06278/assets/figures/visualization/semseg_predictions_jpg/ADE_val_00000134_dt.jpg)

 ![](https://ar5iv.labs.arxiv.org/html/2107.06278/assets/figures/visualization/semseg_predictions_jpg/ADE_val_00001853_gt.jpg)

 ![](https://ar5iv.labs.arxiv.org/html/2107.06278/assets/figures/visualization/semseg_predictions_jpg/ADE_val_00001853_dt.jpg) 

 ![](https://ar5iv.labs.arxiv.org/html/2107.06278/assets/figures/visualization/semseg_predictions_jpg/ADE_val_00000001_gt.jpg)

 ![](https://ar5iv.labs.arxiv.org/html/2107.06278/assets/figures/visualization/semseg_predictions_jpg/ADE_val_00000001_dt.jpg)

 ![](https://ar5iv.labs.arxiv.org/html/2107.06278/assets/figures/visualization/semseg_predictions_jpg/ADE_val_00000939_gt.jpg)

 ![](https://ar5iv.labs.arxiv.org/html/2107.06278/assets/figures/visualization/semseg_predictions_jpg/ADE_val_00000939_dt.jpg) 

 ![](https://ar5iv.labs.arxiv.org/html/2107.06278/assets/figures/visualization/semseg_predictions_jpg/ADE_val_00000485_gt.jpg)

 ![](https://ar5iv.labs.arxiv.org/html/2107.06278/assets/figures/visualization/semseg_predictions_jpg/ADE_val_00000485_dt.jpg)

 ![](https://ar5iv.labs.arxiv.org/html/2107.06278/assets/figures/visualization/semseg_predictions_jpg/ADE_val_00000506_gt.jpg)

 ![](https://ar5iv.labs.arxiv.org/html/2107.06278/assets/figures/visualization/semseg_predictions_jpg/ADE_val_00000506_dt.jpg) 

Figure II: Visualization of MaskFormer semantic segmentation predictions on the ADE20K dataset. We visualize the MaskFormer with Swin-L backbone which achieves 55.6 mIoU (multi-scale) on the validation set. First and third columns: ground truth. Second and fourth columns: prediction.  
图二：ADE20K 数据集上 MaskFormer 语义分割预测的可视化。我们可视化了带有 Swin-L 主干的 MaskFormer，它在验证集上实现了 55.6 mIoU（多尺度）。第一列和第三列：地面实况。第二列和第四列：预测。

## Appendix E Visualization 附录 E 标准化

We visualize sample semantic segmentation predictions of the MaskFormer model with Swin-L \[[29](#bib.bib29)\] backbone (55.6 mIoU) on the ADE20K validation set in Figure [II](#A4.F2 "Figure II ‣ Appendix D Additional ablation studies ‣ Per-Pixel Classification is Not All You Need for Semantic Segmentation").  
我们在图 II 中的 ADE20K 验证集上可视化了具有 Swin-L \[29\] 主干（55.6 mIoU）的 MaskFormer 模型的样本语义分割预测。

## References 引用


*   \[1\] COCO + Places Challenges 2017. [https://places-coco2017.github.io/](https://places-coco2017.github.io/), 2016.  
*   \[2\] Pablo Arbeláez, Jordi Pont-Tuset, Jonathan T Barron, Ferran Marques, and Jitendra Malik. Multiscale combinatorial grouping. In CVPR, 2014.  
*   \[3\] Holger Caesar, Jasper Uijlings, and Vittorio Ferrari. COCO-Stuff: Thing and stuff classes in context. In CVPR, 2018.  
*   \[4\] Nicolas Carion, Francisco Massa, Gabriel Synnaeve, Nicolas Usunier, Alexander Kirillov, and Sergey Zagoruyko. End-to-end object detection with transformers. In ECCV, 2020.  
*   \[5\] Joao Carreira, Rui Caseiro, Jorge Batista, and Cristian Sminchisescu. Semantic segmentation with second-order pooling. In ECCV, 2012.  
*   \[6\] Joao Carreira and Cristian Sminchisescu. CPMC: Automatic object segmentation using constrained parametric min-cuts. PAMI, 2011.  
*   \[7\] Liang-Chieh Chen, George Papandreou, Iasonas Kokkinos, Kevin Murphy, and Alan L Yuille. DeepLab: Semantic image segmentation with deep convolutional nets, atrous convolution, and fully connected CRFs. PAMI, 2018.  
*   \[8\] Liang-Chieh Chen, George Papandreou, Florian Schroff, and Hartwig Adam. Rethinking atrous convolution for semantic image segmentation. arXiv:1706.05587, 2017.  
*   \[9\] Liang-Chieh Chen, Yukun Zhu, George Papandreou, Florian Schroff, and Hartwig Adam. Encoder-decoder with atrous separable convolution for semantic image segmentation. In ECCV, 2018.  
*   \[10\] Bowen Cheng, Liang-Chieh Chen, Yunchao Wei, Yukun Zhu, Zilong Huang, Jinjun Xiong, Thomas S Huang, Wen-Mei Hwu, and Honghui Shi. SPGNet: Semantic prediction guidance for scene parsing. In ICCV, 2019.  
*   \[11\] Bowen Cheng, Maxwell D Collins, Yukun Zhu, Ting Liu, Thomas S Huang, Hartwig Adam, and Liang-Chieh Chen. Panoptic-DeepLab: A simple, strong, and fast baseline for bottom-up panoptic segmentation. In CVPR, 2020.  
*   \[12\] François Chollet. Xception: Deep learning with depthwise separable convolutions. In CVPR, 2017.  
*   \[13\] Dorin Comaniciu and Peter Meer. Robust Analysis of Feature Spaces: Color Image Segmentation. In CVPR, 1997.  
*   \[14\] MMSegmentation Contributors. MMSegmentation: OpenMMLab semantic segmentation toolbox and benchmark. [https://github.com/open-mmlab/mmsegmentation](https://github.com/open-mmlab/mmsegmentation), 2020.  
*   \[15\] Marius Cordts, Mohamed Omran, Sebastian Ramos, Timo Rehfeld, Markus Enzweiler, Rodrigo Benenson, Uwe Franke, Stefan Roth, and Bernt Schiele. The Cityscapes dataset for semantic urban scene understanding. In CVPR, 2016.  
*   \[16\] Jifeng Dai, Kaiming He, and Jian Sun. Convolutional feature masking for joint object and stuff segmentation. In CVPR, 2015.  
*   \[17\] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, et al. An image is worth 16x16 words: Transformers for image recognition at scale. In ICLR, 2021.  
*   \[18\] Mark Everingham, SM Ali Eslami, Luc Van Gool, Christopher KI Williams, John Winn, and Andrew Zisserman. The PASCAL visual object classes challenge: A retrospective. IJCV, 2015.
*   \[19\] Jun Fu, Jing Liu, Haijie Tian, Yong Li, Yongjun Bao, Zhiwei Fang, and Hanqing Lu. Dual attention network for scene segmentation. In CVPR, 2019.
*   \[20\] Bharath Hariharan, Pablo Arbeláez, Ross Girshick, and Jitendra Malik. Simultaneous detection and segmentation. In ECCV, 2014.
*   \[21\] Kaiming He, Georgia Gkioxari, Piotr Dollár, and Ross Girshick. Mask R-CNN. In ICCV, 2017.
*   \[22\] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In CVPR, 2016.
*   \[23\] Zilong Huang, Xinggang Wang, Lichao Huang, Chang Huang, Yunchao Wei, and Wenyu Liu. CCNet: Criss-cross attention for semantic segmentation. In ICCV, 2019.
*   \[24\] Alexander Kirillov, Kaiming He, Ross Girshick, Carsten Rother, and Piotr Dollár. Panoptic segmentation. In CVPR, 2019.
*   \[25\] Scott Konishi and Alan Yuille. Statistical Cues for Domain Specific Image Segmentation with Performance Analysis. In CVPR, 2000.
*   \[26\] Tsung-Yi Lin, Piotr Dollár, Ross Girshick, Kaiming He, Bharath Hariharan, and Serge Belongie. Feature pyramid networks for object detection. In CVPR, 2017.
*   \[27\] Tsung-Yi Lin, Priya Goyal, Ross Girshick, Kaiming He, and Piotr Dollár. Focal loss for dense object detection. In ICCV, 2017.
*   \[28\] Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Dollár, and C Lawrence Zitnick. Microsoft COCO: Common objects in context. In ECCV, 2014.
*   \[29\] Ze Liu, Yutong Lin, Yue Cao, Han Hu, Yixuan Wei, Zheng Zhang, Stephen Lin, and Baining Guo. Swin transformer: Hierarchical vision transformer using shifted windows. arXiv:2103.14030, 2021.
*   \[30\] Jonathan Long, Evan Shelhamer, and Trevor Darrell. Fully convolutional networks for semantic segmentation. In CVPR, 2015.
*   \[31\] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. In ICLR, 2019.
*   \[32\] Ningning Ma, Xiangyu Zhang, Hai-Tao Zheng, and Jian Sun. ShuffleNet V2: Practical guidelines for efficient cnn architecture design. In ECCV, 2018.
*   \[33\] Fausto Milletari, Nassir Navab, and Seyed-Ahmad Ahmadi. V-Net: Fully convolutional neural networks for volumetric medical image segmentation. In 3DV, 2016.
*   \[34\] Gerhard Neuhold, Tobias Ollmann, Samuel Rota Bulò, and Peter Kontschieder. The mapillary vistas dataset for semantic understanding of street scenes. In CVPR, 2017.
*   \[35\] Olga Russakovsky, Jia Deng, Hao Su, Jonathan Krause, Sanjeev Satheesh, Sean Ma, Zhiheng Huang, Andrej Karpathy, Aditya Khosla, Michael Bernstein, Alexander C. Berg, and Li Fei-Fei. ImageNet Large Scale Visual Recognition Challenge. IJCV, 2015.
*   \[36\] Jianbo Shi and Jitendra Malik. Normalized Cuts and Image Segmentation. PAMI, 2000.
*   \[37\] Robin Strudel, Ricardo Garcia, Ivan Laptev, and Cordelia Schmid. Segmenter: Transformer for semantic segmentation. arXiv:2105.05633, 2021.
*   \[38\] Andrew Tao, Karan Sapra, and Bryan Catanzaro. Hierarchical multi-scale attention for semantic segmentation. arXiv:2005.10821, 2020.
*   \[39\] Zhi Tian, Chunhua Shen, and Hao Chen. Conditional convolutions for instance segmentation. In ECCV, 2020.
*   \[40\] Jasper RR Uijlings, Koen EA Van De Sande, Theo Gevers, and Arnold WM Smeulders. Selective search for object recognition. IJCV, 2013.
*   \[41\] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Lukasz Kaiser, and Illia Polosukhin. Attention is all you need. In NeurIPS, 2017.
*   \[42\] Huiyu Wang, Yukun Zhu, Hartwig Adam, Alan Yuille, and Liang-Chieh Chen. MaX-DeepLab: End-to-end panoptic segmentation with mask transformers. In CVPR, 2021.
*   \[43\] Xiaolong Wang, Ross Girshick, Abhinav Gupta, and Kaiming He. Non-local neural networks. In CVPR, 2018.
*   \[44\] Xinlong Wang, Rufeng Zhang, Tao Kong, Lei Li, and Chunhua Shen. SOLOv2: Dynamic and fast instance segmentation. NeurIPS, 2020.
*   \[45\] Yuxin Wu and Kaiming He. Group normalization. In ECCV, 2018.
*   \[46\] Yuxin Wu, Alexander Kirillov, Francisco Massa, Wan-Yen Lo, and Ross Girshick. Detectron2. [https://github.com/facebookresearch/detectron2](https://github.com/facebookresearch/detectron2), 2019.
*   \[47\] Yangxin Wu, Gengwei Zhang, Yiming Gao, Xiajun Deng, Ke Gong, Xiaodan Liang, and Liang Lin. Bidirectional graph reasoning network for panoptic segmentation. In CVPR, 2020.
*   \[48\] Yangxin Wu, Gengwei Zhang, Hang Xu, Xiaodan Liang, and Liang Lin. Auto-panoptic: Cooperative multi-component architecture search for panoptic segmentation. In NeurIPS, 2020.
*   \[49\] Tete Xiao, Yingcheng Liu, Bolei Zhou, Yuning Jiang, and Jian Sun. Unified perceptual parsing for scene understanding. In ECCV, 2018.
*   \[50\] Yuhui Yuan, Xilin Chen, and Jingdong Wang. Object-contextual representations for semantic segmentation. In ECCV, 2020.
*   \[51\] Yuhui Yuan, Lang Huang, Jianyuan Guo, Chao Zhang, Xilin Chen, and Jingdong Wang. OCNet: Object context for semantic segmentation. IJCV, 2021.
*   \[52\] Hengshuang Zhao, Jianping Shi, Xiaojuan Qi, Xiaogang Wang, and Jiaya Jia. Pyramid scene parsing network. In CVPR, 2017.
*   \[53\] Sixiao Zheng, Jiachen Lu, Hengshuang Zhao, Xiatian Zhu, Zekun Luo, Yabiao Wang, Yanwei Fu, Jianfeng Feng, Tao Xiang, Philip HS Torr, et al. Rethinking semantic segmentation from a sequence-to-sequence perspective with transformers. In CVPR, 2021.
*   \[54\] Bolei Zhou, Hang Zhao, Xavier Puig, Sanja Fidler, Adela Barriuso, and Antonio Torralba. Scene parsing challenge 2016. [http://sceneparsing.csail.mit.edu/index_challenge.html](http://sceneparsing.csail.mit.edu/index_challenge.html), 2016.
*   \[55\] Bolei Zhou, Hang Zhao, Xavier Puig, Sanja Fidler, Adela Barriuso, and Antonio Torralba. Scene parsing through ADE20K dataset. In CVPR, 2017.


# 收获
## Dice 损失