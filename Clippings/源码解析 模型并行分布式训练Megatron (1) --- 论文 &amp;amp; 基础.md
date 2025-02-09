---
title: "[源码解析] 模型并行分布式训练Megatron (1) --- 论文 &amp;amp; 基础"
source: "https://www.cnblogs.com/rossiXYZ/p/15840803.html"
author:
  - "[[罗西的思考]]"
published: 2022-01-27T18:50:00.0000000&#x2B;08:00
created: 2025-02-08
description: "NVIDIA Megatron 是一个基于 PyTorch 的分布式训练框架，用来训练超大Transformer语言模型，其通过综合应用了数据并行，Tensor并行和Pipeline并行来复现 GPT3，值得我们深入分析其背后机理。"
tags:
  - "clippings"
---
NVIDIA Megatron 是一个基于 PyTorch 的分布式训练框架，用来训练超大Transformer语言模型，其通过综合应用了数据并行，Tensor并行和Pipeline并行来复现 GPT3，值得我们深入分析其背后机理。

## \[源码解析\] 模型并行分布式训练Megatron (1) --- 论文 & 基础

- [\[源码解析\] 模型并行分布式训练Megatron (1) --- 论文 & 基础](https://www.cnblogs.com/rossiXYZ/p/#%E6%BA%90%E7%A0%81%E8%A7%A3%E6%9E%90-%E6%A8%A1%E5%9E%8B%E5%B9%B6%E8%A1%8C%E5%88%86%E5%B8%83%E5%BC%8F%E8%AE%AD%E7%BB%83megatron-1-----%E8%AE%BA%E6%96%87--%E5%9F%BA%E7%A1%80)
- [0x00 摘要](https://www.cnblogs.com/rossiXYZ/p/#0x00-%E6%91%98%E8%A6%81)
- [0x01 Introduction](https://www.cnblogs.com/rossiXYZ/p/#0x01-introduction)
- [1.1 问题](https://www.cnblogs.com/rossiXYZ/p/#11-%E9%97%AE%E9%A2%98)
- [1.2 数据并行](https://www.cnblogs.com/rossiXYZ/p/#12-%E6%95%B0%E6%8D%AE%E5%B9%B6%E8%A1%8C)
- [1.3 模型并行](https://www.cnblogs.com/rossiXYZ/p/#13-%E6%A8%A1%E5%9E%8B%E5%B9%B6%E8%A1%8C)
- [1.3.1 通信](https://www.cnblogs.com/rossiXYZ/p/#131-%E9%80%9A%E4%BF%A1)
- [1.3.2 张量并行](https://www.cnblogs.com/rossiXYZ/p/#132-%E5%BC%A0%E9%87%8F%E5%B9%B6%E8%A1%8C)
- [1.3.3 流水线并行](https://www.cnblogs.com/rossiXYZ/p/#133-%E6%B5%81%E6%B0%B4%E7%BA%BF%E5%B9%B6%E8%A1%8C)
- [1.4 技术组合](https://www.cnblogs.com/rossiXYZ/p/#14-%E6%8A%80%E6%9C%AF%E7%BB%84%E5%90%88)
- [1.5 指导原则](https://www.cnblogs.com/rossiXYZ/p/#15-%E6%8C%87%E5%AF%BC%E5%8E%9F%E5%88%99)
- [0x02 张量模型并行（Tensor Model Parallelism）](https://www.cnblogs.com/rossiXYZ/p/#0x02-%E5%BC%A0%E9%87%8F%E6%A8%A1%E5%9E%8B%E5%B9%B6%E8%A1%8Ctensor-model-parallelism)
- [2.1 原理](https://www.cnblogs.com/rossiXYZ/p/#21-%E5%8E%9F%E7%90%86)
- [2.1.1 行并行（Row Parallelism）](https://www.cnblogs.com/rossiXYZ/p/#211-%E8%A1%8C%E5%B9%B6%E8%A1%8Crow--parallelism)
- [2.1.2 列并行（Column Parallelism）](https://www.cnblogs.com/rossiXYZ/p/#212-%E5%88%97%E5%B9%B6%E8%A1%8Ccolumn-parallelism)
- [2.2 Model Parallel Transformers](https://www.cnblogs.com/rossiXYZ/p/#22-model-parallel-transformers)
- [2.2.1 Transformer](https://www.cnblogs.com/rossiXYZ/p/#221-transformer)
- [2.2.2 切分 Transformer](https://www.cnblogs.com/rossiXYZ/p/#222-%E5%88%87%E5%88%86-transformer)
- [2.2.3 切分MLP](https://www.cnblogs.com/rossiXYZ/p/#223-%E5%88%87%E5%88%86mlp)
- [2.2.4 切分self attention](https://www.cnblogs.com/rossiXYZ/p/#224-%E5%88%87%E5%88%86self-attention)
- [2.2.5 通信](https://www.cnblogs.com/rossiXYZ/p/#225-%E9%80%9A%E4%BF%A1)
- [2.2.6 小结](https://www.cnblogs.com/rossiXYZ/p/#226-%E5%B0%8F%E7%BB%93)
- [0x03 并行配置](https://www.cnblogs.com/rossiXYZ/p/#0x03-%E5%B9%B6%E8%A1%8C%E9%85%8D%E7%BD%AE)
- [3.1 符号说明](https://www.cnblogs.com/rossiXYZ/p/#31-%E7%AC%A6%E5%8F%B7%E8%AF%B4%E6%98%8E)
- [3.2 Tensor and Pipeline Model Parallelism](https://www.cnblogs.com/rossiXYZ/p/#32-tensor-and-pipeline-model-parallelism)
- [3.3 Data and Model Parallelism](https://www.cnblogs.com/rossiXYZ/p/#33-data-and-model-parallelism)
- [3.3.1 Pipeline Model Parallelism.](https://www.cnblogs.com/rossiXYZ/p/#331-pipeline-model-parallelism)
- [3.3.2 Data and Tensor Model Parallelism.](https://www.cnblogs.com/rossiXYZ/p/#332-data-and-tensor-model-parallelism)
- [3.4 Microbatch Size](https://www.cnblogs.com/rossiXYZ/p/#34-microbatch-size)
- [3.5 对比](https://www.cnblogs.com/rossiXYZ/p/#35-%E5%AF%B9%E6%AF%94)
- [3.5.1 Tensor versus Pipeline Parallelism.](https://www.cnblogs.com/rossiXYZ/p/#351-tensor-versus-pipeline-parallelism)
- [3.5.2 Pipeline versus Data Parallelism.](https://www.cnblogs.com/rossiXYZ/p/#352-pipeline-versus-data-parallelism)
- [3.5.3 Tensor versus Data Parallelism.](https://www.cnblogs.com/rossiXYZ/p/#353-tensor-versus-data-parallelism)
- [0x04 结论](https://www.cnblogs.com/rossiXYZ/p/#0x04-%E7%BB%93%E8%AE%BA)
- [0xFF 参考](https://www.cnblogs.com/rossiXYZ/p/#0xff-%E5%8F%82%E8%80%83)

## 0x00 摘要

NVIDIA Megatron 是一个基于 PyTorch 的分布式训练框架，用来训练超大Transformer语言模型，其通过综合应用了数据并行，Tensor并行和Pipeline并行来复现 GPT3，值得我们深入分析其背后机理。

本系列大概有6～7篇文章，通过论文和源码和大家一起学习研究。

本文把 Megatron 的两篇论文/一篇官方PPT 选取部分内容，糅合在一起进行翻译分析，希望大家可以通过本文对 Megatron 思路有一个基本了解。

## 0x01 Introduction

### 1.1 问题

在NLP领域之中，大模型可以带来更精准强大的语义理解和推理能力，所以随着规模计算的普及和数据集的增大，使得模型的参数数量也以指数级的速度增长。训练这样大的模型非常具有挑战性，具体原因如下：

- (a) 对显存的挑战。即使是最大的GPU的主内存也不可能适合这些模型的参数，比如一个175B的GPT-3模型需要（175B \* 4bytes）就是700GB模型参数空间，从而梯度也是700G，优化器状态是1400G，一共2.8TB。
- (b) 对计算的挑战。即使我们能够把模型放进单个GPU中（例如，通过在主机和设备内存之间交换参数），但是其所需的大量计算操作会导致漫长训练时间（例如，使用单个V100 NVIDIA GPU来训练1750亿个参数的GPT-3需要大约288年）。如何计算可以参见 2104.04473的附录 FLOATING-POINT OPERATIONS。
- (c) 对计算的挑战。不同并行策略对应的通信模式和通信量不同。
- 数据并行：通信发生在后向传播的梯度规约all-reduce操作，通信量是每个GPU之上模型的大小。
- 模型并行：我们在下面会详述。

这就需要采用并行化来加速。使用硬件加速器来横向扩展（scale out）深度神经网络训练主要有两种模式：数据并行，模型并行。

### 1.2 数据并行

数据并行模式会在每个worker之上复制一份模型，这样每个worker都有一个完整模型的副本。输入数据集是分片的，一个训练的小批量数据将在多个worker之间分割；worker定期汇总它们的梯度，以确保所有worker看到一个一致的权重版本。对于无法放进单个worker的大型模型，人们可以在模型之中较小的分片上使用数据并行。

数据并行扩展通常效果很好，但有两个限制：

- a）超过某一个点之后，每个GPU的batch size变得太小，这降低了GPU的利用率，增加了通信成本；
- b）可使用的最大设备数就是batch size，着限制了可用于训练的加速器数量。

### 1.3 模型并行

人们会使用一些内存管理技术，如激活检查点（activation checkpointing）来克服数据并行的这种限制，也会使用模型并行来对模型进行分区来解决这两个挑战，使得权重及其关联的优化器状态不需要同时驻留在处理器上。

模型并行模式会让一个模型的内存和计算分布在多个worker之间，以此来解决一个模型在一张卡上无法容纳的问题，其解决方法是把模型放到多个设备之上。

模型并行分为两种：流水线并行和张量并行，就是把模型切分的方式。

- 流水线并行（pipeline model parallel）是把模型不同的层放到不同设备之上，比如前面几层放到一个设备之上，中间几层放到另外一个设备上，最后几层放到第三个设备之上。
- 张量并行则是层内分割，把某一个层做切分，放置到不同设备之上，也可以理解为把矩阵运算分配到不同的设备之上，比如把某个矩阵乘法切分成为多个矩阵乘法放到不同设备之上。

具体如下图，上面是层间并行（流水线并行），纵向切一刀，前面三层给第一个GPU，后面三层给第二个GPU。下面是层内并行（tensor并行），横向切一刀，每个张量分成两块，分到不同GPU之上。

![](https://img2022.cnblogs.com/blog/1850883/202201/1850883-20220124201659620-672005374.png)

或者从另一个角度看看，两种切分同时存在，是正交和互补的（orthogonal and complimentary）。

![](https://img2022.cnblogs.com/blog/1850883/202201/1850883-20220124201711859-473710086.png)

图来自：[GTC 2020: Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism](https://developer.nvidia.com/gtc/2020/video/s21496)

#### 1.3.1 通信

我们接下来看看模型并行的通信状况。

- 张量并行：通信发生在每层的前向传播和后向传播过程之中，通信类型是all-reduce，不但单次通信数据量大，并且通信频繁。
- 流水线并行：通信在流水线阶段相邻的切分点之上，通信类型是P2P通信，单词通信数据量较少但是比较频繁，而且因为流水线的特点，会产生GPU空闲时间，这里称为流水线气泡（Bubble）。

比如下图之中，上方是原始流水线，下面是模型并行，中间给出了 Bubble 位置。

![](https://img2022.cnblogs.com/blog/1850883/202201/1850883-20220124201730745-73728244.png)

因为张量并行一般都在同一个机器之上，所以通过 NVLink 来进行加速，对于流水线并行，一般通过 Infiniband 交换机进行连接。

![](https://img2022.cnblogs.com/blog/1850883/202201/1850883-20220124201740766-976118624.png)

图来自 Megatron 论文。

#### 1.3.2 张量并行

有些工作在张量（层内）模型并行化（ tensor (intra-layer) model parallelism）做出了一些尝试，即每个transformer 层内的矩阵乘法被分割到多个GPU上，虽然这种方法在NVIDIA DGX A100服务器（有8个80GB-A100 GPU）上对规模不超过200亿个参数的模型效果很好，但对更大的模型就会出现问题。因为较大的模型需要在多个多GPU服务器上分割，这导致了两个问题。

- (a) 张量并行所需的all-reduce通信需要通过服务器间的链接，这比多GPU服务器内的高带宽NVLink要慢；
- (b) 高度的模型并行会产生很多小矩阵乘法（GEMMs），这可能会降低GPU的利用率。

#### 1.3.3 流水线并行

流水线模型并行化是另一项支持大型模型训练的技术。在流水线并行之中，一个模型的各层会在多个GPU上做切分。一个批次（batch）被分割成较小的微批（microbatches），并在这些微批上进行流水线式执行。

通过流水线并行，一个模型的层被分散到多个设备上。当用于具有相同transformer块重复的模型时，每个设备可以被分配相同数量的transformer层。Megatron不考虑更多的非对称模型架构，在这种架构下，层的分配到流水线阶段是比较困难的。在流水线模型并行中，训练会在一个设备上执行一组操作，然后将输出传递到流水线中下一个设备，下一个设备将执行另一组不同操作。

原生（naive）流水线会有这样的问题：一个输入在后向传递中看到的权重更新并不是其前向传递中所对应的。所以，流水线方案需要确保输入在前向和后向传播中看到一致的权重版本，以实现明确的同步权重更新语义。

模型的层可以用各种方式分配给worker，并且对于输入的前向计算和后向计算使用不同的schedule。层的分配策略和调度策略导致了不同的性能权衡。无论哪种调度策略，为了保持严格的优化器语义，优化器操作步骤（step）需要跨设备同步，这样，在每个批次结束时需要进行流水线刷新来完成微批执行操作（同时没有新的微批被注入）。Megatron引入了定期流水线刷新。

在每个批次的开始和结束时，设备是空闲的。我们把这个空闲时间称为流水线bubble，并希望它尽可能的小。根据注入流水线的微批数量，多达50%的时间可能被用于刷新流水线。微批数量与流水线深度（size）的比例越大，流水线刷新所花费的时间就越少。因此，为了实现高效率，通常需要较大的batch size。

一些方法将参数服务器与流水线并行使用。然而，这些都存在不一致的问题。TensorFlow的GPipe框架通过使用同步梯度下降克服了这种不一致性问题。然而，这种方法需要额外的逻辑来处理这些通信和计算操作流水线，并且会遇到降低效率的流水线气泡，或者对优化器本身的更改会影响准确性。

某些异步和bounded-staleness方法，如PipeMare、PipeDream和PipeDream-2BW完全取消了刷新，但这样会放松了权重更新语义。Megatron会在未来的工作中考虑这些方案。

### 1.4 技术组合

用户可以使用各种技术来训练他们的大型模型，每种技术都有不同的权衡。此外，这些技术也可以被结合起来使用。然而，结合这些技术会导致复杂的相互作用，对于系统拓扑是个极大的挑战，不仅要对模型做合理切割（依据算法特点），还需要做软硬件一体的系统架构设计，需要仔细推理以获得良好的性能。因此以下问题就特别重要：

应该如何组合并行技术，以便在保留严格的优化器语义的同时，在给定的batch size下最大限度地提高大型模型的训练吞吐量？

Megatron-LM 开发人员展示了一个如何结合流水线、张量和数据并行，名为PTD-P的技术，这项技术将以良好的计算性能（峰值设备吞吐量的52%）在1000个GPU上训练大型语言模型。PTD-P利用跨多GPU服务器的流水线并行、多GPU服务器内的张量并行和数据并行的组合，在同一服务器和跨服务器的GPU之间具有高带宽链接的优化集群环境中训练具有一万亿个参数的模型，并具有优雅的扩展性。

![](https://img2022.cnblogs.com/blog/1850883/202201/1850883-20220124201816805-1665303573.png)

要实现这种规模化的吞吐量，需要在多个方面进行创新和精心设计：

- 高效的核（kernel）实现，这使大部分计算操作是计算绑定（compute-bound）而不是内存绑定（memory-bound。
- 在设备上对计算图进行智能分割，以减少通过网络发送的字节数，同时也限制设备的空闲时间。
- 实施特定领域的通信优化和使用高速硬件（比如最先进的GPU，并且同一服务器内和不同服务器GPU之间使用高带宽链接）。

### 1.5 指导原则

Megatron 开发者研究了各种组合之间如何影响吞吐量，基于这些研究得出来分布式训练的一些指导原则：

- 不同的并行模式以复杂的方式互相作用：并行化策略影响通信量、核的计算效率，以及worker因流水线刷新（流水线气泡）而等待的空闲时间。例如，张量模型并行在多GPU服务器中是有效的，但大模型必须采用流水线模型并行。
- 用于流水线并行的schdule对通信量、流水线气泡大小和用于存储激活的内存都有影响。Megatron 提出了一个新的交错schdule，与以前提出的schdule相比，它可以在稍微提高内存占用的基础上提高多达10%的吞吐量。
- 超参数的值，如microbatch size，对memory footprint、在worker上执行的核效果和流水线bubble大小有影响。
- 分布式训练是通信密集型的。使用较慢的节点间连接或更多的通信密集型分区会阻碍性能。

## 0x02 张量模型并行（Tensor Model Parallelism）

### 2.1 原理

我们用 GEMM 来看看如何进行模型并行，这里要进行的是 XA = Y，对于模型来说，X 是输入，A是权重，Y是输出。从数学原理上来看，对于`linear`层就是把矩阵分块进行计算，然后把结果合并，对于非`linear`层则不做额外设计。

#### 2.1.1 行并行（Row Parallelism）

我们先看看Row Parallelism，就是把 A 按照行分割成两部分。为了保证运算，同时我们也把 X 按照列来分割为两部分，这里 
$$
X1
$$
的最后一个维度等于 
$$
A1
$$
 最前的一个维度，理论上是：

$$$
X A = \left[\right. X_{1} & X_{2} \left]\right. \left[\right. A_{1} \\ A_{2} \left]\right. = X_{1} A_{1} + X_{2} A_{2} = Y_{1} + Y_{2} = Y
$$$

所以，$X_{1}$和 $A_{1}$ 就可以放到第一个 GPU 之上计算，
$$
X2
$$
 和 
$$
A2
$$
 可以放到第二个 GPU 之上，然后把结果相加。

![](https://img2022.cnblogs.com/blog/1850883/202201/1850883-20220124201834765-1518661121.png)

我们接下来进行计算。第一步是把图上横向红色箭头和纵向箭头进行点积，得到Y中的绿色。

![](https://img2022.cnblogs.com/blog/1850883/202201/1850883-20220124201843190-1398946842.png)

第三步，计算出来一个新的绿色。

![](https://img2022.cnblogs.com/blog/1850883/202201/1850883-20220124201852094-726256529.png)

第四步，计算了输出的一行。

![](https://img2022.cnblogs.com/blog/1850883/202201/1850883-20220124201901325-1183238194.png)

第五步，继续执行，得出了一个 
$$
Y1
$$
。

![](https://img2022.cnblogs.com/blog/1850883/202201/1850883-20220124201910579-1822208580.png)

第六步，得出了蓝色的 
$$
Y2
$$
，此时，可以把 
$$
Y1,Y2
$$
 加起来，得到最终的输出 Y。

![](https://img2022.cnblogs.com/blog/1850883/202201/1850883-20220124201921849-1308485259.png)

#### 2.1.2 列并行（Column Parallelism）

我们接下来看看另外一种并行方式Column Parallelism，就是把 A按照列来分割。

![](https://img2022.cnblogs.com/blog/1850883/202201/1850883-20220124201929688-1253758722.png)

最终计算结果如下：

![](https://img2022.cnblogs.com/blog/1850883/202201/1850883-20220124201938521-1658186920.png)

图来自：[GTC 2020: Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism](https://developer.nvidia.com/gtc/2020/video/s21496)

### 2.2 Model Parallel Transformers

这里Transformer的模型并行，特指层内切分，即 Tensor Model Parallel。

#### 2.2.1 Transformer

自从2018年Google的Attention论文推出之后，近年的模型架构都是在 Transformer基础之上完成，模型有多少层，就意味着模型有多少个Transformer块，所以语言模型的计算量主要是Transformer的计算，而Transformer本质上就是大量的矩阵计算，适合GPU并行操作。

Transformers层由一个Masked Multi Self Attention和Feed Forward两部分构成，Feed Forward 部分是一个MLP网络，由多个全连接层构成，每个全连接层是由矩阵乘操作和GeLU激活层或者Dropout构成。

Megatron 的 Feed Forward 是一个两层多层感知器（MLP），第一层是从 H变成4H，第二层是从 4H 变回到 H，所以Transformer具体架构如下，紫色块对应于全连接层。每个蓝色块表示一个被复制N次的transformer层，红色的 x L 代表此蓝色复制 L 次。

![](https://img2022.cnblogs.com/blog/1850883/202201/1850883-20220124202021702-620140809.png)

#### 2.2.2 切分 Transformer

分布式张量计算是一种正交且更通用的方法，它将张量操作划分到多个设备上，以加速计算或增加模型大小。FlexFlow是一个进行这种并行计算的深度学习框架，并且提供了一种选择最佳并行化策略的方法。最近，Mesh TensorFlow引入了一种语言，用于指定TensorFlow中的一般分布式张量计算。用户在语言中指定并行维度，并使用适当的集合原语编译生成一个计算图。我们采用了Mesh TensorFlow的相似见解，并利用transformer's attention heads 的计算并行性来并行化Transformer模型。然而，Megatron没有实现模型并行性的框架和编译器，而是对现有的PyTorch transformer实现进行了一些有针对性的修改。Megatron的方法很简单，不需要任何新的编译器或代码重写，只是通过插入一些简单的原语来完全实现，

Megatron就是要把 Masked Multi Self Attention 和Feed Forward 都进行切分以并行化，利用Transformers网络的结构，通过添加一些同步原语来创建一个简单的模型并行实现。

#### 2.2.3 切分MLP

我们从MLP块开始。MLP 块的第一部分是GEMM，后面是GeLU：

$$$
Y = G e L U \left(\right. X A \left.\right)
$$$

并行化GEMM的**一个选项**是沿行方向分割权重矩阵A，沿列切分输入X：

$$$
X = \left[\right. X_{1} & X_{2} \left]\right. , A = \left[\right. A_{1} \\ A_{2} \left]\right.
$$$

分区的结果就变成 $Y = G e L U \left(\right. X_{1} A_{1} + X_{2} A_{2} \left.\right)$，括号之中的两项，每一个都可以在一个独立的GPU之上完成，然后通过 all-reduce 操作完成求和操纵。既然 GeLU 是一个非线性函数，那么就有 
$$
GeLU(X1A1+X2A2)≠GeLU(X1A1)+GeLH(X2A2)
$$
，所以这种方案需要在 GeLU 函数之前加上一个同步点。这个同步点让不同GPU之间交换信息。

**另一个选项**是沿列拆分A，得到 
$$
A=[A1，A2]
$$
。该分区允许GeLU非线性独立应用于每个分区GEMM的输出：

$$$
\left[\right. Y_{1} & Y_{2} \left]\right. = \left[\right. G e L U \left(\right. X A_{1} \left.\right) , G e L U \left(\right. X A_{2} \left.\right) \left]\right.
$$$

这个方法更好，因为它删除了同步点，直接把两个 GeLU 的输出拼接在一起就行。因此，我们以这种列并行方式划分第一个GEMM，并沿其行分割第二个GEMM，以便它直接获取GeLU层的输出，而不需要任何其他通信（比如 all-reduce 就不需要了），如图所示。

![](https://img2022.cnblogs.com/blog/1850883/202201/1850883-20220124202101035-1802963939.png)

上图第一个是 GeLU 操作，第二个是 Dropout操作，具体逻辑如下：

1. MLP的整个输入 X 通过 f 放置到每一块 GPU 之上。
2. 对于第一个全连接层：
1. 使用列分割，把权重矩阵切分到两块 GPU 之上，得到 
$$
A1,A2
$$
。
2. 在每一块 GPU 之上进行矩阵乘法得到第一个全连接层的输出 
$$
Y1
$$
 和 
$$
Y2
$$
。
3. 对于第二个全连接层：
1. 使用行切分，把权重矩阵切分到两个 GPU 之上，得到 
$$
B1,B2
$$
。
2. 前面输出 
$$
Y1
$$
 和 
$$
Y2
$$
 正好满足需求，直接可以和 B 的相关部分（
$$
B1,B2
$$
）做相关计算，不需要通信或者其他操作，就得到了 
$$
Z1,Z2
$$
。分别位于两个GPU之上。
4. $$
Z1,Z2
$$
 通过 g 做 all-reduce（这是一个同步点），再通过 dropout 得到了最终的输出 Z。

然后在GPU之上，第二个GEMM的输出在传递到dropout层之前进行规约。这种方法将MLP块中的两个GEMM跨GPU进行拆分，并且只需要在前向过程中进行一次 all-reduce 操作（g 操作符）和在后向过程中进行一次 all-reduce 操作（f 操作符）。这两个操作符是彼此共轭体，只需几行代码就可以在PyTorch中实现。作为示例，f 运算符的实现如下所示：

![](https://img2022.cnblogs.com/blog/1850883/202201/1850883-20220124202114458-1063784615.png)

f算子的实现。g类似于f，在后向函数中使用identity，在前向函数中使用all-reduce。

#### 2.2.4 切分self attention

如下图所示。

- 首先，对于自我注意力块，Megatron 利用了多头注意力操作中固有的并行性，以列并行方式对与键（K）、查询（Q）和值（V）相关联的GEMM进行分区，从而在一个GPU上本地完成与每个注意力头对应的矩阵乘法。这使我们能够在GPU中分割每个attention head参数和工作负载，每个GPU得到了部分输出。
- 其次，对于后续的全连接层，因为每个GPU之上有了部分输出，所以对于权重矩阵B就按行切分，与输入的 
$$
Y1,Y2
$$
 进行直接计算，然后通过 g 之中的 all-reduce 操作和Dropout 得到最终结果 Z。

![](https://img2022.cnblogs.com/blog/1850883/202201/1850883-20220124202130180-63572728.png)

图：具有模型并行性的transformer块。f和g是共轭的。f在前向传播中使用一个identity运算符，在后向传播之中使用了all reduce，而g在前向传播之中使用了all reduce，在后向传播中使用了identity运算符。

#### 2.2.5 通信

来自线性层（在 self attention 层之后）输出的后续GEMM会沿着其行实施并行化，并直接获取并行注意力层的输出，而不需要GPU之间的通信。这种用于MLP和自我注意层的方法融合了两个GEMM组，消除了中间的同步点，并导致更好的伸缩性。这使我们能够在一个简单的transformer层中执行所有GEMM，只需在正向路径中使用两个all-reduce，在反向路径中使用两个all-reduce（见下图）。

![](https://img2022.cnblogs.com/blog/1850883/202201/1850883-20220124202140707-537421751.png)

图：transformer层中的通信操作。在一个单模型并行transformer层的正向和反向传播中总共有4个通信操作。

Transformer语言模型输出了一个嵌入，其维数为隐藏大小（H）乘以词汇量大小（v）。由于现代语言模型的词汇量约为数万个（例如，GPT-2使用的词汇量为50257），因此将嵌入GEMM的输出并行化是非常有益的。然而，在transformer语言模型中，想让输出嵌入层与输入嵌入层共享权重，需要对两者进行修改。

我们沿着词汇表维度 
$$
E=[E1，E2]
$$
（按列）对输入嵌入权重矩阵
$$
EH×v
$$
进行并行化。因为每个分区现在只包含嵌入表的一部分，所以在输入嵌入之后需要一个all-reduce（g操作符）。对于输出嵌入，一种方法是执行并行 
$$
GEMM[Y1，Y2]=[XE1，XE2]
$$
 以获得logit，然后添加一个all-gather 
$$
Y=all−gather([Y1，Y2])
$$
，并将结果发送到交叉熵损失函数。但是，在这种情况下，由于词汇表的很大，all-gather 将传递
$$
b×s×v
$$
 个元素（b是batch size，s是序列长度）。为了减小通信规模，我们将并行
$$
GEMM[Y1，Y2]
$$
的输出与交叉熵损失进行融合，从而将维数降低到
$$
b×s
$$
。

#### 2.2.6 小结

我们的模型并行方法旨在减少通信和控制GPU计算范围的。我们不是让一个GPU计算dropout、layer normalization或 residual connection，并将结果广播给其他GPU，而是选择跨GPU复制计算。

模型并行性与数据并行性是正交的，因此我们可以同时使用二者在来训练大型模型。下图显示了一组用于混合模型并行和数据并行性的GPU。

- 一个模型需要占据8张卡，模型被复制了64分，一共启动了512个即成。
- 模型并行。同一服务器内的多个GPU形成模型并行组（model parallel group），例如图中的GPU 1到8，并包含分布在这些GPU上的模型实例。其余的GPU可能位于同一台服务器内，也可能位于其他服务器中，它们运行其他模型并行组。每个模型并行组内的GPU执行组内所有GPU之间的all-reduce。
- 数据并行。在每个模型并行组中具有相同位置的GPU（例如图中的GPU 1，9，…，505）形成数据并行组（data parallel group），即，具有相同模型参数的进程被分配到同一个数据并行组之中。对于数据并行，每个all-reduce操作在每个模型并行组中一个GPU之上执行。
- 所有通信都是通过pytorch调用NCCL来实现的。

在反向传播过程中，我们并行运行多个梯度all-reduce操作，以规约每个不同数据并行组中的权重梯度。所需GPU的总数是模型和数据并行组数量的乘积。

![](https://img2022.cnblogs.com/blog/1850883/202201/1850883-20220124202155530-142530468.png)

混合模型和数据并行的GPU分组，8路模型并行和64路数据并行。

## 0x03 并行配置

我们接着看如何混合使用各种并行。

### 3.1 符号说明

以下是本文余下使用的符号说明。

![](https://img2022.cnblogs.com/blog/1850883/202201/1850883-20220124202226689-953995445.png)

### 3.2 Tensor and Pipeline Model Parallelism

张量和流水线模型并行性都可以用于在多个GPU上划分模型的参数。如前所述，将流水线并行性与周期性刷新一起使用会产生大小为 
$$
(𝑝−1)/𝑚
$$
的流水线气泡。 让我们假设𝑑 = 1（数据并行大小），因此 𝑡 · 𝑝 = 𝑛。在此情况下，流水线气泡大小是：

$$$
\frac{p - 1}{m} = \frac{n / t - 1}{m}
$$$

假如我们固定𝐵, 𝑏, 和𝑑 (𝑚 = 𝐵/(𝑏 · 𝑑) 也固定下来），当 𝑡 增加时，流水线气泡会相应减小。

不同GPU之间通信量也受𝑝 和𝑡 的影响。管道模型并行具有更便宜的点对点通信。另一方面，张量模型并行性使用更消耗带宽的all-reduce通信（向前和向后传递中各有两个all-reduce操作）。

- 使用流水线并行，在每对连续设备（向前或向后传播）之间为每个微批次执行的通信总量为𝑏𝑠h，𝑠 是序列长度，h是隐藏大小（hidden size）。
- 使用张量模型并行，每个层前向传播和后向传播中，总大小𝑏𝑠h的张量需要在 𝑡 个模型副本之中 all-reduce 两次。

因此，我们看到张量模型并行性增加了设备之间的通信量。因此，当 𝑡 大于单个节点中的GPU数量时，在较慢的节点间链路上执行张量模型并行是不合算的。

因此得到：

结论#1：当考虑不同形式的模型并行时，当使用𝑔-GPU服务器，通常应该把张量模型并行度控制在 𝑔 之内，然后使用流水线并行来跨服务器扩展到更大的模型。

![](https://img2022.cnblogs.com/blog/1850883/202201/1850883-20220124202239526-1516567242.png)

### 3.3 Data and Model Parallelism

然后考虑数据并行和模型并行。

#### 3.3.1 Pipeline Model Parallelism.

我们给定 𝑡 = 1 (tensor-model-parallel size)，那么每个流水线的微批次数目是 
$$
𝑚=𝐵/(𝑑·𝑏)=𝑏′/𝑑,
$$
，这里 
$$
𝑏′:=𝐵/𝑏
$$
。给定 GPU 数目为 n，流水线阶段的数目是 𝑝 = 𝑛/(𝑡 · 𝑑) = 𝑛/𝑑，流水线气泡大小是：

$$$
\frac{p - 1}{m} = \frac{n / d - 1}{b^{'} / d} = \frac{n - d}{b^{'}}
$$$

当 𝑑 变大，𝑛 − 𝑑 变小，因此流水线气泡变小。因为模型训练需要的内存占用可能大于单个加速器的内存容量，所以不可能增加𝑑 一直到𝑛。而数据并行性所需的all-reduce通信不会随着更高的数据并行度而增加。

我们还可以分析 batch size 𝐵 增加带来的影响。 对于给定的并行配置，如批大小𝐵 增加，𝑏′ = 𝐵/𝑏 增加，(𝑛 − 𝑑)/𝑏′ 会相应减少，从而增加吞吐量。数据并行所需的all-reduce也变得更少，从而进一步提高了吞吐量。

#### 3.3.2 Data and Tensor Model Parallelism.

使用张量模型并行，每个微批次都需要执行all-reduce通信。这在多GPU服务器之间可能非常昂贵。另一方面，数据并行性对于每个批次只需执行一次 all-reduce。此外，使用张量模型并行，每个模型并行rank在每个模型层中只执行计算的子集，因此对于不够大的层，现代GPU可能无法以最高效率执行这些子矩阵计算。

结论#2：当使用数据和模型并行时，总的模型并行大小应该为𝑀 = 𝑡 · 𝑝 ，这样模型参数和中间元数据可以放入GPU内存。数据并行性可用于将训练扩展到更多GPU。

![](https://img2022.cnblogs.com/blog/1850883/202201/1850883-20220124202251827-843597518.png)

### 3.4 Microbatch Size

微批尺寸 𝑏 的选择也影响到模型训练的吞吐量。例如，在单个GPU上，如果微批尺寸较大，每个GPU的吞吐量最多可增加1.3倍。现在，在定并行配置（𝑝,𝑡,𝑑）和批量大小𝐵下，我们想确定最佳微批尺寸𝑏。

无论微批大小如何，数据并行通信量将是相同的。鉴于函数 
$$
tf(b)
$$
和 
$$
tb(b)
$$
将微批大小映射到单个微批的前向和后向计算时间，在忽略通信成本的条件下，计算一个batch的总时间为（如前，定义𝑏′为𝐵/𝑑）。

$$$
（ b^{'} / b + p - 1 \left.\right) . \left(\right. t_{f} \left(\right. b \left.\right) + t_{b} \left(\right. b \left.\right) \left.\right)
$$$

因此，微批的大小既影响操作的算术强度，也影响管道 bubble 大小（通过影响𝑚）。

经验之谈#3: 最佳微批尺寸𝑏取决于模型的吞吐量和内存占用特性，以及管道深度𝑝、数据并行尺寸𝑑和批尺寸𝐵。

![](https://img2022.cnblogs.com/blog/1850883/202201/1850883-20220124202259669-175978037.png)

### 3.5 对比

我们接下来看看各种并行机制的对比。

#### 3.5.1 Tensor versus Pipeline Parallelism.

我们观察到，张量模型的并行性在节点（DGX A100服务器）内是最好的，因为它会减少通信量。另一方面，流水线模型并行使用更便宜的点对点通信，可以跨节点执行，而不会限制整个计算。然而，流水线并行性会在流水线气泡中花费大量时间，因此，应限制流水线级的总数，以便流水线中的microbatches数量是流水线深度的合理倍数。因此，当张量并行大小等于单个节点中的GPU数量（8个，DGX A100个节点）时会达到峰值性能。这一结果表明，单独使用张量模型并行性（Megatron V1）和流水线模型并行性（PipeDream）都无法与这两种技术结合使用的性能相匹配。

#### 3.5.2 Pipeline versus Data Parallelism.

通过实验发现，对于每个batch size，吞吐量随着流水线并行规模的增加而降低。流水线模型并行应该主要用于支持不适合单个 worker 的大型模型训练，数据并行应该用于扩大训练规模。

#### 3.5.3 Tensor versus Data Parallelism.

接下来看看数据和张量模型的并行性对性能的影响。在较大的批处理量和微批处理量为1的情况下，数据并行通信并不频繁；张量模型并行需要对批处理中的每个微批进行all-to-all通信。这种all-to-all的通信与张量模型并行主义的通信主导了端到端的训练时间，特别是当通信需要在多GPU节点上进行时。此外，随着张量模型并行规模的增加，我们在每个GPU上执行较小的矩阵乘法，降低了每个GPU的利用率。

我们应该注意到，尽管数据并行可以带来高效的扩展，但我们不能单独使用数据并行来处理训练批量有限的大型模型，因为a）内存容量不足，b）数据并行的扩展限制（例如，GPT-3的训练批量为1536。因此，数据并行性只支持并行到1536个GPU；然而，大约有10000个GPU用来训练这个模型）。

## 0x04 结论

Megatron使用了PTD-P（节点间流水线并行、节点内张量并行和数据并行）在训练具有万亿参数的大型模型时候达到了高聚合吞吐量（502 petaFLOP/s）。

- Tensor模型并行被用于intra-node transformer 层，这样在HGX based系统上高效运行。
- Pipeline 模型并行被用于inter-node transformer 层，其可以有效利用集群中多网卡设计。
- 数据并行则在前两者基础之上进行加持，使得训练可以扩展到更大规模和更快的速度。

## 0xFF 参考

[\[细读经典\]Megatron论文和代码详细分析(2)](https://zhuanlan.zhihu.com/p/388830967)

[\[细读经典\]Megatron论文和代码详细分析(1)](https://zhuanlan.zhihu.com/p/366906920)

[Megatron-LM源码阅读（一）](https://zhuanlan.zhihu.com/p/405883984)

[Megatron-LM源码阅读（二）](https://zhuanlan.zhihu.com/p/407094090)

[megatron学习总结](https://zhuanlan.zhihu.com/p/381326200)

[GTC 2020: Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism](https://developer.nvidia.com/gtc/2020/video/s21496)

www.DeepL.com/Translator

[https://developer.nvidia.com/gtc/2020/slides/s21496-megatron-lm-training-multi-billion-parameter-language-models-using-model-parallelism.pdf](https://developer.nvidia.com/gtc/2020/slides/s21496-megatron-lm-training-multi-billion-parameter-language-models-using-model-parallelism.pdf)

[NVIDIA Megatron：超大Transformer语言模型的分布式训练框架 （一）](https://zhuanlan.zhihu.com/p/420908718)

[NVIDIA Megatron：超大Transformer语言模型的分布式训练框架 (二)](https://zhuanlan.zhihu.com/p/423596659)