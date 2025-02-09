---
title: "[源码解析] 模型并行分布式训练 Megatron (3) ---模型并行实现"
source: "https://www.cnblogs.com/rossiXYZ/p/15871062.html"
author:
  - "[[罗西的思考]]"
published: 2022-02-08T16:31:00.0000000&#x2B;08:00
created: 2025-02-08
description: "NVIDIA Megatron 是一个基于 PyTorch 的分布式训练框架，用来训练超大Transformer语言模型，其通过综合应用了数据并行，Tensor并行和Pipeline并行来复现 GPT3，值得我们深入分析其背后机理。"
tags:
  - "clippings"
---
NVIDIA Megatron 是一个基于 PyTorch 的分布式训练框架，用来训练超大Transformer语言模型，其通过综合应用了数据并行，Tensor并行和Pipeline并行来复现 GPT3，值得我们深入分析其背后机理。

## \[源码解析\] 模型并行分布式训练 Megatron (3) ---模型并行实现

- [\[源码解析\] 模型并行分布式训练 Megatron (3) ---模型并行实现](https://www.cnblogs.com/rossiXYZ/p/#%E6%BA%90%E7%A0%81%E8%A7%A3%E6%9E%90-%E6%A8%A1%E5%9E%8B%E5%B9%B6%E8%A1%8C%E5%88%86%E5%B8%83%E5%BC%8F%E8%AE%AD%E7%BB%83-megatron-3----%E6%A8%A1%E5%9E%8B%E5%B9%B6%E8%A1%8C%E5%AE%9E%E7%8E%B0)
- [0x00 摘要](https://www.cnblogs.com/rossiXYZ/p/#0x00-%E6%91%98%E8%A6%81)
- [0x01 并行Transformer层](https://www.cnblogs.com/rossiXYZ/p/#0x01-%E5%B9%B6%E8%A1%8Ctransformer%E5%B1%82)
- [1.1 初始化](https://www.cnblogs.com/rossiXYZ/p/#11-%E5%88%9D%E5%A7%8B%E5%8C%96)
- [1.2 前向传播](https://www.cnblogs.com/rossiXYZ/p/#12-%E5%89%8D%E5%90%91%E4%BC%A0%E6%92%AD)
- [0x02 并行MLP](https://www.cnblogs.com/rossiXYZ/p/#0x02-%E5%B9%B6%E8%A1%8Cmlp)
- [2.1 命名规范](https://www.cnblogs.com/rossiXYZ/p/#21-%E5%91%BD%E5%90%8D%E8%A7%84%E8%8C%83)
- [2.2 MLP 代码](https://www.cnblogs.com/rossiXYZ/p/#22-mlp-%E4%BB%A3%E7%A0%81)
- [2.2.1 初始化](https://www.cnblogs.com/rossiXYZ/p/#221-%E5%88%9D%E5%A7%8B%E5%8C%96)
- [2.2.2 前向操作](https://www.cnblogs.com/rossiXYZ/p/#222-%E5%89%8D%E5%90%91%E6%93%8D%E4%BD%9C)
- [0x03 ColumnParallelLinear](https://www.cnblogs.com/rossiXYZ/p/#0x03-columnparallellinear)
- [3.1 定义](https://www.cnblogs.com/rossiXYZ/p/#31-%E5%AE%9A%E4%B9%89)
- [3.2 初始化](https://www.cnblogs.com/rossiXYZ/p/#32-%E5%88%9D%E5%A7%8B%E5%8C%96)
- [3.2.1 切分size](https://www.cnblogs.com/rossiXYZ/p/#321-%E5%88%87%E5%88%86size)
- [3.2.2 初始化权重](https://www.cnblogs.com/rossiXYZ/p/#322-%E5%88%9D%E5%A7%8B%E5%8C%96%E6%9D%83%E9%87%8D)
- [3.3 逻辑梳理](https://www.cnblogs.com/rossiXYZ/p/#33-%E9%80%BB%E8%BE%91%E6%A2%B3%E7%90%86)
- [3.3.1 前向传播](https://www.cnblogs.com/rossiXYZ/p/#331-%E5%89%8D%E5%90%91%E4%BC%A0%E6%92%AD)
- [3.3.2 后向传播](https://www.cnblogs.com/rossiXYZ/p/#332-%E5%90%8E%E5%90%91%E4%BC%A0%E6%92%AD)
- [3.4 代码实现](https://www.cnblogs.com/rossiXYZ/p/#34-%E4%BB%A3%E7%A0%81%E5%AE%9E%E7%8E%B0)
- [3.3.1 ColumnParallelLinear](https://www.cnblogs.com/rossiXYZ/p/#331-columnparallellinear)
- [3.3.2 f 操作](https://www.cnblogs.com/rossiXYZ/p/#332-f-%E6%93%8D%E4%BD%9C)
- [3.3.2.1 同步操作](https://www.cnblogs.com/rossiXYZ/p/#3321-%E5%90%8C%E6%AD%A5%E6%93%8D%E4%BD%9C)
- [3.3.2.2 异步 All-Reduce](https://www.cnblogs.com/rossiXYZ/p/#3322-%E5%BC%82%E6%AD%A5-all-reduce)
- [3.3.3 g 操作](https://www.cnblogs.com/rossiXYZ/p/#333-g-%E6%93%8D%E4%BD%9C)
- [3.3.4 基础函数](https://www.cnblogs.com/rossiXYZ/p/#334-%E5%9F%BA%E7%A1%80%E5%87%BD%E6%95%B0)
- [3.3.4.1 gather](https://www.cnblogs.com/rossiXYZ/p/#3341-gather)
- [3.3.4.2 split](https://www.cnblogs.com/rossiXYZ/p/#3342-split)
- [0x04 RowParallelLinear](https://www.cnblogs.com/rossiXYZ/p/#0x04-rowparallellinear)
- [4.1 定义](https://www.cnblogs.com/rossiXYZ/p/#41-%E5%AE%9A%E4%B9%89)
- [4.2 初始化](https://www.cnblogs.com/rossiXYZ/p/#42-%E5%88%9D%E5%A7%8B%E5%8C%96)
- [4.3 逻辑梳理](https://www.cnblogs.com/rossiXYZ/p/#43-%E9%80%BB%E8%BE%91%E6%A2%B3%E7%90%86)
- [4.3.1 前向传播](https://www.cnblogs.com/rossiXYZ/p/#431-%E5%89%8D%E5%90%91%E4%BC%A0%E6%92%AD)
- [4.3.2 后向传播](https://www.cnblogs.com/rossiXYZ/p/#432-%E5%90%8E%E5%90%91%E4%BC%A0%E6%92%AD)
- [4.4 代码实现](https://www.cnblogs.com/rossiXYZ/p/#44-%E4%BB%A3%E7%A0%81%E5%AE%9E%E7%8E%B0)
- [4.4.1 RowParallelLinear](https://www.cnblogs.com/rossiXYZ/p/#441-rowparallellinear)
- [4.4.1 f 操作](https://www.cnblogs.com/rossiXYZ/p/#441-f-%E6%93%8D%E4%BD%9C)
- [4.4.2 g 操作](https://www.cnblogs.com/rossiXYZ/p/#442-g-%E6%93%8D%E4%BD%9C)
- [0x05 Embedding](https://www.cnblogs.com/rossiXYZ/p/#0x05-embedding)
- [0x06 总结](https://www.cnblogs.com/rossiXYZ/p/#0x06-%E6%80%BB%E7%BB%93)
- [6.1 MLP并行](https://www.cnblogs.com/rossiXYZ/p/#61-mlp%E5%B9%B6%E8%A1%8C)
- [6.2 共轭函数](https://www.cnblogs.com/rossiXYZ/p/#62-%E5%85%B1%E8%BD%AD%E5%87%BD%E6%95%B0)
- [0xFF 参考](https://www.cnblogs.com/rossiXYZ/p/#0xff-%E5%8F%82%E8%80%83)

## 0x00 摘要

NVIDIA Megatron 是一个基于 PyTorch 的分布式训练框架，用来训练超大Transformer语言模型，其通过综合应用了数据并行，Tensor并行和Pipeline并行来复现 GPT3，值得我们深入分析其背后机理。

本系列大概有6～7篇文章，通过论文和源码和大家一起学习研究。本文将看看 Megatron 如何处理模型并行。

本系列其他文章为：

\[[源码解析\] 模型并行分布式训练Megatron (1) --- 论文 & 基础](https://www.cnblogs.com/rossiXYZ/p/15840803.html)

\[[源码解析\] 模型并行分布式训练Megatron (2) --- 整体架构](https://www.cnblogs.com/rossiXYZ/p/15868988.html)

## 0x01 并行Transformer层

在论文篇之中，我们了解到，因为模型越来越大，其尺寸远远超过了处理器的内存限制，因此产生了诸如激活检查点（activation checkpointing）这样的内存管理技术。而模型并行则通过对模型进行各种分片来克服单个处理器内存限制，这样模型权重和其关联的优化器状态就可以分散到多个设备之上。

ParallelTransformerLayer 就是对 Transformer 层的并行实现，所以我们接着分析。

### 1.1 初始化

ParallelTransformerLayer 初始化方法之中，建立了如下：

- 生成一个LayerNorm处理输入数据。
- 生成并行Attention。
- 生成处理attention输出的LayerNorm。
- 如果是decoder，则生成一个ParallelAttention。
- 生成一个并行MLP。

```python
class ParallelTransformerLayer(MegatronModule):
    """A single transformer layer.

    Transformer layer takes input with size [b, s, h] and returns an
    output of the same size.
    """

    def __init__(self, init_method, output_layer_init_method,
                 layer_number, layer_type=LayerType.encoder,
                 self_attn_mask_type=AttnMaskType.padding):
        args = get_args()

        super(ParallelTransformerLayer, self).__init__()
        self.layer_number = layer_number
        self.layer_type = layer_type

        self.apply_residual_connection_post_layernorm \
            = args.apply_residual_connection_post_layernorm

        self.bf16 = args.bf16
        self.fp32_residual_connection = args.fp32_residual_connection

        # Layernorm on the input data.
        self.input_layernorm = LayerNorm( # 生成一个LayerNorm处理输入数据
            args.hidden_size,
            eps=args.layernorm_epsilon,
            no_persist_layer_norm=args.no_persist_layer_norm)

        # Self attention.
        self.self_attention = ParallelAttention( # 生成并行Attention
            init_method,
            output_layer_init_method,
            layer_number,
            attention_type=AttnType.self_attn,
            attn_mask_type=self_attn_mask_type)
        self.hidden_dropout = args.hidden_dropout
        self.bias_dropout_fusion = args.bias_dropout_fusion

        # Layernorm on the attention output
        self.post_attention_layernorm = LayerNorm( # 生成处理attention输出的LayerNorm
            args.hidden_size,
            eps=args.layernorm_epsilon,
            no_persist_layer_norm=args.no_persist_layer_norm)

        if self.layer_type == LayerType.decoder: # 如果本层是decoder
            self.inter_attention = ParallelAttention( # 则生成一个ParallelAttention
                init_method,
                output_layer_init_method,
                layer_number,
                attention_type=AttnType.cross_attn)
            # Layernorm on the attention output.
            self.post_inter_attention_layernorm = LayerNorm(
                args.hidden_size,
                eps=args.layernorm_epsilon,
                no_persist_layer_norm=args.no_persist_layer_norm)

        # MLP
        self.mlp = ParallelMLP(init_method, # 生成一个并行MLP
                               output_layer_init_method)
```

对应就是：

![](https://img2022.cnblogs.com/blog/1850883/202202/1850883-20220208141043153-601051421.png)

### 1.2 前向传播

其前向传播方法如下，就是调用各种成员函数进行前向操作。

```python
def forward(self, hidden_states, attention_mask,
            encoder_output=None, enc_dec_attn_mask=None,
            inference_params=None):
    # hidden_states: [b, s, h]

    # Layer norm at the beginning of the transformer layer.
    layernorm_output = self.input_layernorm(hidden_states) # 对输入进行处理
    
    # Self attention.
    attention_output, attention_bias = \ # attention操作
        self.self_attention(
            layernorm_output,
            attention_mask,
            inference_params=inference_params)

    # Residual connection. 残差连接
    if self.apply_residual_connection_post_layernorm:
        residual = layernorm_output #norm之后结果作为X
    else:
        residual = hidden_states # 原始输入X

    # jit scripting for a nn.module (with dropout) is not
    # trigerring the fusion kernel. For now, we use two
    # different nn.functional routines to account for varying
    # dropout semantics during training and inference phases.
    if self.bias_dropout_fusion: # dropout操作
        if self.training:
            bias_dropout_add_func = bias_dropout_add_fused_train
        else:
            bias_dropout_add_func = bias_dropout_add_fused_inference
    else:
        bias_dropout_add_func = get_bias_dropout_add(self.training)

    # re-enable torch grad to enable fused optimization.
    with torch.enable_grad():
        layernorm_input = bias_dropout_add_func( # dropout操作
            attention_output,
            attention_bias.expand_as(residual),
            residual,
            self.hidden_dropout)

    # Layer norm post the self attention.
    layernorm_output = self.post_attention_layernorm(layernorm_input) # 处理attention输出

    if self.layer_type == LayerType.decoder:
        attention_output, attention_bias = \
            self.inter_attention(layernorm_output,
                                 enc_dec_attn_mask,
                                 encoder_output=encoder_output)
        # residual connection
        if self.apply_residual_connection_post_layernorm:
            residual = layernorm_output
        else:
            residual = layernorm_input

        # re-enable torch grad to enable fused optimization.
        with torch.enable_grad():
            layernorm_input = bias_dropout_add_func(
                attention_output,
                attention_bias.expand_as(residual),
                residual,
                self.hidden_dropout)

        # Layer norm post the decoder attention
        layernorm_output = self.post_inter_attention_layernorm(layernorm_input)

    # MLP.
    mlp_output, mlp_bias = self.mlp(layernorm_output) # MLP操作 

    # Second residual connection.
    if self.apply_residual_connection_post_layernorm: # 残差操作
        residual = layernorm_output
    else:
        residual = layernorm_input

    # re-enable torch grad to enable fused optimization.
    with torch.enable_grad():
        output = bias_dropout_add_func( # dropout操作
            mlp_output,
            mlp_bias.expand_as(residual),
            residual,
            self.hidden_dropout)

    return output
```

## 0x02 并行MLP

ParallelTransformerLayer 里面包含了 Attention 和 MLP，因为篇幅所限，我们这里主要对MLP进行分析。对于 Attention 则简单研究一下其行切分机制，毕竟我们想了解的是如何进行模型并行，而非深入理解Transformer。

![](https://img2022.cnblogs.com/blog/1850883/202202/1850883-20220208141100274-550820520.png)

Megatron的并行MLP包含了两个线性层，第一个线性层实现了 hidden size 到 4 x hidden size 的转换，第二个线性层实现了 4 x hidden size 回到 hidden size。具体 MLP 的逻辑如下：

![](https://img2022.cnblogs.com/blog/1850883/202202/1850883-20220208141109167-1520737440.png)

图：具有模型并行性的 MLP。f和g表示和通信切块相关的操作，其是共轭的。f 的前向传播是一个identity运算符，而后向传播是一个all-reduce，g 的前向传播是 all-reduce，后向传播是一个identity运算符。这里的 f 来自 ColumnParallelLinear，g 来自 RowParallelLinear。即，MLP 就是把 ColumnParallelLinear 和 RowParallelLinear 结合起来。

于是，这里焦点问题就是：如何把这两种线性层切开到不同的GPU卡之上？参见前文，这里采用了第二种方案，

> **另一个选项**是沿列拆分A，得到 
> $$
> A=[A1，A2]
> $$
> 。该分区允许GeLU非线性独立应用于每个分区GEMM的输出：
> 
> $$$
> \left[\right. Y_{1} & Y_{2} \left]\right. = \left[\right. G e L U \left(\right. X A_{1} \left.\right) , G e L U \left(\right. X A_{2} \left.\right) \left]\right.
> $$$
> 
> 这个方法更好，因为它删除了同步点，直接把两个 GeLU 的输出拼接在一起就行。因此，我们以这种列并行方式划分第一个GEMM，并沿其行分割第二个GEMM，以便它直接获取GeLU层的输出，而不需要任何其他通信（比如 all-reduce 就不需要了），如图所示。

我们再深入分析一下为何选择这个方案。

按照常规逻辑，MLP 的前向传播应该分为两个阶段，分别对应了下面图之中的两行，

- 第一行是把参数 A 按照列切分，然后把结果按照列拼接起来，得到的结果就是与不使用并行策略完全等价的结果。
- 第二行是把激活 Y 按照列切分，参数B按照行切分做并行，最后把输出做加法，得到 Z。

![](https://img2022.cnblogs.com/blog/1850883/202202/1850883-20220208141133106-1001719462.jpg)

但是每个split会导致两次额外的通信（前向传播和后向传播各一次，下面只给出了前向传播）。因为对于第二行来说，其输入Y其实本质是 XA1，XA2并行的，所以为了降低通信量，我们可以把数据通信延后或者干脆取消通信，就是把第一行最后的 all\_gather 和第二行最初的 split 省略掉，这其实就是数学上的传递性和结合律（局部和之和为全局和）。于是我们就得到了论文之中的第二种方案。

![](https://img2022.cnblogs.com/blog/1850883/202202/1850883-20220208141143429-246459938.jpg)

结合代码，就是：

- ColumnParallelLinear 实现了 MLP 的前半部分或者考虑了这个线性层独立使用的情况。
- RowParallelLinear 实现了 MLP 的后半部分或者考虑了这个线性层独立使用的情况。

j

### 2.1 命名规范

我们首先看看命名规范，后文使用如下：

- h: hidden size
- n: number of attention heads
- p: number of model parallel partitions
- np: n/p
- hp: h/p
- hn: h/n
- b: batch size
- s: sequence length
- l: number of layers
- Transformer 的输入size是 \[s, b, h\]，返回一个同样size的张量，我们使用 hyperparameters 作为transformer 的超参数。

### 2.2 MLP 代码

#### 2.2.1 初始化

megatron/model/transformer.py 之中有 ParallelMLP 定义如下：

- 定义了一个 ColumnParallelLinear 用来进行第一个 H 到 4 H 的转换。
- 然后是一个 gelu。
- 接着是 RowParallelLinear 用来进行 4H 到 H 的转换回来。

dropout操作是在上面ParallelTransformerLayer的forward之中进行。

所以，MLP大致如图，这里A，B是各自的权重矩阵：

![](https://img2022.cnblogs.com/blog/1850883/202202/1850883-20220208141204330-1233527709.jpg)

也就是对应论文之中这个图形。

![](https://img2022.cnblogs.com/blog/1850883/202202/1850883-20220208141212703-2046240816.png)

代码如下。

```python
class ParallelMLP(MegatronModule):
    """MLP.

    MLP will take the input with h hidden state, project it to 4*h
    hidden dimension, perform nonlinear transformation, and project the
    state back into h hidden dimension.
    """

    def __init__(self, init_method, output_layer_init_method):
        super(ParallelMLP, self).__init__()
        args = get_args()

        # Project to 4h.
        self.dense_h_to_4h = mpu.ColumnParallelLinear( # 列切分
            args.hidden_size,
            args.ffn_hidden_size,
            gather_output=False, # 这里是false，采用第二种方案
            init_method=init_method,
            skip_bias_add=True)

        self.bias_gelu_fusion = args.bias_gelu_fusion # gelu
        self.activation_func = F.gelu
        if args.openai_gelu:
            self.activation_func = openai_gelu
        elif args.onnx_safe:
            self.activation_func = erf_gelu

        # Project back to h.
        self.dense_4h_to_h = mpu.RowParallelLinear( # 行切分
            args.ffn_hidden_size,
            args.hidden_size,
            input_is_parallel=True,
            init_method=output_layer_init_method,
            skip_bias_add=True)
```

#### 2.2.2 前向操作

这里分别调用了 ColumnParallelLinear 完成了 H 到 4H 的转换，RowParallelLinear 完成了 4H 到 H 的转换。

```python
def forward(self, hidden_states):

    # [s, b, 4hp]
    intermediate_parallel, bias_parallel = self.dense_h_to_4h(hidden_states) # 纵向切分

    if self.bias_gelu_fusion:
         intermediate_parallel = \
                 bias_gelu_impl(intermediate_parallel, bias_parallel)
    else:
        intermediate_parallel = \
            self.activation_func(intermediate_parallel + bias_parallel)

    # [s, b, h]
    output, output_bias = self.dense_4h_to_h(intermediate_parallel) # 横向切分
    return output, output_bias
```

我们接下来分别介绍 ColumnParallelLinear 和 RowParallelLinear。ColumnParallelLinear 分别可以独立使用或者作为 ParallelMLP 的前半段，RowParallelLinear 也可以独立使用或者作为 ParallelMLP 的后半段。

## 0x03 ColumnParallelLinear

ColumnParallelLinear 就是按列进行切分，也就是纵刀流。注意，这里说的是对权重进行列切分。就是：

$$$
Y = X A = X \left[\right. A_{1} , A_{2} \left]\right. = \left[\right. X A_{1} , X A_{2} \left]\right.
$$$

具体切分如下：

![](https://img2022.cnblogs.com/blog/1850883/202202/1850883-20220208141224634-1492463980.jpg)

### 3.1 定义

因为 Python 语言特性，这里有用的只是注释，从注释中可以看出来，对于 $ Y = XA + b 
$$
，A被以如下方式进行并行化：
$$
 A = \[A\_1, ..., A\_p\] $

```python
class ColumnParallelLinear(torch.nn.Module):
    """Linear layer with column parallelism.

    The linear layer is defined as Y = XA + b. A is parallelized along
    its second dimension as A = [A_1, ..., A_p].

    Arguments:
        input_size: first dimension of matrix A.
        output_size: second dimension of matrix A.
        bias: If true, add bias
        gather_output: If true, call all-gether on output and make Y avaiable
                       to all GPUs, otherwise, every GPU will have its output
                       which is Y_i = XA_i
        init_method: method to initialize weights. Note that bias is always set
                     to zero.
        stride: For the strided linear layers.
        keep_master_weight_for_test: This was added for testing and should be
                                     set to False. It returns the master weights
                                     used for initialization.
        skip_bias_add: This was added to enable performance optimations where bias
                       can be fused with other elementwise operations. we skip 
                       adding bias but instead return it.
    """
```

### 3.2 初始化

初始化代码之中主要是用切分的信息来初始化权重。

```python
def __init__(self, input_size, output_size, bias=True, gather_output=True,
             init_method=init.xavier_normal_, stride=1,
             keep_master_weight_for_test=False,
             skip_bias_add=False):
    super(ColumnParallelLinear, self).__init__()

    # Keep input parameters
    self.input_size = input_size
    self.output_size = output_size
    self.gather_output = gather_output
    # Divide the weight matrix along the last dimension.
    world_size = get_tensor_model_parallel_world_size() # 获得本tensor并行组的world size
    self.output_size_per_partition = divide(output_size, world_size) # 获得本子模型应输出size
    self.skip_bias_add = skip_bias_add

    # Parameters.
    # Note: torch.nn.functional.linear performs XA^T + b and as a result
    # we allocate the transpose.
    # Initialize weight.
    args = get_args()
    if args.use_cpu_initialization:
        # 用切分的size初始化权重
        self.weight = Parameter(torch.empty(self.output_size_per_partition,
                                            self.input_size,
                                            dtype=args.params_dtype))
        self.master_weight = _initialize_affine_weight_cpu( # 初始化权重
            self.weight, self.output_size, self.input_size,
            self.output_size_per_partition, 0, init_method,
            stride=stride, return_master_weight=keep_master_weight_for_test)
    else:
        # 用切分的size初始化权重
        self.weight = Parameter(torch.empty(
            self.output_size_per_partition, self.input_size,
            device=torch.cuda.current_device(), dtype=args.params_dtype))
        _initialize_affine_weight_gpu(self.weight, init_method, # 初始化权重
                                      partition_dim=0, stride=stride)

    if bias:
        if args.use_cpu_initialization:
            # 用切分的size初始化权重
            self.bias = Parameter(torch.empty(
                self.output_size_per_partition, dtype=args.params_dtype))
        else:
            # 用切分的size初始化权重
            self.bias = Parameter(torch.empty(
                self.output_size_per_partition,
                device=torch.cuda.current_device(),
                dtype=args.params_dtype))
        set_tensor_model_parallel_attributes(self.bias, True, 0, stride)
        # Always initialize bias to zero.
        with torch.no_grad():
            self.bias.zero_()
    else:
        self.register_parameter('bias', None)
    self.async_tensor_model_parallel_allreduce = (
            not args.no_async_tensor_model_parallel_allreduce and
            world_size > 1)
```

#### 3.2.1 切分size

`self.output_size_per_partition = divide(output_size, world_size)` 这里有一个分割size操作，得到每个子模型应该拥有的权重大小。

```python
def ensure_divisibility(numerator, denominator):
    """Ensure that numerator is divisible by the denominator."""
    assert numerator % denominator == 0, '{} is not divisible by {}'.format(
        numerator, denominator)

def divide(numerator, denominator):
    """Ensure that numerator is divisible by the denominator and return
    the division value."""
    ensure_divisibility(numerator, denominator)
    return numerator // denominator
```

#### 3.2.2 初始化权重

以下代码实现了初始化权重。

```python
def _initialize_affine_weight_gpu(weight, init_method,
                                  partition_dim, stride=1):
    """Initialize affine weight for model parallel on GPU."""

    set_tensor_model_parallel_attributes(tensor=weight,
                                         is_parallel=True,
                                         dim=partition_dim,
                                         stride=stride)

    with get_cuda_rng_tracker().fork():
        init_method(weight)

def _initialize_affine_weight_cpu(weight, output_size, input_size,
                                  per_partition_size, partition_dim,
                                  init_method, stride=1,
                                  return_master_weight=False):
    """Initialize affine weight for model parallel.

    Build the master weight on all processes and scatter
    the relevant chunk."""

    set_tensor_model_parallel_attributes(tensor=weight,
                                         is_parallel=True,
                                         dim=partition_dim,
                                         stride=stride)

    # Initialize master weight
    master_weight = torch.empty(output_size, input_size,
                                dtype=torch.float,
                                requires_grad=False)
    init_method(master_weight)
    args = get_args()
    master_weight = master_weight.to(dtype=args.params_dtype)

    # Split and copy
    per_partition_per_stride_size = divide(per_partition_size, stride)
    weight_list = torch.split(master_weight, per_partition_per_stride_size,
                              dim=partition_dim)
    rank = get_tensor_model_parallel_rank()
    world_size = get_tensor_model_parallel_world_size()
    my_weight_list = weight_list[rank::world_size]

    with torch.no_grad():
        torch.cat(my_weight_list, dim=partition_dim, out=weight)
    if return_master_weight:
        return master_weight
    return None
```

### 3.3 逻辑梳理

为了更好的分析，我们引入下图（来自参考1），这个图对应了 ColumnParallelLinear 类的前向传播和后向传播过程。**这里的 f 和 g 操作其实是从代码之中抽象出来的，可以理解为 f 是对输入的处理，g 则是处理之后得到最终输出**。此处对应了论文中描述的粗体字：

> Figure 3. Blocks of Transformer with Model Parallelism. f and g are conjugate. **f is an identity operator in the forward pass and all reduce in the backward pass** **while g is an all reduce in the forward pass and identity in the backward pass**.

![](https://img2022.cnblogs.com/blog/1850883/202202/1850883-20220208141253802-102700960.png)

图片来自 [GTC 2020: Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism](https://developer.nvidia.com/gtc/2020/video/s21496)。

我们针对上图，梳理一下逻辑。

#### 3.3.1 前向传播

我们一步一步细化。

首先，总体语义为：Y = XA + b。

其次，前向传播时候的逻辑如下：

- 输入：这里 A 沿着列做切分，X 是全部的输入（每个GPU都拥有相同的X）。
- 计算：经过计算之后，输出的 
$$
Y1,Y2
$$
 也是按照列被切分过的。每个GPU只有自己对应的分区。
- 输出：
$$
Y1,Y2
$$
 只有合并在一起，才能得到最终输出的 Y。

再次，我们使用operator来细化一下：

- 输入：因为每个GPU需要拿到一个完整的输入 X，所以前向操作之中需要把X分发到每个GPU，这样就使用了 Identity 操作。
- 计算：经过计算之后，输出的 
$$
Y1,Y2
$$
 也是按照列被切分过的。每个GPU只有自己对应的分区。
- 输出：因为
$$
Y1,Y2
$$
 需要合并在一起，才能得到最终输出的 Y。所以需要有一个 all-gather 操作来进行聚合，即得到 $Y = \left[\right. Y_{1} , Y_{2} \left]\right.$。

我们把这些逻辑点在上图上用红色方框标示，输入 X 先经过 f 来处理，输出 Y 是 g 整合之后的结果。

![](https://img2022.cnblogs.com/blog/1850883/202202/1850883-20220208141307480-867270427.png)

#### 3.3.2 后向传播

我们接下来看看后向传播，对于上图来说，后向传播是从上至下，梯度先经过 g，最后被 f 处理。

反向传播的逻辑如下：

- 目前得到了反向传播上游传过来的梯度 
$$
∂L∂Y
$$
，现在需要对其进行切分，保证每个GPU之上都有一份梯度 
$$
∂L∂Yi
$$
。操作是
$$
∂L∂Yi(split)
$$
。
- 每个GPU之上会进行关于X的梯度计算，于是每个GPU都有一份对X的梯度（但是其内容不一样）。
- 最后需要把各个 GPU 之上关于X的梯度进行相加，得到完整梯度，这就需要一个 all-reduce 操作。即 $\frac{\partial L}{\partial X} = \frac{\partial L}{\partial X} \left(\left|\right.\right)_{1} + \frac{\partial L}{\partial X} \left(\left|\right.\right)_{2}$

所以我们在图上用蓝色圆角矩形标示出来后向传播对应的算子。

![](https://img2022.cnblogs.com/blog/1850883/202202/1850883-20220208141318520-1832254572.png)

### 3.4 代码实现

我们接下来结合代码来分析。

#### 3.3.1 ColumnParallelLinear

ColumnParallelLinear 的 forward 代码之中，主要是实施了 f 和 g 的forward操作，同时把 f 和 g 的backward 操作搭建起来，具体如下：

- 如果配置了异步操作，则使用 ColumnParallelLinearWithAsyncAllreduce 完成 f 运算符的功能，这一个函数包括了identity 操作，矩阵乘法，搭建后向传播操作。
- 如果是同步操作，则：
- 使用 copy\_to\_tensor\_model\_parallel\_region 完成前向传播 identity 操作，建立反向传播all-reduce，就是图中f的backward。identity 操作 就是把输入 X 完整的拷贝到多个GPU之上，类似 X 通过 f 的前向操作，变成了 \[X, X, ..., X\]。
- 使用 linear 对 \[X, X, ..., X\] 和 权重 A 完成矩阵乘法操作。
- 如果`gather_output`为True，则在前向传播时候把 
$$
Yi
$$
 做all-gather，因为反向传播时需要把完整梯度scatter到对应GPU之上，所以要搭建对于的split操作。MLP实现之中，此处设置为 False，这样每个GPU输出的是自己partition 的 4h/p，直接传送给下一个线性层。

```python
def forward(self, input_):
    # 如果选择忽略bias，就会设置为None，后续就不用处理了
    bias = self.bias if not self.skip_bias_add else None

    # 下面主要是图中的 f 操作
    if self.async_tensor_model_parallel_allreduce:
        # 建立反向传播时候的异步all-reduce
        input_shape = input_.shape
        input_ = input_.view(input_shape[0] * input_shape[1],input_shape[2])
        # Maxtrix multiply with asynchronouse all-reduce execution
        output_parallel = ColumnParallelLinearWithAsyncAllreduce.apply(
                input_, self.weight, bias)
        output_parallel = output_parallel.view(
                input_shape[0], input_shape[1], output_parallel.shape[1])
    else:
        # Set up backprop all-reduce.、
        # 建立反向传播all-reduce，就是图中f的backward
        input_parallel = copy_to_tensor_model_parallel_region(input_) 

        # Matrix multiply.
        output_parallel = F.linear(input_parallel, self.weight, bias) # 矩阵乘法操作

    # 下面就是图中的 g 操作    
    if self.gather_output: # 是否需要聚合操作
        # All-gather across the partitions.
        # 聚合输出，就是图中g的forward
        output = gather_from_tensor_model_parallel_region(output_parallel) #
    else:
        output = output_parallel
        
    output_bias = self.bias if self.skip_bias_add else None # 如果不忽略bias，还得传出去
    return output, output_bias
```

#### 3.3.2 f 操作

F 操作是对输入进行初步处理，具体是：

- 前向传播时候直接拷贝。
- 后向传播做all-reduce。

![](https://img2022.cnblogs.com/blog/1850883/202202/1850883-20220208141327993-1106693503.png)

##### 3.3.2.1 同步操作

这里我们主要分析 copy\_to\_tensor\_model\_parallel\_region，其做了前向copy操作，同时构建了后向 all-reduce。

```python
def copy_to_tensor_model_parallel_region(input_):
    return _CopyToModelParallelRegion.apply(input_)
```

我们还是需要看看 \_CopyToModelParallelRegion。可以看到，其 forward 就是简单的把输入转移到输出，就是对应了前向复制identity。

```python
class _CopyToModelParallelRegion(torch.autograd.Function):
    """Pass the input to the model parallel region."""

    @staticmethod
    def symbolic(graph, input_):
        return input_
    
    @staticmethod
    def forward(ctx, input_):
        return input_ # 简单的把输入转移到输出，就是对应了前向复制identity

    @staticmethod
    def backward(ctx, grad_output):
        return _reduce(grad_output) # 反向传播时候，输入是多个GPU上的梯度整体，通过all-reduce合并
```

对应的后向传播就使用了All-reduce，反向传播时候，输入是多个GPU上的梯度整体，通过all-reduce合并。

```python
def _reduce(input_):
    """All-reduce the input tensor across model parallel group."""

    # Bypass the function if we are using only 1 GPU.
    if get_tensor_model_parallel_world_size()==1:
        return input_

    # All-reduce.
    torch.distributed.all_reduce(input_, group=get_tensor_model_parallel_group())

    return input_
```

##### 3.3.2.2 异步 All-Reduce

ColumnParallelLinearWithAsyncAllreduce 这里把同步之中的乘法操作也放置进来。

```python
class ColumnParallelLinearWithAsyncAllreduce(torch.autograd.Function):
    """
    Column-parallel linear layer execution with asynchronous all-reduce
    execution in backprop.
    """
    @staticmethod
    def forward(ctx, input, weight, bias):
        ctx.save_for_backward(input, weight)
        ctx.use_bias = bias is not None
        output = torch.matmul(input, weight.t()) # 同步时候的乘法也在这里了
        if bias is not None:
            output = output + bias
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.saved_tensors
        use_bias = ctx.use_bias
        grad_input = grad_output.matmul(weight)
        # Asyncronous all-reduce
        handle = torch.distributed.all_reduce( # 反向传播操作
                grad_input, group=get_tensor_model_parallel_group(), async_op=True)
        # Delay the start of weight gradient computation shortly (3us) to have
        # all-reduce scheduled first and have GPU resources allocated
        _ = torch.empty(1, device=grad_output.device) + 1
        grad_weight = grad_output.t().matmul(input)
        grad_bias = grad_output.sum(dim=0) if use_bias else None
        handle.wait()
        return grad_input, grad_weight, grad_bias
```

#### 3.3.3 g 操作

以下对应了图之中的 g 操作。G操作是最终生成输出Y，逻辑是：

- 前向传播时候做 all-gather；
- 后向传播需要执行 split，把梯度scatter到不同GPU之上。

![](https://img2022.cnblogs.com/blog/1850883/202202/1850883-20220208141338114-1994572598.png)

```python
def gather_from_tensor_model_parallel_region(input_):
    return _GatherFromModelParallelRegion.apply(input_)
```

具体代码如下：

```python
class _GatherFromModelParallelRegion(torch.autograd.Function):
    """Gather the input from model parallel region and concatinate."""

    @staticmethod
    def symbolic(graph, input_):
        return _gather(input_)
    
    @staticmethod
    def forward(ctx, input_):
        return _gather(input_)

    @staticmethod
    def backward(ctx, grad_output):
        return _split(grad_output)
```

#### 3.3.4 基础函数

我们接下来看看上面用到的一些基础函数。

##### 3.3.4.1 gather

\_gather 是沿着最后一个维度进行拼接。

```python
def _gather(input_):
    """Gather tensors and concatinate along the last dimension."""

    world_size = get_tensor_model_parallel_world_size()
    # Bypass the function if we are using only 1 GPU.
    if world_size==1:
        return input_

    # Size and dimension.
    last_dim = input_.dim() - 1
    rank = get_tensor_model_parallel_rank() # 获得本worker在tensor并行之中的rank

    tensor_list = [torch.empty_like(input_) for _ in range(world_size)]
    tensor_list[rank] = input_
    # 在本 tensor 进程组之间进行 all-gather操作
    torch.distributed.all_gather(tensor_list, input_, group=get_tensor_model_parallel_group())

    # Note: torch.cat already creates a contiguous tensor.
    output = torch.cat(tensor_list, dim=last_dim).contiguous()

    return output
```

##### 3.3.4.2 split

\_split 完成了张量切分操作。

```python
def split_tensor_along_last_dim(tensor, num_partitions,
                                contiguous_split_chunks=False):
    """Split a tensor along its last dimension.
    Arguments:
        tensor: input tensor.
        num_partitions: number of partitions to split the tensor
        contiguous_split_chunks: If True, make each chunk contiguous
                                 in memory.
    """
    # Get the size and dimension.
    last_dim = tensor.dim() - 1
    last_dim_size = divide(tensor.size()[last_dim], num_partitions) # 得到每个切分的size
    # Split.
    tensor_list = torch.split(tensor, last_dim_size, dim=last_dim) # 对张量进行切分
    # Note: torch.split does not create contiguous tensors by default.
    if contiguous_split_chunks:
        return tuple(chunk.contiguous() for chunk in tensor_list)

    return tensor_list

def _split(input_):
    """Split the tensor along its last dimension and keep the
    corresponding slice."""

    world_size = get_tensor_model_parallel_world_size() # 获取本tensor进程组的world size
    # Bypass the function if we are using only 1 GPU.
    if world_size==1:
        return input_

    # Split along last dimension.
    input_list = split_tensor_along_last_dim(input_, world_size)

    # Note: torch.split does not create contiguous tensors by default.
    rank = get_tensor_model_parallel_rank() # 获取自己的rank
    output = input_list[rank].contiguous() # 获取切分后，自己对应的rank

    return output
```

其中，get\_tensor\_model\_parallel\_rank 作用是获取本进程在tensor并行组的rank。

```python
def get_tensor_model_parallel_rank():
    """Return my rank for the tensor model parallel group."""
    global _MPU_TENSOR_MODEL_PARALLEL_RANK
    if _MPU_TENSOR_MODEL_PARALLEL_RANK is not None:
        return _MPU_TENSOR_MODEL_PARALLEL_RANK
    return torch.distributed.get_rank(group=get_tensor_model_parallel_group())
```

## 0x04 RowParallelLinear

RowParallelLinear 这里是按照行进行切分，就是横刀流，注意这里是对权重A实施行切分。比如公式为 Y = XA，X是输入，A是权重，Y是输出，行切分就是针对A的第一个维度进行切分，这里 
$$
X1
$$
 最后一个维度等于 
$$
A1
$$
 第一个维度。

$$$
X A = \left[\right. X_{1} , X_{2} \left]\right. \left[\right. A_{1} \\ A_{2} \left]\right. = X_{1} A_{1} + X_{2} A_{2} = Y_{1} + Y_{2} = Y
$$$

具体如下：

![](https://img2022.cnblogs.com/blog/1850883/202202/1850883-20220208141352962-130450706.jpg)

### 4.1 定义

定义之中只有注释有用，可以看出来如何切分。

```python
class RowParallelLinear(torch.nn.Module):
    """Linear layer with row parallelism.

    The linear layer is defined as Y = XA + b. A is parallelized along
    its first dimension and X along its second dimension as:
               -   -
              | A_1 |
              | .   |
          A = | .   |        X = [X_1, ..., X_p]
              | .   |
              | A_p |
               -   -
    Arguments:
        input_size: first dimension of matrix A.
        output_size: second dimension of matrix A.
        bias: If true, add bias. Note that bias is not parallelized.
        input_is_parallel: If true, we assume that the input is already
                           split across the GPUs and we do not split
                           again.
        init_method: method to initialize weights. Note that bias is always set
                     to zero.
        stride: For the strided linear layers.
        keep_master_weight_for_test: This was added for testing and should be
                                     set to False. It returns the master weights
                                     used for initialization.
        skip_bias_add: This was added to enable performance optimization where bias
                       can be fused with other elementwise operations. We skip
                       adding bias but instead return it.
    """
```

### 4.2 初始化

和列切分类似，初始化之中主要是获取每个权重分区的大小，然后据此切分权重。

```python
def __init__(self, input_size, output_size, bias=True,
             input_is_parallel=False,
             init_method=init.xavier_normal_, stride=1,
             keep_master_weight_for_test=False,
             skip_bias_add=False):
    super(RowParallelLinear, self).__init__()

    # Keep input parameters
    self.input_size = input_size
    self.output_size = output_size
    self.input_is_parallel = input_is_parallel
    # Divide the weight matrix along the last dimension.
    world_size = get_tensor_model_parallel_world_size()
    self.input_size_per_partition = divide(input_size, world_size) # 获取每个权重分区的大小
    self.skip_bias_add = skip_bias_add

    # Parameters.
    # Note: torch.nn.functional.linear performs XA^T + b and as a result
    # we allocate the transpose.
    # Initialize weight.
    args = get_args()
    if args.use_cpu_initialization:
        self.weight = Parameter(torch.empty(self.output_size,
                                            self.input_size_per_partition,
                                            dtype=args.params_dtype))
        # 切分权重
        self.master_weight = _initialize_affine_weight_cpu(
            self.weight, self.output_size, self.input_size,
            self.input_size_per_partition, 1, init_method,
            stride=stride, return_master_weight=keep_master_weight_for_test)
    else:
        self.weight = Parameter(torch.empty(
            self.output_size, self.input_size_per_partition,
            device=torch.cuda.current_device(), dtype=args.params_dtype))
        # 切分权重
        _initialize_affine_weight_gpu(self.weight, init_method,
                                      partition_dim=1, stride=stride)
    if bias:
        if args.use_cpu_initialization:
            self.bias = Parameter(torch.empty(self.output_size,
                                              dtype=args.params_dtype))
        else:
            self.bias = Parameter(torch.empty(
                self.output_size, device=torch.cuda.current_device(),
                dtype=args.params_dtype))
        # Always initialize bias to zero.
        with torch.no_grad():
            self.bias.zero_()
    else:
        self.register_parameter('bias', None)
```

### 4.3 逻辑梳理

为了更好的分析，我们引入下图（来自参考1），这个图对应了 RowParallelLinear 类的前向传播和后向传播过程。**这里的 f 和 g 操作其实是从代码之中抽象出来的，可以理解为 f 是对输入的处理，g 则是处理之后得到最终输出**。

![](https://img2022.cnblogs.com/blog/1850883/202202/1850883-20220208143055342-1678425157.png)

我们针对上图，梳理一下逻辑。

#### 4.3.1 前向传播

我们一步一步细化。

首先，总体语义为：Y = XA + b。

其次，前向传播时候的逻辑如下：

- 输入：这里 A 沿着行做切分，因为A的维度发生了变化，所以X也需要做相应变化，X就必须按照列做切分，这样 X 每个分块才能与A 每个分块进行相乘。这里如果输入是已经split过的(input\_is\_parallel 为True)，则就不需要再进行split。
- 计算：计算就是 
$$
Y1=X1A1
$$
 和 
$$
Y2=X2A2
$$
。经过计算之后，输出的 
$$
Y1,Y2
$$
 的shape就是最终 Y 的shape。每个GPU只有自己对应的分区。
- 输出：
$$
Y1,Y2
$$
 只有合并在一起，才能得到最终输出的 Y。但是因为 
$$
Y1,Y2
$$
 形状相同，都等于Y的形状，所以只要简单矩阵相加即可。

再次，我们使用operator来细化一下：

- 输入：需要对 X 进行纵向切分，这就是一个split操作，得到了 
$$
[X1,X2]
$$
，这两个分区要分别放到两个GPU之上。
- 计算：经过计算之后，每个GPU只有自己对应的分区。
- 输出：因为
$$
Y1,Y2
$$
 需要合并在一起，才能得到最终输出的 Y。这样需要把 
$$
Y1
$$
 和 
$$
Y2
$$
 相加（因为是两个GPU，所以之间还有等待操作），这就是 all-reduce 操作。

我们把这些逻辑点在上图上用红色方框标示，输入 X 先经过 f 来处理，输出 Y 是 g 整合之后的结果。

![](https://img2022.cnblogs.com/blog/1850883/202202/1850883-20220208141422223-225711603.png)

#### 4.3.2 后向传播

我们接下来看看后向传播，对于上图来说，后向传播是从上至下，梯度先经过 g，最后被 f 处理。

反向传播的逻辑如下：

- 目前得到了反向传播上游传过来的梯度 
$$
∂L∂Y
$$
，因为 
$$
Y1,Y2
$$
 的形状相同，所以直接把梯度 
$$
∂L∂Y
$$
传给每个GPU即可，操作是
$$
∂L∂Yi=∂L∂Y(identity)
$$
。这里解释一下，在前向传播时候，XA 的结果需要 all-reduce，可以理解为 sum operator，所以反向传播时候直接拷贝梯度即可。
- 每个GPU之上会进行关于X的梯度计算，于是每个GPU都有一份对X的梯度（但是其内容不一样）。
- 最后需要把各个 GPU 之上关于X的梯度进行聚合，得到完整梯度，就是forward 之中 split 的反向操作，按照最后一列对梯度进行拼接，即all-gather操作。

所以我们在图上用蓝色圆角矩形标示出来后向传播对应的算子。

![](https://img2022.cnblogs.com/blog/1850883/202202/1850883-20220208141429934-2121632349.png)

### 4.4 代码实现

我们接下来看看代码如何实现。

#### 4.4.1 RowParallelLinear

RowParallelLinear 的 forward 代码之中，主要是实施了 f 和 g 的forward操作，同时把 f 和 g 的backward 操作搭建起来，具体如下：

```python
def forward(self, input_):
    # 这里，输入的张量已经被分割到每个GPU，输出张量是all-reduce之后的整体
    # Set up backprop all-reduce.
    if self.input_is_parallel:  # 是否已经是split的输入
        # Transformer's MLP 到达这里，因为已经split，所以直接就接了输入，不会scatter
        input_parallel = input_
    else:
        # 独立 row parallel 线性层到这里，会进行前向切分和后向拼接
        input_parallel = scatter_to_tensor_model_parallel_region(input_)
        
    # Matrix multiply.
    # 比如 X_i 和 A_i 进行乘法操作
    output_parallel = F.linear(input_parallel, self.weight)
    
    # All-reduce across all the partitions.
    # 进行前向all-reduce操作，这样每个GPU之上都是完整的最新结果，同时搭建了后向的identity操作。
    output_ = reduce_from_tensor_model_parallel_region(output_parallel)
    if not self.skip_bias_add:
        # 加上bias
        output = output_ + self.bias if self.bias is not None else output_
        output_bias = None
    else:
        output = output_
        output_bias = self.bias
    return output, output_bias
```

#### 4.4.1 f 操作

scatter\_to\_tensor\_model\_parallel\_region 对应了f操作，其作用是：

- 前向切分split输入，同时搭建后向的 all-gather 操作。
- 后向操作进行 all-gather 操作。

![](https://img2022.cnblogs.com/blog/1850883/202202/1850883-20220208141442860-1280138623.png)

代码为：

```python
def scatter_to_tensor_model_parallel_region(input_):
    return _ScatterToModelParallelRegion.apply(input_)
```

具体 \_ScatterToModelParallelRegion 完成了实际业务，具体 \_split, \_gather 操作在前面都介绍过。

```python
class _ScatterToModelParallelRegion(torch.autograd.Function):
    """Split the input and keep only the corresponding chuck to the rank."""

    @staticmethod
    def symbolic(graph, input_):
        return _split(input_)

    @staticmethod
    def forward(ctx, input_):
        return _split(input_)

    @staticmethod
    def backward(ctx, grad_output):
        return _gather(grad_output)
```

#### 4.4.2 g 操作

reduce\_from\_tensor\_model\_parallel\_region 对应了 g 操作，作用是:

- 前向操作是 all-reduce之后得到最终输出.
- 反向操作则直接拷贝操作。

![](https://img2022.cnblogs.com/blog/1850883/202202/1850883-20220208141451286-1581646926.png)

代码为：

```python
def reduce_from_tensor_model_parallel_region(input_):
    return _ReduceFromModelParallelRegion.apply(input_)
```

具体业务如下：

```python
class _ReduceFromModelParallelRegion(torch.autograd.Function):
    """All-reduce the input from the model parallel region."""

    @staticmethod
    def symbolic(graph, input_):
        return _reduce(input_)
    
    @staticmethod
    def forward(ctx, input_):
        return _reduce(input_) # 前面有介绍

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output #就是indentity 操作，直接把输入拷贝到两个GPU之上
```

## 0x05 Embedding

我们接下来看看 embedding。为了让内存做到均衡配置，对embedding也会按照vocab维度来做shard操作，最终把分区放到多个GPU之上。这样每个卡上都有嵌入表的一部分。

```python
class VocabParallelEmbedding(torch.nn.Module):
    """Embedding parallelized in the vocabulary dimension.

    This is mainly adapted from torch.nn.Embedding and all the default
    values are kept.
    Arguments:
        num_embeddings: vocabulary size.
        embedding_dim: size of hidden state.
        init_method: method to initialize weights.
    """

    def __init__(self, num_embeddings, embedding_dim,
                 init_method=init.xavier_normal_):
        super(VocabParallelEmbedding, self).__init__()
        # Keep the input dimensions.
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        # Set the detauls for compatibility.
        self.padding_idx = None
        self.max_norm = None
        self.norm_type = 2.
        self.scale_grad_by_freq = False
        self.sparse = False
        self._weight = None
        self.tensor_model_parallel_size = get_tensor_model_parallel_world_size()
        # Divide the weight matrix along the vocaburaly dimension.
        self.vocab_start_index, self.vocab_end_index = \ # 得到分区的起始，终止位置
            VocabUtility.vocab_range_from_global_vocab_size(
                self.num_embeddings, get_tensor_model_parallel_rank(),
                self.tensor_model_parallel_size)
        self.num_embeddings_per_partition = self.vocab_end_index - \ # 得到分区内嵌入数目
            self.vocab_start_index

        # Allocate weights and initialize.
        args = get_args()
        if args.use_cpu_initialization:
            self.weight = Parameter(torch.empty(
                self.num_embeddings_per_partition, self.embedding_dim,
                dtype=args.params_dtype))
            _initialize_affine_weight_cpu( # 对权重进行分区
                self.weight, self.num_embeddings, self.embedding_dim,
                self.num_embeddings_per_partition, 0, init_method)
        else:
            self.weight = Parameter(torch.empty(
                self.num_embeddings_per_partition, self.embedding_dim,
                device=torch.cuda.current_device(), dtype=args.params_dtype))
            _initialize_affine_weight_gpu(self.weight, init_method, # 对权重进行分区
                                          partition_dim=0, stride=1)
```

因为每一个GPU只是获得了总体嵌入的一部分，所以对于每个worker来说，可能有一个输入找不到嵌入，因此需要对embedding最终输出做一个 all-reduce操作，这样可以得到完整embedding。

```python
def forward(self, input_):
        if self.tensor_model_parallel_size > 1:
            # Build the mask.
        		# input_mask 意思是单词不在本worker的 embedding 分区范围内，所以设置为0
            input_mask = (input_ < self.vocab_start_index) | \
                         (input_ >= self.vocab_end_index)
            # Mask the input.
            masked_input = input_.clone() - self.vocab_start_index
            masked_input[input_mask] = 0
        else:
            masked_input = input_
            # Get the embeddings.
        output_parallel = F.embedding(masked_input, self.weight,
                                      self.padding_idx, self.max_norm,
                                      self.norm_type, self.scale_grad_by_freq,
                                      self.sparse)
        # Mask the output embedding.
        if self.tensor_model_parallel_size > 1:
            output_parallel[input_mask, :] = 0.0
        # Reduce across all the model parallel GPUs.
        output = reduce_from_tensor_model_parallel_region(output_parallel)
        return output
```

## 0x06 总结

### 6.1 MLP并行

我们总结一下MLP的并行实现，具体如下图，其中逻辑如下：

- 中间灰色的是论文中的概念图。
- 联系代码之后，我们可以知道，其是由一个 ColumnParallelLinear 接上一个 RowParallelLinear 完成的，我们把概念图转化为图左侧两个方框。
- ColumnParallelLinear 是对权重进行列切分，RowParallelLinear 是对权重进行行切分。
- 其中 ColumnParallelLinear 的 
$$
Y1,Y2
$$
 没有经过 all-gather 操作（就是略过了 g 操作），而是直接输入到了 RowParallelLinear 之中，接到了RowParallelLinear 的 
$$
X1,X2
$$
，即，RowParallelLinear 没有 f 操作。
- 概念图之中的 f 就是ColumnParallelLinear 的 f，g 就是 RowParallelLinear 的 g。具体逻辑如图上所示。

![](https://img2022.cnblogs.com/blog/1850883/202202/1850883-20220208141512681-1907431207.png)

### 6.2 共轭函数

论文之中提到了共轭函数。

> f and g are conjugate. f is an identity operator in the forward pass and all reduce in the backward pass while g is an all reduce in the forward pass and identity in the backward pass.

我们前面代码之中也有使用到，我们整理出来如下，其中两两互为共轭函数。

- copy\_to\_tensor\_model\_parallel\_region 是前向操作copy(identity)，后向操作 all-reduce。
- reduce\_from\_tensor\_model\_parallel\_region 是前向操作 all-reduce，后向操作 copy(identity)。

其实，就是MLP之中的 f，g 操作，这两个是共轭函数。

类似，gather\_from\_tensor\_model\_parallel\_region 是前向操作 all-gather，后向操作 scatter，这和scatter\_to\_tensor\_model\_parallel\_region 也是共轭函数。

这些函数代码具体如下：

```python
def copy_to_tensor_model_parallel_region(input_):
    return _CopyToModelParallelRegion.apply(input_)

def reduce_from_tensor_model_parallel_region(input_):
    return _ReduceFromModelParallelRegion.apply(input_)

def scatter_to_tensor_model_parallel_region(input_):
    return _ScatterToModelParallelRegion.apply(input_)

def gather_from_tensor_model_parallel_region(input_):
    return _GatherFromModelParallelRegion.apply(input_)
```

至此，我们已经完成了对模型并行实现的分析，下一篇我们看看在源码之中如何设定各种并行配置。

## 0xFF 参考

[https://developer.nvidia.com/gtc/2020/slides/s21496-megatron-lm-training-multi-billion-parameter-language-models-using-model-parallelism.pdf](https://developer.nvidia.com/gtc/2020/slides/s21496-megatron-lm-training-multi-billion-parameter-language-models-using-model-parallelism.pdf)

[\[细读经典\]Megatron论文和代码详细分析(2)](https://zhuanlan.zhihu.com/p/388830967)

[\[细读经典\]Megatron论文和代码详细分析(1)](https://zhuanlan.zhihu.com/p/366906920)

[Megatron-LM源码阅读（一）](https://zhuanlan.zhihu.com/p/405883984)

[Megatron-LM源码阅读（二）](https://zhuanlan.zhihu.com/p/407094090)

[megatron学习总结](https://zhuanlan.zhihu.com/p/381326200)

[GTC 2020: Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism](https://developer.nvidia.com/gtc/2020/video/s21496)

[大规模训练之 transformer 中的张量模型并行](https://zhuanlan.zhihu.com/p/450689346)