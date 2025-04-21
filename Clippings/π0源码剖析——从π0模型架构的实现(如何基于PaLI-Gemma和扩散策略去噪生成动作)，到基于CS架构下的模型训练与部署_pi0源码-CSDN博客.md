---
title: "π0源码剖析——从π0模型架构的实现(如何基于PaLI-Gemma和扩散策略去噪生成动作)，到基于C/S架构下的模型训练与部署_pi0源码-CSDN博客"
source: "https://blog.csdn.net/v_JULY_v/article/details/146068251"
author:
published:
created: 2025-04-09
description: "文章浏览阅读4.2k次，点赞42次，收藏45次。ChatGPT出来后的两年多，也是疯狂写博的两年多，年初deepseek更引爆了下从曾经15年创业后每年2-6篇的，干到23年30篇、24年65篇，25年前两月18篇，成了我在大模型和具身的原始技术积累如今一转眼已到25年3月初，纪念这两年多，然近期和团队接了好几个大客户订单，使得3月起 不得不全力加速落地，自己也得每天抠paper、搞代码，今年可能没法像去年那样干65篇，不过，我还是争取保持月月更新。_pi0源码"
tags:
  - "clippings"
---
## 前言

ChatGPT出来后的两年多，也是我疯狂写博的两年多(年初deepseek更引爆了下)，比如从创业起步时的15年到后来22年之间 每年2-6篇的，干到了23年30篇、24年65篇、25年前两月18篇，成了我在大模型和具身的原始技术积累

如今一转眼已到25年3月初，时光走得太快，近期和团队接了好几个大客户订单，使得3月起 不得不全力加速落地，自己也得每天抠paper、搞代码

so，为何在明明如此之忙 一天当两天用的情况下，还要继续努力更新博客呢？

原因在于

1. 一方面，我 **确实喜欢分享，因为写博的这10多年下来 确实可以帮到很多、很多人** ，不然本博客也不会有如今如此巨大的访问量与影响力  
	更何况有些文章是之前既定计划中的，在本文之前，上一篇关于π0的文章是π0\_fast《 [π0开源了且推出自回归版π0-FAST——打造机器人动作专用的高效Tokenizer：比扩散π0的训练速度快5倍但效果相当](https://blog.csdn.net/v_JULY_v/article/details/145475733 "π0开源了且推出自回归版π0-FAST——打造机器人动作专用的高效Tokenizer：比扩散π0的训练速度快5倍但效果相当") 》，文中提到，会解读π0的源码「 *至于什么是π0，详见此文《 [π0——用于通用机器人控制的VLA模型：一套框架控制7种机械臂(基于PaliGemma和流匹配的3B模型)](https://blog.csdn.net/v_JULY_v/article/details/143472442 "π0——用于通用机器人控制的VLA模型：一套框架控制7种机械臂(基于PaliGemma和流匹配的3B模型)") 》* 」
2. 二方面，我司「七月在线」在做一系列工厂落地场景的过程中，我们也希望团结到可以和我们一块做的朋友，而若想团结，便需要借助博客 顺带分享我们每个季度在重点做的业务场景

比如过去一周，我把lerobot、reflect vlm、π0的 [仿真环境](https://so.csdn.net/so/search?q=%E4%BB%BF%E7%9C%9F%E7%8E%AF%E5%A2%83&spm=1001.2101.3001.7020) 都在我自己本地电脑上跑了下( *过程中，GitHub copilot这种AI编程工具在环境的安装上帮了我很大的忙——各种环境 只要几句命令，直接帮我装好，真心不错* )

![](https://i-blog.csdnimg.cn/direct/4b0cf4e63f814932a1444d31179b12fb.png)

如此硬着头皮冥思苦想、摸索了好几天，随后使得我自己知道怎么带队完成『太多工厂希望实现的一个生产线任务』了，3月初先仿真训练，2-3个月内部署到真机

> 当然了，也不单纯只是「这几天的想」就能想出来的，这几天之前
> 
> 1. 有把过去一年当三年用的具身技术积累
> 2. 有一年多来，和同事们 如姚博士，以及朋友们许多的讨论
> 3. 有去年十几个工厂对我们的支持与信任
> 
> 我们正在不断壮大队伍
> 
> - 有我司内部同事，亦有我联合带的北理、中南等985的具身研究生，及一块合作开发的朋友，很快会把多个生产线任务并行开发起来
> - 且无论哪个项目，都是不断长期迭代的，故过程中少不了科研层面的突破，欢迎更多伙伴加入我们(全、兼、实习皆可，有意者，敬请私我)，和我们一块开发

话休絮烦，本文便按照如下图所示的源码结构，重点解读一下π的整个源码 「 *π0及π0-FAST的GitHub地址： [github.com/Physical-Intelligence/openpi](http://github.com/Physical-Intelligence/openpi "github.com/Physical-Intelligence/openpi")* 」

1. π0的源码结构非常清晰、可读性高，不愧是成熟的商业化公司，是我司七月的学习榜样之一  
	另，我在解读时，除了尽可能像解读 [iDP3](https://blog.csdn.net/v_JULY_v/article/details/143180794 "iDP3") 那样，比如特意在分析代码文件之前，贴一下对应的代码结构截图——避免只是堆砌代码，我还会尽可能把模块之间、模块内部的函数之间彼此的联系及互相调用的关系 都阐述出来  
	如此，不但 **从宏观上做到一目了然，更从微观上做到抽丝剥茧** ，看到彼此的联系与调用关系
2. 我身边的很多朋友目前都在做π0的微调及二次开发，相信本文无论对我身边的朋友，还是对更多人的学习与工作，都会起到比较大的提升

> ![](https://i-blog.csdnimg.cn/direct/96e6175951b14f6083cc2c2837bbd92c.png)
> 
> ---
> 
> [前言](https://blog.csdn.net/v_JULY_v/article/details/#t0)
> 
> [第一部分 π0模型架构的实现：src下models的全面分析与解读](https://blog.csdn.net/v_JULY_v/article/details/#t1)
> 
> [1.1 models/model.py：核心基础模型的定义](https://blog.csdn.net/v_JULY_v/article/details/#t2)
> 
> [1.2 models/pi0.py的实现](https://blog.csdn.net/v_JULY_v/article/details/#t3)
> 
> [1.2.1 make\_attn\_mask：注意力掩码生成函数](https://blog.csdn.net/v_JULY_v/article/details/#t4)
> 
> [1.2.2 posemb\_sincos：位置编码函数](https://blog.csdn.net/v_JULY_v/article/details/#t5)
> 
> [1.2.3 class Pi0Config：定义动作专家底层结构gemma\_300m，且含inputs\_spec、get\_freeze\_filter(决定对VLM和action expect的哪部分微调，还是都微调)](https://blog.csdn.net/v_JULY_v/article/details/#t6)
> 
> [1.2.3.1 模型配置参数的定义](https://blog.csdn.net/v_JULY_v/article/details/#2.1.3.1%C2%A0%E6%A8%A1%E5%9E%8B%E9%85%8D%E7%BD%AE%E5%8F%82%E6%95%B0%E7%9A%84%E5%AE%9A%E4%B9%89)
> 
> [1.2.3.2 inputs\_spec：定义了π0模型本身接收的输入数据格式编辑](https://blog.csdn.net/v_JULY_v/article/details/#2.1.3.2%C2%A0inputs_spec%EF%BC%9A%E5%AE%9A%E4%B9%89%E4%BA%86%CF%800%E6%A8%A1%E5%9E%8B%E6%9C%AC%E8%BA%AB%E6%8E%A5%E6%94%B6%E7%9A%84%E8%BE%93%E5%85%A5%E6%95%B0%E6%8D%AE%E6%A0%BC%E5%BC%8F%E2%80%8B%E7%BC%96%E8%BE%91)
> 
> [1.2.3.3 get\_freeze\_filter：决定是否LoRA微调——决定微调时只调整动作专家的参数，还是和VLM的参数也调整](https://blog.csdn.net/v_JULY_v/article/details/#2.1.3.3%C2%A0get_freeze_filter%EF%BC%9A%E9%92%88%E5%AF%B9%E6%98%AF%E5%90%A6LoRA%E7%9A%84%E5%A4%84%E7%90%86)
> 
> [1.2.4 class Pi0：含损失函数(训练去噪的准确性)、推理(去噪生成动作)](https://blog.csdn.net/v_JULY_v/article/details/#t7)
> 
> [1.2.4.1 初始化方法 \`\_\_init\_\_\`](https://blog.csdn.net/v_JULY_v/article/details/#2.1.4.1%20%E5%88%9D%E5%A7%8B%E5%8C%96%E6%96%B9%E6%B3%95%20%60__init__%60)
> 
> [1.2.4.2 特征嵌入方法：embed\_prefix(图像和文本输入)、embed\_suffix(状态和动作信息)编辑](https://blog.csdn.net/v_JULY_v/article/details/#2.1.4.2%20%E7%89%B9%E5%BE%81%E5%B5%8C%E5%85%A5%E6%96%B9%E6%B3%95%EF%BC%9Aembed_prefix%28%E5%9B%BE%E5%83%8F%E5%92%8C%E6%96%87%E6%9C%AC%E8%BE%93%E5%85%A5%29%E3%80%81embed_suffix%28%E7%8A%B6%E6%80%81%E5%92%8C%E5%8A%A8%E4%BD%9C%E4%BF%A1%E6%81%AF%29%E2%80%8B%E7%BC%96%E8%BE%91)
> 
> [1.2.4.3 损失函数compute\_loss：训练模型去噪的准确率](https://blog.csdn.net/v_JULY_v/article/details/#2.1.4.3%20%E6%8D%9F%E5%A4%B1%E5%87%BD%E6%95%B0%20%60compute_loss%60)
> 
> [注解 LeRobotDataset：训练数据集的来源(即训练数据集长什么样)](https://blog.csdn.net/v_JULY_v/article/details/#t8)
> 
> [1.2.4.4 推理函数 \`sample\_actions\`：基于扩散模型逆向采样(即去噪)，生成机器人动作序列](https://blog.csdn.net/v_JULY_v/article/details/#2.1.4.4%20%E6%8E%A8%E7%90%86%E5%87%BD%E6%95%B0%20%60sample_actions%60%EF%BC%9A%E5%9F%BA%E4%BA%8E%E6%89%A9%E6%95%A3%E6%A8%A1%E5%9E%8B%E9%80%86%E5%90%91%E9%87%87%E6%A0%B7%EF%BC%8C%E7%94%9F%E6%88%90%E6%9C%BA%E5%99%A8%E4%BA%BA%E5%8A%A8%E4%BD%9C%E5%BA%8F%E5%88%97)
> 
> [1.3 语言模型实现：models/gemma.py](https://blog.csdn.net/v_JULY_v/article/details/#t9)
> 
> [1.4 视觉模型实现：models/siglip.py](https://blog.csdn.net/v_JULY_v/article/details/#t10)
> 
> [1.5 其他支持模块：LoRA、tokenizer、vit的实现](https://blog.csdn.net/v_JULY_v/article/details/#t11)
> 
> [第二部分 模型训练的配置：src下training模块的全面分析与解读](https://blog.csdn.net/v_JULY_v/article/details/#t12)
> 
> [2.1 配置系统 (config.py)](https://blog.csdn.net/v_JULY_v/article/details/#t13)
> 
> [2.1.1 基础配置类AssetsConfig、DataConfig](https://blog.csdn.net/v_JULY_v/article/details/#t14)
> 
> [2.1.2 数据集配置：包含ALOHA、Libero两套数据集](https://blog.csdn.net/v_JULY_v/article/details/#t15)
> 
> [2.1.3 训练配置TrainConfig：模型、数据、优化器等训练参数的设置](https://blog.csdn.net/v_JULY_v/article/details/#t16)
> 
> [2.1.4 预定义配置：基于ALOHA/Libero数据集微调π0——比如完成aloha\_sim\_transfer\_cube\_human](https://blog.csdn.net/v_JULY_v/article/details/#t17)
> 
> [2.2 数据加载系统 data\_loader.py](https://blog.csdn.net/v_JULY_v/article/details/#t18)
> 
> [2.2.1 FakeDataset类](https://blog.csdn.net/v_JULY_v/article/details/#t19)
> 
> [2.2.2 create\_dataset：创建适合训练的数据集](https://blog.csdn.net/v_JULY_v/article/details/#t20)
> 
> [2.2.3 transform\_dataset：对数据集应用转换，比如数据清洗等(创建TransformedDataset实例)](https://blog.csdn.net/v_JULY_v/article/details/#t21)
> 
> [2.2.4 create\_data\_loader：创建用于训练的数据加载器](https://blog.csdn.net/v_JULY_v/article/details/#t22)
> 
> [2.3 优化器系统 (optimizer.py)](https://blog.csdn.net/v_JULY_v/article/details/#t23)
> 
> [2.4 检查点系统 (checkpoints.py)](https://blog.csdn.net/v_JULY_v/article/details/#t24)
> 
> [2.5 模型分片系统(sharding.py)：含FSDP的实现](https://blog.csdn.net/v_JULY_v/article/details/#t25)
> 
> [2.6 权重加载系统 (weight\_loaders.py)](https://blog.csdn.net/v_JULY_v/article/details/#t26)
> 
> [2.7 辅助工具(utils.py)](https://blog.csdn.net/v_JULY_v/article/details/#t27)
> 
> [第三部分 模型的训练与部署：基于客户端-服务器C/S架构](https://blog.csdn.net/v_JULY_v/article/details/#t28)
> 
> [3.1 packages/openpi-client：帮真机或Sim与策略服务器进行通信和交互](https://blog.csdn.net/v_JULY_v/article/details/#t29)
> 
> [3.1.1 核心接口层](https://blog.csdn.net/v_JULY_v/article/details/#t30)
> 
> [3.1.2 通信层](https://blog.csdn.net/v_JULY_v/article/details/#t31)
> 
> [3.1.3 数据处理层](https://blog.csdn.net/v_JULY_v/article/details/#t32)
> 
> [3.1.4 运行时系统层](https://blog.csdn.net/v_JULY_v/article/details/#t33)
> 
> [3.1.5 工具支持](https://blog.csdn.net/v_JULY_v/article/details/#t34)
> 
> [3.2 scripts(策略服务器)：包含数据处理、模型训练、模型推理的多个脚本](https://blog.csdn.net/v_JULY_v/article/details/#t35)
> 
> [3.2.1 \_\_init\_\_.py](https://blog.csdn.net/v_JULY_v/article/details/#t36)
> 
> [3.2.2 compute\_norm\_stats.py：计算数据的归一化统计信息](https://blog.csdn.net/v_JULY_v/article/details/#t37)
> 
> [3.2.3 serve\_policy.py：启动策略服务，用于模型推理](https://blog.csdn.net/v_JULY_v/article/details/#t38)
> 
> [3.2.4 train\_test.py：训练和测试模型](https://blog.csdn.net/v_JULY_v/article/details/#t39)
> 
> [3.2.5 train.py：训练模型——损失函数计算、梯度下降、参数更新](https://blog.csdn.net/v_JULY_v/article/details/#t40)
> 
> [3.2.6 scripts/docker](https://blog.csdn.net/v_JULY_v/article/details/#t41)
> 
> [第四部分 策略适配接口：src下policy的全面分析与解读](https://blog.csdn.net/v_JULY_v/article/details/#t42)
> 
> [4.1 policy.py：实现了Policy类和 PolicyRecorder类](https://blog.csdn.net/v_JULY_v/article/details/#t43)
> 
> [4.1.1 Policy 类](https://blog.csdn.net/v_JULY_v/article/details/#t44)
> 
> [4.1.2 \`PolicyRecorder\`](https://blog.csdn.net/v_JULY_v/article/details/#t45)
> 
> [4.2 policy\_config.py](https://blog.csdn.net/v_JULY_v/article/details/#t46)
> 
> [4.2.1 PolicyConfig 数据类](https://blog.csdn.net/v_JULY_v/article/details/#t47)
> 
> [4.2.2 create\_trained\_policy 函数](https://blog.csdn.net/v_JULY_v/article/details/#t48)
> 
> [4.3 policies/aloha\_policy.py](https://blog.csdn.net/v_JULY_v/article/details/#t49)
> 
> [4.3.1 make\_aloha\_example：输入示例——状态向量、图像数据、文本prompt](https://blog.csdn.net/v_JULY_v/article/details/#t50)
> 
> [4.3.2 AlohaInputs：定义Aloha 策略的输入数据结构](https://blog.csdn.net/v_JULY_v/article/details/#t51)
> 
> [4.3.3 AlohaOutputs：定义Aloha 策略的输出数据结构](https://blog.csdn.net/v_JULY_v/article/details/#t52)
> 
> [4.3.4 多个辅助函数：数据的标准化、反标准化、关节角度翻转](https://blog.csdn.net/v_JULY_v/article/details/#t53)
> 
> [第五部分 examples ：各种机器人平台及策略客户端的示例实现](https://blog.csdn.net/v_JULY_v/article/details/#t54)

## 第一部分 π0模型架构的实现：src下models的全面分析与解读

接下来，我们来看核心src下的各个模块

![](https://i-blog.csdnimg.cn/direct/9e22f9f36fef4185a795b59e65f1bb83.png)

首先是其中的src/openpi/models

![](https://i-blog.csdnimg.cn/direct/f32c4eb312e846e98721b1b659a5284e.png)

### 1.1 models/model.py：核心基础模型的定义

这是模型框架的核心文件，定义了基础的抽象类和数据结构：

1. \`BaseModelConfig\`: 所有模型配置的抽象基类
2. \`BaseModel\`: 所有模型实现的抽象基类
3. \`Observation\`: 保存模型输入的数据类
4. \`Actions\`: 定义动作数据格式
5. 提供了通用功能如\`preprocess\_observation\`和\`restore\_params\`

// 待更

### 1.2 models/pi0.py的实现

Pi0是一个多模态扩散模型：继承自\`BaseModel\`，使用SigLIP处理视觉输入、使用Gemma处理语言输入，实现了基于扩散的动作生成系统，且包含\`compute\_loss\`和\`sample\_actions\`方法的实现

总之，Pi0结合了多模态输入(图像和文本)来生成机器人动作序列。下面是对代码的详细解析：

#### 1.2.1 make\_attn\_mask：注意力掩码生成函数

这个函数生成transformer中使用的注意力掩码，控制 token 之间的注意力流动方式

```python
def make_attn_mask(input_mask, mask_ar):

    """

    从big_vision项目改编的注意力掩码生成函数

    

    Token可以关注那些累积mask_ar小于等于自己的有效输入token。

    这样\`mask_ar\` bool[?B, N]可用于设置几种类型的注意力，例如：

    

      [[1 1 1 1 1 1]]: 纯因果注意力。

    

      [[0 0 0 1 1 1]]: 前缀语言模型注意力。前3个token之间可以互相关注，

                      后3个token有因果注意力。第一个条目也可以是1，不改变行为。

    

      [[1 0 1 0 1 0 0 1 0 0]]: 4个块之间的因果注意力。一个块的token可以

                              关注所有之前的块和同一块内的所有token。

    

    参数:

      input_mask: bool[B, N] 如果是输入的一部分则为true，如果是填充则为false

      mask_ar: bool[?B, N] 如果前面的token不能依赖于它则为true，

               如果它共享与前一个token相同的注意力掩码则为false

    """

 

    # 将mask_ar广播到与input_mask相同的形状

    mask_ar = jnp.broadcast_to(mask_ar, input_mask.shape)  

 

    # 计算mask_ar在序列维度上的累积和

    cumsum = jnp.cumsum(mask_ar, axis=1)  

 

    # 创建注意力掩码：当目标位置的累积值<=查询位置的累积值时，允许注意力流动

    attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]  

 

    # 创建有效掩码：只有有效的输入位置之间才能有注意力

    valid_mask = input_mask[:, None, :] * input_mask[:, :, None]  

 

    # 结合注意力掩码和有效掩码

    return jnp.logical_and(attn_mask, valid_mask)
```

它支持多种注意力模式：

1. 纯因果注意力（每个 token 只能关注自己和之前的 token）
2. 前缀语言模型注意力（允许前缀内部自由注意，后缀部分使用因果注意力）
3. 块状因果注意力（在块内自由注意，块之间是因果的）

#### 1.2.2 posemb\_sincos：位置编码函数

使用正弦余弦函数实现位置编码

```python
def posemb_sincos(

    pos: at.Real[at.Array, Any], embedding_dim: int, min_period: float, max_period: float

) -> at.Float[at.Array, f"b {embedding_dim}"]:

    """计算标量位置的正弦余弦位置嵌入向量"""

    if embedding_dim % 2 != 0:      # 检查嵌入维度是否为偶数

        raise ValueError(f"embedding_dim ({embedding_dim}) must be divisible by 2")

 

    fraction = jnp.linspace(0.0, 1.0, embedding_dim // 2)  # 创建均匀分布的分数值

    period = min_period * (max_period / min_period) ** fraction  # 计算周期值，对数空间中均匀分布

    sinusoid_input = jnp.einsum(

        "i,j->ij",

        pos,

        1.0 / period * 2 * jnp.pi,                      # 计算角频率

        precision=jax.lax.Precision.HIGHEST,            # 使用最高精度进行计算

    )

 

    # 连接sin和cos值，形成完整的位置编码

    return jnp.concatenate([jnp.sin(sinusoid_input), jnp.cos(sinusoid_input)], axis=-1)
```

#### 1.2.3 class Pi0Config：定义动作专家底层结构gemma\_300m，且含inputs\_spec、get\_freeze\_filter(决定对VLM和action expect的哪部分微调，还是都微调)

##### 1.2.3.1 模型配置参数的定义

**首先** ，这个类定义了模型的配置参数 ，比如PaLI-Gemma 变体：\`gemma\_2b，尤其值得注意的是 在本π0的官方实现中，动作专家的底层结构用的 **300M大小的gemma模型变体**

```cobol
class Pi0Config(_model.BaseModelConfig):

    dtype: str = "bfloat16"  # 设置数据类型为bfloat16

    paligemma_variant: _gemma.Variant = "gemma_2b"          # 设置PaLI-Gemma变体为2B参数版本

    action_expert_variant: _gemma.Variant = "gemma_300m"    # 设置动作专家为gemma的300M变体版本

 

    # 设置模型特定的默认值

    action_dim: int = 32          # 设置动作维度为32

    action_horizon: int = 50      # 设置动作序列长度为50步

    max_token_len: int = 48       # 设置最大token长度为48
```

##### 1.2.3.2 inputs\_spec：定义了π0模型本身接收的输入数据格式

**其次** ，通过inputs\_spec函数定义了π0模型本身接收的输入数据格式， 函数采用关键字参数 \`batch\_size\`（默认为1），返回一个包含观察规格和动作规格的元组

```cobol
def inputs_spec(self, *, batch_size: int = 1) -> Tuple[Type[_model.Observation], Type[_model.Actions]]
```
1. 其支持多种输入，比如  
	视觉输入(三个不同视角的RGB图像) 、 语言输入(分词后的文本prompt) 、 状态输入(当前机器人状态)
2. 输出上  
	**则是一个时序动作序列(包含50个连续的动作向量，每个动作向量有32个维度，可能对应关节角度或其他控制信号)**

具体而言该函数进行如下4个操作  
**一、创建图像规格**

```cobol
image_spec = jax.ShapeDtypeStruct([batch_size, *_model.IMAGE_RESOLUTION, 3], jnp.float32)
```

其中的

1. \`\[batch\_size, \*\_model.IMAGE\_RESOLUTION, 3\]\` 定义了图像张量的形状：比如  
	批次大小  
	图像分辨率（ *从 \`\_model.IMAGE\_RESOLUTION\` 获取，可能是如 \[224, 224\] 这样的值* ）  
	3 个颜色通道 (RGB)
2. \`jnp.float32\` 指定了数据类型为 32 位浮点数

**二、创建图像掩码规格**

```cobol
image_mask_spec = jax.ShapeDtypeStruct([batch_size], jnp.bool_)
```

其定义了图像掩码规格，每个批次中的每个图像都有一个布尔值，这个掩码用于指示哪些图像是有效的（\`True\`）或无效的（\`False\`）

**三、创建观察规格 ：包含视觉输入、机器人状态、指令输入**  
\`at.disable\_typechecking()\` 临时禁用类型检查，可能是因为这里创建的是类型规格而不是实际的数据，且观察规格包含多个组件：

1. 多视角图像  
	base\_0\_rgb: 机器人底座/身体视角的RGB图像  
	left\_wrist\_0\_rgb: 左手腕视角的RGB图像  
	right\_wrist\_0\_rgb: 右手腕视角的RGB图像
	```cobol
	with at.disable_typechecking():
	            observation_spec = _model.Observation(
	                images={
	                    "base_0_rgb": image_spec,
	                    "left_wrist_0_rgb": image_spec,
	                    "right_wrist_0_rgb": image_spec,
	                },
	```
2. 图像掩码  
	对应每个视角图像的有效性掩码
3. 机器人状态：  
	形状为 \`\[batch\_size, self.action\_dim\]\` 的浮点数张量，其中的\`self.action\_dim\` 默认为32，表示状态向量的维度
	```cobol
	state=jax.ShapeDtypeStruct([batch_size, self.action_dim], jnp.float32),
	```
4. 分词后的文本prompt  
	形状为 \`\[batch\_size, self.max\_token\_len\]\` 的整数张量  
	\`self.max\_token\_len\` 默认为48，表示最大token数量  
	数据类型为 \`jnp.int32\`，表示token ID
5. 提示掩码  
	与分词提示相同形状的布尔张量，用于指示哪些位置有有效的token
	```cobol
	state=jax.ShapeDtypeStruct([batch_size, self.action_dim], jnp.float32),
	                tokenized_prompt=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.int32),
	                tokenized_prompt_mask=jax.ShapeDtypeStruct([batch_size, self.max_token_len], bool),
	            )
	```

**四、创建动作规格**

```cobol
action_spec = jax.ShapeDtypeStruct([batch_size, self.action_horizon, self.action_dim], jnp.float32)
```

其定义了动作数据的形状和类型：

- \`batch\_size\`: 批次大小
- \`self.action\_horizon\`: 动作序列长度，默认为50
- \`self.action\_dim\`: 每个动作的维度，默认为32
- \`jnp.float32\` 指定了数据类型为32位浮点数

然后返回

```kotlin
return observation_spec, action_spec
```

##### 1.2.3.3 get\_freeze\_filter：决定是否LoRA微调——决定微调时只调整动作专家的参数，还是和VLM的参数也调整

**此外** ，该配置类还实现了get\_freeze\_filter这个函数 ， 作用是如果选择LoRA微调(冻结原始预训练模型的参数，只更新新添加的低秩适应层参数)，则需要对模型中的某些参数做冻结

三种可能的情况：

1. 只对 PaLI-Gemma 使用 LoRA：冻结 Gemma 参数（但排除动作专家参数）
2. 只对动作专家使用 LoRA：冻结动作专家参数
3. 对两者都使用 LoRA：冻结两者的基础参数

如此，可以选择性地微调模型的特定部分(语言部分或动作预测部分）

具体而言

1. 首先，定义函数
	```ruby
	def get_freeze_filter(self) -> nnx.filterlib.Filter:
	        """返回基于模型配置的冻结过滤器"""
	```
2. 其次，初始化变量
	```python
	filters = []      # 初始化过滤器列表
	        has_lora = False  # 初始化LoRA标志
	```
3. 接着，创建参数过滤器
	```csharp
	# 匹配所有LLM参数的正则表达式，用于选择 Gemma 语言模型的参数
	        gemma_params_filter = nnx_utils.PathRegex(".*llm.*")  
	 
	        # 匹配动作专家参数的正则表达式
	        action_expert_params_filter = nnx_utils.PathRegex(".*llm.*_1.*")
	```
4. 接下来是对PaLI-Gemma变体的处理
	```python
	# 如果PaLI-Gemma使用LoRA
	        if "lora" in self.paligemma_variant:
	            filters.append(
	                gemma_params_filter,  # 添加Gemma参数过滤器
	            )
	            if "lora" not in self.action_expert_variant:
	                # 如果只冻结Gemma参数，排除动作专家参数
	                filters.append(
	                    nnx.Not(action_expert_params_filter),
	                )
	            has_lora = True
	```
5. 再下来是对动作专家变体的处理
	```python
	elif "lora" in self.action_expert_variant:
	            # 如果动作专家使用LoRA
	            filters.append(
	                action_expert_params_filter,
	            )
	            has_lora = True
	```

> 值得注意的是，也是我之前看到这里思考过的一个问题，即在训练 π0 的动作预测能力时
> 
> 1. 默认会同时调整 VLM 和动作专家的参数
> 2. 如果需要只调整动作专家的参数，可以通过修改 \`get\_freeze\_filter\` 方法来冻结 VLM 的参数  
> 	证据之一是这个函数 **get\_freeze\_filter** 的最后是这样子的
> 	```kotlin
> 	if has_lora:
> 	            # If any lora is used, exclude all lora params.
> 	            filters.append(
> 	                nnx.Not(nnx_utils.PathRegex(".*lora.*")),
> 	            )
> 	 
> 	        if not filters:
> 	            # 关键行：如果没有指定过滤器，返回nnx.Nothing，表示不冻结任何参数
> 	            return nnx.Nothing  
> 	        return nnx.All(*filters)
> 	```
> 	*此外，下文的《1.2.4.3 损失函数compute\_loss：训练模型去噪的准确率》一节的最后还会再次详细分析这个问题*

#### 1.2.4 class Pi0：含损失函数(训练去噪的准确性)、推理(去噪生成动作)

核心模型类，继承自 \`\_model.BaseModel\`，实现了：

1. 多模态输入处理  
	处理多视角图像（基础视角、左手腕视角、右手腕视角）  
	处理文本提示（如指令）  
	处理机器人当前状态
2. 扩散过程  
	训练时：将干净动作添加噪声，让模型学习去噪  
	推理时：从纯噪声开始，逐步降噪生成动作序列
3. 注意力机制  
	使用精心设计的注意力掩码控制信息流动  
	前缀（图像和文本）内部使用全注意力  
	后缀（状态和动作）使用特殊的注意力模式

##### 1.2.4.1 初始化方法 \`\_\_init\_\_\`

```haskell
class Pi0(_model.BaseModel):

    def __init__(self, config: Pi0Config, rngs: nnx.Rngs):

        # 初始化基类

        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)

        

        # 获取PaLI-Gemma和动作专家配置

        paligemma_config = _gemma.get_config(config.paligemma_variant)

        action_expert_config = _gemma.get_config(config.action_expert_variant)
```

其组合了多个核心组件：

一个是PaLI-Gemma 模型 ：结合了 Gemma 语言模型和 SigLIP 视觉模型

![](https://i-blog.csdnimg.cn/direct/df385da6f9df42049101ee5d8cfbe2de.png)

1. 先是对语言模型的初始化
	```cobol
	# 创建并初始化语言模型
	        # TODO: 用NNX重写Gemma，目前使用桥接
	        llm = nnx_bridge.ToNNX(
	            _gemma.Module(
	                configs=[paligemma_config, action_expert_config],  # 配置两个Gemma模型
	                embed_dtype=config.dtype,          # 设置嵌入数据类型
	            )
	        )
	        llm.lazy_init(rngs=rngs, method="init")    # 延迟初始化LLM
	```
2. 然后是对视觉模型的初始化
	```cobol
	# 创建并初始化图像模型
	        img = nnx_bridge.ToNNX(
	            _siglip.Module(
	                num_classes=paligemma_config.width,  # 设置图像特征维度与语言模型宽度相匹配
	                variant="So400m/14",  # 使用400M参数SigLIP模型
	                pool_type="none",  # 不使用池化，保留所有图像token
	                scan=True,  # 启用扫描优化
	                dtype_mm=config.dtype,  # 设置矩阵乘法数据类型
	            )
	        )
	 
	        # 使用假观察中的图像初始化图像模型
	        img.lazy_init(next(iter(config.fake_obs().images.values())), train=False, rngs=rngs)
	```
3. 最后，把语言模型和视觉模型组合成PaLI-Gemma多模态模型
	```cobol
	# 组合LLM和图像模型为PaLI-Gemma多模态模型
	        self.PaliGemma = nnx.Dict(llm=llm, img=img)
	```

另一个是线性投影层 ：用于

1. 状态投影
	```python
	# 状态投影层：将机器人状态投影到模型维度
	self.state_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
	```
2. 动作投影
	```python
	# 动作输入投影层：将动作投影到模型维度
	        self.action_in_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
	```
3. 时间-动作混合等
	```python
	# 动作-时间MLP输入层：将连接的动作和时间特征投影到模型维度
	self.action_time_mlp_in = nnx.Linear(2 * action_expert_config.width, action_expert_config.width, rngs=rngs)
	 
	# 动作-时间MLP输出层
	self.action_time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
	 
	# 动作输出投影层：将模型输出投影回动作维度
	self.action_out_proj = nnx.Linear(action_expert_config.width, config.action_dim, rngs=rngs)
	```

##### 1.2.4.2 特征嵌入方法：embed\_prefix(图像和文本输入)、embed\_suffix(状态和动作信息)

- \`embed\_prefix\`：处理图像和文本输入( *图像通过SigLip模型编码，文本通过Gemma LLM编码* )，创建前缀 token，皆为双向注意力，用ar\_mask = false表示
- \`embed\_suffix\`：处理机器人状态信息、噪声化的动作信息( *状态和噪声动作经过线性投影和MLP处理* )，创建后缀 token  
	其中  
	状态为单个token，和第一个动作token均设置为 单向注意力，用ar\_mask = true表示  
	其余动作tokens之间设置为双向注意力，用ar\_mask = false表示

**首先，对于前者embed\_prefix**

```cobol
def embed_prefix(

        self, obs: _model.Observation

    ) -> Tuple[at.Float[at.Array, Any], at.Bool[at.Array, Any], at.Bool[at.Array, Any]]:

        """嵌入前缀部分（图像和文本）"""

        input_mask = []           # 初始化输入掩码列表

        ar_mask = []              # 初始化自回归掩码列表

        tokens = []               # 初始化token列表
```

其工作流程为

1. **图像处理：说白了，就是把图像token化**  
	使用SigLip视觉模型处理每个图像，生成图像tokens序列
	```python
	# 嵌入图像
	        for name in obs.images:
	            # 通过图像模型获取图像token
	            image_tokens, _ = self.PaliGemma.img(obs.images[name], train=False)
	 
	            tokens.append(image_tokens)      # 添加图像token
	```
2. 图像掩码扩展  
	将图像掩码扩展到与图像tokens相同的序列长度，使用einops.repeat进行形状变换，这些掩码会 **指示哪些图像是有效的，而哪些是填充的**
	```csharp
	# 重复图像掩码以匹配token维度
	            input_mask.append(
	                einops.repeat(
	                    obs.image_masks[name],
	                    "b -> b s",               # 调整形状：批次维度保持不变，添加序列维度
	                    s=image_tokens.shape[1],  # 序列长度等于图像token数
	                )
	            )
	```
3. 自回归掩码设置  
	设置图像tokens之间的注意力为双向(False表示双向注意力)，原因在于图像内容通常是非时序性的数据
	```cobol
	# 图像token之间互相关注（非自回归）
	            ar_mask += [False] * image_tokens.shape[1]
	```
4. 文本处理  
	使用LLM模型对文本输入tokenized\_inputs进行嵌入
	```python
	# 添加语言（即分词后的输入）
	        if obs.tokenized_prompt is not None:
	            # 通过语言模型嵌入分词后的提示
	            tokenized_inputs = self.PaliGemma.llm(obs.tokenized_prompt, method="embed")
	            tokens.append(tokenized_inputs)                  # 添加文本token
	            input_mask.append(obs.tokenized_prompt_mask)     # 添加提示掩码
	```
	且同样设置为双向注意力，相当于 **语言token可以关注图像token，图像token反过来亦可关注语言token，最终实现多模态融合**
	```cobol
	# 图像和语言输入之间完全关注（非自回归）
	            ar_mask += [False] * tokenized_inputs.shape[1]
	```
5. 最后，连接所有token和掩码，其中包含了  
	多模态信息的融合表示tokens——图像token和语言token  
	以及指示哪些token是有效信息的input\_mask  
	和如何在这些token之间进行注意力计算规则的ar\_mask——相当于控制信息流动的方向
	```cobol
	# 连接所有token和掩码
	        tokens = jnp.concatenate(tokens, axis=1)    # 在序列维度上连接token
	        input_mask = jnp.concatenate(input_mask, axis=1)  # 在序列维度上连接输入掩码
	        ar_mask = jnp.array(ar_mask)                # 转换自回归掩码为数组
	        
	        return tokens, input_mask, ar_mask          # 返回token、输入掩码和自回归掩码
	```

> 顺便，再回顾下此图 ![](https://i-blog.csdnimg.cn/direct/934a260f9c7f480f9cda7804608e53a0.png)

**其次，对于后者embed\_suffix**

定义如下，其参数包括obs(一般包含图像和机器人状态)、noisy\_actions、timestep

```cobol
def embed_suffix(

        self, obs: _model.Observation, noisy_actions: _model.Actions, timestep: at.Float[at.Array, Any]

    ) -> Tuple[at.Float[at.Array, Any], at.Bool[at.Array, Any], at.Bool[at.Array, Any]]:

        """嵌入后缀部分（状态和动作）"""

        input_mask = []           # 初始化输入掩码列表

        ar_mask = []              # 初始化自回归掩码列表

        tokens = []               # 初始化token列表
```

其工作流程为

1. 状态处理  
	将状态信息投影到embedding空间
	```cobol
	# 添加单个状态token
	        state_token = self.state_proj(obs.state)[:, None, :]  # 投影状态并添加序列维度
	        tokens.append(state_token)                            # 添加状态token
	 
	        # 添加状态掩码（全为1），表示这个状态token是有效的
	        input_mask.append(jnp.ones((obs.state.shape[0], 1), dtype=jnp.bool_))
	```
	并设置为单向注意力(True)，表明图像和语言输入不能关注状态信息，因为image/language do not attend to state or actions
	```cobol
	# 图像/语言输入不关注状态或动作（自回归）
	        ar_mask += [True]
	```
2. 时间步嵌入，使用正弦-余弦位置编码生成时间步嵌入
	```cobol
	# 使用正弦余弦位置编码嵌入时间步，敏感度范围为[0, 1]
	        time_emb = posemb_sincos(timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0)
	```
3. 动作和时间信息融合，比如通过action\_time\_tokens连接：「带噪声的动作」和「时间token」
	```cobol
	# 混合时间步 + 动作信息，使用MLP
	        action_tokens = self.action_in_proj(noisy_actions)  # 投影带噪声的动作
	 
	        # 重复时间嵌入以匹配动作序列长度
	        time_tokens = einops.repeat(time_emb, "b emb -> b s emb", s=self.action_horizon)
	 
	        # 连接动作和时间token
	        action_time_tokens = jnp.concatenate([action_tokens, time_tokens], axis=-1)
	```
4. MLP处理 ![](https://i-blog.csdnimg.cn/direct/934a260f9c7f480f9cda7804608e53a0.png)  
	使用两层MLP和swish激活函数对「动作和时间的组合表示」进行非线性变换，以进一步融合：(噪声)动作和时间信息
	```cobol
	# 通过MLP处理
	        action_time_tokens = self.action_time_mlp_in(action_time_tokens)   # 输入层
	        action_time_tokens = nnx.swish(action_time_tokens)                 # Swish激活函数
	        action_time_tokens = self.action_time_mlp_out(action_time_tokens)  # 输出层
	```
5. 注意力掩码设置  
	第一个动作token设置为单向注意力「 *上面说过了的， 单向注意力，用ar\_mask = true表示* 」，其余动作tokens之间设置为双向注意力
	```cobol
	# 添加动作时间token
	        tokens.append(action_time_tokens)
	 
	        # 添加掩码（全为1），表示所有动作token都是有效的
	        input_mask.append(jnp.ones(action_time_tokens.shape[:2], dtype=jnp.bool_))  
	 
	        # 图像/语言/状态输入不关注动作token（动作第一个是自回归的——单向，其余不是——双向）
	        ar_mask += [True] + ([False] * (self.action_horizon - 1))
	```
6. 最后连接所有token和掩码
	```cobol
	# 连接所有token和掩码
	        tokens = jnp.concatenate(tokens, axis=1)          # 在序列维度上连接token
	        input_mask = jnp.concatenate(input_mask, axis=1)  # 在序列维度上连接输入掩码
	        ar_mask = jnp.array(ar_mask)        # 转换自回归掩码为数组
	        
	        return tokens, input_mask, ar_mask  # 返回token、输入掩码和自回归掩码
	```

##### 1.2.4.3 损失函数compute\_loss：训练模型去噪的准确率

**总的来讲**

1. 训练的时候，对其中的「原始动作action」数据加噪，最后去预测所添加的真实噪声，预测噪声的结果为，然后计算预测噪声与真实噪声之间的均方误差  
	***也就是说，训练时的本质 其实是为了让模型具备生成真正想要动作的能力，以确保在推理时，能得到真正想要动作的能力***  
	  
	*那可能有同学疑问了，既然通过对* *原始动作* *加* *噪* *，然后* *预测噪声* *，最后* ***噪声动作*** *减掉预测* *噪声* *便是所预测的* *原始动作* *，那为何不对比实际的原始动作，与所预测的原始动作 是否一致呢*  
	*其实我之前在此文《 [图像生成发展起源：从VAE、扩散模型DDPM、DDIM到DETR、ViT、Swin transformer](https://blog.csdn.net/v_JULY_v/article/details/130361959 "图像生成发展起源：从VAE、扩散模型DDPM、DDIM到DETR、ViT、Swin transformer") 》中的「2.1.1 从扩散模型概念的提出到DDPM(含U-Net网络的简介)、DDIM」已经讲了，原因在于  
	1 对噪声的预测，比对动作的预测更容易，一者 预测噪声收敛更稳定，二者 噪声通常是标准化的，比如高斯噪声的均值为0 方差为1，使得模型预测噪声时不需要适应不同尺度的输出  
	2* *\-prediction 和 -prediction其实理论上也是等价的，毕竟 +* *\=*
2. 如此，便可以在推理的时候，针对一个随机生成的纯噪声，基于observation(包含图像和机器人状态)，逐步去噪生成机器人的动作序列

具体而言， compute\_loss 实现了扩散模型的训练损失计算

1. 对输入观察进行预处理，其中  
	preprocess\_rng用于观察预处理(比如图像增强等)  
	noise\_rng用于生成噪声  
	time\_rng用于从beta分布采样时间步
	```cobol
	def compute_loss(
	        self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions, *, train: bool = False
	    ) -> at.Float[at.Array, Any]:
	        """计算扩散模型的损失函数"""
	        # 分割随机数生成器为三部分，用于不同的随机操作
	        preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
	```
2. 生成随机噪声并采样时间点 t
	```cobol
	# 获取动作的批次形状
	        batch_shape = actions.shape[:-2]
	 
	        # 生成与动作相同形状的高斯噪声
	        noise = jax.random.normal(noise_rng, actions.shape)
	 
	        # 从Beta分布采样时间点，范围为[0.001, 1]，Beta(1.5, 1)偏向较低的值
	        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
	 
	        # 扩展时间维度以匹配动作形状
	        time_expanded = time[..., None, None]
	```
3. 创建带噪动作序列 x\_t，相当于 **x\_t是噪声化的动作** ，随着时间从0到1， 原始动作 逐渐添加 **真实噪声** ，变为 **纯噪声**  
	**而 代表所加的真实噪声** ，便是咱们所要预测噪声的ground truth  
	故 **所添加的噪声** 即 = **加满噪声的动作** - 原始动作
	```cobol
	# 创建带噪声的动作：t * noise + (1-t) * actions
	        x_t = time_expanded * noise + (1 - time_expanded) * actions
	 
	        # 计算真实噪声减去动作的差异，这是模型需要预测的目标
	        u_t = noise - actions
	```
4. 嵌入前缀和后缀
	```cobol
	# 一次性前向传递前缀+后缀
	        # 嵌入前缀（图像和文本）
	        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
	 
	        # 嵌入后缀（状态和带噪声的动作）
	        suffix_tokens, suffix_mask, suffix_ar_mask = self.embed_suffix(observation, x_t, time)
	```
5. 构建注意力掩码和位置编码  
	根据下图  
	![](https://i-blog.csdnimg.cn/direct/934a260f9c7f480f9cda7804608e53a0.png)  
	可得
	```cobol
	# 连接掩码：通过链接前缀和后缀的掩码，从而创建完整的输入掩码
	        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
	        ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
	 
	        # 创建注意力掩码make_attn_mask，从而控制不同token之间的可见性
	        attn_mask = make_attn_mask(input_mask, ar_mask)
	 
	        # 计算位置编码
	        positions = jnp.cumsum(input_mask, axis=1) - 1
	```
6. 模型前向传播，即 **调用PaliGemma进行推理，处理前缀和后缀token**  
	当然了，输出中我们只关注与后缀相关的部分，因为其中包含了我们想要的动作预测的部分
	```cobol
	# 通过PaLI-Gemma模型处理token
	        _, suffix_out = self.PaliGemma.llm(
	            [prefix_tokens, suffix_tokens], mask=attn_mask, positions=positions
	        )
	```
7. 预测噪声
	```php
	# 将模型输出投影回动作空间
	        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])
	```
8. 计算预测噪声与实际噪声间的均方误差
	```cobol
	# 返回预测噪声和真实噪声之间的均方误差
	        return jnp.mean(jnp.square(v_t - u_t), axis=-1)
	```

> 在上文此节《 1.2.3.3 get\_freeze\_filter：决定是否LoRA——决定训练时只调整动作专家的参数，还是和VLM的参数也调整 》的最后，我们分析过：在训练 π0 的动作预测能力时
> 
> 1. 默认会同时调整 VLM 和动作专家的参数
> 2. 如果需要只调整动作专家的参数，可以通过修改 \`get\_freeze\_filter\` 方法来冻结 VLM 的参数
> 
> 那可能还是有同学有疑问，可否进一步分析、可否给更多证据？下面，我就给大家更多证据，即从以下几段代码可以看出，默认情况下会同时调整 VLM 和动作专家的参数：
> 
> 1. \`compute\_loss\` 方法中的前向传播逻辑（\`pi0.py\` 文件）：
> 	```cobol
> 	# 使用VLM生成表示
> 	    (prefix_out, suffix_out), _ = self.PaliGemma.llm(  
> 	        [prefix_tokens, suffix_tokens], mask=attn_mask, positions=positions
> 	    )
> 	 
> 	    # 使用动作专家生成动作预测
> 	    v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])
> 	```
> 	\- 这里调用了 \`self.PaliGemma.llm\`，即 VLM 的部分， *用于处理前缀prefix\_tokens(来自embed\_prefix)，和后缀的嵌入suffix\_tokens(来自embed\_suffix)*  
> 	\- 同时，\`self.action\_out\_proj\` 是动作专家的输出层，用于生成动作预测值  
> 	\- 这 表明 VLM 和动作专家的参数都参与了前向传播
> 2. \`get\_freeze\_filter\` 方法的默认返回值（\`pi0.py\` 文件）：
> 	```kotlin
> 	if not filters:
> 	    return nnx.Nothing
> 	```
> 	\- 如果没有指定 \`lora\` 相关的配置，\`filters\` 列表为空，\`get\_freeze\_filter\` 返回 \`nnx.Nothing\`  
> 	\- 这意味着没有任何参数被冻结，所有参数（包括 VLM 和动作专家）都会被调整
> 3. \`PaliGemma\` 的初始化（\`pi0.py\` 文件）：
> 	```cobol
> 	self.PaliGemma = nnx.Dict(llm=llm, img=img)
> 	```
> 	\- \`PaliGemma\` 包含了 VLM（\`llm\`）和图像处理模块（\`img\`）  
> 	\- 在训练过程中，这些模块的参数默认不会被冻结
> 
> 综上所述，代码中没有明确冻结 VLM 的参数，因此默认情况下会同时调整 VLM 和动作专家的参数

#### 注解 LeRobotDataset：训练数据集的来源(即训练数据集长什么样)

不知道有没有同学会疑问这段代码里面的数据集 是从哪来的，比如原始动作action 从哪来的，我暂且不管有没有疑惑，假设有人有此疑惑，故我来解释下数据集的来源途径

π0主要使用两种数据集：

- FakeDataset - 生成随机数据用于测试
- LeRobotDataset - 真实的机器人操作数据

> LeRobotDataset 是一个专为机器人学习设计的数据集格式，来自\`lerobot.common.datasets.lerobot\_dataset\`模块。这个数据集包含了训练π0模型所需的观察数据和动作数据，其包含
> 
> 1. Aloha数据集，侧重双臂协同的精确操作，适合特定任务的模仿学习，比如这个是 [打开笔帽](https://huggingface.co/datasets/physical-intelligence/aloha_pen_uncap_diverse "打开笔帽") 的任务
> 2. [Libero数据集](https://huggingface.co/datasets/physical-intelligence/libero "Libero数据集") ，注重多样化任务和泛化能力，适合语言引导的通用机器人控制  
> 	![](https://i-blog.csdnimg.cn/direct/d4debff8aa904f24ae4f59eb89b67f31.png)
> 
> ---
> 
> LeRobotDataset 数据通常包含以下几个关键部分：
> 
> 1. 观察数据 (Observation)  
> 	图像数据：来自不同摄像头的图像
> 	```csharp
> 	"observation.images.cam_high"
> 	"observation.images.cam_low"
> 	"observation.images.cam_left_wrist"
> 	"observation.images.cam_right_wrist"
> 	```
> 	状态数据：机器人的关节角度等状态信息
> 	```csharp
> 	"observation.state"
> 	```
> 2. 动作数据 (Actions)  
> 	动作序列：每个时间步的机器人动作指令
> 	```csharp
> 	"action"
> 	```
> 	时间戳信息：通过\`delta\_timestamps\`定义的时间间隔
> 3. 任务信息  
> 	任务描述：可用于生成提示(prompt)  
> 	元数据：包括帧率(fps)等信息
> 
> 数据集示例
> 
> 1. ALOHA数据集  
> 	physical-intelligence/aloha\_pen\_uncap\_diverse
> 	```cobol
> 	{
> 	    "observation": {
> 	        "images": {
> 	            "cam_high": np.ndarray(shape=(3, 224, 224), dtype=np.uint8),
> 	            "cam_left_wrist": np.ndarray(shape=(3, 224, 224), dtype=np.uint8),
> 	            "cam_right_wrist": np.ndarray(shape=(3, 224, 224), dtype=np.uint8)
> 	        },
> 	        "state": np.ndarray(shape=(14,), dtype=np.float32)
> 	    },
> 	    "action": np.ndarray(shape=(14,), dtype=np.float32),
> 	    "prompt": "uncap the pen"
> 	}
> 	```
> 	其中，14维机器人状态向量的含义
> 	```cobol
> 	[
> 	    # 左臂关节角度 (6维)
> 	    left_shoulder_pitch,
> 	    left_shoulder_roll,
> 	    left_shoulder_yaw,
> 	    left_elbow_pitch,
> 	    left_elbow_roll,
> 	    left_wrist_pitch,
> 	 
> 	    # 左手爪状态 (1维)
> 	    left_gripper,
> 	 
> 	    # 右臂关节角度 (6维)
> 	    right_shoulder_pitch,
> 	    right_shoulder_roll,
> 	    right_shoulder_yaw,
> 	    right_elbow_pitch,
> 	    right_elbow_roll,
> 	    right_wrist_pitch,
> 	 
> 	    # 右手爪状态 (1维)
> 	    right_gripper
> 	]
> 	```
> 2. 一个LeRobotDataset的样本可能看起来像这样  
> 	比如Libero数据集：physical-intelligence/libero
> 	```cobol
> 	{
> 	    "observation": {
> 	        "images": {
> 	            # 高视角RGB图像，224x224x3
> 	            "cam_high": np.ndarray(shape=(224, 224, 3), dtype=np.uint8),
> 	            # 低视角RGB图像
> 	            "cam_low": np.ndarray(shape=(224, 224, 3), dtype=np.uint8),
> 	            # 左手腕视角RGB图像
> 	            "cam_left_wrist": np.ndarray(shape=(224, 224, 3), dtype=np.uint8),
> 	            # 右手腕视角RGB图像
> 	            "cam_right_wrist": np.ndarray(shape=(224, 224, 3), dtype=np.uint8)
> 	        },
> 	 
> 	        # 机器人状态向量，包含关节角度等信息
> 	        "state": np.ndarray(shape=(14,), dtype=np.float32),  
> 	    },
> 	 
> 	    # 动作序列，50个时间步，每步14维动作向量
> 	    "actions": np.ndarray(shape=(50, 14), dtype=np.float32),
> 	 
> 	    # 任务描述文本
> 	    "prompt": "fold the towel"
> 	}
> 	```
> 	再比如
> 	```cobol
> 	{
> 	    "observation": {
> 	        "images": {
> 	            "cam_high": <224x224x3 RGB image of robot workspace from above>,
> 	            "cam_left_wrist": <224x224x3 RGB image from left gripper>,
> 	            "cam_right_wrist": <224x224x3 RGB image from right gripper>
> 	        },
> 	        "state": [0.1, -0.5, 0.3, ...],  # 14维机器人关节状态
> 	    },
> 	    "actions": [
> 	        [0.1, -0.2, 0.3, ...],  # t=0时刻的动作
> 	        [0.15, -0.25, 0.35, ...],  # t=1时刻的动作
> 	        ...  # 共50个时间步
> 	    ],
> 	    "prompt": "pick up the blue cube and place it in the red bowl"
> 	}
> 	```

真实数据来自\`lerobot\_dataset\`模块，通过以下代码加载—— *下文「2.2.2 create\_dataset：创建适合训练的数据集」还会详解* ：

```cobol
dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(repo_id, local_files_only=data_config.local_files_only)

dataset = lerobot_dataset.LeRobotDataset(

    data_config.repo_id,

    delta_timestamps={

        key: [t / dataset_meta.fps for t in range(model_config.action_horizon)]

        for key in data_config.action_sequence_keys

    },

    local_files_only=data_config.local_files_only,

)
```

这里的\`repo\_id\`指向一个特定的数据仓库，是Hugging Face上的数据集或其他存储位置。数据集通过配置文件中的参数指定，例如我们在\`config.py\`中看到的配置—— *下文「2.1 配置系统 (config.py)」还会详解* ：

```cobol
# Inference Aloha configs.

    #

    TrainConfig(

        name="pi0_aloha",

        model=pi0.Pi0Config(),

        data=LeRobotAlohaDataConfig(

            assets=AssetsConfig(asset_id="trossen"),

        ),

    ),
```

以下是对数据流程总结

1. 从LeRobot数据集加载原始数据，包含观察(observation)和动作(action)
2. 通过数据转换管道处理数据，包括重打包和归一化
3. 在训练期间，向原始动作添加噪声
4. 模型学习预测添加的噪声，而不是直接预测原始动作
5. 在推理时，模型从纯噪声开始，通过迭代去噪过程生成动作序列

这种基于扩散的方法允许π0从噪声中逐步精炼动作，最终生成平滑且符合任务要求的机器人动作序列

##### 1.2.4.4 推理函数 \`sample\_actions\`：基于扩散模型逆向采样(即去噪)，生成机器人动作序列

**sample\_actions** 函数是Pi0模型的核心推理方法，实现了基于扩散模型的逆向采样过程—— *说白了 就是去噪，它从纯噪声开始，通过多步骤逐渐"去噪"，最终生成符合条件分布的机器人动作序列*

函数的核心是一个基于while循环的迭代过程，每一步都使用训练好的神经网络预测从当前噪声化动作到目标动作的方向—— *从噪声到目标的方向 代表速度场，毕竟咱们去噪的方向得对 不然就去歪了*

总之，这个函数将观察数据（图像和可选的文本提示）转换为具体的动作轨迹，是模型部署时的主要接口，简言之，其包含以下流程

1. 首先从纯噪声开始 (t=1)
2. 通过重复迭代降噪步骤，逐步将噪声转化为有意义的动作序列
3. 使用KV缓存优化推理速度
4. 实现了一个迭代降噪过程
5. 最终返回完全降噪后的动作序列 x\_0

具体而言，包含如下步骤

**第一，初始化**

首先，函数对输入观察数据进行预处理，包括标准化图像大小等操作

```cobol
def sample_actions(

    self,

    rng: at.KeyArrayLike,               # 随机数生成器

    observation: _model.Observation,    # 观察输入，包含图像和文本等

    *,

    num_steps: int = 10,                # 扩散过程的步数，默认为10步

) -> _model.Actions:                    # 返回生成的动作序列

 

    # 对观察数据进行预处理，不进行训练时的数据增强

    observation = _model.preprocess_observation(None, observation, train=False)
```

然后 **设置时间步长\`dt\`为负值（因为是从t=1向t=0方向演化），生成初始随机噪声作为起点，且时间上约定："t=1是噪声，t=0是目标分布"** ，这是扩散文献中常见的约定，不过与Pi0论文相反

```cobol
# 注意：这里使用扩散模型文献中更常见的约定，t=1是噪声，t=0是目标分布

    # 这与pi0论文相反

    dt = -1.0 / num_steps                       # 计算时间步长，从1到0

    batch_size = observation.state.shape[0]     # 获取批次大小

 

    # 生成初始噪声，形状为[批次大小, 动作序列长度, 动作维度]

    noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))
```

**第二，Key-Value缓存初始化** (预计算并存储前缀表示，减少冗余计算)

处理观察数据，得到前缀表示和相关掩码

```cobol
# 首先通过前缀的前向传递填充KV缓存

    # 获取前缀的token表示和掩码

    prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)

 

    # 创建前缀的注意力掩码

    prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)

 

    # 计算位置编码

    positions = jnp.cumsum(prefix_mask, axis=1) - 1
```

然后使用PaliGemma语言模型进行一次前向传递，生成Key-Value缓存（\`kv\_cache\`）—— *这是一个性能优化：因为前缀部分在整个采样过程中保持不变，预先计算并缓存它们的表示可以避免重复计算*

```cobol
# 进行前向传递，获取KV缓存

    _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)
```

**第三，通过step函数构建注意力掩码系统并让PaliGemma做推理**

核心迭代通过 \`jax.lax.while\_loop\` 实现

> 根据源码
> 
> ![](https://i-blog.csdnimg.cn/direct/d89c6e9235a74b13bed623b21213e21e.png)
> 
> 可知，该 ***class Pi0(\_model.BaseModel)*** 类的最后两行是
> 
> ```cobol
> # 使用while循环进行迭代采样，从t=1（噪声）开始
> 
>     x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
> 
>  
> 
>     # 返回最终的去噪结果（生成的动作序列）
> 
>     return x_0
> ```

具体而言，包含 \`step\` 函数和 \`cond\` 函数，其中，\`step\` 函数是每次迭代的核心

首先 ，step函数通过 \`embed\_suffix\` 处理当前状态，包括状态信息嵌入、噪声化动作、时间步编码

```python
def step(carry):

        """定义单步去噪函数"""

        x_t, time = carry  # carry数组包含当前状态和时间

 

        # 将时间广播到批次维度，并嵌入后缀（状态和动作）

        suffix_tokens, suffix_mask, suffix_ar_mask = self.embed_suffix(

            observation, x_t, jnp.broadcast_to(time, batch_size)

        )
```

其次 ，构建复杂的注意力掩码系统，处理前缀-后缀之间的注意力关系—— *这个复杂的掩码系统允许后缀token（包括状态和动作）有选择地关注前缀token（图像和文本），实现了条件生成* ，具体而言，其构建了三层注意力掩码：

- 后缀内部注意力掩码，控制后缀token（状态和动作）之间的注意力关系
```csharp
# 创建后缀内部的注意力掩码，形状为(批次, 后缀长度, 后缀长度)

        suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
```
- 前缀-后缀注意力掩码，控制后缀token如何关注前缀token（图像和文本输入）
```cobol
# 创建后缀对前缀的注意力掩码，形状为(批次, 后缀长度, 前缀长度)

        prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
```
- 完整注意力掩码，将前两个掩码组合，形成完整的注意力控制机制
```cobol
# 组合掩码，形状为(批次, 后缀长度, 前缀长度+后缀长度)

        # 控制后缀token（生成查询）如何关注完整序列（生成键和值）

        full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
```

当然了，过程中还做了形状检查，确保张量维度正确

```cobol
# 验证掩码形状正确

        assert full_attn_mask.shape == (

            batch_size,

            suffix_tokens.shape[1],

            prefix_tokens.shape[1] + suffix_tokens.shape[1],

        )
```

接着 ，计算位置编码，为后缀token计算其在完整序列中的位置，这对于Transformer模型理解序列顺序很重要

```cobol
# 计算后缀token的位置编码

        positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1
```

之后 ，模型推理，使用PaliGemma语言模型进行推理，利用缓存的前缀信息（\`kv\_cache\`）提高效率

```python
# 使用KV缓存进行高效的前向传递

        (prefix_out, suffix_out), _ = self.PaliGemma.llm(

            [None, suffix_tokens], mask=full_attn_mask, positions=positions, kv_cache=kv_cache

        )

 

        # 且确保前缀输出为None（因为使用了KV缓存）

        assert prefix_out is None
```

**第四，step函数中做最后的速度预测与动作更新(去噪)**

在每一步中，模型预测速度场 \`v\_t\`（从噪声到目标的方向），并通过类欧拉法更新动作表示——使用简单而有效的欧拉方法进行轨迹采样

本质就是对去噪，而便是预测的噪声，是时间步长——如上面说过的「 *时间步长\`dt\`为负值（因为是从t=1向t=0方向演化），生成初始随机噪声作为起点，且时间上约定："t=1是噪声，t=0是目标分布"* 」

具体而言

- 一方面，提取模型输出并预测速度场\`v\_t\`—— *相当于本质是通过PaliGemma模型预测去噪方向 \`v\_t\`*
```php
# 预测噪声

        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])
```
- 二方面，使用欧拉法更新动作状态和时间步
```cobol
# 使用欧拉方法更新状态和时间

        return x_t + dt * v_t, time + dt
```

至于cond函数确定何时停止迭代，通过检查时间是否接近零(当然，要考虑浮点精读可能存在的误差)

```python
def cond(carry):

        """定义循环终止条件"""

        x_t, time = carry

 

        # 考虑浮点误差，当时间接近0时停止

        return time >= -dt / 2
```

### 1.3 语言模型实现：models/gemma.py

src/openpi/models/gemma.py实现了Gemma语言模型的核心组件，定义了RMSNorm、Embedder、Attention、FeedForward等模块，且提供了不同规模Gemma模型的配置（300M, 2B等）

// 待更

### 1.4 视觉模型实现：models/siglip.py

\`siglip.py\`: 实现了视觉编码器，基于Vision Transformer (ViT)，定义了位置编码、注意力池化等组件，支持不同大小的模型变体

// 待更

### 1.5 其他支持模块：LoRA、tokenizer、vit的实现

- \`lora.py\` 实现了LoRA (Low-Rank Adaptation)微调方法
- \`tokenizer.py\`: 提供文本tokenization功能
- \`vit.py\`: Vision Transformer实现

// 待更

## 第二部分 模型训练的配置：src下training模块的全面分析与解读

training模块是 OpenPI 项目中负责训练相关功能的核心部分，该目录下包含了以下主要文件：

1. checkpoints.py - 检查点管理
2. config.py - 配置系统
3. data\_loader.py - 数据加载器
4. data\_loader\_test.py - 数据加载器测试
5. optimizer.py - 优化器实现
6. sharding.py - 模型分片工具
7. utils.py - 通用工具函数
8. weight\_loaders.py - 模型权重加载器

### 2.1 配置系统 (config.py)

定义了训练过程的各种配置类型，包括：

1. \`TrainConfig\`：顶级训练配置，包含模型、数据、优化器等所有训练参数
2. \`DataConfigFactory\`：抽象工厂类，用于创建特定环境的数据配置
3. \`AssetsConfig\`：管理资产（如归一化统计数据）的位置
4. 预定义了多种常用配置（如 ALOHA、DROID、LIBERO 等环境的配置）
5. 通过 \`get\_config\` 函数根据名称检索预定义配置

在配置流程上

\- 训练脚本通过 \`\_config.cli()\` 或 \`\_config.get\_config()\` 获取配置  
\- 配置系统加载预定义的训练参数，确定训练环境和模型参数  
\- 数据配置通过工厂模式创建，根据不同环境（ALOHA、DROID 等）提供不同的预处理流程

#### 2.1.1 基础配置类AssetsConfig、DataConfig

一个是AssetsConfig

```python
class AssetsConfig:

    """用于确定数据pipeline所需资产(如归一化统计信息)的位置"""

    assets_dir: str | None = None      # 资产目录

    asset_id: str | None = None        # 资产ID
```

一个是DataConfig

```python
@dataclasses.dataclass(frozen=True)

class DataConfig:

    repo_id: str | None = None            # 数据集仓库ID

    asset_id: str | None = None           # 资产ID

    norm_stats: dict[str, _transforms.NormStats] | None = None  # 归一化统计信息

    repack_transforms: _transforms.Group  # 数据重打包转换

    data_transforms: _transforms.Group    # 数据预处理转换

    model_transforms: _transforms.Group   # 模型特定转换
```

#### 2.1.2 数据集配置：包含ALOHA、Libero两套数据集——LeRobotLiberoDataConfig

涉及两个配置

- 一个是LeRobotAlohaDataConfig
	```python
	@dataclasses.dataclass(frozen=True)
	class LeRobotAlohaDataConfig(DataConfigFactory):
	    """ALOHA数据集配置"""
	    use_delta_joint_actions: bool = True      # 是否使用关节角度增量
	    default_prompt: str | None = None         # 默认提示语
	    adapt_to_pi: bool = True                  # 是否适配到π内部运行时
	```
- 一个是LeRobotLiberoDataConfig
	```python
	@dataclasses.dataclass(frozen=True)
	class LeRobotLiberoDataConfig(DataConfigFactory):
	    """Libero数据集配置"""
	```

对于后者的结构，详见下图

![](https://i-blog.csdnimg.cn/direct/10e28e6bccbc496296c1a0d70ad57e28.png)

1. \`LeRobotLiberoDataConfig\` 是一个用于机器人控制系统的数据配置类，它负责定义整个数据管道中不同阶段的数据转换操作。这个类通过 \`@dataclasses.dataclass(frozen=True)\` 装饰器声明为不可变数据类，确保配置一旦创建就不能被修改，增强了数据处理的稳定性
2. 该类重写了基类 \`DataConfigFactory\` 的 \`create\` 方法，该方法是整个配置系统的核心，负责构建完整的数据配置
	```php
	def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
	        # 重写父类方法，创建数据配置。参数包括资产目录路径和模型配置，返回DataConfig对象
	        # ..
	```
	方法接收两个关键参数：存放数据资产的目录路径和模型配置对象，然后返回一个完整的 \`DataConfig\` 对象
3. 在方法内部，首先定义了 \`repack\_transform\`，这是一个仅在训练阶段应用的转换器，用于将数据集中的键名映射到推理环境期望的键名  
	例如，将 \`"observation/image"\` 映射到 \`"image"\`。这种转换确保了训练数据和推理环境之间的一致性，是适配不同数据源的关键步骤
4. 接下来，\` **data\_transforms** \` 配置了同时应用于训练和推理阶段的转换操作  
	它使用 \`libero\_policy.LiberoInputs\` 处理输入数据，\`libero\_policy.LiberoOutputs\` 处理输出数据
	```cobol
	# 数据转换应用于来自数据集的数据和推理过程中的数据
	        # 下面，定义了进入模型的数据转换（"inputs"）和从模型输出的数据转换（"outputs"）（后者仅在推理时使用）
	        # 这些转换在\`libero_policy.py\`中定义
	        # 一旦创建了自己的转换，你可以用自己的替换下面的转换
	        data_transforms = _transforms.Group(
	             # 定义输入转换，使用LiberoInputs处理器
	            inputs=[libero_policy.LiberoInputs(action_dim=model_config.action_dim, model_type=model_config.model_type)], 
	 
	            # 定义输出转换，使用LiberoOutputs处理器
	            outputs=[libero_policy.LiberoOutputs()],  
	        )
	```
	这些转换器负责将原始数据调整为模型能够处理的格式
5. 特别值得注意的是关于动作表示的转换：该配置支持将绝对动作（如具体的关节角度）转换为相对动作（相对于初始状态的变化量）  
	通过 \` **delta\_action\_mask** \` 创建一个布尔掩码，指定哪些动作维度需要进行转换（这里是前6个维度对应机器人关节，保留最后一个维度对应夹爪不变）
	```cobol
	# 创建动作掩码，指定哪些维度需要转换为相对动作（前6个关节），哪些保持绝对值（夹爪）
	        # 创建布尔掩码，前6个维度为True，最后一个维度为False
	        delta_action_mask = _transforms.make_bool_mask(6, -1)
	```
	这对于训练基于相对动作的模型（如Pi0模型）非常重要
6. 最后，\` **model\_transforms** \` 处理模型特有的转换操作，比如提示文本的token化和图像尺寸调整
	```csharp
	# 使用模型配置创建模型转换——处理提示文本的token化和其他模型特定的转换
	        model_transforms = ModelTransformFactory()(model_config)
	```
	这些转换由 \`ModelTransformFactory\` 根据模型类型动态创建，支持不同类型的模型（Pi0或Pi0\_FAST）
7. 整个方法通过 \`dataclasses.replace\` 将这些转换器与基础配置（通过 \`create\_base\_config\` 创建）合并，生成最终的数据配置对象
	```cobol
	return dataclasses.replace(
	            self.create_base_config(assets_dirs),         # 创建基础配置
	            repack_transforms=repack_transform,           # 设置重新打包转换
	            data_transforms=data_transforms,              # 设置数据转换
	            model_transforms=model_transforms,            # 设置模型转换
	        )
	```

#### 2.1.3 训练配置TrainConfig：模型、数据、优化器等训练参数的设置

```python
class TrainConfig:

    name: str                              # 配置名称

    project_name: str = "openpi"           # 项目名称

    exp_name: str                          # 实验名称

    model: _model.BaseModelConfig          # 模型配置

    batch_size: int = 32                   # 批次大小

    num_train_steps: int = 30_000          # 训练步数

    lr_schedule: _optimizer.LRScheduleConfig      # 学习率调度

    optimizer: _optimizer.OptimizerConfig         # 优化器配置
```

#### 2.1.4 预定义配置：基于ALOHA/Libero数据集微调π0——比如完成aloha\_sim\_transfer\_cube\_human

文件最后定义了多个具体的训练配置：

- 比如ALOHA的
	```cobol
	TrainConfig(
	    name="pi0_aloha_pen_uncap",      # 配置名称，反映模型和数据集
	    model=pi0.Pi0Config(),           # 使用pi0模型配置
	    data=LeRobotAlohaDataConfig(     # 使用LeRobotAloha数据集配置
	 
	        # 数据集仓库ID
	        repo_id="physical-intelligence/aloha_pen_uncap_diverse",  
	 
	        # 资产配置
	        assets=AssetsConfig(  
	            # 资产目录
	            assets_dir="s3://openpi-assets/checkpoints/pi0_base/assets",  
	            # 资产ID
	            asset_id="trossen",          
	        ),
	        # 默认提示语
	        default_prompt="uncap the pen",  
	 
	        # 数据重打包转换
	        repack_transforms=_transforms.Group(      
	            inputs=[
	                # 重打包转换
	                _transforms.RepackTransform(      
	                    {
	                        "images": {
	                            # 高视角摄像头图像
	                            "cam_high": "observation.images.cam_high",  
	 
	                            # 左手腕摄像头图像
	                            "cam_left_wrist": "observation.images.cam_left_wrist",
	 
	                            # 右手腕摄像头图像  
	                            "cam_right_wrist": "observation.images.cam_right_wrist",                  
	                        },
	 
	                        # 机器人状态
	                        "state": "observation.state",  
	 
	                        # 动作
	                        "actions": "action",           
	                    }
	                )
	            ]
	        ),
	 
	        base_config=DataConfig(
	            # 是否只使用本地数据集，False表示允许从Hugging Face下载
	            local_files_only=False,  
	        ),
	    ),
	 
	    # 加载预训练权重
	    weight_loader=weight_loaders.CheckpointWeightLoader("s3://openpi-assets/checkpoints/pi0_base/params"),  
	 
	    # 训练步数为20,000步
	    num_train_steps=20_000,  
	),
	```
	当然，这里面还涉及到ALOHA中一个仿真环境中的操作任务
	```cobol
	# 这个配置用于演示如何在简单的模拟环境中进行训练
	TrainConfig(
	    name="pi0_aloha_sim",          # 配置名称
	    model=pi0.Pi0Config(),         # 使用pi0模型配置
	    data=LeRobotAlohaDataConfig(   # 使用LeRobotAloha数据集配置
	 
	         # 数据集仓库ID
	        repo_id="lerobot/aloha_sim_transfer_cube_human", 
	        default_prompt="Transfer cube",      # 默认提示语
	        use_delta_joint_actions=False,       # 是否使用关节角度增量
	    ),
	    weight_loader=weight_loaders.CheckpointWeightLoader("s3://openpi-assets/checkpoints/pi0_base/params"),        # 加载预训练权重
	    num_train_steps=20_000,                  # 训练步数为20,000步
	),
	```
- 再比如Libero的
	```cobol
	TrainConfig(
	    # 更改名称以反映你的模型和数据集
	    name="pi0_libero",
	    
	    # 在这里定义模型配置 - 这个例子中我们使用pi0作为模型架构并执行完整微调
	    # 在后面的例子中我们会展示如何修改配置来执行低内存(LORA)微调
	    # 以及如何使用pi0-FAST作为替代架构
	    model=pi0.Pi0Config(),
	    
	    # 在这里定义要训练的数据集。这个例子中我们使用Libero数据集
	    # 对于你自己的数据集，你可以更改repo_id指向你的数据集
	    # 同时修改DataConfig以使用你为数据集创建的新配置
	    data=LeRobotLiberoDataConfig(
	        # 指定数据集的Hugging Face仓库ID
	        repo_id="physical-intelligence/libero",
	        
	        # 基础配置设置
	        base_config=DataConfig(
	            # 是否只使用本地数据集，False表示允许从Hugging Face下载
	            local_files_only=False,  
	            
	            # 这个标志决定是否从LeRobot数据集的task字段加载提示(即任务指令)
	            # 如果设为True，提示将会出现在输入字典的prompt字段中
	            # 推荐设置为True
	            prompt_from_task=True,
	        ),
	    ),
	    
	    # 在这里定义要加载哪个预训练检查点来初始化模型
	    # 这应该与你上面选择的模型配置匹配 - 即在这种情况下我们使用pi0基础模型
	    weight_loader=weight_loaders.CheckpointWeightLoader(
	        "s3://openpi-assets/checkpoints/pi0_base/params"
	    ),
	    
	    # 在下面你可以定义其他超参数，如学习率、训练步数等
	    # 查看TrainConfig类以获取完整的可用超参数列表
	    num_train_steps=30_000,  # 设置训练步数为30,000步
	),
	```

### 2.2 数据加载系统 data\_loader.py

定义了数据集和数据加载器的接口（\`Dataset\` 和 \`DataLoader\`）

1. 实现了数据转换管道，将原始数据转换为模型可用的格式
2. 支持各种数据源：真实数据集（通过 LeRobot 数据集接口）、模拟数据（使用 \`FakeDataset\`）
3. 提供数据归一化和转换功能

> 在数据加载流程上
> 
> TrainConfig  
> └── data (DataConfigFactory)  
> ├── create() → DataConfig  
> │ ├── repo\_id: 数据集 ID  
> │ ├── norm\_stats: 归一化统计数据  
> │ ├── repack\_transforms: 数据重包装转换  
> │ ├── data\_transforms: 特定于环境的转换  
> │ └── model\_transforms: 特定于模型的转换  
> └── \_load\_norm\_stats() → 归一化统计数据
> 
>   
> create\_data\_loader(config)  
> ├── data\_config = config.data.create()  
> ├── dataset = create\_dataset(data\_config, config.model)  
> ├── dataset = transform\_dataset(dataset, data\_config)  
> └── return DataLoaderImpl(data\_config, TorchDataLoader(...))

#### 2.2.1 FakeDataset类

#### 2.2.2 create\_dataset：创建适合训练的数据集

\`create\_dataset\` 函数是一个关键的数据准备工具，负责根据配置参数创建适合模型训练的数据集。这个函数通过处理不同数据源和应用必要的转换，为模型提供标准化的训练数据。

1. 首先，函数检查 \`data\_config.repo\_id\` 的值，这个参数指定了数据仓库的标识符
	```cobol
	def create_dataset(data_config: _config.DataConfig, model_config: _model.BaseModelConfig) -> Dataset:
	    """创建用于训练的数据集"""
	    # 从数据配置中获取仓库ID
	    repo_id = data_config.repo_id
	```
	如果 \`repo\_id\` 为 \`None\`，函数会抛出 \`ValueError\` 异常，明确指出无法创建数据集。这是一种防御性编程的体现，确保基本的配置参数存在
	```python
	# 如果仓库ID为空，抛出错误
	    if repo_id is None:
	        raise ValueError("Repo ID is not set. Cannot create dataset.")
	```
	如果 \`repo\_id\` 的值为 "fake"，函数则创建并返回一个 \`FakeDataset\` 实例，其样本数设为 1024。这种虚拟数据集在测试模型架构、调试训练流程或者进行性能基准测试时非常有用，无需加载真实数据即可快速验证系统功能
	```cobol
	# 如果是fake数据集，返回包含1024个样本的假数据集
	    if repo_id == "fake":
	        return FakeDataset(model_config, num_samples=1024)
	```
	对于其他情况（即使用真实数据），函数首先创建 \`LeRobotDatasetMetadata\` 对象来获取数据集的元信息
	```cobol
	# 创建数据集元数据对象，包含数据集的基本信息（如fps等）
	    dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(
	        repo_id, 
	        local_files_only=data_config.local_files_only
	    )
	```
	然后初始化 \` LeRobotDataset \` 实例
	```cobol
	# 创建LeRobot数据集实例
	    dataset = lerobot_dataset.LeRobotDataset(
	        data_config.repo_id,
	        # 创建时间戳字典，用于采样动作序列
	        delta_timestamps={
	            # 对每个动作序列键，根据模型的动作视界长度和数据集的fps生成时间戳列表
	            key: [t / dataset_meta.fps for t in range(model_config.action_horizon)]
	            for key in data_config.action_sequence_keys
	        },
	        # 是否只使用本地文件
	        local_files_only=data_config.local_files_only,
	    )
	```
	特别值得注意的是，函数会根据模型的 \`action\_horizon\`（动作预测的时间步长）和数据集的帧率（fps）计算 \`delta\_timestamps\`，这些时间戳用于在时序数据中定位动作序列。这种计算确保了动作序列的时间间隔与模型预期一致，无论原始数据的采样率如何
2. 最后，如果 \`data\_config.prompt\_from\_task\` 设置为 \`True\`，函数会将原始数据集包装在 \`TransformedDataset\` 中，并应用 \`PromptFromLeRobotTask\` 转换
	```csharp
	# 如果配置指定从任务中提取提示信息
	    if data_config.prompt_from_task:
	        # 创建转换后的数据集，应用PromptFromLeRobotTask转换，将任务描述转换为提示
	        dataset = TransformedDataset(
	            dataset, 
	            [_transforms.PromptFromLeRobotTask(dataset_meta.tasks)]
	        )
	```
	这个转换可能将任务描述转换为自然语言提示，增强模型对任务上下文的理解能力  
	然后返回处理好的数据集
	```csharp
	# 返回处理后的数据集
	    return dataset
	```

#### 2.2.3 transform\_dataset：对数据集应用转换，比如数据清洗等(创建TransformedDataset实例)

\`transform\_dataset\` 函数是数据预处理管道中的关键组件，负责对原始数据集应用一系列转换操作，以满足模型训练的需求。该函数接收一个原始数据集、数据配置对象以及一个可选的控制标志，并返回经过转换的新数据集

首先，函数会处理数据归一化统计信息（normalization statistics）。对于实际数据集（非"fake"数据集），如果没有显式跳过归一化统计（\`skip\_norm\_stats=False\`），函数会检查数据配置中是否包含必要的归一化统计数据。如果这些统计数据缺失，函数会抛出一个明确的错误信息，提示用户需要运行特定脚本来计算这些统计数据。这种检查机制确保了数据归一化步骤能够正确执行，避免了训练过程中可能出现的数值问题

核心转换逻辑通过创建一个 \`TransformedDataset\` 实例来实现，该实例封装了原始数据集和一系列转换函数。这些转换函数按照特定顺序应用：

1. 首先是数据重新打包转换（\`repack\_transforms\`），可能用于调整数据的基本结构
2. 接着是一般数据转换（\`data\_transforms\`），处理数据清洗、增强等操作
3. 然后应用归一化转换（\`Normalize\`），使用前面获取的统计数据
4. 最后是模型特定的转换（\`model\_transforms\`），针对特定模型架构的数据格式要求

#### 2.2.4 create\_data\_loader：创建用于训练的数据加载器

\`create\_data\_loader\` 函数是整个数据处理流水线的核心组件，它协调多个模块共同工作，创建一个用于模型训练的数据加载器

整个函数的工作流程可以分为三个主要阶段：

1. **第一阶段：数据集准备**  
	函数首先通过调用 \`data\_config.create()\` 方法创建数据配置对象，该对象包含了所有数据处理相关的配置信息  
	  
	随后，通过 \`create\_dataset\` 函数创建原始数据集，这可能是一个真实的机器人数据集或者是一个用于测试的假数据集（当 \`repo\_id\` 为 "fake" 时）  
	  
	然后，调用 \`transform\_dataset\` 函数应用一系列数据转换，包括数据重新打包、数据清洗、归一化和模型特定转换。这些转换确保了原始数据被正确处理为模型所需的格式
2. **第二阶段：PyTorch 数据加载器创建**  
	接下来，函数实例化一个 \`TorchDataLoader\` 对象，这是对 PyTorch 数据加载器的封装。这个过程涉及多个关键参数设置：计算各进程的本地批量大小（通过全局批量大小除以进程数）  
	配置数据分片策略（sharding）用于分布式训练  
	设置是否打乱数据、工作进程数和随机种子等  
	  
	\`TorchDataLoader\` 的设计支持无限迭代数据（当 \`num\_batches\` 为 \`None\` 时）或限定批次数的迭代，这对于训练和评估场景都很适用。其内部使用 JAX 的分片机制确保数据在分布式环境中正确分布
3. **第三阶段：接口适配器实现**  
	最后，函数通过定义嵌套类 \`DataLoaderImpl\` 来适配 \`DataLoader\` 协议接口。这个类封装了前面创建的 \`TorchDataLoader\` 实例，并提供了两个关键方法：  
	1\. \`data\_config()\` 返回数据配置信息，便于训练代码访问数据处理的元信息  
	  
	2\. \`\_\_iter\_\_()\` 生成器方法对数据批次进行最后的格式转换：  
	将字典格式的观察数据转换为结构化的 \`Observation\` 对象（通过 \`Observation.from\_dict\`）提取动作数据  
	以元组形式 \`(observation, actions)\` 返回每个批次

这种设计实现了关注点分离，使数据加载、转换和格式适配各自独立，同时又协同工作，为模型训练提供了一个干净的数据流接口。函数还处理了多进程环境、数据分片和内存效率等复杂问题，这些都是大规模机器学习训练中的关键挑战

### 2.3 优化器系统 (optimizer.py)

定义了多种学习率调度策略：

1. \`CosineDecaySchedule\`：余弦衰减学习率
2. \`RsqrtDecaySchedule\`：反平方根衰减学习率

实现了常用优化器配置：

1. \`AdamW\`：带有权重衰减的 Adam 优化器
2. \`SGD\`：随机梯度下降优化器

通过 \`create\_optimizer\` 函数统一创建优化器实例

### 2.4 检查点系统 (checkpoints.py)

负责模型状态的保存和恢复，比如管理训练状态的序列化，包括：

1. 模型参数
2. 优化器状态
3. EMA 参数（如果使用）

且使用 Orbax 库实现高效的检查点存储

| 模型初始化流程 | 训练步骤流程 | 与 models 模块的交互 | 检查点管理流程 |
| --- | --- | --- | --- |
| init\_train\_state(config, rng, mesh)   ├── 创建模型：model = config.model.create(rng)   ├── 加载权重：partial\_params = config.weight\_loader.load(params)   ├── 设置冻结参数：params = state\_map(params, config.freeze\_filter,...)   ├── 创建优化器：tx = create\_optimizer(config.optimizer, config.lr\_schedule)   └── 返回 TrainState | train\_step(config, rng, state, batch)   ├── 计算梯度：loss, grads = value\_and\_grad(model.compute\_loss )()   ├── 更新参数：updates, new\_opt\_state = state.tx.update(grads, state.opt\_state, params)   ├── 应用更新：new\_params = optax.apply\_updates(params, updates)   ├── 更新 EMA 参数（如果配置）   └── 返回 new\_state, info | \- 训练系统加载模型定义 (\`BaseModel\`)   \- 处理模型参数的保存和加载   \- 调用模型的 \`compute\_loss\` 方法计算损失—— *详见上文的「1.2.4.3 损失函数 \`compute\_loss\`」* | save\_state(checkpoint\_manager, state, data\_loader, step)   ├── \_split\_params(state) → 分离训练状态和推理参数   ├── 保存归一化统计数据到 assets 目录   └── checkpoint\_manager.save() → 保存检查点      restore\_state(checkpoint\_manager, state, data\_loader)   ├── checkpoint\_manager.restore() → 恢复检查点   └── \_merge\_params() → 合并恢复的参数 |

// 待更

### 2.5 模型分片系统(sharding.py)：含FSDP的实现

实现分布式训练时的模型参数分片

1. 提供 \`fsdp\_sharding\` 函数用于全参数数据并行(FSDP)的实现
2. 基于 JAX 的分片机制，优化大规模模型的训练性能
3. 通过 \`activation\_sharding\_constraint\` 处理激活值的分片

### 2.6 权重加载系统 (weight\_loaders.py)

定义了 \`WeightLoader\` 协议，用于加载预训练权重，且实现了多种加载策略：

1. \`NoOpWeightLoader\`：不加载权重（用于从头训练）
2. \`CheckpointWeightLoader\`：从检查点加载完整权重
3. \`PaliGemmaWeightLoader\`：从官方 PaliGemma 检查点加载权重

另，还支持权重合并功能，可以部分加载权重（如 LoRA 微调）

### 2.7 辅助工具(utils.py)

定义了 \`TrainState\` 数据类，封装了训练过程的状态

1. 提供日志记录和调试功能
2. 实现了 PyTree 转换和可视化功能

// 待更

## 第三部分 模型的训练与部署：基于客户端-服务器C/S架构

packages/openpi-client，是一个独立的客户端库openpi-client 库，主要负责：

1. 提供与策略服务器通信的接口：使用 WebSocketClientPolicy 连接服务器
2. 处理观察数据(图像、状态等)的发送，和动作数据的接收
3. 管理客户端运行时环境
4. 被各种机器人平台(如 ALOHA、DROID)使用来与服务器交互

scripts这个模块提供了服务器端的各种工具和脚本，主要包括：

1. 策略服务相关——serve\_policy.py：启动策略服务器，处理来自客户端的请求
2. 训练相关——train.py: 模型训练的入口点
3. 数据处理——compute\_norm\_stats.py: 计算数据归一化统计信息
4. 部署相关：提供 Docker 相关的配置和安装脚本

总的来说，这是一个典型的分布式系统设计：packages/openpi-client 提供轻量级的客户端接口，而 scripts/ 则提供服务器端的功能实现，两者通过 WebSocket 协议进行通信，形成了一个完整的策略部署和执行系统

> 所谓客户端-服务器架构——Client-server model，也称C/S架构、主从zòng式架构，是一种将客户端与服务器分割开来的分布式架构。 *每一个客户端软件的实例都可以向一个服务器或应用程序服务器发出请求。有很多不同类型的服务器，例如文件服务器、游戏服务器等*
> 
> ![](https://i-blog.csdnimg.cn/direct/39e43507d4d54ee2bf575a8a71ccd947.png)
> 
> ---
> 
> 客户端的特征：
> 
> 1. 主动的角色（主）
> 2. 发送请求
> 3. 等待直到收到响应
> 
> 服务端的特征：
> 
> 1. 被动的角色（从）
> 2. 等待来自客户端的请求
> 3. 处理请求并传回结果

### 3.1 packages/openpi-client：帮真机或Sim与策略服务器进行通信和交互

该模块的目录结构如下

![](https://i-blog.csdnimg.cn/direct/6f08b6645da74b6a8eac7425a88b60d7.png)

这个客户端包的设计非常模块化，具有良好的扩展性，主要用于：

1. 连接到 OpenPI 服务器
2. 处理观察数据和动作序列
3. 管理机器人或仿真环境的运行
4. 提供事件监控和记录功能

它的设计允许在不同的机器人平台上灵活部署，支持实时控制和异步通信，是 OpenPI 项目中连接模型服务器和实际机器人执行系统的重要桥梁

#### 3.1.1 核心接口层

\`BasePolicy\`: 定义策略接口  
\`Environment\`: 定义环境接口  
\`Agent\`: 定义代理接口

#### 3.1.2 通信层

1. \`WebsocketClientPolicy\`: 实现与服务器的 WebSocket 通信
2. \`msgpack\_numpy\`: 处理数据序列化

#### 3.1.3 数据处理层

1. \`ActionChunkBroker\`: 处理动作序列的分块和缓存
2. \`image\_tools\`: 提供图像处理和优化功能

#### 3.1.4 运行时系统层

1. \`Runtime\`: 核心运行时系统
2. \`Subscriber\`: 事件订阅系统
3. \`agents\`: 具体代理实现

#### 3.1.5 工具支持

1. 图像处理工具
2. 数据类型转换
3. 网络通信优化

### 3.2 scripts(策略服务器)：包含数据处理、模型训练、模型推理的多个脚本

根据下图

![](https://i-blog.csdnimg.cn/direct/8939e104f1b44ad485587f7123af1902.png)

可知，scripts 目录包含多个 Python 脚本，这些脚本用于数据处理、模型训练和服务部署等任务，每个脚本通常对应一个特定的功能或任务

1. \_\_init\_\_.py
2. compute\_norm\_stats.py: 计算数据的归一化统计信息
3. serve\_policy.py：启动策略服务，提供模型推理接口
4. train\_test.py: 训练和测试模型
5. train.py: 训练模型

#### 3.2.1 \_\_init\_\_.py

#### 3.2.3 serve\_policy.py：启动策略服务，用于模型推理

1. 在这个代码片段中，首先导入了一些必要的模块和库，包括 \`policy\`、\`policy\_config\`、\`websocket\_policy\_server\` 和 \`config\`，这些模块来自 \`openpi\` 项目
	```coffeescript
	from openpi.policies import policy as _policy       # 导入 openpi.policies.policy 模块并重命名为 _policy
	from openpi.policies import policy_config as _policy_config  # 导入 openpi.policies.policy_config 模块并重命名为 _policy_config
	from openpi.serving import websocket_policy_server  # 导入 openpi.serving.websocket_policy_server 模块
	from openpi.training import config as _config       # 导入 openpi.training.config 模块并重命名为 _config
	```
	接下来定义了一个枚举类 \`EnvMode\`，它表示支持的环境类型，包括 \`ALOHA\`、\`ALOHA\_SIM\`、\`DROID\` 和 \`LIBERO\`
	```python
	class EnvMode(enum.Enum):
	    """支持的环境。"""
	    ALOHA = "aloha"              # ALOHA 环境
	    ALOHA_SIM = "aloha_sim"      # ALOHA 模拟环境
	    DROID = "droid"              # DROID 环境
	    LIBERO = "libero"            # LIBERO 环境
	```
2. 然后定义了几个数据类  
	\`Checkpoint\` 类用于从训练好的检查点加载策略，包含两个字段：\`config\`（训练配置名称）和 \`dir\`（检查点目录）  
	\`Default\` 类表示使用默认策略  
	\`Args\` 类定义了脚本的参数，包括环境类型、默认提示、端口、是否记录策略行为以及如何加载策略
3. 接下来定义了一个字典 \`DEFAULT\_CHECKPOINT\`，它为每个环境类型指定了默认的检查点配置
	```cobol
	# 每个环境应使用的默认检查点
	DEFAULT_CHECKPOINT: dict[EnvMode, Checkpoint] = {
	    EnvMode.ALOHA: Checkpoint(
	        config="pi0_aloha",
	        dir="s3://openpi-assets/checkpoints/pi0_base",
	    ),
	    EnvMode.ALOHA_SIM: Checkpoint(
	        config="pi0_aloha_sim",
	        dir="s3://openpi-assets/checkpoints/pi0_aloha_sim",
	    ),
	    EnvMode.DROID: Checkpoint(
	        config="pi0_fast_droid",
	        dir="s3://openpi-assets/checkpoints/pi0_fast_droid",
	    ),
	    EnvMode.LIBERO: Checkpoint(
	        config="pi0_fast_libero",
	        dir="s3://openpi-assets/checkpoints/pi0_fast_libero",
	    ),
	}
	```
	\`create\_default\_policy\` 函数根据环境类型创建默认策略，如果环境类型不支持，则抛出异常
	```cobol
	def create_default_policy(env: EnvMode, *, default_prompt: str | None = None) -> _policy.Policy:
	    """为给定环境创建默认策略 """
	    if checkpoint := DEFAULT_CHECKPOINT.get(env):              # 获取环境对应的默认检查点
	        return _policy_config.create_trained_policy(
	            _config.get_config(checkpoint.config), checkpoint.dir, default_prompt=default_prompt
	        )  # 创建训练好的策略
	    raise ValueError(f"Unsupported environment mode: {env}")   # 如果环境不支持，抛出异常
	```
	\`create\_policy\` 函数根据传入的参数创建策略，如果参数中指定了检查点，则从检查点加载策略，否则使用默认策略
	```cobol
	def create_policy(args: Args) -> _policy.Policy:
	    """根据给定的参数创建策略 """
	    match args.policy:          # 匹配策略类型
	        case Checkpoint():      # 如果是 Checkpoint 类型
	            return _policy_config.create_trained_policy(
	                _config.get_config(args.policy.config), args.policy.dir, default_prompt=args.default_prompt
	            )      # 创建训练好的策略
	        case Default():          # 如果是 Default 类型
	            return create_default_policy(args.env, default_prompt=args.default_prompt)      # 创建默认策略
	```
4. \`main\` 函数是脚本的入口点，它首先调用 \`create\_policy\` 函数创建策略，然后记录策略的元数据
	```ruby
	def main(args: Args) -> None:
	    policy = create_policy(args)           # 创建策略
	    policy_metadata = policy.metadata      # 获取策略的元数据
	```
	如果参数中指定了记录策略行为，则使用 \`PolicyRecorder\` 包装策略
	```csharp
	# 记录策略的行为
	    if args.record:
	        # 使用 PolicyRecorder 记录策略行为
	        policy = _policy.PolicyRecorder(policy, "policy_records")
	```
	接着获取主机名和本地 IP 地址
	```perl
	hostname = socket.gethostname()              # 获取主机名
	    local_ip = socket.gethostbyname(hostname)    # 获取本地 IP 地址
	    logging.info("Creating server (host: %s, ip: %s)", hostname, local_ip)  # 记录服务器创建信息
	```
	并创建一个 WebSocket 服务器来提供策略服务，最后调用 \`serve\_forever\` 方法启动服务器
	```cobol
	server = websocket_policy_server.WebsocketPolicyServer(
	        policy=policy,
	        host="0.0.0.0",
	        port=args.port,
	        metadata=policy_metadata,
	    )  # 创建 WebSocket 策略服务器
	    server.serve_forever()      # 启动服务器，永远运行
	```
5. 在脚本的最后，使用 \`logging\` 模块配置日志记录，并调用 \`main\` 函数启动脚本，参数通过 \`tyro.cli\` 解析

#### 3.2.4 train\_test.py：训练和测试模型

#### 3.2.5 train.py：训练模型——损失函数计算、梯度下降、参数更新

这段代码是一个基于JAX的分布式训练脚本，集成了模型初始化、训练循环、日志记录、实验跟踪和检查点管理等功能。以下是对代码的模块化解读：

一开始先后涉及日志初始化 (\`init\_logging\`)、Weights & Biases 初始化 (\`init\_wandb\`)、权重加载与验证 (\`\_load\_weights\_and\_validate\`)

之后是训练状态初始化 (\`init\_train\_state\`)

1. 创建优化器（\`tx\`）和模型实例
2. 合并预训练参数（若有）到模型状态
3. 参数类型转换（如冻结参数转\`bfloat16\`）
4. 定义分布式分片策略（\`fsdp\_sharding\`）
5. 返回值：包含模型参数、优化器状态、EMA参数的\`TrainState\`对象及分片信息

再之后，是单步训练\`train\_step\`

1. 前向计算：模型计算损失(启用训练模式)，loss\_fn中调用的损失函数来自—— *1.2.4.3 损失函数compute\_loss：训练模型去噪的准确率(含训练数据集的来源介绍)*
	```php
	def train_step(
	    config: _config.TrainConfig,
	    rng: at.KeyArrayLike,
	    state: training_utils.TrainState,
	    batch: tuple[_model.Observation, _model.Actions],
	) -> tuple[training_utils.TrainState, dict[str, at.Array]]:
	    """执行单个训练步骤"""
	    # 合并模型定义和参数
	    model = nnx.merge(state.model_def, state.params)
	    model.train()  # 设置模型为训练模式
	 
	    @at.typecheck
	    def loss_fn(
	        model: _model.BaseModel, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions
	    ):
	        """损失函数"""
	        # 计算每个数据项的损失
	        chunked_loss = model.compute_loss(rng, observation, actions, train=True)
	        return jnp.mean(chunked_loss)  # 返回平均损失
	```
2. 随机数生成
	```cobol
	# 根据当前步数折叠随机数种子，确保每步使用不同随机数
	    train_rng = jax.random.fold_in(rng, state.step)
	 
	    # 解包批次数据
	    observation, actions = batch
	```
3. 梯度计算：通过\`nnx.value\_and\_grad\`获取梯度，仅更新可训练参数
	```cobol
	# 过滤出可训练参数
	    diff_state = nnx.DiffState(0, config.trainable_filter)
	 
	    # 计算损失和梯度
	    loss, grads = nnx.value_and_grad(loss_fn, argnums=diff_state)(model, train_rng, observation, actions)
	```
4. 参数更新：应用优化器更新，合并新参数到模型
	```csharp
	# 过滤出可训练参数
	    params = state.params.filter(config.trainable_filter)
	 
	    # 使用优化器更新参数
	    updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
	    new_params = optax.apply_updates(params, updates)
	 
	    # 更新模型参数并返回新的完整状态
	    nnx.update(model, new_params)
	    new_params = nnx.state(model)
	```
5. EMA维护：指数平滑更新关键参数
	```cobol
	# 创建新的训练状态，更新步数、参数和优化器状态
	    new_state = dataclasses.replace(state, step=state.step + 1, params=new_params, opt_state=new_opt_state)
	    if state.ema_decay is not None:
	        # 如果使用EMA，更新EMA参数
	        new_state = dataclasses.replace(
	            new_state,
	            ema_params=jax.tree.map(
	                lambda old, new: state.ema_decay * old + (1 - state.ema_decay) * new, state.ema_params, new_params
	            ),
	        )
	 
	    # 过滤出核心参数（不包括偏置、缩放等）
	    kernel_params = nnx.state(
	        model,
	        nnx.All(
	            nnx.Param,  # 必须是参数
	            nnx.Not(nnx_utils.PathRegex(".*/(bias|scale|pos_embedding|input_embedding)")),  # 排除特定名称
	            lambda _, x: x.value.ndim > 1,  # 必须是多维的
	        ),
	    )
	```
6. 指标收集：损失、梯度范数、参数范数（过滤非核参数）
	```csharp
	# 收集训练信息
	    info = {
	        "loss": loss,  # 损失值
	        "grad_norm": optax.global_norm(grads),              # 梯度范数
	        "param_norm": optax.global_norm(kernel_params),     # 参数范数
	    }
	    return new_state, info
	```

最后是主函数\`main\`

1. 环境初始化：日志、JAX配置、随机种子、设备分片
2. 数据准备：分布式数据加载器，分片策略（数据并行）
3. 状态恢复：检查点管理器处理恢复逻辑。
4. 训练循环：  
	JIT编译的分布式训练步骤（\`ptrain\_step\`）  
	定期日志记录（控制台 + W&B）  
	检查点保存（间隔保存 + 最终保存）
5. 清理：等待异步保存操作完成

// 待更

#### 3.2.6 scripts/docker

好的，下面是对 \`openpi-main/scripts/docker\` 目录的详细分析。这个目录通包含与 Docker 相关的脚本和配置文件，用于构建和管理 Docker 容器，具体而言，包含以下文件和子目录：

![](https://i-blog.csdnimg.cn/direct/afc0e1b9d3124282af4c61505410f0b1.png)

主要文件和功能如下所示

1. docker/compose.yml
2. docker/install\_docker\_ubuntu22.sh
3. docker/install\_nvidia\_container\_toolkit.sh
4. docker/serve\_policy.Dockerfile

// 待更

## 第四部分 策略适配接口：src下policy的全面分析与解读

src/openpi/policies目录包含以下文件：

BasePolicy (policy.py)  
├── Policy  
│ ├── BaseModel  
│ └── transforms.py  
├── AlohaPolicy (aloha\_policy.py)  
├── DroidPolicy (droid\_policy.py)  
└── LiberoPolicy (libero\_policy.py)

此外，每个特定机器人都有自己的策略文件，如

- aloha\_policy.py
- droid\_policy.py
- libero\_policy.py

这些文件定义了特定于机器人的输入和输出转换函数，处理数据格式、规范化和特定的转换需求

1. 比如每种机器人（ALOHA、DROID、LIBERO）的策略文件定义了特定的输入/输出转换类
2. 这些转换类作为 \`transforms\` 参数传递给 \`Policy\` 构造函数，例如，\`AlohaInputs\` 处理 ALOHA 机器人特有的状态和图像格式，\`AlohaOutputs\` 处理对应的输出转换

### 4.1 policy.py：实现了Policy类和 PolicyRecorder类

#### 4.1.1 Policy 类

policy.py 定义了基本的 \`Policy\` 类和 \`PolicyRecorder\` 类，它们继承自\`openpi\_client.base\_policy.BasePolicy\`

首先，做一系列初始化

```python
class Policy(BasePolicy):  # 定义Policy类，继承自BasePolicy

    def __init__(

        self,

        model: _model.BaseModel,  # 模型参数，必须是BaseModel的实例

        *,  # 之后的所有参数必须使用关键字传递

        rng: at.KeyArrayLike | None = None,  # 随机数生成器，可选

        # 输入转换函数序列，默认为空

        transforms: Sequence[_transforms.DataTransformFn] = (),  

        # 输出转换函数序列，默认为空

        output_transforms: Sequence[_transforms.DataTransformFn] = (),  

        # 传递给sample_actions的额外参数，可选

        sample_kwargs: dict[str, Any] | None = None,  

        metadata: dict[str, Any] | None = None,  # 元数据字典，可选

    ):

 

        # 使用JIT编译model的sample_actions方法提高性能

        self._sample_actions = nnx_utils.module_jit(model.sample_actions)  

 

        # 组合所有输入转换函数为一个函数

        self._input_transform = _transforms.compose(transforms)  

 

        # 组合所有输出转换函数为一个函数

        self._output_transform = _transforms.compose(output_transforms)  

        self._rng = rng or jax.random.key(0)       # 设置随机数生成器，如果未提供则创建一个新的

        self._sample_kwargs = sample_kwargs or {}  # 存储采样参数，如果未提供则使用空字典

        self._metadata = metadata or {}            # 存储元数据，如果未提供则使用空字典
```

其次，对于infer 方法——在策略内部流程上

1. 复制输入观察数据
	```ruby
	def infer(self, obs: dict) -> dict:  # type: ignore[misc]  # 推理方法，接收观察字典，返回动作字典
	        # 复制输入，因为转换可能会修改输入
	        inputs = jax.tree.map(lambda x: x, obs)  # 使用JAX树映射创建输入的深拷贝
	```
2. 应用输入转换  
	Policy.infer\` 方法首先应用输入转换：self.\_input\_transform，将客户端提供的观察转换为模型所需的格式
	```cobol
	inputs = self._input_transform(inputs)  # 应用输入转换函数处理输入数据
	```
3. 将数据转换为批处理格式并转为 JAX 数组
	```python
	# 将输入转换为批处理格式并转为jax数组
	        inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)  # 添加批次维度并转为JAX数组
	```
	生成新的随机数键
	```cobol
	self._rng, sample_rng = jax.random.split(self._rng)  # 分割随机数键以保持随机性
	```
4. 模型推理  
	调用模型的 \`sample\_actions\` 方法「 *该方法的实现，详见上文的 **1.2.4.4 推理函数 \`sample\_actions\`：基于扩散模型逆向采样，生成机器人动作序列*** 」进行推理，即获取动作预测
	```cobol
	outputs = {
	            "state": inputs["state"],  # 保留状态信息
	            "actions": self._sample_actions(sample_rng, _model.Observation.from_dict(inputs), **self._sample_kwargs),  # 使用模型生成动作
	        }
	```
5. 解除批处理并转换为 NumPy 数组
	```python
	# 移除批次维度并转换为NumPy数组
	        outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)  # 取第一个样本并转为NumPy数组
	```
6. 输出转换  
	最后应用输出转换 (\`self.\_output\_transform\`)，将模型输出转换为客户端期望的格式
	```php
	return self._output_transform(outputs)  # 应用输出转换并返回结果
	```

#### 4.1.2 \`PolicyRecorder\`

PolicyRecorder是一个装饰器类，它包装了一个基础策略，并在执行策略的同时将所有的输入和输出保存到磁盘，用于记录策略的行为

对于初始化函数：\`policy\`，涉及被包装的基础策略、record\_dir\`：保存记录的目录路径

对于infer 方法

1. 调用被包装策略的 \`infer\` 方法获取结果
2. 将输入和输出数据组织为字典
3. 使用 Flax 的 \`flatten\_dict\` 函数将嵌套字典展平
4. 构建输出文件路径
5. 将数据保存为 NumPy 数组文件
6. 返回策略结果

// 待更

### 4.2 policy\_config.py

policy\_config.py 定义了 \`PolicyConfig\` 类和 \`create\_trained\_policy\` 函数  
\`create\_trained\_policy\` 函数用于从训练好的检查点创建策略实例，加载模型参数、归一化统计数据，并配置转换函数

*相当于客户端代码会实例化一个 \`Policy\` 对象，通常是通过 \`create\_trained\_policy\` 函数，客户端通过调用 \`policy.infer(obs)\` 方法获取策略输出*

#### 4.2.1 PolicyConfig 数据类

\`PolicyConfig\` 是一个使用 \`@dataclasses.dataclass\` 装饰的数据类，用于存储创建策略所需的所有配置信息：

```python
# 定义策略配置类

class PolicyConfig:     

    model: _model.BaseModel      # 模型实例，必须是BaseModel类型

    norm_stats: dict[str, transforms.NormStats]        # 归一化统计信息，键是特征名称，值是归一化统计数据

 

    input_layers: Sequence[transforms.DataTransformFn]      # 输入数据转换函数序列

    output_layers: Sequence[transforms.DataTransformFn]     # 输出数据转换函数序列

 

    model_type: _model.ModelType = _model.ModelType.PI0     # 模型类型，默认为PI0

    default_prompt: str | None = None                  # 默认提示文本，可选

    sample_kwargs: dict[str, Any] | None = None        # 采样参数字典，可选
```

这个类主要是作为配置容器，将所有策略创建时需要的参数组织在一起

#### 4.2.2 create\_trained\_policy 函数

\`create\_trained\_policy\` 函数是从训练好的检查点创建可用策略的工厂函数

```rust
def create_trained_policy(

    train_config: _config.TrainConfig,       # 训练配置对象，包含训练时的所有参数设置

    checkpoint_dir: pathlib.Path | str,      # 检查点目录路径，可以是Path对象或字符串

    *,  # 强制后续参数使用关键字传递

    repack_transforms: transforms.Group | None = None,  # 可选的重新打包转换组

    sample_kwargs: dict[str, Any] | None = None,        # 采样参数，可选

    default_prompt: str | None = None,                  # 默认提示文本，可选

    norm_stats: dict[str, transforms.NormStats] | None = None,  # 归一化统计信息，可选

) -> _policy.Policy:                         # 返回类型是Policy对象
```

函数的核心流程是：

1. 处理输入参数，确保 \`repack\_transforms\` 不为空  
	且检查并可能下载检查点目录
	```cobol
	repack_transforms = repack_transforms or transforms.Group()      # 确保repack_transforms不为空，如果未提供则创建空Group
	    checkpoint_dir = download.maybe_download(str(checkpoint_dir))    # 检查并可能下载检查点目录
	```
2. 使用 \`train\_config\` 加载模型参数
	```cobol
	logging.info("Loading model...")  # 记录日志，表示正在加载模型
	 
	    # 加载模型参数并创建模型实例，使用bfloat16数据类型
	    model = train_config.model.load(_model.restore_params(checkpoint_dir / "params", dtype=jnp.bfloat16))
	```
3. 创建数据配置
	```python
	data_config = train_config.data.create(train_config.assets_dirs, train_config.model)  # 创建数据配置
	    if norm_stats is None:  # 如果未提供归一化统计信息
	        # 我们从检查点而非配置资源目录加载归一化统计信息，以确保策略使用与原始训练过程相同的归一化统计信息
	```
4. 如果未提供 \`norm\_stats\`，从检查点加载归一化统计信息
	```python
	if data_config.asset_id is None:  # 如果数据配置中没有asset_id
	            raise ValueError("Asset id is required to load norm stats.")  # 抛出异常，需要asset_id来加载归一化统计信息
	        norm_stats = _checkpoints.load_norm_stats(checkpoint_dir / "assets", data_config.asset_id)  # 从检查点加载归一化统计信息
	```
5. 构建并返回 \`Policy\` 实例，将所有转换函数组织为有序的处理流程：
	```csharp
	return _policy.Policy(  # 创建并返回Policy实例
	        model,  # 传入模型
	```
	输入处理：重新打包转换 → 注入默认提示 → 数据转换 → 归一化 → 模型特定转换
	```cobol
	transforms=[  # 输入转换函数序列
	            *repack_transforms.inputs,          # 展开重打包转换的输入部分
	            transforms.InjectDefaultPrompt(default_prompt),  # 注入默认提示
	            *data_config.data_transforms.inputs,   # 展开数据转换的输入部分
	            transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),      # 添加归一化转换
	            *data_config.model_transforms.inputs,  # 展开模型特定转换的输入部分
	        ],
	```
	输出处理：模型特定转换 → 反归一化 → 数据转换 → 重新打包转换
	```cobol
	output_transforms=[  # 输出转换函数序列
	            *data_config.model_transforms.outputs,     # 展开模型特定转换的输出部分
	            transforms.Unnormalize(norm_stats, use_quantiles=data_config.use_quantile_norm),          # 添加反归一化转换
	            *data_config.data_transforms.outputs,      # 展开数据转换的输出部分
	            *repack_transforms.outputs,           # 展开重打包转换的输出部分
	        ],
	        sample_kwargs=sample_kwargs,              # 设置采样参数
	        metadata=train_config.policy_metadata,    # 设置策略元数据
	    )
	```

\`create\_trained\_policy\` 函数是框架中连接训练过的模型与实际部署使用的关键桥梁，它通过组合各种转换函数，创建出可直接用于推理的 \`Policy\` 实例

### 4.3 policies/aloha\_policy.py

这段代码实现了一个用于 Aloha 策略的输入输出处理和数据转换的模块

#### 4.3.1 make\_aloha\_example：输入示例——状态向量、图像数据、文本prompt

首先，\`make\_aloha\_example\` 函数创建了一个随机的输入示例，包括一个14维的状态向量和四个摄像头的图像数据（高、低、左腕、右腕视角），以及一个文本提示信息

```cobol
# 定义一个函数，创建Aloha策略的随机输入示例

def make_aloha_example() -> dict:  

    # 返回一个字典，包含状态、图像和提示信息

    return {  

        # 创建一个14维的状态向量，所有值为1

        "state": np.ones((14,)),  

 

        # 创建一个包含四个摄像头图像的字典

        "images": {  

            # 高位摄像头图像

            "cam_high": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),  

 

            # 低位摄像头图像

            "cam_low": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8), 

 

            # 左手腕摄像头图像 

            "cam_left_wrist": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),  

 

            # 右手腕摄像头图像

            "cam_right_wrist": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),  

        },

        "prompt": "do something", 

    }
```

这些数据将用于测试和验证 Aloha 策略的输入处理

> 可能有的同学对上面的4个摄像头有疑问，简单，详见此文《 [一文通透动作分块算法ACT：斯坦福ALOHA团队推出的动作序列预测算法(Action Chunking with Transformers)](https://blog.csdn.net/v_JULY_v/article/details/135454242 "一文通透动作分块算法ACT：斯坦福ALOHA团队推出的动作序列预测算法(Action Chunking with Transformers)") 》的「1.2 硬件套装：ALOHA——低成本的开源硬件系统，用于手动远程操作」
> 
> ---
> 
> 如下图所示
> 
> ![](https://i-blog.csdnimg.cn/blog_migrate/115485854b4046f62f250e1f26198e2b.png)
> 
> - 左侧为前、顶部和两个手腕摄像机的视角( *这4个相机的视角分别用 从当前往后的蓝线 、 从顶向下的绿线 、 从左往右的红线 、* *从右往左的红线* *表示* )，以及ALOHA双手工作空间的示意图  
> 	  
> 	具体而言，总计4个Logitech C922x网络摄像头，每个流输出480×640 RGB图像  
> 	其中两个网络摄像头安装在跟随机器人手腕上，以提供夹具的近距离视角( *allowing for a close-up view of the grippers* )  
> 	剩下的两个相机分别安装在桌面的前方(front camera)和桌子上方的顶部位置(top camera)，遥控操作和数据记录均以50Hz频率进行

#### 4.3.2 AlohaInputs：定义Aloha 策略的输入数据结构

接下来，\`AlohaInputs\` 类定义了 Aloha 策略的输入数据结构

```python
class AlohaInputs(transforms.DataTransformFn):  # 定义AlohaInputs类，继承自transforms.DataTransformFn

    """Inputs for the Aloha policy.

    # 预期输入格式

    # 图像字典，键是名称，值是形状为[channel, height, width]的图像

    - images: dict[name, img]

     # 状态向量，长度为14

    - state: [14] 

    # 动作矩阵，形状为[action_horizon, 14]

    - actions: [action_horizon, 14]  

    """

 

    # 模型的动作维度，将用于填充状态和动作

    action_dim: int  # 动作维度

 

    # 如果为True，将关节和夹持器值从标准Aloha空间转换为pi内部运行时使用的空间

    # pi内部运行时使用的空间用于训练基础模型

    # 是否适配pi内部运行时，默认为True

    adapt_to_pi: bool = True  

 

    # 预期的摄像头名称，所有输入摄像头必须在此集合中。缺失的摄像头将用黑色图像替代

    # 缺失的摄像头将用黑色图像替代，对应的\`image_mask\`将设置为False

    # 预期的摄像头名称集合

    EXPECTED_CAMERAS: ClassVar[tuple[str, ...]] = ("cam_high", "cam_low", "cam_left_wrist", "cam_right_wrist")
```
1. 这个类使用 \`dataclasses.dataclass\` 装饰器来简化类的定义，并确保实例是不可变的（\`frozen=True\`）
2. 类中定义了输入数据的预期格式，包括图像、状态和动作数据

\_\_call\_\_ 方法，实现了对Aloha策略输入数据的标准化处理。该方法将 **原始输入数据转换为模型可接受的格式** ，包括多项关键处理步骤，比如进行必要的解码和填充操作，并检查图像数据是否包含预期的摄像头视角

1. 首先，方法通过调用\`\_decode\_aloha\`函数对输入数据进行初步解码，根据\`adapt\_to\_pi\`参数决定是否将数据适配到π内部运行时环境
	```cobol
	# 定义__call__方法，处理输入数据
	    def __call__(self, data: dict) -> dict:  
	 
	        # 解码Aloha数据，根据adapt_to_pi参数进行适配
	        data = _decode_aloha(data, adapt_to_pi=self.adapt_to_pi)
	```
	这一步主要处理状态向量以及将图像格式从\`\[channel, height, width\]\`转换为\`\[height, width, channel\]\`
2. 接着，方法将14维的状态向量使用零填充扩展到模型所需的动作维度(\`action\_dim\`)
	```cobol
	# 获取状态数据，将其从14维填充到模型的动作维度
	        # 使用transforms.pad_to_dim函数填充状态数据
	        state = transforms.pad_to_dim(data["state"], self.action_dim)
	```
	随后，进行输入图像的验证：检查输入图像的键集合是否超出了预期的摄像头列表范围，若发现未知摄像头视角则抛出\`ValueError\`
	```python
	# 获取输入图像数据
	        in_images = data["images"]  
	 
	        # 检查输入图像是否包含所有预期的摄像头
	        if set(in_images) - set(self.EXPECTED_CAMERAS):  
	            # 如果缺少预期的摄像头，抛出异常
	            raise ValueError(f"Expected images to contain {self.EXPECTED_CAMERAS}, got {tuple(in_images)}")
	```
3. 在构建输出字典时，方法首先假定"cam\_high"（高视角摄像头）图像必定存在
	```csharp
	# 假设基础图像总是存在，获取高位摄像头图像
	        base_image = in_images["cam_high"]
	```
	并将其作为基础图像（\`base\_0\_rgb\`）
	```csharp
	# 创建图像字典
	        images = {  
	            # 基础图像
	            "base_0_rgb": base_image,  
	        }
	```
	同时创建了相应的图像掩码字典，标记该图像为有效
	```cobol
	# 创建图像掩码字典
	        image_masks = {  
	            # 基础图像掩码为True
	            "base_0_rgb": np.True_,  
	        }
	```
4. 对于其他摄像头视角（左腕和右腕），方法使用映射关系字典进行处理：
	```csharp
	# 添加额外的图像
	        # 额外图像名称映射
	        extra_image_names = {  
	            # 左手腕图像
	            "left_wrist_0_rgb": "cam_left_wrist",  
	 
	            # 右手腕图像
	            "right_wrist_0_rgb": "cam_right_wrist",  
	        }
	```
	如果相应的源图像存在，则将其添加到输出图像字典并标记为有效；
	```bash
	# 遍历额外图像名称映射
	        for dest, source in extra_image_names.items():  
	 
	            # 如果输入图像中包含该图像
	            if source in in_images:  
	                # 添加到图像字典
	                images[dest] = in_images[source]  
	 
	                # 设置图像掩码为True
	                image_masks[dest] = np.True_
	```
	若不存在，则创建一个与基础图像相同大小的全零图像（黑图），并标记为无效
	```cobol
	# 如果输入图像中不包含该图像
	            else:  
	                # 用黑色图像替代
	                images[dest] = np.zeros_like(base_image)  
	 
	                # 设置图像掩码为False
	                image_masks[dest] = np.False_
	```
	这种处理方式确保了模型在缺失某些视角图像时仍能正常工作
	```perl
	# 创建输入字典
	        inputs = {  
	            "image": images,              # 图像数据
	            "image_mask": image_masks,    # 图像掩码
	            "state": state,               # 状态数据
	        }
	```
5. 方法还会处理训练时特有的数据，如动作序列  
	若输入数据包含"actions"字段，则将其转换为NumPy数组，应用\`\_encode\_actions\_inv\`进行编码转换，并使用零填充扩展到模型动作维度
	```cobol
	# 动作数据仅在训练期间可用
	        # 如果输入数据中包含动作数据
	        if "actions" in data:  
	             # 将动作数据转换为NumPy数组
	            actions = np.asarray(data["actions"]) 
	 
	            # 编码动作数据，根据adapt_to_pi参数进行适配
	            actions = _encode_actions_inv(actions, adapt_to_pi=self.adapt_to_pi)  
	 
	            # 填充动作数据到模型的动作维度
	            inputs["actions"] = transforms.pad_to_dim(actions, self.action_dim)
	```
	最后，如果输入包含"prompt"文本提示，也会将其添加到输出字典中，然后返回处理后的输入数据
	```csharp
	# 如果输入数据中包含提示信息
	        if "prompt" in data:  
	            # 添加提示信息到输入字典
	            inputs["prompt"] = data["prompt"]  
	 
	        # 返回处理后的输入数据
	        return inputs
	```

整体而言，这个方法实现了从多样化的原始输入到标准化模型输入的转换流程，处理了数据格式转换、缺失数据补充、维度调整等核心问题，确保了Aloha策略模型能够接收一致的输入格式，从而实现稳定的推理和训练

#### 4.3.3 AlohaOutputs：定义Aloha 策略的输出数据结构

\`AlohaOutputs\` 类定义了 Aloha 策略的输出数据结构，同样使用 \`dataclasses.dataclass\` 装饰器

```python
# 定义AlohaOutputs类，继承自transforms.DataTransformFn

class AlohaOutputs(transforms.DataTransformFn):  

 

    # 如果为True，将关节和夹持器值从标准Aloha空间转换为pi内部运行时使用的空间

    # pi内部运行时使用的空间用于训练基础模型

    adapt_to_pi: bool = True  # 是否适配pi内部运行时，默认为True
```

\`\_\_call\_\_\` 方法处理输出数据，仅返回前14个维度的动作数据，并进行必要的编码转换

```cobol
# 定义__call__方法，处理输出数据

    def __call__(self, data: dict) -> dict:  

        # 仅返回前14维的动作数据，即将动作数据转换为NumPy数组，并取前14维

        actions = np.asarray(data["actions"][:, :14])  

 

        # 编码动作数据并返回字典

        return {"actions": _encode_actions(actions, adapt_to_pi=self.adapt_to_pi)}
```

#### 4.3.4 多个辅助函数：数据的标准化、反标准化、关节角度翻转

此外，代码中还包含多个辅助函数，用于数据的标准化、反标准化、关节角度翻转、夹持器位置的线性和角度转换等  
  
这些函数确保了数据在不同控制系统之间的兼容性和一致性

// 待更

## 第五部分 examples ：各种机器人平台及策略客户端的示例实现

根据π0对应examples模块的结构

![](https://i-blog.csdnimg.cn/direct/f921f22cfa17438886e495257d20695d.png)

其涉及以下模块

1. aloha\_real/：真实机器人ALOHA的示例
2. aloha\_sim/：ALOHA模拟器的示例
3. droid/：DROID机器人的示例
4. libero/：LIBERO基准测试的示例
5. simple\_client/：简单客户端的示例
6. ur5/：UR5机器人的示例
7. inference.ipynb：推理示例的Jupyter Notebook
8. policy\_records.ipynb：策略记录示例的Jupyter Notebook

// 待更