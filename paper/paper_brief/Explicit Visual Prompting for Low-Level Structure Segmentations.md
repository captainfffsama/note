# 其他信息
- 会议：CVPR 2023
- 论文： <https://arxiv.org/abs/2303.10883>
- 代码：<https://github.com/NiFangBaAGe/Explicit-Visual-Prompt>

# 论文内容

## 关键信息摘抄
### Introcduction

作者认为预训练模型已经包含有足够多的语义信息, 因此通过引入单个个体图像的低级特征, 作为将模型迁移到特定任务上的特定知识可能更有效.

本文中作者考虑两种引入特征, 一是冻结的 patch 嵌入, 这对于改变原始模型的分布很重要; 二是图像的高频信息成分,, 因为预训练的模型是可以通过数据增强来习得这些特征的不变性.