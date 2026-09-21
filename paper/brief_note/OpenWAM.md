---
tags:
  - "#具身智能"
  - "#VLA"
  - "#世界模型"
source: https://arxiv.org/abs/2609.07398v1
---
# OpenWAM: An Open, Modular Exploration Towards Systematic World-Action Model Pretraining
- 论文：<https://arxiv.org/abs/2609.07398v1>
- 代码：<https://github.com/OpenWAM-Official/OpenWAM>

# 核心贡献

WAM 需要解决三个问题：

1. WAM 应该继承什么世界知识
2. 如何构建世界宇动作学习之间的协同
3. 协同效应如何跨领域扩展
通过 OpenWAM-Study 研究的三个核心发现：
- Backbone 能力够，视觉特征够紧密丰富可以有效利用上游世界知识
- 单靠叠参数来涌现不靠谱，还是需要动作部分专用的模块容量够，有显式的世界和动作信息交互，以及联合去噪来促成
- 具身预训练主要改善 OOD 性能，分布内拟合不是主要目的，人类第一人称视角的视频拓宽了世界覆盖面，机器人轨迹提供了可执行的动作知识

# 相关工作

当前对于 WAM 中哪些模块负责传递世界知识，哪些交互负责世界知识和动作协同，哪些知识可以跨领域保持，仍不明确。

StarVLA 提供了 VLA 模块化和高性能的设计选择，XPolicyLab 则为 policy 评估和部署贡献了规范。

多模态领域：

- Cambrian-l 和 Beyond Language Modeling 系统研究了视觉表征、模态专用容量、数据组成与统一预训练。
- Towards Physics of Multimodal Pretraining 进一步隔离了知识流、协同与竞争、以及模态统一的时机

# OpenWAM-Infra