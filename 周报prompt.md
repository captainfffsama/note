 以下是我 25 年 3 月 28 日到 4 月 3 日的工作内容简介：

```
2025.03.28~2025.04.03

1. 完成19篇具身智能论文材料的优化
2. 精读OpenVLA论文，了解OpenVLA模型细节，形成笔记，并发给其他同事交流。为具身智能模型设计提供了思路
3. 精读Pi0论文，了解目前我们模型设计上的思路细节
4. 针对具身智能模型设计要求，思考并调研并汇总了诸如RADIO，Recon，spatialLM等相关工作，形成脑图，协助顾工细化具身智能架构中的一些细节设计。
5. 对接C3实验中心，了解硬件环境情况，梳理缺失的部分底层依赖库，离线升级了显卡驱动，成功部署了conda 
```

结合以上工作内容简单总结，我撰写了如下给领导的，25 年 3 月 28 日到 4 月 3 日的周报：

```
# 本周完成工作:

1. 完成19篇具身智能论文材料的优化
2. 精读OpenVLA论文，了解OpenVLA模型细节，形成笔记，并发给其他同事交流。为具身智能模型设计提供了思路
3. 精读Pi0论文，了解目前我们模型设计上的思路细节
4. 针对具身智能模型设计要求，思考并调研并汇总了诸如RADIO，Recon，spatialLM等相关工作，形成脑图，协助顾工细化具身智能架构中的一些细节设计。
5. 对接C3实验中心，了解硬件环境情况，梳理缺失的部分底层依赖库，离线升级了显卡驱动，成功部署了conda

# 本周工作总结:

具身智能材料：

1. 优化19篇具身智能论文汇总材料，优化部分论文架构解释，补充部分论文工作，整理下载原始论文

2.  结合我们的具身智能模型设计方向，梳理其中VLM模型的模块在设计时可能要考虑的一些潜在要点，整理相关的论文，并用脑图记录。

3. 和欧阳，顾工针对网络设计进行讨论，结合2中自己的梳理和思考，对顾工的初版模型架构图进行具体化，确定每个模型具体使用什么模型，是否需要信息压缩等等。

算法设计和研发：

1. 精读 OpenVLA，熟知其中算法细节和原始作者决策，形成笔记，并发给其他同事交流，为后续算法优化做准备

算法训练：

1. 和C3实验中心对接，了解并记录硬件和系统环境，并形成说明文档。识别并补齐一些环境依赖，成功离线升级显卡驱动，并部署conda。 下周工作计划:
2. 完成Pi0算法docker镜像打包测试，并导入A800环境
3. 完成Pi0初版训练
4. 针对我们确定的模型架构设计，在Pi0基础上进行修改，制定实验计划，进行消融实验以确认设计有效性

# 需协调与帮助:

# 编译算法docker镜像，需要pull的算法基镜像通常比较大，纯torch镜像也有9个多G，手机热点流量不够用。
```

请仿照上面周报的格式和文风，结合我 25 年 6 月 30 日到 7 月 04 日的工作内容简介，写一篇给领导的工作周报。以下是我 25 年 6 月 30 日到 7 月 04 日的工作内容简介：

```
2025.06.30~2025.07.04
1. 测试最新调整loss计算方式的smolvla，以及pi0模型，pi0模型性能仍然存在问题，smolvla优化loss版本的模型，抓取效果有提升，由于机械臂过流问题，未能完全测试挂的效果
2. 协助排查smolvla推理代码中的夹爪问题，机械臂传参问题，目前已经修复推理过程中夹爪问题，机械臂过流可以确认和传参无关。
3. 冻结pi0大部分参数，仅训练action expert部分，进行消融实验。
4. 梳理pi0代码，发现官方最新代码中的bug，修复其中问题，重新训练了pi0基线，目前看loss有较大程度下降，但目前机械臂故障没有测试环境。
5. 基于修复bug的代码，重做3中的实验。
6. 阅读PCP-MAE论文并梳理总结，后续尝试给smolvla或pi0加入更好的点云backbone，并支持加载预训练权重。

下周计划：
1. 搭建PCP-MAE环境，梳理阅读代码
2. 尝试将PCP-MAE 接入smolvla
3. 调研了解 Improving Vision-Language-Action Model with Online Reinforcement Learning 和 VLA-RL，形成记录文档，学习如何将强化学习应用 VLA 的思路
```