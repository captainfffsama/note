# PhysHSI: Towards a Real-World Generalizable and Natural Humanoid-Scene Interaction System

#具身智能 #manipulation #强化学习 

- 论文：[[2510.11072v1] PhysHSI: Towards a Real-World Generalizable and Natural Humanoid-Scene Interaction System](https://arxiv.org/abs/2510.11072v1)
- 代码：[InternRobotics/PhysHSI: Official implementation of the paper: "PhysHSI: Towards a Real-World Generalizable and Natural Humanoid-Scene Interaction System"](https://github.com/InternRobotics/PhysHSI)
- 项目：[PhysHSI: Towards a Real-World Generalizable and Natural Humanoid-Scene Interaction System](https://why618188.github.io/physhsi/)

# 动机

构建人机场景交互 (humanoid-scene interaction, HSI) 难度大。现有方法中：基于运控或者轨迹优化，计算成本高，前提假设多。基于 RL 的方法，若从头学 policy，奖励设计负担大；使用动捕（MoCap）做先验依赖于完美场景观测，现实观测不全，sim2real 难。

# 方法

![](../../Attachments/PhysHSI_fig2.png)

整体方法分三部分：

1. 数据准备
2. AMP 训练
3. 真实世界迁移
## 数据准备

数据重定向方法比较简单，通过缩放优化将 AMASS 和 SAMP 的 SMPL 动作定位到人形上，用平滑滤波抑制抖动，得到数据集 $M_{Robo}$ 

手动标注接触起始帧（拾取 $\phi_1$ ,放置 $\phi_2$ ），拾取前，物体设定一个位置，携带中，设置物体位置在双手中点，方向和机器人基座对齐，放置后，物体再设定一个位置。这样产生一个增强的人形数据集 $M$ ,其中其对象位置一致且物理上连贯

# 参考
- [PhysHSI——搬运箱子，不在话下：仿真中AMP训练，现实中结合内置的LiDAR和头部的外置相机做视觉感知、定位-CSDN博客](https://blog.csdn.net/v_JULY_v/article/details/153319617?spm=1001.2014.3001.5502)