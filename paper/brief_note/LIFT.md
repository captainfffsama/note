---
tags:
  - "#VLA"
  - "#力觉"
---
# Never Too Late for Force: Accelerating VLA Post-Training with Reactive Force Injection
- 论文：<https://arxiv.org/abs/2607.14236v2>
- 代码：<https://github.com/y-wng/lift>
- 翻译：[LIFT_2607.14236v2_zh](../paper_trans/LIFT/LIFT_2607.14236v2_zh.md)

## 动机

结合力进行后训练要关注三个问题：

1. 力频率高维度低，如何注入？
2. 如何保护预训练好的 VLA？
3. 力分布和具体任务有关，离线训练数据可能无法覆盖所有力作用状态怎么办？

## 相关工作
1. RDP,ImplicitRDP
2. FA-RDP 则利用学习得到的多模态信息来调整信息处理频率与采样策略，从而在保证多样化的预接触动作的同时实现快速的反应。
3. ForceVLA,TA-VLA,
4. ForceVLA 2 则结合了力数据与多尺度融合技术，生成混合形式的力 - 位置输出
5. FAVLA 将力的处理过程与动作的执行时机联系了起来：它将一个处理速度较慢的视觉 - 语言模型与一个处理速度较快的“力处理专家”系统相结合，并利用预测到的力变化来决定模型的更新时机

## 方法
### 三个目标
1. 将近期力注入 policy，能对接触事件有反应，而非开环执行动作块
2. 保持模型泛化能力
3. 能够联合利用视觉任务对齐数据和力反馈在线修正数据

### 目标 1 解法
![fig2_architecture](../paper_trans/LIFT/figures/fig2_architecture.png)
