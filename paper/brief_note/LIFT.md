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

> 图 2

如图，预训练 PI 05 上额外接一个同权重的 action expert。力通过 GRU 编码成 token 通过 cross attention 引入。

#### Reactive action expert 的具体推理方式

依据附录 A，设置超参 L=3，即反应动作 $r_i$ 可以读取的 GRU encoder 的 embedding 是 $m_0,…,m_{i-3}$ .即进行完整的一次 VLM+ 前向去噪的大致延迟。

实际推理中，设置 action chunk H=10，在开始 t 0 时，设置 6 D wrench history 的 tensor 为 1 x 10 x 6，第 0 位是真实的 $F_0$ ，剩下 9 位占位符，推理生成原始 pi 05 的基础动作 $a_{0-9}$ 和反应动作 $r_{0-9}$ ,先顺序执行 $r _0,r_ 1,r_ 2$ ,得到 $F_1,F_2,F_3$ 。然后更新 6 D wrench 为$F_0，F_1，F_2，F_3$ ，剩下 6 个占位符。

然后复用 VLM 的 KV cache，推理去噪得到新一轮的 action chunk 基础动作 $a^{(1)}_{0-9}$ 和反应动作 $r^{(1)}_{0-9}$ 。然后执行新得到的 $r^{(1)}_3$ ,得到 $F_4$ ,然后再次更新 6 D wrench 为 $F_{0-4}$ ，剩下 5 个占位符，再次复用 VLM KV Cache，去噪等到新的 action chunk.

即**调用过程**如下：

| 轮次  | 传入的真实力前缀            | 模型内部重新计算          | Policy 参数             | 实际取出的动作             |
| --- | ------------------- | ----------------- | --------------------- | ------------------- |
| 初次  | `F0` + 9 个占位        | `a⁽⁰⁾0～9、r⁽⁰⁾0～9` | `current_step=0, k=3` | `r⁽⁰⁾0、r⁽⁰⁾1、r⁽⁰⁾2` |
| 第二次 | `F0～F3` + **6** 个占位 | `a⁽¹⁾0～9、r⁽¹⁾0～9` | `current_step=3, k=1` | `r⁽¹⁾3`             |
| 第三次 | `F0～F4` + **5** 个占位 | `a⁽²⁾0～9、r⁽²⁾0～9` | `current_step=4, k=1` | `r⁽²⁾4`             |
| 第四次 | `F0～F5` + 4 个占位     | `a⁽³⁾0～9、r⁽³⁾0～9` | `current_step=5, k=1` | `r⁽³⁾5`             |

两个 action expert 的注意力方式如下：

![fig3_shifted_causal_attention](../paper_trans/LIFT/figures/fig3_shifted_causal_attention.png)> 图 3

绿色 1 区是原始 pi 05 的注意力方式。因为一开始是权重复制的，所以 r=a。注意这里 黄色 3 区中 $r_0$ 不关注 $a_0$ ，是为了保持和原始 pi 05 一样的注意力上下文。

这样在进行力预训练之前，反应式专家的输出 r 和原始 action expert 是一样的。

图 2 中 cross-attention 之后的 $W_o$ 设置为 0，这样输出就和原始 action expert 一样。

### 目标 3 异构数据训练

纯视觉数据使用简易的类 UMI 方案采集。带力觉数据是通过已部署策略 DAgger 收集的。

训练时，原始动作流和反应式共享参数，FM 独立采样噪声，但是去噪时间步调度一致。

## 实验结论
1. 论文中 GRU 输入的 history 为 0，即 GRU 从 0 状态开始，那么调用的就是无反应力的 LIFT。单帧力极易受到瞬时噪声或分布外冲击力的误导。
2. 残差式网络效果有限，弱的 base policy 需要大幅修正。
3. DAgger 的时候，接管时机不好判断，实验仅在单臂上进行。

## 硬件设置

Robotiq 2 F-85 平行爪夹爪的 Flexiv Rizon 4 S 7-DoF 协作机器人执行。机械臂在 10 Hz 下以位置控制模式运行。来自策略的轨迹目标由机器人内部的阻抗控制器进行跟踪。

内置 6 D 末端力\力矩传感，原始超 1000 Hz，降采样到 10 Hz，腕部 IPhone 以 10 Hz 采集 RGB。单相机。

遥操 SDK 就是厂家的 Flexiv TDK。主从遥操，从臂力会给到主臂。

## 训练超参
#### 表 1：网络架构超参数 (Table 1: Architecture hyperparameters)

| 参数项 (Parameter) | 设定值 (Value) |
| :--- | :--- |
| 主干网络 (Backbone) | 基于 PaliGemma 视觉 - 语言大模型与动作专家的 $\pi_{0.5}$ |
| 动作分块时域 (Action chunk horizon) | 10 |
| 对齐动作维度 (Padded action dimension) | 32 |
| 力觉输入维度 (Force input dimension) | 6D 力与力矩旋量 (6D wrench) |
| 力编码器结构 (Force encoder) | 单层因果 GRU 串联线性投影层 |
| GRU 隐层维度 (Force encoder hidden width) | 512 |
| 力特征输出维度 (Force encoder output width) | $d_m = 1024$ |
| 时延对齐偏移量 (Latency-alignment offset) | 3 |
| 反应式动作专家初始化 (Reactive expert initialization) | 从基础动作专家权重完全复制 |
| 交叉注意力初始化 (Cross-attention initialization) | 输出投影层零初始化 (Zero-initialized) |

#### 表 2：优化与训练超参数 (Table 2: Optimization hyperparameters)

| 参数项 (Parameter) | 设定值 (Value) |
| :--- | :--- |
| 批处理大小 (Batch size) | 32 |
| 总训练步数 (Total training steps) | 30,000 |
| 学习率调度策略 (Learning-rate schedule) | 10,000 步预热 (warmup)，余弦退火衰减 |
| 峰值与最终学习率 (Peak and final learning rate) | $5 \times 10^{-5}$；衰减周期 1,000,000 步 |
| 优化器 (Optimizer) | AdamW |
| AdamW $\beta_1$ | 0.9 |
| AdamW $\beta_2$ | 0.95 |
| AdamW $\varepsilon$ | $10^{-8}$ |
| 权重衰减系数 (Weight decay) | $10^{-10}$ |
| 梯度裁剪阈值 (Gradient clipping norm) | 1.0 |

8 xA 800 训练。

## 代码模块

![](../../Attachments/LIFT%20网络总览与接口.drawio.svg)
