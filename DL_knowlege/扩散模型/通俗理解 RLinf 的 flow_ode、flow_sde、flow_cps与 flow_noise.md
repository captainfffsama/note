---
tags:
  - "#RLinf"
---

# 通俗理解 RLinf 的 `flow_ode`、`flow_sde`、`flow_cps` 与 `flow_noise`

> 本文面向希望读懂 RLinf Flow Matching 强化学习实现的读者。分析基于仓库 commit `eff293a7`，核心代码是 `rlinf/models/embodiment/openpi/openpi_action_model.py`。本文使用 OpenPI 的时间方向：`t=1` 是纯噪声端，`t=0` 是动作端。GR 00 T 的时间方向相反，但背后的思想相同。

## 1. 一句话理解四种方法

可以把 Flow Matching 想成“从一团随机噪声出发，按照导航器给出的速度，走到一个可执行动作”。四种方法的区别是：**每走一步时，要不要随机晃一下，以及晃动幅度从哪里来。**

| 方法 | 通俗理解 | 去噪步是否随机 | 噪声大小由谁决定 | 转移均值是否同时调整 |
|---|---|---:|---|---:|
| `flow_ode` | 完全按照导航路线走 | 否 | 标准差为 0 | 否 |
| `flow_sde` | 按理论设计的时间表随机走，并修正主路线 | 是 | 时间、步长和 `noise_level` | 是 |
| `flow_cps` | 用正弦/余弦旋钮混合确定性路线与随机扰动 | 是 | 时间和 `noise_level` | 是 |
| `flow_noise` | 让一个小网络根据当前情况决定每个维度该晃多少 | 是 | `ExploreNoiseNet` 学习 | 否，保留 ODE 均值 |

若只记一个选择原则：

- 想要确定性推理或基线：`flow_ode`；
- 想要结构明确、无需再训练噪声网络的 RL 探索：`flow_sde`；
- 想试一种平滑、无除零奇点的解析扰动：`flow_cps`；
- 想让探索强度随状态、时间和动作维度自适应：`flow_noise`。

## 2. 先理解 Flow Matching 在做什么

### 2.1 从噪声走到动作

设：

- `a`：数据中的真实动作；
- `epsilon ~ N(0,I)`：与动作形状相同的单位高斯噪声；
- `t in [0,1]`：流的时间；
- `x_t`：时间 `t` 的中间状态。

OpenPI 训练时使用线性路径：

```text
x_t = t * epsilon + (1 - t) * a
目标速度 u_t = epsilon - a
```

所以：

- `t=1` 时，`x_t=epsilon`，是纯噪声；
- `t=0` 时，`x_t=a`，是目标动作；
- 模型学习在任意中间位置应该沿哪个速度方向移动。

训练代码见 `rlinf/models/embodiment/openpi/openpi_action_model.py:423-428`。

### 2.2 基础噪声与 RL 探索噪声不是一回事

即使选择 `flow_ode`，生成过程通常仍从 `x_1 ~ N(0,I)` 开始。这个随机初态是生成模型的**源分布**。

本文比较的四种方法，主要是在讨论每个离散去噪步骤的转移：

```text
x_t  ->  x_(t-Delta)
```

是否还要加入额外的、能计算概率密度的随机性。不要把“初始单位高斯”与“去噪步骤中的 RL 探索噪声”混为一谈。

## 3. 四种方法共用的计算骨架

### 3.1 时间网格

假设共有 `N` 个去噪步骤，OpenPI 构造：

```text
t_0, t_1, …, t_N = 1, 1-1/N, …, 1/N, 0
Delta = t_i - t_(i+1) > 0
```

代码中的 `num_steps` 决定 `N`。

### 3.2 模型先预测速度

给定当前状态 `x_t`、观测和时间 `t`，动作专家预测速度：

```text
v_t = velocity_model(observation, x_t, t)
```

根据线性流的几何关系，代码进一步构造两个端点估计：

```text
x0_pred = x_t - t * v_t
x1_pred = x_t + (1 - t) * v_t
```

这里：

- `x0_pred` 是模型认为最终动作端 `t=0` 在哪里；
- `x1_pred` 是模型认为噪声端 `t=1` 在哪里。

### 3.3 每种方法只是在定义不同的 `mu` 和 `sigma`

四种方法最终都放进同一个更新框架：

```text
x_(t-Delta) = mu(x_t,t) + sigma(x_t,t) * z
z ~ N(0,I)
```

- `mu`：下一步最可能落到的位置；
- `sigma`：探索强度；
- `z`：实际采样的标准高斯噪声。

当 `sigma>0` 时，条件转移是显式高斯分布，RL 可以计算：

```text
log p(x_(t-Delta) | x_t, observation)
```

PPO/GRPO 用新旧策略下的这个 log probability 形成概率比，从而训练 Flow Matching 策略。

核心分支见 `openpi_action_model.py:1122-1153`；真正采样发生在 `openpi_action_model.py:1037-1040`。

## 4. `flow_ode`：不在去噪步骤中额外加噪

### 4.1 通俗原理

把速度模型想成导航器。`flow_ode` 每一步都严格沿导航器给出的方向前进，没有横向随机晃动。

它解决的是常微分方程：

```text
dx/dt = v_theta(x,t,observation)
```

代码使用离散 Euler 步近似求解。

### 4.2 计算公式

令下一时刻是 `t_next=t-Delta`：

```text
mu_ode = [1 - t_next] * x0_pred + t_next * x1_pred
sigma_ode = 0
```

代入端点预测后可以化简成：

```text
mu_ode = x_t - Delta * v_t
x_(t-Delta) = mu_ode
```

这正是逆时间 Euler 更新。代码见 `openpi_action_model.py:1127-1130`。

### 4.3 逐步计算

1. 输入当前 `x_t`、观测和 `t`。
2. 模型预测速度 `v_t`。
3. 计算 `x0_pred` 和 `x1_pred`。
4. 按 `t-Delta` 对两个端点预测做线性组合，得到 `mu_ode`。
5. 设置 `sigma=0`。
6. 下一状态直接等于 `mu_ode`，不进行额外高斯采样。
7. 重复直到 `t=0`，得到动作。

### 4.4 特点

优点：

- 去噪路径最稳定、最容易复现；
- 没有额外探索方差，评估结果波动较小；
- 计算最简单，不需要噪声网络；
- 是其他三种方法的基准均值路径。

局限：

- `sigma=0`，该步不是普通的连续高斯随机策略；
- 代码把确定性维度的 log probability 设为 0，因此这些步骤本身不能像随机高斯步骤那样提供 PPO 概率比；
- 在需要主动探索未知动作的在线 RL 中，仅靠源分布随机性可能不足。

### 4.5 适用场景

- 验证、部署或 benchmark 评估；
- 想观察预训练 Flow Matching 策略本身的能力；
- 调试速度场、时间方向和动作归一化；
- 作为 `flow_sde`、`flow_cps`、`flow_noise` 的对照实验；
- `joint_logprob=False` 时，未被选中加噪的其他去噪步骤；
- NFT 等不依靠逐步高斯探索的训练路径。

不太适合单独承担需要强探索的 PPO/GRPO 在线训练。

## 5. `flow_sde`：解析噪声调度 + 漂移修正

### 5.1 通俗原理

`flow_sde` 不只是“在 ODE 结果后面撒点高斯噪声”。它同时做两件事：

1. 加一个随时间变化的随机扩散项；
2. 修改确定性均值，也就是修正漂移。

可以类比成：为了探索，司机会随机左右偏一点；但导航器同时修正主路线，使整体交通流仍尽量保持原来从噪声到动作的分布演化。

这就是从 probability-flow ODE 构造随机 SDE 的核心思想：**增加扩散时，必须配套修改漂移，不能只加噪声而不改均值。**

### 5.2 代码中的噪声调度

OpenPI 定义：

```text
g(t) = noise_level * sqrt(t / (1 - t))
step_std = sqrt(Delta) * g(t)
```

其中：

- `noise_level` 是整体噪声旋钮；
- `sqrt(t/(1-t))` 让噪声端附近的扩散更强、动作端附近更弱；
- `sqrt(Delta)` 是 Euler-Maruyama 离散 SDE 时应有的步长缩放。

代码在 `t=1` 时用下一个时间点代替分母中的时间，避免除零。见 `openpi_action_model.py:1131-1135`。

### 5.3 转移均值

设 `t_next=t-Delta`，代码计算：

```text
mu_sde = [1 - t_next] * x0_pred
       + [t_next - g(t)^2 * Delta / (2t)] * x1_pred

sigma_sde = sqrt(Delta) * g(t)
```

和 `flow_ode` 相比，`x1_pred` 的权重多了：

```text
- g(t)^2 * Delta / (2t)
```

这就是漂移修正。最终采样：

```text
x_(t-Delta) = mu_sde + sigma_sde * z
```

### 5.4 逐步计算

1. 模型根据当前 `x_t` 预测 `v_t`。
2. 由 `v_t` 算出 `x0_pred`、`x1_pred`。
3. 读取 `noise_level`；若开启退火，先根据训练步数更新它。
4. 计算时间相关扩散系数 `g(t)`。
5. 计算带漂移修正的 `mu_sde`。
6. 计算单步标准差 `sigma_sde=sqrt(Delta)*g(t)`。
7. 采样与动作张量同形状的 `z~N(0,I)`。
8. 得到下一去噪状态，并用 `Normal(mu_sde,sigma_sde)` 计算 log probability。

### 5.5 噪声退火

若 `noise_anneal=True`：

```text
noise_level(k)
  = start + (end-start) * min(k,K)/K
```

配置：

```yaml
noise_anneal: true
noise_params: [0.7, 0.3, 400]  # start, end, anneal_steps
```

表示前 400 个 global step 从 0.7 线性下降到 0.3，之后保持 0.3。不开退火时直接使用 `noise_level`。实现见 `openpi_action_model.py:1622-1638`。

### 5.6 特点

优点：

- 噪声随 flow 时间自动变化，而不是所有步骤使用同一强度；
- 漂移和扩散成套设计，比简单的动作高斯扰动更贴合 flow 结构；
- 不需要额外的噪声预测网络；
- 每个随机步骤有明确的条件高斯概率，适合 PPO/GRPO；
- 可以用 `noise_level` 和退火直接控制探索。

局限：

- 噪声调度是手工设定的，不能根据具体观测或动作维度自适应；
- 噪声端附近存在 `t/(1-t)` 的边界问题，需要代码特殊处理；
- `noise_level` 太大时，均值修正和随机项都会快速变强，可能破坏预训练策略；
- 时间方向写反会得到完全错误的调度。OpenPI 是 `1 -> 0`，GR 00 T 是 `0 -> 1`。

### 5.7 适用场景

- 在预训练 π0/π0.5、GR 00 T 等 Flow Matching VLA 上做 PPO/GRPO 微调；
- 希望尽量保留原 flow 的分布结构，同时获得可计算 log probability 的探索；
- 不希望引入额外噪声网络参数；
- 希望训练前期探索较强、后期逐步收敛，可以配合噪声退火；
- 训练样本不多，希望先使用结构化先验而不是让噪声强度完全从数据学习。

## 6. `flow_cps`：正弦/余弦配对控制

### 6.1 通俗原理

`flow_cps` 把 `noise_level` 当成一个角度旋钮：

- `cos` 控制确定性端点成分保留多少；
- `sin` 控制随机噪声加入多少。

两者满足：

```text
cos(theta)^2 + sin(theta)^2 = 1
```

所以它像是在“确定性信号”和“随机信号”之间做旋转式配比。仓库代码没有解释 `CPS` 缩写的全称，本文只按实际公式说明，不对名称作额外推断。

### 6.2 计算公式

代码令：

```text
theta = pi * noise_level / 2
t_next = t - Delta
```

然后：

```text
mu_cps = [1 - t_next] * x0_pred
       + [t_next * cos(theta)] * x1_pred

sigma_cps = t_next * sin(theta)
```

最终：

```text
x_(t-Delta) = mu_cps + sigma_cps * z
```

见 `openpi_action_model.py:1139-1145`。

### 6.3 为什么它会在动作端自动停噪

当 `t_next -> 0`：

```text
sigma_cps = t_next * sin(theta) -> 0
```

所以越靠近最终动作，随机扰动越小；到动作端时自然变成确定性。这不需要 `flow_sde` 那样的分式时间调度。

### 6.4 逐步计算

1. 模型预测 `v_t`。
2. 计算 `x0_pred`、`x1_pred`。
3. 将 `noise_level` 转成角度 `theta`。
4. 计算 `cos(theta)` 和 `sin(theta)`。
5. 用余弦缩放 `x1_pred` 的均值权重。
6. 用正弦与 `t_next` 计算标准差。
7. 采样 `z~N(0,I)` 并得到下一状态。
8. 用该高斯转移计算 log probability。

### 6.5 特点

优点：

- 公式简单、连续；
- 没有 `1/(1-t)` 一类边界奇点；
- 噪声在动作端自然衰减到 0；
- `noise_level=0` 时严格退化为 `flow_ode`；
- `noise_level` 在 `[0,1]` 时，确定性/随机性比例容易直观理解。

局限：

- 它不仅增加随机方差，还通过 `cos(theta)` 改变均值，因此不是“保留 ODE 均值再加噪”；
- 代码没有限制 `noise_level` 的范围。若超出常用 `[0,1]`，三角函数会继续周期变化，直觉上的“噪声越大越随机”不再单调；
- 当前仓库主流示例主要使用 `flow_sde` 和 `flow_noise`，`flow_cps` 的现成训练配置和验证覆盖相对少；
- 噪声强度仍是手工规则，不能根据状态自适应。

### 6.6 适用场景

- 研究不同随机 flow 参数化方式；
- 希望避免 SDE 调度边界奇点；
- 希望扰动在动作终点平滑归零；
- 做 `flow_sde` 的消融对照；
- 可以承担额外实验验证成本的场景。

如果目标是直接复现仓库中已有的主流结果，优先从示例更丰富的 `flow_sde` 或 `flow_noise` 开始。

## 7. `flow_noise`：让网络学习每一步该加多少噪声

### 7.1 通俗原理

`flow_sde` 和 `flow_cps` 像使用统一交通规则：到某个时间点，所有样本按同一公式决定噪声强度。

`flow_noise` 则像给每辆车配一个风险评估员。它观察当前隐藏特征，分别决定：

- 这个状态是否需要更多探索；
- 当前去噪时间是否适合探索；
- 哪个动作维度可以多探索；
- 哪个动作维度应该更保守。

因此标准差不再只是标量，而通常是与动作张量同形状的条件输出。

### 7.2 转移公式

`flow_noise` 完全保留 ODE 均值：

```text
mu_noise = mu_ode
```

标准差由噪声网络预测：

```text
sigma_noise = ExploreNoiseNet(suffix_out)
```

最终：

```text
x_(t-Delta) = mu_ode + sigma_noise * z
```

其中 `suffix_out` 是动作专家在当前观测、`x_t` 和 `t` 下产生的隐藏特征，因此噪声网络可以间接感知状态、时间和当前动作 latent。代码见 `openpi_action_model.py:1146-1149`。

### 7.3 噪声网络怎样保证标准差合法

`ExploreNoiseNet` 先输出任意实数 `raw_logvar`，再做三步约束：

```text
h = tanh(raw_logvar)                        # 压到 [-1,1]
logvar = logvar_min
       + (logvar_max-logvar_min)*(h+1)/2   # 映射到合法区间
sigma = exp(0.5*logvar)                    # 从 log variance 还原 std
```

配置虽然叫 `noise_logvar_range`，填写的其实是标准差边界：

```yaml
noise_logvar_range: [0.08, 0.16]  # [min_std, max_std]
```

模块内部先计算：

```text
logvar_min = log(0.08^2)
logvar_max = log(0.16^2)
```

因此最终输出的 `sigma` 一定落在 `[0.08,0.16]`。实现见 `rlinf/models/embodiment/modules/explore_noise_net.py:36-99`。

### 7.4 逐步计算

1. 模型预测速度 `v_t`，同时得到隐藏特征 `suffix_out`。
2. 计算 `x0_pred`、`x1_pred`。
3. 按 `flow_ode` 公式计算 `mu_ode`。
4. 将 `suffix_out` 输入 `ExploreNoiseNet`。
5. 网络输出每个动作位置和维度的 `raw_logvar`。
6. 通过 `tanh`、区间映射和指数运算得到有界标准差。
7. 采样逐元素高斯噪声 `z`。
8. 得到下一状态，并计算条件高斯 log probability 和熵。
9. RL 梯度不仅更新速度模型，也可以更新噪声网络。

### 7.5 为什么通常配 `joint_logprob=True`

OpenPI 有两种去噪链训练方式：

- `joint_logprob=False`：随机选一个去噪步使用噪声方法，其余步骤走 ODE；
- `joint_logprob=True`：所有去噪步都使用随机转移，并把初始单位高斯的概率也纳入整条链。

`flow_noise` 的目标是学习整条生成轨迹上的自适应探索，因此示例通常配置：

```yaml
noise_method: flow_noise
joint_logprob: true
```

代码也只对 `flow_noise` 返回实际的高斯熵；其他方法的 entropy tensor 被置零。见 `openpi_action_model.py:1280-1323`。

### 7.6 特点

优点：

- 能随观测、时间、动作位置和动作维度自适应；
- 保留 ODE 均值，只学习探索尺度，职责较清晰；
- 可直接计算高斯 log probability 和 entropy；
- 适合结合熵奖励，让策略自己学习“何时大胆、何时谨慎”；
- 能表达解析标量调度无法表达的异方差探索。

局限：

- 多了一个需要训练的噪声网络；
- 训练数据不足或熵系数不合适时，标准差可能长期贴住上限或下限；
- 全链 `joint_logprob` 的计算和显存成本通常高于单步随机化；
- 对超参数更敏感，需要同时关注学习率、熵系数、标准差范围；
- `noise_params` 不是它的标准差范围；真正相关的是 `noise_logvar_range`。

### 7.7 适用场景

- 不同状态的探索需求差异明显；
- 不同动作维度风险不同，例如平移可多探索、夹爪或接触方向需保守；
- 任务多样、动作分布多峰，固定噪声调度不够灵活；
- 有足够 rollout 数据训练额外噪声头；
- 使用 entropy bonus，希望显式优化随机性；
- 愿意使用 `joint_logprob=True` 对完整随机去噪链建模。

## 8. 一个完整的数值例子

为了直观看到四种方法的差异，考虑单个动作维度，假设：

```text
t = 0.75
Delta = 0.25
t_next = 0.50
x_t = 0.60
v_t = 0.80
```

先计算端点预测：

```text
x0_pred = 0.60 - 0.75*0.80 = 0.00
x1_pred = 0.60 + 0.25*0.80 = 0.80
```

为了比较实际采样，再假设本次标准高斯样本恰好为 `z=0.20`。

### 8.1 `flow_ode`

```text
mu = 0.50*0.00 + 0.50*0.80 = 0.40
sigma = 0
x_next = 0.40
```

没有额外随机性。

### 8.2 `flow_sde`

令 `noise_level=0.5`：

```text
g(t) = 0.5*sqrt(0.75/0.25) = 0.8660
sigma = sqrt(0.25)*0.8660 = 0.4330

x1_weight
  = 0.50 - 0.8660^2*0.25/(2*0.75)
  = 0.375

mu = 0.50*0.00 + 0.375*0.80 = 0.30
x_next = 0.30 + 0.4330*0.20 = 0.3866
```

可以看到：它既把均值从 ODE 的 `0.40` 修正为 `0.30`，也增加了标准差 `0.4330`。

### 8.3 `flow_cps`

同样令 `noise_level=0.5`：

```text
theta = pi*0.5/2 = pi/4
cos(theta) = sin(theta) = 0.7071

mu = 0.50*0.00 + 0.50*0.7071*0.80 = 0.2828
sigma = 0.50*0.7071 = 0.3536
x_next = 0.2828 + 0.3536*0.20 = 0.3536
```

它也同时改变均值和标准差，但使用的是正弦/余弦配对。

### 8.4 `flow_noise`

假设噪声网络针对当前状态预测 `sigma=0.12`：

```text
mu = mu_ode = 0.40
sigma = 0.12
x_next = 0.40 + 0.12*0.20 = 0.424
```

它保持 ODE 均值不变，只让网络决定当前标准差。

这个例子只用于解释单步差异。实际代码中的 `x_t`、`v_t`、`mu`、`sigma` 都是批量动作张量，而不是标量。

## 9. 如何选择

### 9.1 快速决策表

| 你的目标 | 更适合的选择 | 原因 |
|---|---|---|
| 评估、部署、稳定复现 | `flow_ode` | 无逐步随机扰动 |
| 第一次给预训练 FM 策略做 PPO | `flow_sde` | 结构明确、示例多、额外参数少 |
| 想从强探索逐渐降到弱探索 | `flow_sde` + anneal | 有直接的线性退火配置 |
| 想避免分式时间调度的边界问题 | `flow_cps` | 正弦/余弦公式无除零点 |
| 做随机 flow 参数化研究或消融 | `flow_cps` | 与 SDE、learned noise 形成不同基线 |
| 每个状态/维度需要不同探索强度 | `flow_noise` | 学习异方差标准差 |
| 希望使用 entropy bonus | `flow_noise` | 代码显式返回其高斯熵 |
| 数据很少或不想多训练一个网络 | `flow_sde` | 标准差不依赖额外学习 |
| 追求最低计算与显存开销 | `flow_ode` 或单步 `flow_sde` | 无噪声头或无需全链随机化 |

### 9.2 实践建议

1. 先用 `flow_ode` 验证动作、环境和归一化管线正确。
2. 需要 PPO 探索时，从仓库已有配置对应的 `flow_sde` 开始。
3. 先用较小 `noise_level`，观察 success rate、KL、ratio、entropy 和动作平滑性。
4. 若固定调度在不同状态下表现差异很大，再尝试 `flow_noise`。
5. 使用 `flow_noise` 时同时监控预测 `sigma` 是否贴住 `[min_std,max_std]` 的边界。
6. `flow_cps` 更适合研究和消融；使用前确认目标模型确实实现了该分支。
7. 不要把某个模型的配置直接复制给另一个模型。OpenPI、GR 00 T、Dexbotic 等的支持分支并不完全一致。

## 10. 重要配置项

```yaml
openpi:
  noise_method: flow_sde       # flow_ode / flow_sde / flow_cps / flow_noise
  noise_level: 0.5             # flow_sde、flow_cps 的强度
  noise_anneal: false          # 是否退火 noise_level
  noise_params: [0.7, 0.3, 400]
  noise_logvar_range: [0.08, 0.16]  # flow_noise 的 [min_std,max_std]
  joint_logprob: false         # 单步随机化或全链随机化
  ignore_last: false           # 单步模式下是否排除最后一步
```

需要特别注意：

- `noise_params` 只有在 `noise_anneal=True` 时才参与 `noise_level` 退火；
- `flow_noise` 的上下界由 `noise_logvar_range` 控制；
- `joint_logprob=True` 通常配 `flow_noise`；
- `joint_logprob=False` 的实现只随机化一处去噪转移，其余步骤走 `flow_ode`；
- 并非所有模型都支持四个字符串，必须检查具体 action model 的分支。

## 11. 常见误区

### 误区一：`flow_ode` 完全没有随机性

错误。它没有**逐步探索噪声**，但生成初态通常仍来自 `N(0,I)`。

### 误区二：`flow_sde` 就是 `ODE结果 + 固定高斯噪声`

错误。它的标准差随时间和步长变化，而且均值含漂移修正。

### 误区三：`flow_cps` 只改变方差

错误。它还用 `cos(theta)` 修改了端点预测的均值权重。

### 误区四：`flow_noise` 的噪声大小由 `noise_level` 控制

错误。OpenPI 中它的标准差由 `ExploreNoiseNet` 预测，并由 `noise_logvar_range` 限制。

### 误区五：`noise_logvar_range=[0.08,0.16]` 表示 log variance 范围

错误。尽管名字如此，当前实现把这两个数字当作标准差上下界，内部再平方取对数。

### 误区六：四种模式在所有 Flow Matching 模型中都能直接使用

错误。OpenPI 明确实现四种模式；其他模型可能只有其中一部分，或使用 `reinflow` 等不同名称。

## 12. 代码索引

- Flow Matching 训练插值：`rlinf/models/embodiment/openpi/openpi_action_model.py:418-428`
- 初始高斯噪声与 rollout：`rlinf/models/embodiment/openpi/openpi_action_model.py:924-1045`
- 四种方法的核心公式：`rlinf/models/embodiment/openpi/openpi_action_model.py:1085-1153`
- 高斯 log probability：`rlinf/models/embodiment/openpi/openpi_action_model.py:1241-1255`
- 单步/全链概率与 entropy：`rlinf/models/embodiment/openpi/openpi_action_model.py:1260-1323`
- 噪声退火：`rlinf/models/embodiment/openpi/openpi_action_model.py:1622-1638`
- 可学习噪声网络：`rlinf/models/embodiment/modules/explore_noise_net.py:36-99`
- OpenPI 默认参数：`rlinf/models/embodiment/openpi/openpi_action_model.py:45-68`
- Flow-Noise 示例配置：`examples/embodiment/config/maniskill_ppo_openpi_pi05.yaml:118-130`