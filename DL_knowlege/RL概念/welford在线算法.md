#强化学习 

以下回答来自 gemini 3 pro。

在 RL 中，输入观测的归一化往往是动态的，rsl_rl 用的是一个 Normalize 层。而更常见的是在每次交互时，统计交互数据的观测的均值和方差，然后动态更新。这里均值和方差动态计算的方式就是 welford 在线算法的并行版本。

它的核心逻辑是：**“如何仅利用‘旧的统计值’和‘新进来的这一批数据的统计值’，算出‘总的统计值’？”**

---

### 一、 核心逻辑：两个高斯分布的融合

想象一下，你有一个巨大的历史数据库（即你已经训练过的所有步数），我们称之为 **Set A**。现在新进来了一个 Batch 的数据（比如这一个 iteration 采样的 4096 个样本），我们称之为 **Set B**。

我们需要维护三个核心变量（通常保存在 Buffer 中）：

1. **Count ($N$):** 至今为止见过的样本总数。
2. **Mean ($\mu$):** 至今为止所有样本的均值。
3. **Var ($M_2$ 或 $\sigma^2$):** 至今为止所有样本的方差（或者平方差之和）。

当 Set A（历史）遇到 Set B（新数据）时，更新逻辑如下：

#### 1. 更新总数量 (Count)

这是最简单的：

$$N_{new} = N_{old} + N_{batch}$$

#### 2. 更新均值 (Mean)

新的均值是“旧均值”和“当前 Batch 均值”的加权平均：

$$\mu_{new} = \mu_{old} + \frac{N_{batch}}{N_{new}} (\mu_{batch} - \mu_{old})$$

或者更直观的理解：

$$\mu_{new} = \frac{N_{old} \cdot \mu_{old} + N_{batch} \cdot \mu_{batch}}{N_{new}}$$

#### 3. 更新方差 (Variance) —— **这是难点**

方差不能直接加权平均！因为“混合后的整体波动”不仅包含各自内部的波动，还包含“两个集合中心之间的偏差”。

公式如下（ Chan et al. Parallel Algorithm）：

$$M_{2, new} = M_{2, old} + M_{2, batch} + \frac{N_{old} \cdot N_{batch}}{N_{new}} (\mu_{old} - \mu_{batch})^2$$

- $M_2$ 代表平方差之和（Sum of Squared Differences），即 $Var \times N$。
    
- **直观解释：** 总方差 = 历史方差 + 新 Batch 方差 + **(历史均值和新均值差的平方贡献)**。
    
- 最后算出方差：$\sigma^2_{new} = M_{2, new} / N_{new}$。
    

---

### 二、 代码层面的实现（伪代码解析）

在 `rsl_rl` 或 `stable-baselines3` 中，你会看到类似 `RunningMeanStd` 的类。其实际运行逻辑如下：

Python

```
class RunningMeanStd(nn.Module):
    def __init__(self, epsilon=1e-4):
        # 注册为 buffer，这样 torch.save 会自动保存，但不会被优化器更新
        self.register_buffer("mean", torch.zeros(input_dim))
        self.register_buffer("var", torch.ones(input_dim))
        self.register_buffer("count", torch.tensor(1e-4)) # 防止除以0

    def forward(self, x):
        if self.training:
            # 1. 算出当前这个 Batch 的统计信息
            batch_mean = x.mean(dim=0)
            batch_var = x.var(dim=0)
            batch_count = x.shape[0]

            # 2. 调用上面的数学公式更新全局的 self.mean 和 self.var
            self.update_from_moments(batch_mean, batch_var, batch_count)

        # 3. 标准化：(x - mean) / std
        return (x - self.mean) / torch.sqrt(self.var + 1e-8)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        # 这里就是上面数学公式的代码版
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        # 更新均值
        new_mean = self.mean + delta * batch_count / tot_count
        
        # 更新平方差和 (M2)
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta**2 * self.count * batch_count / tot_count
        
        # 保存新的值
        self.mean = new_mean
        self.var = M2 / tot_count
        self.count = tot_count
```

### 三、 为什么不直接用 Momentum (EMA)？

你可能会问：_“为什么不像 Batch Normalization 那样，用 `mean = 0.9 * old + 0.1 * new` 这种指数移动平均？”_

**原因：强化学习的数据分布是剧烈变化的（Non-stationary）。**

1. **RL 初期：** 机器人像无头苍蝇乱撞，观测值的方差极大，均值很乱。
    
2. **RL 后期：** 机器人学会走路了，状态集中在某个稳定的步态附近，方差变小。
    

如果我们用 EMA（指数移动平均），它会**“遗忘”初期的数据，只记住最近的数据。

在 RL 中，我们希望 Observation Normalizer 代表的是“整个环境的全局统计特性”**，而不仅仅是最近几步的特性。如果只记住最近的，当机器人突然做一个很久没做的动作时，归一化可能会出错。

所以，Welford 算法这种**累计式（Cumulative）**的更新，能保证 $N=100万$ 时的均值，真的是这 100 万 个数据的精确平均值，权重完全平等。

### 四、 总结

1. **算法：** 使用 **并行 Welford 算法**。
    
2. **输入：** 旧的 `(Mean, Var, Count)` + 新 Batch 的 `(Mean, Var, Count)`。
    
3. **输出：** 新的全局 `(Mean, Var, Count)`。
    
4. **特点：** 这是一个增量式更新，不需要存储历史数据，且计算非常快，对 GPU 友好。

# 哪些库实现用了 welford 在线算法

事实上，**几乎所有**主流的、处理连续控制（Continuous Control）的深度强化学习库，在涉及 `Observation Normalization` 时，底层使用的数学原理都是 **Welford's Algorithm（及其并行/Batched 变体）**。

这是因为在 RL 中，我们需要在不消耗巨大内存（不存储历史数据）的前提下，计算**全局精确**的均值和方差，Welford 是唯一的数学最优解。

以下是具体的库及其实现位置，你可以直接去翻源码验证：

### 1. OpenAI Baselines / Stable Baselines3 (SB3) —— 行业标准

这是最经典的实现，几乎被后来所有库（包括 rsl_rl）所借鉴。

* **库名：** `stable-baselines3`
* **类名：** `RunningMeanStd` 和 `VecNormalize`
* **实现逻辑：**
* SB3 有一个 `VecNormalize` 包装器（Wrapper）。
* 它的核心是一个叫 `RunningMeanStd` 的类。
* 这个类并没有使用 `for` 循环逐个处理数据（经典的 Welford），而是使用了 **Chan et al.** 提出的 **Parallel Welford Algorithm**，能够一次性合并一个 Batch 的统计数据。

* **源码位置：** `stable_baselines3/common/running_mean_std.py`

```python
# SB3 源码片段示意
def update_from_moments(self, batch_mean, batch_var, batch_count):
    self.mean, self.var, self.count = update_mean_var_count_from_moments(
        self.mean, self.var, self.count, batch_mean, batch_var, batch_count
    )

```

### 2. RSL_RL (你正在用的) —— GPU 高性能版

专门为 NVIDIA Isaac Gym/Lab 设计的库，完全基于 PyTorch 并在 GPU 上运行。

* **库名：** `rsl_rl`
* **实现位置：** 通常在算法的 `storage` 或 `modules` 处理中，或者直接嵌入在 `ActorCritic` 的预处理层。
* **特点：**
* 它实现了 Welford 算法的 **Tensor 版本**。
* 它不需要把数据拉回 CPU，直接在 GPU 上利用矩阵运算完成 `mean` 和 `var` 的增量更新。

* **代码证据：** 在 `rsl_rl` 的 `EmpiricalNormalization` 或类似模块中，你会看到 `update_mean_var` 函数，里面的公式就是  那个公式。

### 3. Ray RLLib —— 分布式工业级

RLLib 需要处理成百上千个 CPU Worker 同时采样的情况，它的 Filter 机制也是基于 Welford。

* **库名：** `ray.rllib`
* **类名：** `MeanStdFilter`
* **特点：**
* 它的挑战在于：Worker A 和 Worker B 同时传回来不同的 Batch。
* 它利用 Welford 算法的**可合并性（Associativity）**，先在每个 Worker 内部算局部统计，汇聚到中心节点后，再用 Welford 公式把所有 Worker 的统计量合并成一个全局统计量。

### 4. CleanRL —— 教学与单文件实现

这是一个以代码极其简洁著称的库，非常适合学习原理。

* **库名：** `cleanrl`
* **实现：** 它们通常直接把 SB3 的 `RunningMeanStd` 代码复制粘贴到一个文件里。
* **意义：** 如果你想看最纯粹、没有复杂类继承关系的 Welford 算法在 RL 中怎么写，看 CleanRL 的 PPO 实现是最好的选择。

### 5. PyTorch (官方算法库)

哪怕是 PyTorch 官方的一些 RL 示例，在处理 Input Normalization 时，也是手写了这个算法。

---

### 为什么不用指数移动平均 (EMA)？

有些简单的代码或 Batch Normalization 层使用 EMA：

但在强化学习的 **Observation Normalization** 中，**极少使用 EMA，而是一边倒地支持 Welford**。

**原因如下：**

1. **EMA 会遗忘：** EMA 权重随时间指数衰减，它只代表“最近”的数据分布。
2. **RL 需要全局观：** 机器人在 Episode 早期可能经常倒地（方差大），后期站起来了（方差小）。
* 如果是 **EMA**：它会把早期的惨痛经验忘掉，认为方差很小。万一机器人偶然摔倒，观测值剧烈变化，EMA 归一化后的值就会爆炸（因为分母太小）。
* 如果是 **Welford**：它记得从出生到现在的所有数据。即使后期方差变小了，全局方差依然保留了早期的记录，使得归一化后的数值更保守、更安全，不容易梯度爆炸。

### 总结

只要你看到 **`RunningMeanStd`** 这个名字出现在 RL 库里，它背后 100% 就是 **Welford's Online Algorithm (Parallel Version)**。这是 RL 领域的“默认配置”。