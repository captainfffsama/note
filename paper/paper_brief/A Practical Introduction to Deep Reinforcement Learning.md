#强化学习 #教程

# A Practical Introduction to Deep Reinforcement Learning
- Paper：[[2505.08295v1] A Practical Introduction to Deep Reinforcement Learning](https://arxiv.org/abs/2505.08295v1)
以下为其中内容的注解

## Alg1 注解

![](../../Attachments/APIDRL_alg1.png)

这里就是将 $s_t$ 所有的回报都存起来，然后最后求平均就是 $V_{\pi}(s_t)$ 

# 注意点

## 方法分类

如果使⽤蒙特卡洛⽅法来估计预期收益，那么相应的算法被称为“蒙特卡洛策略梯度算法”（Monte Carlo Policy Gradient Algorithm），也就是我们之前讨论过的 REINFORCE 算法。在本节中，我们将讨论那些使⽤时序差分⽅法来估计预期收益的算法；这类算法通常被称为“Actor-Critic 算法”（Actor-Critic Algorithms）

## 终止时状态的处理方法

一些场景下比如某个动作导致立即终止没有了 $s_{T+1}$ ，可以用 $s_T$ 来计算 $r_T$ :

$$
r_T = \begin{cases} 
r_T & \text{coincides with failure} \\ 
r_T + \gamma \hat{V}_\pi(\boldsymbol{s}_{T+1}) & \text{simply timeout} 
\end{cases}
$$

## 优势函数归一化方法

为了降低策略梯度（policy gradients）的⽅差并提⾼训练的稳定性，可以对优势函数（advantage function）应⽤批量归⼀化（batch normalization）技术。

$$
\hat{A}_\pi \leftarrow \frac{\hat{A}_\pi - \mathrm{mean}\big(\hat{A}_\pi\big)}{\mathrm{std}\big(\hat{A}_\pi\big) + \epsilon}
$$

## 观测数据的标准化

在监督学习中，对神经⽹络的输⼊数据应⽤批量归⼀化（batch normalization）可以提⾼训练效果。但在强化学习 （reinforcement learning）中，批量归⼀化通常并不适⽤。相反，通常会通过动态计算状态的均值和⽅差来在训练过程中对输⼊数据进⾏归⼀化处理。

$$
\begin{aligned} 
\mu &\leftarrow \mu + \alpha \cdot (\hat{\mu} - \mu) \\ 
\sigma^2 &\leftarrow \sigma^2 + \alpha \cdot (\hat{\sigma}^2 - \sigma^2 + \hat{\mu} - \mu \cdot \hat{\mu}) \\ 
s &\leftarrow \frac{s - \mu}{\sqrt{\sigma^2} + \epsilon} 
\end{aligned}
$$

$\hat{\mu}$ 和 $\hat{\sigma}^2$ 分别代表当前输入数据集的均值（mean）和方差（variance）。

但也有些实现在统计观测的输入时使用了 [welford在线算法](../../DL_knowlege/welford在线算法.md) 来更精确的实现观测的归一化

## Policy 熵处理方法

在强化学习为保持 policy 的探索性，会在损失函数中添加一个策略的熵奖励项：

$$
\mathcal{L} = \mathcal{L}^{\text{CLIP}} + c_1 \mathcal{L}^V - c_2 H(\pi(\boldsymbol{s}))
$$

即原始 actor 剪裁之后的损失 + 价值函数损失 -policy 的熵。由于 policy 多半是个多维高斯，所以计算方式如下：

$$
H_{total} = \sum_{i=1}^{dim} (\log \sigma_i + \text{const})
$$

详细参见 [强化学习中损失中熵奖励](../../DL_knowlege/强化学习中损失中熵奖励.md)

## 自适应学习率

一般会计算新旧 policy 的 KL 散度 (一般使用 policy 输出的 log 近似计算), 小了就增大 lr. 在 rsl_rl 中通常是使用 1.5 乘除. 而 sb3 中通常是 KL 大了就早停. 具体参见 [强化学习中自适应学习率调整方法](../../DL_knowlege/强化学习中自适应学习率调整方法.md)

## 梯度裁剪

$$
\text{if } \| \mathbf{g} \| > \nu \text{ then } \mathbf{g} \leftarrow \nu \frac{\mathbf{g}}{\| \mathbf{g} \|}
$$

即将所有梯度视为一个向量，计算其 L2 范数，然后等比缩放。具体参见 [强化学习中梯度剪裁方式](../../DL_knowlege/强化学习中梯度剪裁方式.md)
