#具身  #强化学习 #生成模型 #DiffusionPolicy 

# Flow Matching Policy Gradients
- 论文：<https://arxiv.org/html/2507.21053v2>
- Blog: [Flow Matching Policy Gradients](https://flowreinforce.github.io/)

# 核心流程
### Algorithm 1 Flow Policy Optimization (FPO)

策略参数 $\theta$ , 价值函数参数 $\phi$ , 裁剪超参 $\epsilon$ , 蒙特卡洛采样样本数 $N_{mc}$

当算法尚未收敛时，继续执行循环：

- 使用任意流模型采样器收集数据轨迹，并计算相应的优势值（advantages） $\hat{A}_t$
- 每次动作，存 $N_{mc}$ 个时间步长 - 噪声对 $\{(\tau_i, \epsilon_i)\}$ 并计算出 $\ell_{\theta}(\tau_i, \epsilon_i)$ （作为下面公式中的 $ell_{\theta_{old}}(\tau_i, \epsilon_i))$ ）
- $\theta_{old} \leftarrow \theta$
- 对于每一个优化周期（optimization epoch），执行以下操作：
    - 从收集到的轨迹数据中抽取一个小型样本（即一个包含若干条轨迹的数据子集）
    - 对于每一对状态 - 动作对 $(o_t, a_t)$ 及其对应的马尔可夫决策（MC）样本 $\{(\tau_i, \epsilon_i)\}$ ，执行以下操作：
        - 使用已存储的 $(\tau_i, \epsilon_i)$ 来计算 $\ell_{\theta}(\tau_i, \epsilon_i)$

        - $$\hat{r}_{\theta} \leftarrow \exp \left( -\frac{1}{N_{mc}} \sum_{i=1}^{N_{mc}} (\ell_{\theta}(\tau_i, \epsilon_i) - \ell_{\theta_{old}}(\tau_i, \epsilon_i)) \right)$$

        - $$L^{FPO}(\theta) \leftarrow \min(\hat{r}_{\theta}\hat{A}_t, \text{clip}(\hat{r}_{\theta}, 1 \pm \epsilon)\hat{A}_t)$$

    - 结束循环（end for）
    - $\theta \leftarrow \text{Optimizer}(\theta, \nabla_{\theta} \sum L^{FPO}(\theta))$
- 结束循环（end for）
- 如标准 PPO 算法一样，更新价值函数参数 $\phi$ 
结束 `while` 循环

# Q&A
## 3.5 节和 DPPO 等去噪 MDP 的 对比

**Q: 文章最后提到去噪 MDP 需要自定义采样器，和额外的环境步数指的什么？**

A: 

去噪 MDP 方法中，把 N 步去噪（比如 50 步）看作是环境中 50 个动作子步骤。这就是额外的环境步数，原本环境 step 可能只有 100 步。使用去噪 MDP 变成了 5000 步。而 FPO 不算去噪过程，只看最终采样的动作和奖励，通过流匹配损失一次性更新。

这里的自定义采样器应该指的是 DDPM 中采样的公式（即如何 $x_\tau \to x_{\tau-1}$ 的公式 ），要把这个过程改造成能记录这个转换的精确对数似然。

## 如何理解公式 14

**Q：文章算法中，flow matching 的损失函数使用公式 14，如何理解**

$$ \ell_\theta^m(\tau, \epsilon) = \frac{1}{2} w(\lambda_\tau) \cdot \left( -\frac{d\lambda}{d\tau} \right) \cdot \left| \hat{\epsilon}_\theta(a_t^\tau; \lambda_\tau) - \epsilon \right|_2^2. $$

A：

参见 3.4 节。这里 $\lambda_\tau$ 表示噪声水平 $\tau$ 下的对数信噪比（log-SNR）.即在流匹配和扩散模型中，我们通常将原始动作 $a_t$ 与噪声 $\epsilon$ 进行插值，生成加噪后的中间状态 $a_t^\tau$ ：

$$
a_t^\tau = \alpha_\tau a_t + \sigma_\tau \epsilon
$$

**信噪比 (SNR)**：定义为信号方差与噪声方差的比值，即 $\text{SNR}_\tau = \frac{\alpha_\tau^2}{\sigma_\tau^2}$

因此：

$$\lambda_\tau = \log\left(\frac{\alpha_\tau^2}{\sigma_\tau^2}\right) = 2 \log\left(\frac{\alpha_\tau}{\sigma_\tau}\right)$$

 在实际训练中， $\lambda_\tau$ 衡量了在流步长 $\tau$ 处，信号（动作信息）相对于噪声的强度。

 $\frac{d\lambda}{d\tau}$ 代表了**对数信噪比随时间 $\tau$ 变化的瞬时变化率** 。

 在具体算的噪声调度上，两者的计算方式为：

若为：

**A. 最优传输路径 (Optimal Transport / Linear Schedule)**

这是 Flow Matching 最常用的默认设置，采用线性插值 ：

- 定义： $\alpha_\tau = \tau$ ，$\sigma_\tau = 1-\tau$ 。

- 计算 $\lambda_\tau$ $$\lambda_\tau = 2 \log\left(\frac{\tau}{1-\tau}\right)$$

- 计算 $\frac{d\lambda}{d\tau}$ ：
    对上述公式求导： $\frac{d\lambda}{d\tau} = \frac{2}{\tau(1-\tau)}$ 。

 **B. 方差保持路径 (Variance Preserving / DDPM Schedule)**

常用于标准扩散模型 ：

- 定义： $\alpha_\tau = \sqrt{\bar{\alpha}_\tau}$ ， $\sigma_\tau = \sqrt{1-\bar{\alpha}_\tau}$ 。
    
- 计算 $\lambda_\tau$ ：

    $$\lambda_\tau = \log\left(\frac{\bar{\alpha}_\tau}{1-\bar{\alpha}_\tau}\right)$$

- 计算 $\frac{d\lambda}{d\tau}$ ：
    取决于具体预设的 $\bar{\alpha}_\tau$ 函数（如线性的 $\beta$ 调度或余弦调度）。

公式中 $w(\lambda_\tau)$ 为损失权重，具体参见附录 A.1.1.它的计算方式和时间调度方案有关：

1. 若是标准扩散模型（采用均匀的权重分配），则 $w(\lambda_\tau)=1$
2. 若是最优传输模型（基于线性插值的时间调度方案，flow matching），则 $w(\lambda_{\tau}) = e^{-\lambda/2}$
3. 若是余弦函数的时间调度方案（用于速度预测，半步蒸馏 DDPM），则 $w(\lambda_{\tau}) = e^{-\lambda/2}$