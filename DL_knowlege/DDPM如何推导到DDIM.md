#扩散模型 

DDPM 相关推导可以参见 [DDPM原理公式相关问题](DDPM原理公式相关问题.md)

以下回答均来自 Grok3

# 1. DDPM 的基本原理

DDPM（Denoising Diffusion Probabilistic Models）通过前向加噪和反向去噪过程生成数据。我们先回顾 DDPM 的核心公式。

## 前向过程（加噪）

前向过程是一个马尔可夫链，逐步向数据 $x_0$ 添加高斯噪声，生成一系列中间状态 $x_1, x_2, \dots, x_T$，直到 $x_T$ 接近纯高斯噪声。假设每一步的噪声强度由参数 $\beta_t$ 控制，前向过程的条件概率为：

$$
q(x_t | x_{t-1}) = \mathcal{N}(x_t; \sqrt{1-\beta_t}x_{t-1}, \beta_t I)
$$

这表示在第 $t-1$ 步的 $x_{t-1}$ 基础上，添加均值为 0、方差为 $\beta_t$ 的高斯噪声，得到 $x_t$。

为了简化，DDPM 提供了一个直接从 $x_0$ 到任意 $x_t$ 的公式。定义累积方差参数：

$$
\alpha_t = 1 - \beta_t, \quad \bar{\alpha}_t = \prod_{s=1}^t \alpha_s
$$

那么，前向过程可以写为：

$$
q(x_t | x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t}x_0, (1-\bar{\alpha}_t)I)
$$

简单来说，$x_t$ 是 $x_0$ 的缩放版本（乘以 $\sqrt{\bar{\alpha}_t}$）加上高斯噪声：

$$
x_t = \sqrt{\bar{\alpha}_t}x_0 + \sqrt{1-\bar{\alpha}_t}\epsilon, \quad \epsilon \sim \mathcal{N}(0, I)
$$

这里，$\epsilon$ 是随机噪声，$\sqrt{1-\bar{\alpha}_t}$ 控制噪声强度。

## 反向过程（去噪）

DDPM 的反向过程试图从 $x_T$（纯噪声）逐步恢复到 $x_0$（原始数据）。反向过程的条件概率也是高斯分布：

$$
p_\theta(x_{t-1} | x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \Sigma_\theta(t))
$$

其中，$\mu_\theta(x_t, t)$ 是神经网络预测的均值，$\Sigma_\theta(t)$ 是方差（通常固定）。DDPM 的训练目标是让 $p_\theta$ 逼近真实的反向分布 $q(x_{t-1} | x_t)$。

关键是，DDPM 假设 $\mu_\theta(x_t, t)$ 预测的是噪声 $\epsilon$。具体来说，神经网络 $\epsilon_\theta(x_t, t)$ 预测噪声，均值公式为：

$$
\mu_\theta(x_t, t) = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}} \epsilon_\theta(x_t, t) \right)
$$

训练通过优化以下损失函数（简化的形式）：

$$
L = \mathbb{E}_{x_0, \epsilon, t} \left[ \| \epsilon - \epsilon_\theta(x_t, t) \|^2 \right]
$$

这个过程需要逐步采样 $x_{t-1}, x_{t-2}, \dots, x_0$，每步依赖随机噪声，步数多（通常 $T=1000$）。

# 2. 从 DDPM 到 DDIM 的推导

DDIM（Denoising Diffusion Implicit Models）改进了 DDPM 的反向过程，使其从概率性（随机采样）变为确定性（无随机性），从而允许跳步。以下是推导过程。

## 目标：确定性反向过程

DDPM 的反向过程 $p_\theta(x_{t-1} | x_t)$ 是概率性的，依赖随机噪声采样。DDIM 的核心思想是构造一个确定性的反向过程，仍然保持与 DDPM 等价的数据分布。

我们从前向过程的公式开始：

$$
x_t = \sqrt{\bar{\alpha}_t}x_0 + \sqrt{1-\bar{\alpha}_t}\epsilon
$$

目标是从 $x_t$ 推导出 $x_{t-1}$，但不引入随机性。假设我们能直接估计 $x_0$，然后利用前向过程的结构反推 $x_{t-1}$。

### 步骤 1：估计 $x_0$

从前向过程公式：

$$
x_t = \sqrt{\bar{\alpha}_t}x_0 + \sqrt{1-\bar{\alpha}_t}\epsilon
$$

我们可以解出 $x_0$ 的估计值：

$$
x_0 = \frac{x_t - \sqrt{1-\bar{\alpha}_t}\epsilon}{\sqrt{\bar{\alpha}_t}}
$$

在 DDPM 中，神经网络 $\epsilon_\theta(x_t, t)$ 预测噪声 $\epsilon$。用 $\epsilon_\theta(x_t, t)$ 替代 $\epsilon$，得到估计的 $x_0$:

$$
\hat{x}_0 = \frac{x_t - \sqrt{1-\bar{\alpha}_t}\epsilon_\theta(x_t, t)}{\sqrt{\bar{\alpha}_t}}
$$

### 步骤 2：推导 $x_{t-1}$

现在，我们希望从 $x_t$ 确定性地推导出 $x_{t-1}$。根据前向过程，$x_{t-1}$ 的表达式为：

$$
x_{t-1} = \sqrt{\bar{\alpha}_{t-1}}x_0 + \sqrt{1-\bar{\alpha}_{t-1}}\epsilon
$$

将估计的 $\hat{x}_0$ 代入：

$$
x_{t-1} = \sqrt{\bar{\alpha}_{t-1}} \cdot \frac{x_t - \sqrt{1-\bar{\alpha}_t}\epsilon_\theta(x_t, t)}{\sqrt{\bar{\alpha}_t}} + \sqrt{1-\bar{\alpha}_{t-1}}\epsilon
$$

这里，DDIM 做了一个关键假设：为了让反向过程确定性，我们可以选择让 $x_{t-1}$ 的噪声与 $x_t$ 的噪声相同（即复用 $\epsilon_\theta(x_t, t)$），而不是引入新的随机噪声。也就是说，设 $\epsilon$ 在两步中一致：

$$
x_{t-1} = \sqrt{\bar{\alpha}_{t-1}} \cdot \frac{x_t - \sqrt{1-\bar{\alpha}_t}\epsilon_\theta(x_t, t)}{\sqrt{\bar{\alpha}_t}} + \sqrt{1-\bar{\alpha}_{t-1}}\epsilon_\theta(x_t, t)
$$

整理后，得到 DDIM 的反向过程公式：

$$
x_{t-1} = \sqrt{\bar{\alpha}_{t-1}} \cdot \frac{x_t - \sqrt{1-\bar{\alpha}_t}\epsilon_\theta(x_t, t)}{\sqrt{\bar{\alpha}_t}} + \sqrt{1-\bar{\alpha}_{t-1}}\epsilon_\theta(x_t, t)
$$

为了进一步控制灵活性，DDIM 引入一个参数 $\sigma_t$，允许调整噪声量。当 $\sigma_t = 0$，公式完全确定性：

$$
x_{t-1} = \sqrt{\bar{\alpha}_{t-1}} \cdot \frac{x_t - \sqrt{1-\bar{\alpha}_t}\epsilon_\theta(x_t, t)}{\sqrt{\bar{\alpha}_t}} + \sqrt{1-\bar{\alpha}_{t-1} - \sigma_t^2} \cdot \epsilon_\theta(x_t, t)
$$

当 $\sigma_t = 0$，去噪过程完全确定，无随机性。

### 步骤 3：跳步的数学基础

DDIM 的公式允许从任意 $x_t$ 推导到 $x_s$（其中 $s < t$），而不需要计算中间步骤。假设我们想从 $x_t$ 直接跳到 $x_s$，可以用相同的逻辑：

$$
x_s = \sqrt{\bar{\alpha}_s} \cdot \frac{x_t - \sqrt{1-\bar{\alpha}_t}\epsilon_\theta(x_t, t)}{\sqrt{\bar{\alpha}_t}} + \sqrt{1-\bar{\alpha}_s}\epsilon_\theta(x_t, t)
$$

这里的 $\bar{\alpha}_s$ 和 $\bar{\alpha}_t$ 是根据时间步定义的累积方差，公式允许直接计算任意步之间的关系。这是因为 DDIM 的反向过程不依赖马尔可夫链的逐步随机采样，而是通过确定性公式直接“跳”到目标步。

# 3. 简单示例

假设我们有 3 步扩散过程（$T=3$），$\beta_1=0.1, \beta_2=0.2, \beta_3=0.3$，计算累积方差：

$$
\alpha_1 = 0.9, \quad \alpha_2 = 0.8, \quad \alpha_3 = 0.7
$$

$$
\bar{\alpha}_1 = 0.9, \quad \bar{\alpha}_2 = 0.9 \times 0.8 = 0.72, \quad \bar{\alpha}_3 = 0.72 \times 0.7 = 0.504
$$

**DDPM**：
- 从 $x_3$ 到 $x_2$，用概率采样：

$$
\mu_\theta(x_3, 3) = \frac{1}{\sqrt{\alpha_3}} \left( x_3 - \frac{\beta_3}{\sqrt{1-\bar{\alpha}_3}} \epsilon_\theta(x_3, 3) \right)
$$

- 采样 $x_2 \sim \mathcal{N}(\mu_\theta, \beta_3 I)$，然后继续到 $x_1$，每步随机。

**DDIM**：
- 从 $x_3$ 直接到 $x_1$（跳过 $x_2$），设 $\sigma_t = 0$：

$$
x_1 = \sqrt{\bar{\alpha}_1} \cdot \frac{x_3 - \sqrt{1-\bar{\alpha}_3}\epsilon_\theta(x_3, 3)}{\sqrt{\bar{\alpha}_3}} + \sqrt{1-\bar{\alpha}_1}\epsilon_\theta(x_3, 3)
$$

- 代入数值：

$$
x_1 = \sqrt{0.9} \cdot \frac{x_3 - \sqrt{1-0.504}\epsilon_\theta(x_3, 3)}{\sqrt{0.504}} + \sqrt{1-0.9}\epsilon_\theta(x_3, 3)
$$

这个公式直接从 $x_3$ 计算 $x_1$，无需中间的 $x_2$，这就是跳步。

# 4. 为什么 DDIM 可以跳步？
- **DDPM**：反向过程是马尔可夫链，每步依赖随机采样，跳步会破坏概率分布的连续性。
- **DDIM**：反向过程是确定性公式，基于 $x_0$ 的估计和噪声复用，允许直接从 $x_t$ 计算任意 $x_s$，无需中间步骤。
- 跳步的数学基础是 DDIM 公式对时间步 $t$ 和 $s$ 的通用性，依赖 $\bar{\alpha}_t$ 和 $\bar{\alpha}_s$，可以在任意步之间建立关系。


# 5. 总结
- **DDPM**：概率性反向过程，逐步采样，公式依赖随机噪声，步数多。
- **DDIM**：确定性反向过程，通过复用噪声和估计 $x_0$，推导出任意步的公式，允许跳步。
- **推导关键**：从 DDPM 的概率采样公式，简化为 DDIM 的确定性公式，去掉随机性，引入跳步灵活性。
