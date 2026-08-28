#生成模型

阅读本文之前，建议先阅读：[零推导理解Diffusion和Flow Matching](https://zhuanlan.zhihu.com/p/11228697012) 和 [零推导理解Diffusion Flow Matching（二）：随机性](https://zhuanlan.zhihu.com/p/1947597090311090928) 。本文因为涉及到很多推导，没办法作为零推导理解系列的文章了，后面有机会再更新这一系列。

本博客对应论文：[Coefficients-Preserving Sampling for Reinforcement Learning with Flow Matching](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/2509.05952)

代码（整理中）：[https://github.com/IamCreateAI/FlowCPS](https://link.zhihu.com/?target=https%3A//github.com/IamCreateAI/FlowCPS)

# 一 、[Flow-SDE](https://zhida.zhihu.com/search?content_id=262795179&content_type=Article&match_order=1&q=Flow-SDE&zhida_source=entity) 的噪声问题

最近在做 Diffusion-RL 相关的研究，借着 [LLM-RL](https://zhida.zhihu.com/search?content_id=262795179&content_type=Article&match_order=1&q=LLM-RL&zhida_source=entity) 的东风，将各种方法迁移到 Diffusion 或者 Flow Matching 的研究也是逐渐兴起了。我一开始是基于 Flow-GRPO\[1\] 做的，抛开 reward hacking 暂且不谈，这 reward 分数确实是嗷嗷涨：

![](https://pic1.zhimg.com/v2-4d98890d88cb0457415f7d3e9607d70c_1440w.jpg)

> Flow-GRPO 文章中的图，代码开源可复现，点赞

但我在做实验的时候发现一个问题，怎么训练时采样出来的图片上噪声这么大：

![](https://pica.zhimg.com/v2-aa27b26d913ea0b01cbcdad4f74df348_1440w.jpg)

训练时采样到的图片，有显著的噪声

按理说就算训练时减少了采样步骤，这图像应该是变糊才对，不应该像这样有这么多噪声，这反而像是去噪不彻底的表现。后来直接在 4 步的 [FLUX.1-schnell](https://zhida.zhihu.com/search?content_id=262795179&content_type=Article&match_order=1&q=FLUX.1-schnell&zhida_source=entity) 上进行实验，也发现了类似的现象，说明不是步数少带来的问题。

那问题就只能出在采样方法上了，RL 训练需要多样性的样本，众所周知 Flow Matching 直接建模了一个 ODE，为了引入随机性，改用 SDE 是非常自然的想法。Flow-GRPO 中使用了 Flow-SDE\[2\] 作为采样器，采样公式为：

$$
x_{t-\Delta t} = x_t - [\hat{v}_{\theta}(x_t, t) + \frac{\sigma_t^2}{2t}\underbrace{(x_t+(1-t)\hat{v}_{\theta}(x_t, t))}_{\text{predicted noise} }]\Delta t + \sigma_t \sqrt{\Delta t} \epsilon
$$

中间的 $x_t+(1-t)\hat{v}_{\theta}(x_t, t)$ 我们上篇文章说过，其实就是预测的噪声项。这个公式看起来比较复杂，我们需要将其像上篇文章一样拆解成预测样本、预测噪声、新加入噪声三项。首先对一个 ODE 的 Flow Matching 采样进行分解：

$$
\begin{align} 
\hat{x}_{t - \Delta t} 
&= x_t - \hat{v}_{\theta}(x_t,t) \Delta t \notag \\ 
&= \left(1-(t-\Delta t)\right)\underbrace{\left(x_t - t\hat{v}_{\theta}(x_t,t)\right)}_{\text{predicted }\hat{x}_0 } + (t-\Delta t)\underbrace{\left(x_t + (1-t)\hat{v}_{\theta}(x_t,t)\right)}_{\text{predicted }\hat{x}_1 } \notag \\ 
&= \underbrace{\left(1-(t-\Delta t)\right)}_{\text{coefficient of sample}} \hat{x}_0 + \underbrace{(t-\Delta t)}_{\text{coefficient of noise}}\hat{x}_1, 
\end{align}
$$

有了这个公式之后，对 Flow-SDE 也进行分解：

$$
\begin{align} 
x_{t-\Delta t} 
& = x_t - [\hat{v}_{\theta}(x_t, t) + \frac{\sigma_t^2}{2t}\underbrace{(x_t+(1-t)\hat{v}_{\theta}(x_t, t))}_{\text{predicted }\hat{x}_1 }]\Delta t + \sigma_t \sqrt{\Delta t} \epsilon \notag\\ 
&=\underbrace{x_t - \hat{v}_{\theta}(x_t, t) \Delta t}_{\text{above equation}} - \frac{\sigma_t^2 \Delta t}{2t}\hat{x}_1+ \sigma_t \sqrt{\Delta t} \epsilon \notag\\ 
&= \left(1-(t-\Delta t)\right) \hat{x}_0 + (t-\Delta t - \frac{\sigma_t^2 \Delta t}{2t})\hat{x}_1+ \sigma_t \sqrt{\Delta t} \epsilon 
\end{align}
$$

对于噪声这里，也是像 [DDIM](https://zhida.zhihu.com/search?content_id=262795179&content_type=Article&match_order=1&q=DDIM&zhida_source=entity) 那样减去了一项，后面再加入了新的一项，看起来没什么大问题…吗？

其实问题还挺严重的，DDIM 两个噪声项系数的平方和，恰好等于它当前步骤应该有的噪声方差水平，但这里后两项的平方和，显然不等于 $(t - \Delta t)^2$ ，如果这里加得比 $(t - \Delta t)^2$ 高了，相当于每一步都加了更多的噪声，那最终生成的图像岂不是也会多很多噪声？我们推导一下，这里后两项相加的噪声标准差为：

$$
\begin{align} 
\sigma_{total} 
&= \sqrt{(t-\Delta t - \frac{\sigma_t^2 \Delta t}{2t})^2 + \sigma_t^2 \Delta t} \notag\\ 
&= \sqrt{(t-\Delta t)^2 -\frac{\sigma_t^2 \Delta t}{t}(t-\Delta t)+ (\frac{\sigma_t^2 \Delta t}{2t})^2 + \sigma_t^2 \Delta t} \notag\\ 
&=\sqrt{(t-\Delta t)^2 +\frac{(\sigma_t \Delta t)^2}{t}+ (\frac{\sigma_t^2 \Delta t}{2t})^2} \notag\\ &\ge t-\Delta t, 
\end{align}
$$

这里虽然写的是大于等于，但其实等于号只有在 $\sigma_t = 0$ 时，也就是没有随机性时能取到。所以说 Flow-SDE 每一步都多加了 $\sqrt{\frac{(\sigma_t \Delta t)^2}{t}+ (\frac{\sigma_t^2 \Delta t}{2t})^2}$ 这么多的噪声！尤其是这个误差项还会除以 t，在 t 很小的时候，这个误差还是非常大的。

# 二 、解决方案

发现问题之后，解决问题就简单很多了。如果说 SDE 减少的那项 $\frac{\sigma_t^2 \Delta t}{2t}\hat{x}_1$ 跟后面新加入的噪声 $\sigma_t \sqrt{\Delta t} \epsilon$ 不匹配的话，我们换掉它，让第二项减去的幅度与新加入的噪声幅度相符就好了：

$$
x_{t-\Delta t} = \left(1-(t-\Delta t)\right) \hat{x}_0 + \sqrt{(t-\Delta t)^2 - \sigma_t^2\Delta t}\hat{x}_1+ \sigma_t\sqrt{\Delta t}\epsilon
$$

这个公式形式其实跟 DDIM 蛮像的，在每一步，样本前的系数和噪声前的系数（多个噪声要取平方和开根号）都完美符合 scheduler。其实这也与训练相符，因为训练时是按 $x_t = (1-t) x_0 + t x_1$ 采样网络输入的，那在测试时每一步网络的输入也应该符合这样一个形式。神经网络是个黑盒，OOD 的样本输出会带来的什么样的输出是不可预知的，大概率不会有什么好结果。

为了能够做到训练和测试一致，我们提出“系数保持采样（Coefficients-Preserving Sample，CPS）”的概念，只有当一个采样过程中每一步的样本前系数与噪声前系数都与 scheduler 一致的情况下，才可以称之为“系数保持采样”。可以看到 DDIM 和我们提出的采样算法，都是符合系数保持采样要求的，而 Flow-SDE 有一定的误差，不能被称作系数保持采样。

上边这个采样公式其实还是保守了一些，其中 $\sqrt{\Delta t}$ 项是为了形成 [维纳过程](https://zhida.zhihu.com/search?content_id=262795179&content_type=Article&match_order=1&q=%E7%BB%B4%E7%BA%B3%E8%BF%87%E7%A8%8B&zhida_source=entity)（Wiener Process，也叫布朗运动），加入这一项之后，可以保证在整个采样过程中，加入的总噪声幅度不超过 1。但在 RL 中，我们希望的是多样性越高越好，所以完全不必加入这个限制，我每一步都采样一个新鲜的噪声一样能正常的产生图像。所以拿掉 $\sqrt{\Delta t}$ 之后采样公式变为：

$$
x_{t-\Delta t} = \left(1-(t-\Delta t)\right) \hat{x}_0 + \sqrt{(t-\Delta t)^2 - \sigma_t^2}\hat{x}_1+ \sigma_t\epsilon
$$

之后我又注意到这个公式其实不太好用， $\sigma_t$ 如果设置成全局一致的话，前几步就太小了，而后几步又加得太多，如果设置得过大超过了 $t - \Delta t$ 还会导致根号下出现负数的问题。考虑到这就是一个圆形公式，没有什么形式比 sin cos 更适合的了，令 $\sigma_t = (t - \Delta t ) \sin(\frac{\eta \pi}{2})$ ：

$$
x_{t-\Delta t} = \left(1-(t-\Delta t)\right) \hat{x}_0 + (t - \Delta t)\cos(\frac{\eta \pi}{2})\hat{x}_1+ (t - \Delta t)\sin(\frac{\eta \pi}{2})\epsilon
$$

这里 $\eta \in [0,1]$ ，代表的物理意义就是噪声总和与预测噪声之间的夹角，画出图来也很好看：

![](https://pic4.zhimg.com/v2-a11268da89fe3524bdb69b6ec09e781f_1440w.jpg)

跟上篇文章的区别就在于这里加入多少新噪声由角度来控制了

# 三 、误差的根源

如果只是做工程的话，问题已经解决，这篇文章到这里也就结束了。但我仍然有一个疑问：SDE 采样究竟错在哪了？这可是 Diffusion 的理论基石之一，怎么会有问题？

于是我打开了 Score-SDE 这篇文章\[3\]，开始认真地啃它的理论推导，其实也不算很难：

![](https://pica.zhimg.com/v2-1bd71f06b38c7e11986430ea4579b9fe_1440w.jpg)

来自于\[3\] 的 Appendix B, VP SDE 对应 DDPM

微分方程大致上应该符合这么一个形式： $dx = f(x,t)dt$ ，Eq. 24 里有两个约等于，第一步是为了凑出来一个 $x(t)$ 移到左边形成 $dx$ 而使用了泰勒展开，另一个是为了消除掉一个 $\Delta t$ 避免同一项里出现两个 $\Delta t$ ，这样进行了两步近似之后，形成了一个微分方程 Eq. 25。

注意到这里的两个约等于成立的条件都是 $\Delta t \to 0$ ，这对于当年的 DDPM 的 1000 步采样来说自然是没问题的，但对于目前大家用的十几步甚至蒸馏到 4 步的模型来说，就不再成立了。其中的泰勒展开用的还只是一阶展开，其误差在 $\Delta t$ 较大的时候确实会比较大。

那对于 Flow Matching 的 SDE 来说是不是也是这样近似出来的呢？利用泰勒展开：$\sqrt{t^2 - x} = t - \frac{x}{2t} + O(x)$ ，对我们上边提出的采样公式做一下变形：

$$
\begin{align} 
x_{t-\Delta t} 
&= \left(1-(t-\Delta t)\right) \hat{x}_0 + \sqrt{(t-\Delta t)^2 - \sigma_t^2\Delta t}\hat{x}_1+ \sigma_t\sqrt{\Delta t}\epsilon \\
 & \approx \left(1-(t-\Delta t)\right) \hat{x}_0 + \left(t-\Delta t - \frac{\sigma_t^2\Delta t}{2(t-\Delta t)}\right)\hat{x}_1+ \sigma_t\sqrt{\Delta t}\epsilon \notag\\ 
 & \approx \left(1-(t-\Delta t)\right) \hat{x}_0 + \left(t-\Delta t - \frac{\sigma_t^2\Delta t}{2t}\right)\hat{x}_1+ \sigma_t\sqrt{\Delta t}\epsilon, 
 \end{align}
$$

与 VP SDE 的推导一样，第一步我也是用了泰勒展开，第二步也是省略掉了一个 $\Delta t$ ，最后竟然能得到跟 Flow-SDE 一模一样的采样公式！所以说 Flow-SDE 其实也只能用于 $\Delta t \to 0$ 的情况，对于较大的步长，误差会比较明显。

这里我们画出了 Flow-GRPO 和一个同期工作 Dance-GRPO\[4\] 的采样误差图：

![](https://pic2.zhimg.com/v2-fa127207d1b66a4df2bd6bf26e4d0f39_1440w.jpg)

分别是 1000 步、16 步和 4 步

可以看到误差随着步数的降低而显著增加。另外注意到，Flow-GRPO 在 t=1 处（第 1 步）有较高的误差，而 Dance-GRPO 在 t=0 附近（最后几步）有较高的误差，这个误差的罪魁祸首也是泰勒展开带来的这一项 $\frac{\sigma_t^2 \Delta t}{2t}\hat{x}_1$ ，Flow-GRPO 对 $\sigma_t$ 的定义为 $\sigma_t = \eta\sqrt{\frac{t}{1-t}}$ ，恰好抵消掉了一个 $t$，变成除以 $1-t$ ，所以它在 $t=1$ 处误差较大；而 Dance-GRPO 定义 $\sigma_t = \eta$ ，所以它在 $t=0$ 处误差较大。

后来我又翻阅了更多的资料，发现泰勒展开其实在微分方程中有很广泛的应用，包括 Ito's Lemma、Fokker-Planck 方程等，都在使用泰勒展开，所以在应用这些理论的同时，不自觉就已经做了某种近似，毕竟 $\Delta t \to 0$ 这个条件在微分方程中天然是满足的，这么做一般不会有什么问题。但如果这个条件不再满足了，就会出现误差。使用 SDE 采样且步长较长时，应该使用积分形式来避免这个误差，而不是直接离散化求解。

ps：其实我也尝试了用 LLM 帮我推导一个积分形式出来，结果发现虽然误差小了，但/t 带来的数值问题依旧，所以泰勒展开也会带来一些工程上的副作用，仍然不能随便乱用。。

# 四 、实验结果

按照我们的采样公式 Flow-CPS，即使过程中新注入的随机性再大，也不会让输出的图片上带有显著的噪声：

![](https://pic2.zhimg.com/v2-c4400ff5de950e172564380fc8015ae1_1440w.jpg)

而这对 GRPO 的训练也很有帮助，毕竟很多 reward model 都是基于人类对美学的判断进行训练的，而带噪的图像我们很难说它是美观的。

![](https://picx.zhimg.com/v2-78a36fb29ce1b41c4aed530220a18f7d_1440w.jpg)

PickScore 作为 reward，它是根据人类偏好训练的一个模型

注意到这里的训练曲线，我们提出的方法 Flow-CPS 的 reward 几乎一直比 Flow-SDE 的 reward 要高，最终的结果也是我们的方法得到的更高的验证集 reward 值。在其他的一些任务上，我们也取得了比 Flow-SDE 更好的结果：

![](https://picx.zhimg.com/v2-d824ce645edcc17d485cf4d1d7885103_1440w.jpg)

# 五 、总结

本文中，我们首先发现了 Flow-SDE 采样图像带有显著噪声的问题，随后我们通过分析提出了系数保持采样的概念，并证明了 Flow-SDE 不符合系数保持采样的要求。之后我们提出了一个符合系数保持采样的公式，它在高噪声强度下仍然能产生干净的图像。我们还分析了 Flow-SDE 采样会过噪的原因，根源来自于其推导过程中的泰勒展开，而且泰勒展开不仅带来了误差，还引入了/t 项带来了数值问题，所以要尽量避免使用。最后，我们通过实验验证了我们方法对于 reward 的计算和优化都有很大的帮助，显著高于基于 Flow-SDE 采样的结果。

参考文献

\[1\] Flow-GRPO: Training Flow Matching Models via Online RL

\[2\] [Diffusion Meets Flow Matching](https://link.zhihu.com/?target=https%3A//diffusionflow.github.io/)

\[3\] Score-Based Generative Modeling through Stochastic Differential Equations

\[4\] DanceGRPO: Unleashing GRPO on Visual Generation