#扩散模型 #图像生成

视频地址： <https://www.youtube.com/watch?v=r-fgrZ0Ve74>

# 关键观点摘要

 DDPM 取得成功的一个重要点是对不同时间步损失重新进行了加权

$$
\mathcal{L} = \mathbb{E}_{t \sim p(t), x \sim p(x)} \lambda_t \left[ \| v_{\text{target}} - v_\Theta(z_t; t) \|_2^2 \right]
$$

上式是 DDPM 的 $L_{simple}$ 损失。通过对更加相关或者不太相关的步骤进行重新加权，加速了模型的收敛

## Diffusion 通过拼接其他表示到潜变量中来改善收敛

[[2502.02492] VideoJAM: Joint Appearance-Motion Representations for Enhanced Motion Generation in Video Models]( https://arxiv.org/abs/2502.02492 ) 使用一个现成的光流预测网络来预测每一帧光流，然后将光流和 RGB concate 在一起然后去噪，这被证明可以提升视频生成中关于运动的性能。

[2504.16064 ReDi](https://arxiv.org/pdf/2504.16064) 则使用 Dinov2 的 PCA 特征然后拼接到 VAE 的潜变量，然后去噪

## Diffusion 通过在损失上引入其他学习目标，比如和其他特征对齐

$$
\mathcal{L} = \mathbb{E}_{t \sim p(t), x \sim p(x)} \left[ \| v_{\text{target}} - v_\Theta(z_t; t) \|_2^2 + \lambda L_{\text{align}} \right]
$$

上式为 [REPA]( https://arxiv.org/abs/2410.06940 ) 的目标，即在损失上加一个显式的回归。比如在某层 DiT 上使用一个 trainable 的 MLP 与某个预训练的视觉编码器特征对齐。但也有文章 [[2505.16792] REPA Works Until It Doesn't: Early-Stopped, Holistic Alignment Supercharges Diffusion Training]( https://arxiv.org/abs/2505.16792 ) 发现这种做法在最初是可以加速，但是训练到一定阶段将不在有效，甚至可能导致两种损失互相抵消。

[REPA-E](https://arxiv.org/abs/2504.10483) 则是集大成之作，它把各种损失纳入到一起考量。如下所示：

$$

L(\theta, \phi, \omega) = \underbrace{L_{\text{DIFF}}(\theta)}_{\text{diffusion, stop-grad}} + \lambda \underbrace{L_{\text{REPA}}(\theta, \phi, \omega)}_{\text{cosine alignment to DINO $\Phi$ feats}} + \eta \underbrace{L_{\text{REG}}(\phi)}_{\text{MSE + LPIPS + GAN + KL}}

$$

## 从蒸馏角度，使用更少步的去噪

[[2505.13447] Mean Flows for One-step Generative Modeling]( https://arxiv.org/abs/2505.13447 )

## 引入 GAN 

[[2311.17042] Adversarial Diffusion Distillation](https://arxiv.org/abs/2311.17042)

![](../Attachments/Adversarial%20Diffusion%20Distillation_fig2.png)

上一种在像素空间中进行扩散蒸馏的方法并不高效，它使用一个现成的编码器，512 像素大小也不支持多个宽高比，这导致它难扩展。但是使用对抗扩散蒸馏，由于我们可以重复利用预训练的教师模型，这要高效得多。

![](../Attachments/Pasted%20image%2020250623222556.png)相应的文章还包括 [[2403.12015] Fast High-Resolution Image Synthesis with Latent Adversarial Diffusion Distillation]( https://arxiv.org/abs/2403.12015 )

## 如何保持图片生成的上下文一致性

Flux. 1 Kontext