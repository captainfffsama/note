对于一个 token 数量为 10，hidden_dim 为 100 的张量 query，应用 RMSNorm：

对单个 token 应用以下公式计算其均方差： $RMS(x)=\sqrt{\frac{1}{100} \sum_{i=1}^{100}{x_i^2}+\theta}​$

其中 $\theta$ 是防止除零的小常数（如 10−6）。

使用 RMS 值对每个 token 特征向量归一化，再乘上一个可学习缩放参数 $g$ ：

$RMSNorm(x)=g * \frac{x}{RMS(x)}$

表示逐元素相乘，每个 token 独立处理
