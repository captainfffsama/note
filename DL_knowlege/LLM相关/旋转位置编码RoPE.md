#LLM

# 背景

原始 attention 中词向量并没有考虑单个 token 在序列中的位置。

原始 transformer 中虽然使用 Sinusoidal 位置编码，但在后续工作中其实较少使用：

$$
\begin{equation}\left\{\begin{aligned}&\boldsymbol{p}_{k,2i}=\sin\Big(k/10000^{2i/d}\Big)\\
&\boldsymbol{p}_{k, 2i+1}=\cos\Big(k/10000^{2i/d}\Big) \end{aligned}\right.\end{equation}
$$

Bert 中使用了一个可以学习的位置编码。

# RoPE
## 仅考虑二维情况下

对于一个 $d_{embedding}=2$ 的词向量 $x_1$ ，其添加 RoPE 之后的词向量为 $x_1'$ ,应用在 q 上， $m$ ,是一个和 token 位置相关的系数， $\theta$ 是一个旋转的基本角度， 可以用以下公式表示：

$$
x_1'=W_q x_m e^{im \theta}=(W_q x_m)e^{im \theta}=q_m e^{im \theta}
$$

对 $e^{im \theta}$ 应用欧拉公式，将 q 拆开，则：

$$
x_1' =\begin{pmatrix}\cos m\theta & -\sin m\theta\\ \sin m\theta & \cos m\theta\end{pmatrix} \begin{pmatrix}q_0 \\ q_1\end{pmatrix}
$$

同理假设另外一个 token $x_2$ ,则：

$$
x_2'=q_n e^{in \theta}=\begin{pmatrix}\cos n\theta & -\sin n\theta\\ \sin n\theta & \cos n\theta\end{pmatrix} \begin{pmatrix}k_0 \\ k_1\end{pmatrix}
$$

将两者按照 attention 中那样，做两个向量的点积得到 attention 矩阵中的一个位置：

$$
x'^T_1 x'_2=\begin{pmatrix}q_0 & q_1\end{pmatrix} \begin{pmatrix}\cos (m-n)\theta & -\sin (m-n)\theta\\ \sin (m-n)\theta & \cos (m-n)\theta\end{pmatrix} \begin{pmatrix}k_0  \\ k_1\end{pmatrix}
$$

# 扩展到多维

由于内积满足线性叠加性，所以高维的词向量直接在 d 维度上两两一组做这个操作就行了，这也就是为何 RoPE 要求词向量维度数是偶数，即：

$$
\scriptsize{\underbrace{\begin{pmatrix} 
\cos m\theta_0 & -\sin m\theta_0 & 0 & 0 & \cdots & 0 & 0 \\ 
\sin m\theta_0 & \cos m\theta_0 & 0 & 0 & \cdots & 0 & 0 \\ 
0 & 0 & \cos m\theta_1 & -\sin m\theta_1 & \cdots & 0 & 0 \\ 
0 & 0 & \sin m\theta_1 & \cos m\theta_1 & \cdots & 0 & 0 \\ 
\vdots & \vdots & \vdots & \vdots & \ddots & \vdots & \vdots \\ 
0 & 0 & 0 & 0 & \cdots & \cos m\theta_{d/2-1} & -\sin m\theta_{d/2-1} \\ 
0 & 0 & 0 & 0 & \cdots & \sin m\theta_{d/2-1} & \cos m\theta_{d/2-1} \\ 
\end{pmatrix}}_{\boldsymbol{\mathcal{R}}_m} \begin{pmatrix}q_0 \\ q_1 \\ q_2 \\ q_3 \\ \vdots \\ q_{d-2} \\ q_{d-1}\end{pmatrix}}
$$

因为这里矩阵很稀疏，所以可以直接相乘用以下形式计算更加方便：

$$
\begin{pmatrix}q_0 \\ q_1 \\ q_2 \\ q_3 \\ \vdots \\ q_{d-2} \\ q_{d-1} 
\end{pmatrix}\otimes\begin{pmatrix}\cos m\theta_0 \\ \cos m\theta_0 \\ \cos m\theta_1 \\ \cos m\theta_1 \\ \vdots \\ \cos m\theta_{d/2-1} \\ \cos m\theta_{d/2-1} 
\end{pmatrix} + \begin{pmatrix}-q_1 \\ q_0 \\ -q_3 \\ q_2 \\ \vdots \\ -q_{d-1} \\ q_{d-2} 
\end{pmatrix}\otimes\begin{pmatrix}\sin m\theta_0 \\ \sin m\theta_0 \\ \sin m\theta_1 \\ \sin m\theta_1 \\ \vdots \\ \sin m\theta_{d/2-1} \\ \sin m\theta_{d/2-1} 
\end{pmatrix}
$$

这里 $\theta_i=10000^{-2i/d},i \in [1,2,…,\frac{d}{2}]$

这里注意有一个远程衰减的特性，即隔得近的 token 之间内积的增速会随着 token 之间距离变小。

## 上述公式代码实现

对于 $\theta_i=10000^{-2i/d},i \in [1,2,…,\frac{d}{2}]$ ，其实现如下：

```python
# shape: d/2
theta=1./（10000**(torch.arange(0,self.d,2).float()/self.d)).to(x.device)
```

获取 token 在 prompt 中的顺序位置\[0,1,…, seq_len-1\]:

```python
#shape:seq_len
seq_idx=torch.arange(seq_len,device=x.device).float().to(x.device)
```

将位置 m 和 $\theta_i$ 融合, $m\theta_i$ ：

```python
#shape: seq_len*d/2
idx_theta=torch.einsum('m,d->md',seq_idx,theta)
```

对于第 m 个字的词向量，其 cos 的系数是 $m\theta_0,m\theta_1,…,m\theta_i, m\theta_0,m\theta_1,…,m\theta_i,$ 这里简单起见重排了位置：

```python
#shape: seq_len*d
idx_theta2=torch.cat([idx_theta,idx_theta],dim=1)
```

接下来计算 cos 和 sim, 直接：

```python
cos_cached=idx_theta2.cos()[:,None,None,:]
sin_cached=idx_theta2.sin()[:,None,None,:]
```

在 ChatGLM 中：

```python
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[…, :x.shape[-1]//2]
    x2 = x[…, x.shape[-1]//2:]
    return torch.cat((-x2,x1),dim=-1)
x_rope,x_pass=x[...,:self.d],x[...,self.d:]
neg_half_x=rotate_half(x_rope)
x_rope=(x_rope*cos_cached[:x.shape[0]])+(neg_half_x*sin_cached[:x.shape[0]])
```

注意这里不是向原文公式一样正负正负的 sin，而是一半直接负一半正，因为神经元无序，所以这里也不依赖维度顺序

在 LLama 中，RoPe 是每层 transformers 都加，在 ChatGLM 中只有第一层加。

# 参考
- [通俗易懂-大模型的关键技术之一：旋转位置编码rope （2）_哔哩哔哩_bilibili](https://www.bilibili.com/video/BV1Tr421p7By/?spm_id_from=333.337.search-card.all.click&vd_source=c6acf7e2d08361599bddd176f227d590)
- [Transformer升级之路：2、博采众长的旋转式位置编码 - 科学空间|Scientific Spaces](https://spaces.ac.cn/archives/8265)