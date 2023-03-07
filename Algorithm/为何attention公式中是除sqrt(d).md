[toc]

# attention公式

$$
Attention(Q,K,V)=softmax(\frac{QK^T}{\sqrt{d_k}})V
$$
## attention is all you need 原文
>The two most commonly used attention functions are additive attention, and dot-product (multiplicative) attention. Dot-product attention is identical to our algorithm, except for the scaling factor of $\frac{1}{\sqrt{d_k}}$. Additive attention computes the compatibility function using a feed-forward network with a single hidden layer. While the two are similar in theoretical complexity, dot-product attention is much faster and more space-efficient in practice, since it can be implemented using highly optimized matrix multiplication code. 
>While for small values of $d_k$ the two mechanisms perform similarly, additive attention outperforms dot product attention without scaling for larger values of $d_k$ . We suspect that for large values of $d_k$, the dot products grow large in magnitude, pushing the softmax function into regions where it has extremely small gradients . To counteract this effect, we scale the dot products by $\frac{1}{\sqrt{d_k}}$.

加性注意力和点乘（乘性）注意力是最常用的两种注意力函数。本文使用了点乘注意力，并使用 $\frac{1}{\sqrt{d_k}}$ 进行缩放。加型注意力使用了一层的神经前馈网络来计算。虽然加性注意力和点乘的理论复杂度差不多，但是实践中点积注意力通过使用高度优化的矩阵乘法代码来实现，可以更快更省空间。

当 $d_k$ 较小的时候，两种方法差不多，但是当 $d_k$ 较大时，不缩放的情况下加性注意力时优于点积注意力的。我们猜想是较大的$d_k$会使得点积大小变大，导致 softmax函数陷入到具有极小梯度的区域。为了抵消这种影响，我们使用$\frac{1}{\sqrt{d_k}}$ 缩放了点积。

# 为何是使用 sqrt(d)而非d缩放
在统计学中，若$X$和$Y$独立且都是随机分布变量,则:
$$
E[X+Y]=E[X]+E[Y]
$$
$$
Var(X+Y)=Var(X)+Var(Y)
$$
$$
E[XY]=E[X]E[Y]
$$
$$
Var(XY)=(Var(X)+E[X]^2)(Var(Y)+E[Y]^2)-E[X]^2E[Y]^2
$$
我们假定 $Q$ 和 $K$ 都是 $d_k\times d_k$ 大小的矩阵,每个元素都服从正态分布并相互独立.
由于 $Q$ 和 $K$ 都是独立分布的,因此我们不妨仅仅关注其中一个元素,比如最左上角的元素.那么其余元素的结果是雷同的.

那么$QK$的最左上角值就是$\sum^{d_k}_{i=0}{Q_{1,i}K_{i,1}}$ .
由于 $Q$ 和 $K$ 独立
$$
E[Q_{1,i}K_{i,1}]=E[Q_{1,i}]E[K_{i,1}]=0
$$
$$
Var(Q_{1,i}K_{i,1})=(Var(Q_{1,i})+E[Q_{1,i}]^2)(Var(K_{i,1})+E[K_{i,1}]^2)-E[Q_{1,i}]^2E[K_{i,1}]^2=1
$$

我们把$d_k$维加起来得到
$$
E[\sum^{d_k}_{i=0}{Q_{1,i}K_{i,1}}]=\sum^{d_k}_{i=0}{E[Q_{1,i}K_{i,1}]}=0
$$
$$
Var(\sum^{d_k}_{i=0}{Q_{1,i}K_{i,1}})=\sum^{d_k}_{i=0}{Var(Q_{1,i}K_{i,1})}=d_k
$$

我们对 $QK$ 归一化就是:
$$
\frac{QK-0}{\sqrt{d_k}}
$$

# 参考
- <[neural networks - Why does this multiplication of $Q$ and $K$ have a variance of $d_k$, in scaled dot product attention? - Artificial Intelligence Stack Exchange](https://ai.stackexchange.com/questions/21237/why-does-this-multiplication-of-q-and-k-have-a-variance-of-d-k-in-scaled)>