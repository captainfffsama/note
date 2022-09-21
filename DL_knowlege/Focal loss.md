#损失函数 
#机器学习 

[原文](https://wmathor.com/index.php/archives/1548/)

对于二分类模型，我们总希望模型能够给正样本输出 1，负样本输出 0，但限于模型的拟合能力等问题，一般来说做不到这一点。而事实上在预测中，我们也是认为大于 0.5 的就是正样本了，小于 0.5 的就是负样本。这样就意味着，我们可以 “有选择” 地更新模型，**比如，设定一个阈值为 0.6，那么模型对某个正样本的输出大于 0.6，我就不根据这个样本来更新模型了，模型对某个负样本的输出小于 0.4，我也不根据这个样本来更新模型了，只有在 0.4~0.6 之间的，才让模型更新，这时候模型会更 “集中精力” 去关心那些 “模凌两可” 的样本，从而使得分类效果更好**，这跟传统的 SVM 思想是一致的

不仅如此，这样的做法**理论上**还能防止过拟合，因为它防止了模型专门挑**那些容易拟合的样本**来 "拼命" 拟合（是的损失函数下降），这就好比老师只关心优生，希望优生能从 80 分提高到 90 分，而不想办法提高差生的成绩，这显然不是一个好老师

# 修正的交叉熵损失（硬截断）

怎样才能达到我们上面说的目的呢？很简单，调整损失函数即可，这里主要借鉴了 hinge loss 和 triplet loss 的思想。一般常用的交叉熵损失函数是（$y$ 为 one-hot 表示，$\hat{y}$ 为经过 softmax 后的输出）：

$$
L_{old}=−\sum ylog⁡\hat{y}
$$

> 实际上交叉熵损失函数的严格定义为：
$$
L_{y,\hat{y}}=−\sum (ylog⁡\hat{y}+(1-y)log⁡1-\hat{y})
$$
> 但 PyTorch 官方实现的 `CrossEntropyLoss()` 函数的形式是：
$$ 
L_{y,\hat{y}} =−\sum ylog⁡\hat{y}
$$

> 实际上究竟用哪个公式并不重要，这里提一下是为了避免读者误将 PyTorch 版交叉熵损失认为是原本的交叉熵损失的样貌

假设阈值选定为 $m=0.6$，这个阈值原则上大于 0.5 均可。引入单位阶跃函数 $θ(x)$：

$$
θ(x)=
	\begin{cases}
		1,x>0  \\
		\frac{1}{2},x=0  \\
		0,x<0 \\
	\end{cases}
$$
那么，考虑新的损失函数：
$$
L_{new}=−\sum_y λ(y,\hat{y})ylog⁡\hat{y}
$$
其中
$$
λ(y,\hat{y}) =1−θ(y−m)θ(\hat{y}−m)−θ(1−m−y)θ(1−m−\hat{y})
$$
即

$$
λ(y,\hat{y})=
	\begin{cases}
		0,(y=1且\hat{y}>m)或(y=0且\hat{y}<1−m) \\
		1,其他情形  \\
	\end{cases}
$$
$L_{new}$ 就是在交叉熵的基础上加入了修正项 $λ(y,\hat{y})$，这一项意味着，当进入一个正样本时，那么 $y=1$，显然
$$
λ(1,\hat{y})=1−θ(\hat{y}−m)
$$
这时候，如果 $\hat{y}>m$，那么 $λ(1,\hat{y})=0$，这时交叉熵自动为 0（达到最小值）；反之，$\hat{y}<m$ 则有 $λ(1,\hat{y})=1$，此时保持交叉熵，也就是说，**如果正样本的输出已经大于 $m$ 了，那就不更新参数了，小于 $m$ 才继续更新；类似地可以分析负样本的情形，如果负样本的输出已经小于 $1−m$ 了，那就不更新参数了，大于 $1−m$ 才继续更新**

这样一来，只要将原始的交叉熵损失，换成修正的交叉熵 $L_{new}$，就可以达到我们设计的目的了。下面是笔者利用 PyTorch 实现的多分类 Loss（支持二分类），Keras 版本请查看苏剑林大佬的[这篇博客](https://kexue.fm/archives/4293)

```python
import torch
import numpy as np
import torch.nn as nn
theta = lambda t: (torch.sign(t) + 1.) / 2.
sigmoid = lambda t: (torch.sigmoid(1e9 * t))
class loss(nn.Module):
def __init__(self, theta, num_classes=2, reduction='mean', margin=0.5):
    super().__init__()
    self.theta = theta
    self.num_classes = num_classes
    self.reduction = reduction
    self.m = margin
def forward(self, pred, y):
'''
    pred: 2-D [batch, num_classes]. Softmaxed, no log
    y: 1-D [batch]. Index, but one-hot
    '''
    y_onehot = torch.tensor(np.eye(self.num_classes)[y]) 
    lambda_y_pred = 1 - self.theta(y_onehot - self.m) * \
                             self.theta(pred - self.m) \
                           - self.theta(1 - self.m - y_onehot) * \
                             self.theta(1 - self.m - pred)
    weight = torch.sign(torch.sum(lambda_y_pred, dim = 1)).unsqueeze(0)
    cel = y_onehot * torch.log(pred)
if self.reduction == 'sum':
return -torch.mean(torch.mm(weight, cel).squeeze(0))
else:
return -torch.sum(torch.mm(weight, cel).squeeze(0))
y_pred = torch.randn(3, 3)
y_pred_softmax = nn.Softmax(dim=1)(y_pred)
y_pred_softmax.clamp_(1e-8, 0.999999)
label = torch.tensor([0, 2, 2])
loss_fn = loss(theta, 3, reduction='mean', margin=0.6)
print(loss_fn(y_pred_softmax, label).item())
```

修正后的交叉熵损失看上去很好，同样的情况下在测试集上的表现确有提升，但是所需要的迭代次数会大大增加

原因是这样的：以正样本为例，**我只告诉模型正样本的预测值大于 0.6 就不更新了，却没有告诉它要 "保持" 大于 0.6**，所以下一阶段，它的预测值很有可能变回小于 0.6 了，虽然在下一个回合它还能被更新，这样反复迭代，理论上可以达到目的，但是迭代次数会大大增加。所以，要想改进的话，重点是**除了告诉模型正样本的预测值大于 0.6 就不更新了，还要告诉模型当其大于 0.6 后继续保持**。（好比老师看到一个学生及格了就不管了，这显然是不行的。如果学生已经及格，那么应该要想办法让他保持目前这个状态甚至变得更好，而不是不管）

#### 软化 Loss

硬截断会出现不足，关键在于因子 $λ(y,\hat{y})$ 是不可导的，或者说我们认为它导数为 0，因此这一项不会对梯度有任何帮助，从而我们不能从它这里得到合理的反馈（也就是模型不知道 "保持" 意味着什么）

解决这个问题的一个方法就是 "软化" 这个 loss，**"软化" 就是把一些本来不可导的函数用一些可导函数来近似**，数学角度应该叫 "光滑化"。这样处理之后本来不可导的东西就可导

其实 $λ(y,\hat{y})$ 中不可导的部分是 $θ(x)$，因此我们只要 "软化" $θ(x)$ 即可，而软化它再容易不过了，只需要利用 sigmoid 函数！我们有

$$
θ(x)=\underset{K→+∞}{lim}σ(Kx)
$$

所以很显然，我们只需要将 $θ(x)$ 替换为 $σ(Kx)$ 即可：

$$
λ(y,\hat{y})=1−σ(K(y−m))σ(K(\hat{y}−m))−σ(K(1−m−y))σ(K(1−m−\hat{y}))
$$

#### Focal Loss

由于 Kaiming 大神的 Focal Loss 一开始是基于图像的二分类问题所提出的，所以下面我们首先以二分类的损失函数为例，并且设 $m=0.5$（为什么 Kaiming 大神不是 NLPer......）

二分类问题的标准 loss 是交叉熵

$$
L_{ce}=−ylog⁡\hat{y}−(1−y)log⁡(1−\hat{y})=
	\begin{cases}
		−log⁡(\hat{y}),当y=1  \\
		−log⁡(1−\hat{y}),当y=0 \\
	\end{cases}
$$
其中 $y∈{0,1}$ 是真实标签，$\hat{y}$ 是预测值。当然，对于二分类函数我们几乎都是用 sigmoid 函数激活 $\hat{y}=σ(x)$，所以相当于
$$
L_{ce}=−ylog⁡σ(x)−(1−y)log⁡σ(−x)=
	\begin{cases}
		−log⁡σ(x),当y=1  \\
		−log⁡σ(−x),当y=0  \\
	\end{cases}
$$
> $1−σ(x)=σ(−x)$

引入硬截断后的二分类 loss 形式为
$$
L^∗=λ(y,\hat{y})⋅L_{ce}
$$
其中
$$
λ(y,\hat{y})=
	\begin{cases}
		1−θ(\hat{y}−0.5),当y=1  \\
		1−θ(0.5−\hat{y}),当y=0 \\
	\end{cases}
$$
$$
=
\begin{cases}
	θ(0.5−\hat{y}),当y=1 \\
	θ(\hat{y}−0.5),当y=0 \\
\end{cases}
$$
实际上，它也等价于
$$
λ(y,\hat{y})=
	\begin{cases}
		θ(−x),当y=1 \\
		θ(x),当y=0 \\
	\end{cases}
$$

> 注意这里我并没有说 "等于"，而是 "等价于"，因为 $θ(0.5−\hat{y})$ 表示 $\hat{y}>0.5$ 时取 0，小于 0.5 时取 1；而 $θ(−x)$ 表示 $x>0$ 时取 0，小于 0 时取 1。$\hat{y}>0.5$ 和 $\hat{y}<0.5$ 分别刚好对应 $x>0$ 和 $x<0$

因为 $θ(x)=\underset{K→+∞}{lim}σ(Kx)$，所以很显然有

$$
L^∗=
	\begin{cases}
		−σ(−Kx)log⁡σ(x),当y=1  \\
		−σ(Kx)log⁡σ(−x),当y=0  \\
	\end{cases}
$$

***
以上仅仅只是我们根据已知内容推导的二分类交叉熵损失，Kaiming 大神的 Focal Loss 形式如下：
$$
L_{fl}=
	\begin{cases}
		−(1−\hat{y})^γlog⁡\hat{y},当y=1  \\
		−\hat{y}^γlog⁡(1−\hat{y}),当y=0  \\
	\end{cases}
$$
带入 $\hat{y}=σ(x)$ 则有

$$
L_{fl}=
	\begin{cases}
		−σ^γ(−x)log⁡σ(x),当y=1  \\
		−σ^γ(x)log⁡σ(−x),当y=0  \\
	\end{cases}
$$
特别地，**如果 $K$ 和 $γ$ 都取 1，那么 $L^∗=L_{fl}$ ！**

事实上 $K$ 和 $γ$ 的作用是一样的，都是为了调节权重曲线的陡度，只是调节的方式不太一样。注意 $L_∗$ 或 $L_{fl}$ 实际上都已经包含了对不均衡样本的解决办法，或者说，类别不均衡本质上就是分类难度差异的体现。**比如负样本远比正样本多的话，模型肯定会倾向于数目多的负类（可以想像模型直接无脑全部预测为负类），这时负类的 $\hat{y}^γ$ 或 $σ(Kx)$ 都很小，而正类的 $(1−\hat{y})^γ$ 或 $σ(−Kx)$ 都很大，这时模型就会开始集中精力关注正样本**

当然，Kaiming 大神还发现对 $L_{fl}$ 做个权重调整，结果会有微小提升

$$
L_{fl}=
	\begin{cases}
		−α(1−\hat{y})^γlog⁡\hat{y},当y=1 \\
		−(1−α)\hat{y}^γlog⁡(1−\hat{y}),当y=0 \\
	\end{cases}
$$

通过一系列调参，得到 $α=0.25,γ=2$（在他的模型上）的效果最好。注意在他的任务中，正样本是少数样本，也就是说，本来正样本难以 “匹敌” 负样本，但经过 $(1−\hat{y})^γ$ 和 $y^γ$ 的 "操控" 后，也许形势还逆转了，因此要对正样本降权。不过我认为这样调整只是经验结果，理论上很难有一个指导方案来决定 $α$ 的值，如果没有大算力调参，倒不如直接让 $α=0.5$（均等）

#### 多分类

Focal Loss 在多分类中的形式也很容易得到，其实就是
$$
L_{fl}=−(1−\hat{y})^γlog⁡\hat{y_t}
$$
其中，$\hat{y_t}$ 是目标的预测值，一般是经过 Softmax 后的结果

#### 为什么 Focal Loss 有效？

这一节我们试着理解为什么 Focal Loss 有效，下图展示了不同 γ 值下 Focal Loss 曲线。特别地，当 $γ=0$ 时，其形式就是 CrossEntropy Loss

![](https://z3.ax1x.com/2021/05/05/gKA4ds.png#shadow)

在上图中，"蓝色" 线表示交叉熵损失，X 轴表示预测真实值的概率，Y 轴是给定预测值下的损失值。从图像中可以看出，当模型以 0.6 的概率预测真实值时，交叉熵损失仍在 0.5 左右。因此为了减少损失，我们要求模型必须以更高的概率预测真实值。换句话说，交叉熵损失要求模型对真实值的预测结果非常有信心，但这反过来实际上会对性能产生负面影响

> 模型实际上可能变得过于自信（或者说过拟合），因此该模型无法更好的推广（鲁棒性不强）

Focal Loss 不同于上述方案，从图中可以看出，使用 $γ>1$ 的 Focal Loss 可以减少模型预测正确概率大于 0.5 的损失。因此，在类别不平衡的情况下，Focal Loss 会将模型的注意力转向稀有类别。实际上仔细观察上图我们还能分析得到：**更大的 $γ$ 值对模型预测概率的 "宽容度" 越高**。如何理解这句话？我们对比 $γ=2$ 和 $γ=5$ 的两条曲线，$γ=5$ 时，模型预测概率只要大于 0.3，Loss 就非常小了；$γ=2$ 时，模型预测概率至少要大于 0.5，Loss 才非常小，所以这变相是在人为规定置信度

下面是基于 PyTorch 实现支持多分类的 Focal Loss 代码，源自 <https://github.com/yatengLG/Focal-Loss-Pytorch>，由于代码年久失修，有些在 issue 中提出的 bug 作者还没来得及修改，这里我贴出的代码是经过修改后的，其中需要注意的是 $α$ 这个参数，**样本较多的类别应该分配一个较大的权重，而样本数较少的类别应该分配一个较小的权重**。这里我默认 $α=0.75$ 相当于默认多分类中，第 0 个类别样本数比较大，如果举个具体的例子，在 NER 任务中，`other` 这个这个类别对应的索引是 0，而且 `other` 这个类别一般来说都特别多（大部分情况下是最多的），所以 `other` 分配到的权重应该是 $α=0.75$，而其他类别的权重均为 $1−α=0.25$

```python
import torch
from torch import nn
from torch.nn import functional as F
class focal_loss(nn.Module):
def __init__(self, alpha=0.25, gamma=2, num_classes=3, size_average=True):
"""
        focal_loss损失函数, $-\alpha (1-\hat{y})^{\gamma} * CrossEntropyLoss(\hat{y}, y)$
        alpha: 类别权重. 当α是列表时, 为各类别权重, 当α为常数时, 类别权重为[α, 1-α, 1-α, ....]
        gamma: 难易样本调节参数.
        num_classes: 类别数量
        size_average: 损失计算方式, 默认取均值
        """
        super(focal_loss,self).__init__()
        self.size_average = size_average
if isinstance(alpha, list):
assert len(alpha) == num_classes   
            print(" --- Focal_loss alpha = {}, 将对每一类权重进行精细化赋值 --- ".format(alpha))
            self.alpha = torch.Tensor(alpha)
else:
assert alpha < 1
            print(" --- Focal_loss alpha = {} --- ".format(alpha))
            self.alpha = torch.zeros(num_classes)
            self.alpha[0] += alpha
            self.alpha[1:] += (1-alpha) 
        self.gamma = gamma
def forward(self, preds, labels):
"""
        preds: 预测类别. size:[B, C] or [B, S, C] B 批次, S长度, C类别数
        labels: 实际类别. size:[B] or [B, S] B批次, S长度
        """
        labels = labels.view(-1, 1) 
        preds = preds.view(-1, preds.size(-1)) 
        self.alpha = self.alpha.to(preds.device)
        preds_logsoft = F.log_softmax(preds, dim=1) 
        preds_softmax = torch.exp(preds_logsoft)    
        print(labels)
        print(preds_softmax)
        preds_softmax = preds_softmax.gather(1, labels)   
        preds_logsoft = preds_logsoft.gather(1, labels)
        alpha = self.alpha.gather(0, labels.view(-1))
        loss = -torch.mul(torch.pow((1-preds_softmax), self.gamma), preds_logsoft)  
        loss = torch.mul(alpha, loss.t())
if self.size_average:
            loss = loss.mean()
else:
            loss = loss.sum()
return loss
y_pred = torch.randn(3, 4, 5) 
label = torch.tensor([[0, 3, 4, 1], [2, 1, 0, 0], [0, 0, 4, 1]]) 
loss_fn = focal_loss(alpha = 0.75, num_classes=5)
print(loss_fn(y_pred, label).item())
```

### Focal Loss 使用问题总结

在笔者研究 Focal Loss 的这几天，看了不少文章，其中也提到很多关于 Focal Loss 的问题，这里一一列出进行记录

#### 关于参数 $α$（非常重要，仔细阅读）

很多读者可能想当然认为应该给样本较少的类别赋予一个较大的权重，实际上如果没有 $(1−\hat{y}^γ)$ 以及 $\hat{y}^γ$ 这两项，这么做确实没问题。但由于引入了这两项，本来样本少的类别应该是难分类的，结果随着模型的训练，样本多的类别变得难分类了，在这种情况下，我们应该给样本少的类别一个较小的权重，而给样本多的类别一个较大的权重

简单来说，添加 $(1−\hat{y}^γ)$ 以及 $\hat{y}^γ$ 是为了平衡正负样本，而添加 $α$ 和 $(1−α)$ 又是为了平衡 $\hat{y}^γ$ 以及 $\hat{y}^γ$，有一种套娃的思想在里面，平衡来平衡去

#### 训练过程

建议一开始训练不要使用 Focal Loss。对于一般的分类问题，开始时先使用正常的 CrossEntropyLoss，训练一段时间，确保网络的权重更新到一定的时候再更换为 Focal Loss

#### 初始化参数

有一个非常小的细节，对于分类问题，我们一般会在最后通过一个 Linear 层，而这个 Linear 层的 bias 设置是有讲究的，一般初始化设为
$$
b=−log⁡ \frac{1−π}{π}
$$
其中，假设二分类中样本数少的类别共有 $m$ 个，样本数多的类别共有 $n$ 个（ $m+n$ 等于总样本数），则 $π= \frac{m}{m+n}$，为什么这样设计？

首先我们知道最后一层的激活函数是 
$$
σ : \frac{1}{1+e^{−(wx+b)}}
$$
因为默认初始化的情况下 $w,b$ 均为 0，此时不管你提取到的特征是什么，或者说不管你输入的是什么，经过激活之后的输出都是 0.5（正类和负类都是 0.5），这会带来什么问题？

假设我们使用二分类的 CrossEntropyLoss
$$
L=−log⁡(p)−(1−y)log⁡(p)
$$
那么刚开始的时候，不管输入的是正样本还是负样本（假设负样本特别多），他们的误差都是 $−log⁡(0.5)$，而负样本的个数多得多，这么看，刚开始训练的时候，loss 肯定要被负样本的误差带偏（模型会想方设法尽力全部预测成负样本，以降低 loss）

但是如果我们对最后一层的 bias 使用上面的初始化呢？把 b 带入到 $σ$ 中
$$
\frac{1}{1+e^{log⁡(\frac{1−π}{π})}}=\frac{1}{1+(\frac{1−π}{π})}=π
$$
对于正样本来说，$L=−log⁡(π)$；对于负样本来说，$L=−log⁡(1−π)$。由于 $0<π<1−π<1$，所以 $log⁡(1−π)<log⁡(π)$，这样做了以后，虽然可能所有负样本联合起来的损失仍然比正样本大，但相较于不初始化 bias 的情况要好很多

实际上我本人写代码的时候，尤其在 `nn.Linear` 中喜欢设置 `bias=False`，即不添加 bias，因为我认为 `nn.Linear` 多数情况下只是为了转换一下维度，进行一个线性变换的操作，所以加上 bias 可能会使得原本特征矩阵内的值变得怪怪的，但是这里最好还是加上

#### References

*   [文本情感分类（四）：更好的损失函数](https://kexue.fm/archives/4293)
*   [从 loss 的硬截断、软化到 focal loss](https://kexue.fm/archives/4733)
*   [What is Focal Loss and when should you use it?](https://amaarora.github.io/2020/06/29/FocalLoss.html#so-why-did-that-work-what-did-focal-loss-do-to-make-it-work)
*   [focal loss 理解与初始化偏置 b 设置解释](https://zhuanlan.zhihu.com/p/63626711)
*   [使用 focal loss 训练数据不平衡的模型](https://zhuanlan.zhihu.com/p/258506276)
