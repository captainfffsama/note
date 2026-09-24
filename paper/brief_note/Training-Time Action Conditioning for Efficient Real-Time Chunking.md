---
tags:
  - "#具身智能"
  - "#VLA"
---
# Training-Time Action Conditioning for Efficient Real-Time Chunking
- 论文：<https://arxiv.org/abs/2512.05964>

可以把这篇论文里的 **RTC（Real-Time Chunking，实时动作分块）** 理解成一句话：

> **机器人一边执行上一批已经规划好的动作，一边计算下一批动作；下一批动作必须“接着上一批动作往下走”，不能突然跳变。**
这篇论文进一步提出了一个更简洁的版本：**不要在推理时费力地把“上一段动作”硬塞进生成过程，而是在训练时就教会模型：“前几步动作已经确定了，你只需要接着生成后面的动作。”** 论文称它为 **training-time RTC / training-time action conditioning**。
下面我从最底层的数学开始，把它拆开。



# 一、RTC 到底在解决什么问题？

假设一个机器人控制频率是 50 Hz，也就是：$1\text{ step}=20\text{ms}$

VLA 每次不是只预测一个动作，而是一次预测一串：$A_t=[a_t,a_{t+1},a_{t+2},\dots,a_{t+H-1}]$

这里：

- $A_t$ ：从时刻 $t$ 开始的一整个 action chunk
- $a_t$ ：第 $t$ 步动作
- $H$ ：prediction horizon，一次预测多少步
论文也是这样定义 action chunk 的。
例如：$H=8$

那么模型一次输出：

$$
A_t=
[a_t,a_{t+1},a_{t+2},a_{t+3},a_{t+4},a_{t+5},a_{t+6},a_{t+7}]
$$

问题是：

**VLA 推理不是瞬间完成的。**

比如推理花了 100 ms，而机器人是 50 Hz：$d=\frac{100\text{ms}}{20\text{ms}}=5$

也就是说：

> 模型在 $t$ 时刻开始计算，但直到 $t+5$ 才拿到结果。
论文把这个推理延迟定义为 $d$ ，单位就是 controller timestep。
于是出现一个很关键的问题。
模型在 $t$ 时刻预测：$a_t,a_{t+1},…,a_{t+7}$

但等预测出来时已经： $t+5$ 。所以：$a_t,a_{t+1},…,a_{t+4}$

这 5 个动作已经“过期”了。

机器人不可能为了等模型计算 100 ms 原地不动。

这就是 RTC 要解决的问题。

# 二、RTC 的基本思想：上一块执行，下一块偷偷算

假设上一轮生成： $A_{t-s}$ ，机器人先执行它。

与此同时，在 $t$ 时刻启动下一次 VLA 推理：$A_t$

模型算 $A_t$ 的这段时间里，机器人继续执行上一段 chunk 中剩余的动作。

论文 Figure 1 就是在画这个过程。

可以简单画成：

```
上一 chunk:
           ↓ t
[ -------- | a_t a_{t+1} a_{t+2} … ]

新 chunk:
           ↓ t
           [ a_t a_{t+1} a_{t+2} a_{t+3} … ]
```

如果推理延迟是： $d=3$ 。那么新 chunk 真正算出来的时候已经到了： $t+3$ 。于是：

```
t        t+1      t+2      t+3
|---------|---------|---------|

机器人执行：
上一chunk   上一chunk   上一chunk

模型：
[--------- 正在算新chunk --------]
                              ↓
                           算完
```

所以真正需要新模型决定的是： $a_{t+3},a_{t+4},…$ 。而： $a_t,a_{t+1},a_{t+2}$ 实际上已经被旧 chunk 决定并执行了。

这 $d$ 个动作就是：Action Prefix

论文明确把这 $d$ 个来自上一 chunk 的重叠动作叫做 **action prefix**。

# 三、为什么不能直接把新 chunk 前 $d$ 个动作扔掉？

这是理解 RTC 最重要的一点。

你可能会想：

> 既然新模型算出来时前 5 步已经过去了，那我直接执行第 6 步不就好了？

问题是：

**新模型不知道机器人前 5 步实际上执行了什么。**

例如旧 chunk 规划： $[0.1,\ 0.2,\ 0.3,\ 0.4,\ 0.5,\dots]$ 。新模型独立预测： $[-0.3,\ -0.2,\ -0.1,\ 0.7,\ 1.2,\dots]$ 虽然前三步不会执行，但新模型生成后半段动作的时候，是按照自己的前半段轨迹想象出来的。

也就是说它可能认为：$\text{机器人走了}-0.3\rightarrow-0.2\rightarrow-0.1$

但现实机器人其实走的是：$0.1\rightarrow0.2\rightarrow0.3$

于是当你突然切到： $0.7$ 。轨迹可能直接产生一个大跳：

```
真实旧轨迹：
0.1 → 0.2 → 0.3

新模型幻想的轨迹：
-0.3 → -0.2 → -0.1 → 0.7
                    ↑
                   它认为自己从这里接

实际执行：
0.1 → 0.2 → 0.3 → 0.7
                  ↑
               可能突然跳变
```

论文把这种跨 chunk 的不连续问题描述为可能导致 OOD 的 “jerks”。

所以真正需要做的是：

> **在生成新 chunk 时告诉模型：前 $d$ 个动作已经确定了，请在这些动作的基础上继续生成。**

数学上： $p(A_{t+d:H}\mid o_t,A_{t:t+d})$ 。论文就是这么写的。

可以理解成：

$$
\boxed{
\text{未来剩余动作}
=
f(
\text{当前视觉},
\text{已经确定的前几步动作}
)
}
$$

# 四、RTC 背后的第二个核心：Flow Matching

这篇论文的 π0.6 action expert 使用的是 flow matching。

如果不理解 flow matching，后面的 RTC 数学很容易显得莫名其妙。

其实可以把它理解得非常简单。

## 4.1 Flow Matching 在干嘛？

我们真正想生成的是一个动作：$A$

但生成模型一开始没有动作，而是从随机噪声开始：$\epsilon\sim\mathcal N(0,I)$

然后逐渐把： $\epsilon$ 。变成： $A$

类似于：

```
随机噪声
   ↓
乱七八糟的动作
   ↓
大致合理
   ↓
更加合理
   ↓
真实动作
```

定义一个连续变量：$\tau\in[0,1]$

其中：$\tau=0$

代表纯噪声：$A^0=\epsilon$

而：$\tau=1$

代表真实动作：$A^1=A$

论文采用最简单的线性路径：

$$
\boxed{
A^\tau=\tau A+(1-\tau)\epsilon
}
$$

例如真实动作：$A=10$

噪声：$\epsilon=2$

那么：$\tau=0\Rightarrow A^\tau=2$，$\tau=0.25\Rightarrow 0.25\times10+0.75\times2=4$，$\tau=0.5\Rightarrow6$，$\tau=0.75\Rightarrow8$，$\tau=1\Rightarrow10$

因此：

```
τ = 0                          τ = 1
噪声 --------------------------> 真动作
 2        4        6        8       10
```

---

# 五、模型究竟学习什么？

上面的轨迹为：$A^\tau=\tau A+(1-\tau)\epsilon$

对 $\tau$ 求导：$\frac{dA^\tau}{d\tau}=A-\epsilon$

所以沿着这条直线走时，速度永远是：

$$
\boxed{
v=A-\epsilon
}
$$

这就是 flow matching 中所谓的 **velocity**。

模型：$v_\theta(A^\tau,o,\tau)$

要做的就是：

> 看见当前这个“半噪声动作” $A^\tau$ 、机器人观察 $o$ 、以及现在处于生成过程的哪个时间 $\tau$ ，预测“应该往哪个方向走”。
训练时让它逼近：$A-\epsilon$

代码中实际也是：$\text{loss}=(\hat v-(A-\epsilon))^2$

论文 Algorithm 1 的实现非常直接：

```
pred_v_t = model(observation, x_t, time)
loss = (pred_v_t - (action_chunk - noise))**2
```

所以 Flow Matching 的核心其实就是：

$$
\boxed{
\text{模型学习一个速度场}
}
$$

告诉你：

> “现在在这里，下一小步应该往哪里移动。”

---

# 六、普通 Flow Matching 如何生成动作？

推理开始时：$x_0\sim\mathcal N(0,I)$

例如：$x_0=[-0.4,\ 1.7,\ -0.2,\dots]$

然后不断：$v_\theta(x_\tau,o,\tau)$

再用数值积分：

$$
\boxed{
x_{\tau+\Delta\tau}
=
x_\tau+\Delta\tau\,v_\theta
}
$$

最终：$x_1\approx A$

论文 Algorithm 1 也是最简单的 Euler integration：

```
v_t = model(…)
x_t = x_t + dt * v_t
```

假设 5 次 denoising：$\Delta\tau=\frac15=0.2$

就是：

```
τ=0
 x0
 ↓ model
τ=0.2
 x1
 ↓ model
τ=0.4
 x2
 ↓
τ=0.6
 ↓
τ=0.8
 ↓
τ=1.0
最终动作
```

---

# 七、Training-time RTC 最漂亮的数学技巧出现了

现在关键问题来了。

假设：$H=8$

延迟：$d=3$

我们已经知道前三步动作：$[a_t,a_{t+1},a_{t+2}]$

想让模型生成：$[a_{t+3},…,a_{t+7}]$

怎么让 flow matching 知道：

> 前三步已经确定，不要生成它们？
答案非常巧妙：

$$
\boxed{
\text{把 prefix 的 }\tau\text{ 直接设成 }1
}
$$

---

# 八、为什么 $\tau=1$ 就代表“这个动作已经确定”？

回到：$A^\tau=\tau A+(1-\tau)\epsilon$

当：$\tau=1$

得到：$A^1=A$

噪声部分完全消失。

所以：

$$
\boxed{
\tau=1
\iff
\text{这个动作已经是最终动作}
}
$$

这非常关键。

正常 postfix：$\tau=0.4$

比如：$x=0.4A+0.6\epsilon$

还是 noisy。

但是 prefix：$\tau=1$

所以：$x=A$

完全干净。

于是一个 action chunk 的 timestep 可以不同：

$$
\boxed{
\tau_i=
\begin{cases}
1,&i<d\\
\tau,&i\ge d
\end{cases}
}
$$

假设：$H=8,\ d=3,\ \tau=0.4$

那么：$[\tau_0,\tau_1,\dots,\tau_7]=[1,1,1,0.4,0.4,0.4,0.4,0.4]$

这正是论文 Figure 2 的核心。论文指出，prefix 输入真实且不加噪的动作，并把对应 flow-matching timestep 设为 1；postfix 则维持正常噪声过程。

---

# 九、具体计算一次你就明白了

假设训练样本：$A=[1,2,3,4,5,6,7,8]$

采样：$d=3$

所以：$\text{prefix}=[1,2,3]$，$\text{postfix}=[4,5,6,7,8]$

假设这次 flow timestep：$\tau=0.5$

并采样噪声：$\epsilon=[-1,4,0,2,-3,8,1,5]$

那么普通 Flow Matching 本来会计算：$x_i=\tau A_i+(1-\tau)\epsilon_i$

但是 RTC 把前三个位置的：$\tau_i=1$

所以：$\tau=[1,1,1,0.5,0.5,0.5,0.5,0.5]$

于是前三个：$x_0=1$，$x_1=2$，$x_2=3$

完全保持真实动作。

后五个：$x_3=0.5\times4+0.5\times2=3$，$x_4=0.5\times5+0.5\times(-3)=1$，$x_5=0.5\times6+0.5\times8=7$

……

因此模型输入大概是：

$$
\boxed{
[1,\ 2,\ 3,\ 3,\ 1,\ 7,\dots]
}
$$

但注意含义完全不同：

```
1   2   3  | 3    1    7 …
↑   ↑   ↑    ↑    ↑    ↑
确定动作    noisy postfix
prefix
```

于是模型自然学会：

> “前三步是条件，不需要修改；根据它们以及 observation，把后面那部分生成出来。”

---

# 十、为什么 loss 不能计算 prefix？

这是另一个非常核心的地方。

模型已经被明确告诉：$a_t,a_{t+1},a_{t+2}$

是已知答案。

所以我们的任务不是：

> 预测整条轨迹。
而是：

> 已知前三步，预测后五步。
数学上目标是：

$$
\boxed{
p(A_{\text{postfix}}\mid o,A_{\text{prefix}})
}
$$

因此 loss 应该只作用于：$i\ge d$

定义：

$$
m_i=
\begin{cases}
0&i<d\\
1&i\ge d
\end{cases}
$$

最终：

$$
\boxed{
L=
\frac{
\sum_i
m_i
\left\|
v_\theta(x_i,o,\tau_i)
-
(A_i-\epsilon_i)
\right\|^2
}{
\sum_i m_i
}
}
$$

这就是所谓：

> **mask prefix loss**

论文明确规定 loss 只计算 postfix。

代码也是：

```
postfix_mask = logical_not(prefix_mask)
loss = sum(loss * postfix_mask) / sum(postfix_mask)
```

换成人话：

```
prefix:
老师已经把答案写黑板上了
→ 不考

postfix:
学生必须自己做
→ 计算loss
```

---

# 十一、完整的训练步骤

现在可以把 Training-time RTC 整个数学过程串起来了。

假设训练集中一个真实 chunk：$A=[a_0,\dots,a_{H-1}]$

### 第一步：随机采一个推理延迟

$d\sim p(d)$

例如：$d=3$

论文特别强调真实推理延迟并非固定，所以训练时随机采样 $d$ 。

于是：$A_\text{prefix}=A_{0:d}$，$A_\text{postfix}=A_{d:H}$

---

### 第二步：采样 Flow Matching 时间

$\tau\sim U(0,1)$

例如：$\tau=0.37$

---

### 第三步：采高斯噪声

$\epsilon\sim N(0,I)$

---

### 第四步：构造 prefix mask

$M_i=1[i<d]$

例如：$d=3,H=8$

得到：$M=[1,1,1,0,0,0,0,0]$

---

### 第五步：修改每个 action token 的时间

原本：$[\tau,\tau,\tau,\tau,\dots]$

RTC 改成：

$$
\boxed{
[1,1,1,\tau,\tau,\tau,\tau,\tau]
}
$$

---

### 第六步：构造模型输入

统一用：$x_i=\tau_iA_i+(1-\tau_i)\epsilon_i$

对于 prefix：$\tau_i=1$

所以：$x_i=A_i$

对于 postfix：$x_i=\tau A_i+(1-\tau)\epsilon_i$

所以整体：

$$
\boxed{
x=
[
A_{\text{prefix}},
A_{\text{postfix}}^\tau
]
}
$$

也就是：

```
干净 prefix | 加噪 postfix
```

论文 Algorithm 1 对应代码就是：

```
prefix_mask = arange(ah) < delay
time = where(prefix_mask, 1.0, time)

x_t =
    time * action_chunk
  + (1-time) * noise
```

---

### 第七步：模型预测 velocity

$\hat v=v_\theta(o,x,\tau_{\text{masked}})$

---

### 第八步：计算目标速度

$v^*=A-\epsilon$

---

### 第九步：只对 postfix 算误差

$$
L
=
\sum_{i=d}^{H-1}
\|\hat v_i-v_i^*\|^2
$$

然后反向传播。

久而久之模型学到的不是普通的：$p(A\mid o)$

而是：

$$
\boxed{
p(A_{d:H}\mid o,A_{0:d},d)
}
$$

也就是：

> **给我已经确定的未来前几步动作，我会把它自然地续写下去。**

---

# 十二、真正推理时怎么算？

训练学会后，推理反而非常简单。

现在假设：$d=3$

上一 chunk 已经给出了：$A_\text{prefix}=[a_t,a_{t+1},a_{t+2}]$

这些动作在 VLA 计算期间会被实际执行。

---

## 第一步：postfix 从噪声开始

初始化：$x^0\sim N(0,I)$

但 prefix 已经知道了。

---

## 第二步：每次迭代都强制把 prefix 写回去

论文代码：

```
x_t = where(prefix_mask, action_prefix, x_t)
```

所以假设当前：$x=[-0.5,1.4,-0.2,0.8,\dots]$

而真正 prefix：$[0.1,0.15,0.2]$

那么直接改成：$x=[0.1,0.15,0.2,0.8,\dots]$

---

## 第三步：prefix 的 time 固定为 1

$$
\tau_i=
\begin{cases}
1&i<d\\
\tau&i\ge d
\end{cases}
$$

于是模型知道：

```
这些位置 = 已经确定
这些位置 = 还在生成
```

---

## 第四步：模型预测速度

$v=v_\theta(o,x,\tau)$

---

## 第五步：积分

postfix 逐渐：$x\leftarrow x+\Delta t\,v$

然后：$\tau\leftarrow\tau+\Delta t$

重复 $N$ 次。

论文的采样过程完整写在 Algorithm 1。

最终得到：

$$
\boxed{
[
\underbrace{a_t,a_{t+1},a_{t+2}}_{\text{旧 chunk 给定}}
,
\underbrace{
\hat a_{t+3},\hat a_{t+4},…
}_{\text{新模型生成}}
]
}
$$

---

# 十三、为什么这样就能让两个 chunk 平滑连接？

因为新轨迹不是凭空产生的。

普通 async：$p(A_\text{new}\mid o)$

它并不知道：$A_\text{old}$

所以可能：

```
旧chunk
────────╮
        │
        ╰──────── 新chunk
         ↑ jump
```

RTC 则是：$p(A_\text{postfix}\mid o,A_\text{prefix})$

也就是：

```
已知旧轨迹：
──────────

请从这里继续：
          ─────────────
```

这跟语言模型特别像。

普通 action generation 类似：

> 请从头写一句话。
RTC 类似：

> 已知前半句话是 “今天天气很好，所以我准备……”
> 请续写。
你得到的后半句自然会和前半句衔接。

---

# 十四、 $d\le H-s$ 又是怎么来的？

论文 Figure 1 给出了：

$$
\boxed{d\le H-s}
$$

这个其实完全不用死记。

假设：$H=8$

每个 chunk 真正执行：$s=4$

上一 chunk 是：$[a_0,a_1,a_2,a_3,a_4,a_5,a_6,a_7]$

执行前 4 步后：$a_0,a_1,a_2,a_3$

已经执行。

还有：$a_4,a_5,a_6,a_7$

4 步可以“撑住”VLA 推理。

因此：$d_{\max}=4$

也就是：$H-s=8-4=4$

如果：$d=5$

那么上一 chunk 只剩 4 个动作：

```
a4 a5 a6 a7
```

但新模型要 5 个 timestep 才回来：

```
step1 step2 step3 step4 step5
```

第 5 步的时候：

> 没动作可以执行了。
所以必须：

$$
\boxed{
d\le H-s
}
$$

这是一个非常物理直觉的约束。

---

# 十五、为什么训练时 $d$ 要随机采样？

现实中推理延迟可能：

```
这一帧 80ms
下一帧 102ms
下一帧 91ms
下一帧 120ms
```

所以：$d$

不是固定值。

如果你永远训练：$d=5$

那么模型只学会：

> 给我前 5 步，我续写。
突然：$d=7$

模型可能没见过。

所以训练时：

$$
\boxed{
d\sim p(d)
}
$$

让模型同时学：$p(A_{1:H}\mid A_{0:1})$，$p(A_{2:H}\mid A_{0:2})$，$p(A_{3:H}\mid A_{0:3})$

……

真实 π0.6 实验里论文采样：$d\in[0,10]$

这在 50 Hz 机器人上对应最大约：$10\times20\text{ms}=200\text{ms}$

延迟。

---

# 十六、它和原始 inference-time RTC 到底区别在哪里？

两者目的完全一样：

$$
\boxed{
\text{让新 chunk 与旧 chunk 连续}
}
$$

区别是“什么时候施加这个条件”。

---

### 原始 inference-time RTC

模型本身训练时不会 prefix conditioning。

推理过程中通过 **pseudoinverse guidance / inpainting** 强行要求：$A_\text{new,prefix}\approx A_\text{old,prefix}$

而且原版 RTC 不仅使用必须锁死的 $d$ 个 prefix，还可以利用更多 overlap action，并通过逐渐减小的权重进行 **soft masking**。

好处：

> 灵活。
坏处：

> 每个 denoising step 都需要 vector-Jacobian product，也就是额外反向传播，因此推理更贵、更慢。

---

### Training-time RTC

训练时就让模型学：$p(A_\text{postfix}\mid o,A_\text{prefix})$

推理的时候不需要额外 optimization。

只需要：

```
prefix 强制覆盖
↓
正常 flow inference
```

所以：

$$
\boxed{
\text{把 inference-time constraint}
\rightarrow
\text{变成 learned conditional generation}
}
$$

我认为这是理解这篇论文最重要的一句话。

---

# 十七、从概率角度再理解一次，会更透彻

普通 VLA 学：

$$
\boxed{
p(A_t\mid o_t)
}
$$

意思：

> 看到当前画面，从头生成未来轨迹。
而 RTC 的现实情况是：

> 未来最靠前的一小段，其实已经由上一轮预测决定了。
所以真正应该学习的是：

$$
\boxed{
p(
A_{t+d:t+H}
\mid
o_t,
A_{t:t+d}
)
}
$$

论文对此有明确表述。

这其实是一个 **conditional completion problem**：

```
 observation
      ↓
┌────────────────────────┐
│  fixed prefix | ???    │
└────────────────────────┘
                 ↑
              要生成
```

因此 RTC 从数学本质来说并不神秘：

$$
\boxed{
\text{RTC = 带已知动作前缀的条件轨迹补全}
}
$$

而 Flow Matching 只是实现这个“轨迹补全”的生成工具。

---

# 十八、把整个 RTC 压缩成一个流程图

训练时：

```
真实动作块 A
[a0 a1 a2 a3 a4 a5 a6 a7]
              │
              │ 随机采 delay d=3
              ↓
[a0 a1 a2 | a3 a4 a5 a6 a7]
  prefix   |      postfix
              │
      prefix: τ=1
      postfix: τ=random
              ↓
[a0 a1 a2 | noisy noisy noisy …]
              │
              ↓
       Flow Matching Model
              │
              ↓
    预测整个 velocity field
              │
              ↓
       只算 postfix loss
              │
              ↓
模型学会：
“看到固定前缀后如何自然续写”
```

推理时：

```
上一 chunk
       ↓
获得接下来 d 步已确定动作
       ↓
prefix = [a_t … a_{t+d-1}]
       ↓
postfix 初始化为 noise
       ↓
[prefix | noise noise noise …]
       ↓
模型
       ↓
[prefix | better actions …]
       ↓
模型
       ↓
[prefix | final postfix …]
       ↓
新 chunk
```

---

# 十九、最后把核心数学只留下 5 个式子

如果你最后只想记住数学骨架，其实就是下面五步。

**1. 动作块**

$$
A_t=[a_t,\dots,a_{t+H-1}]
$$

**2. Flow Matching 加噪**

$$
A^\tau
=
\tau A+(1-\tau)\epsilon
$$

**3. prefix 强制作为完整真实数据**

$$
\tau_i=
\begin{cases}
1&i<d\\
\tau&i\ge d
\end{cases}
$$

于是：

$$
x_i=
\begin{cases}
A_i&i<d\\
\tau A_i+(1-\tau)\epsilon_i&i\ge d
\end{cases}
$$

**4. 模型学 velocity**

$$
v_\theta(x,o,\tau)
\approx
A-\epsilon
$$

但只对：$i\ge d$

计算 loss。

**5. 推理积分**

$$
x\leftarrow x+\Delta\tau\,v_\theta(x,o,\tau)
$$

并且每一步始终把：$x_{0:d}=A_\text{prefix}$

重新锁死。

最终模型实现的就是：

$$
\boxed{
A_\text{postfix}
\sim
p(
A_\text{postfix}
\mid
o,
A_\text{prefix},
d
)
}
$$

这就是这篇论文里 **training-time RTC 最核心的数学原理**。

有一点需要特别区分：你上传的这篇论文对**原始 inference-time RTC 的 pseudoinverse guidance** 只做了概括，没有把原 RTC 那套 Jacobian / pseudoinverse guidance 的完整公式推导重新展开。因此上面关于原版 RTC 我只按本文明确说明的机制解释，没有补造它未给出的公式；本文真正完整给出的数学和代码，是 training-time RTC。