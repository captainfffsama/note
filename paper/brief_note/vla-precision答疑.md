[toc]

#  Q:结合论文 docs/paper/vla-precision.md，告诉我在代码中公式（9）具体 advantage 是如何计算的？ 请详细且通俗，不要修改代码

公式（9）里的 advantage，并不是 PPO/GAE 中由 reward 手工计算出的 $G_t-V(s_t)$。在这套代码里，它是 critic 网络直接输出的“动作相关分数”，然后用“人类纠正后的好动作”减去“被覆盖的原始坏动作”，构造相对 advantage。

## 1. 论文公式（9）在表达什么

论文中的公式是：

$$
\Delta A_{t,k}^{\mathrm{pair}}
=
A_{\phi_k}(\omega_t,\hat a_t^{\mathrm{exec}})
-
A_{\phi_k}(\omega_t,\hat a_t^{\mathrm{prop}})
$$
$$
\mathcal L_{\mathrm{rank}}
=
\mathbb E\left[
\frac{1}{K}
\sum_{k=1}^{K}
\left[m_c-\Delta A_{t,k}^{\mathrm{pair}}\right]_+^2
\right]
$$

其中：

- $\omega_t$：当前视觉观测和机器人状态。
- $\hat a_t^{\mathrm{exec}}$：最终真正执行的动作。发生人工干预时，通常就是人类纠正动作。
- $\hat a_t^{\mathrm{prop}}$：机器人在人工干预前原本提出的动作。
- $A_{\phi_k}$：第 $k$ 个 critic 的 advantage 分支。
- $m_c$：希望好动作至少领先坏动作的 margin。
- $K$：critic ensemble 的数量，当前代码默认是 2。

通俗来说，公式（9）要求：

> 在同一个状态下，人类纠正后真正执行的动作，其 advantage 至少要比机器人原本想执行的动作高 $m_c$。

---

## 2. 单个动作的 advantage 从哪里来

critic 在 [networks.py](/data/VLA-Precision/src/vla_precision/acob/networks.py:38) 中被拆成两个分支：

$$
Q_{\phi_k}(\omega,a)
=
V_{\phi_k}(\omega)
+
A_{\phi_k}(\omega,a)
$$

代码逻辑等价于：

```python
obs_enc = encoder(observations)

value = value_head(value_network(obs_enc))

advantage_inputs = concatenate([obs_enc, actions])
advantage = advantage_head(advantage_network(advantage_inputs))

q = value + advantage
```

所以：

### Value 分支

$$
V_{\phi_k}(\omega_t)
$$

只接收状态编码，不接收动作，回答的是：

> 当前这个状态总体上有多好？

### Advantage 分支

$$
A_{\phi_k}(\omega_t,\hat a_t)
$$

同时接收状态编码和动作，回答的是：

> 在当前状态下，选择这个具体动作的相对好坏分数是多少？

一个很重要的实现细节是：

> 代码并不是先算出 $Q$，再显式执行 `advantage = Q - V`。

而是网络直接产生：

```python
value
advantage
q = value + advantage
```

因此，`advantage` 是神经网络 advantage head 的原始输出，并不是代码运行时用 return 减 value 临时计算出来的数值。

---

## 3. 公式（9）中的两个 advantage 怎样得到

核心实现位于 [agent.py](/data/VLA-Precision/src/vla_precision/acob/agent.py:424)。

首先准备两组动作：

```python
good_actions = batch["actions"]
bad_actions = batch["intervention_bad_actions"]
```

含义分别是：

- `batch["actions"]`：最终执行的动作，即论文中的 $a_t^{\mathrm{exec}}$。
- `batch["intervention_bad_actions"]`：发生人工干预前，策略原本提出但被覆盖的动作，即 $a_t^{\mathrm{prop}}$。

两者都会先转换成 critic 使用的扁平动作表示：

```python
good_actions = self._env_actions_to_critic_actions(...)
bad_actions = self._env_actions_to_critic_actions(...)
```

随后，把相同的 observation 分别和两个动作送进同一组 critic：

```python
predicted_good_advantages
bad_outputs = self.forward_critic_components(
    observations,
    bad_actions,
)
bad_advantages = bad_outputs["advantage"]
```

于是，对于每一个 critic $k$，得到：

$$
A_{t,k}^{\mathrm{good}}
=
A_{\phi_k}(\omega_t,\hat a_t^{\mathrm{exec}})
$$

以及：

$$
A_{t,k}^{\mathrm{bad}}
=
A_{\phi_k}(\omega_t,\hat a_t^{\mathrm{prop}})
$$

代码紧接着相减：

```python
margin_advantages = predicted_good_advantages - bad_advantages
```

这正对应：

$$
\Delta A_{t,k}^{\mathrm{pair}}
=
A_{t,k}^{\mathrm{good}}-A_{t,k}^{\mathrm{bad}}
$$

注意，这里是每个 critic 内部分别相减，不是先对 ensemble 求最小值或平均值再相减。

---

## 4. 为什么这里可以直接比较 advantage

对于同一个 critic、同一个状态：

$$
Q(\omega,a_{\mathrm{good}})
=
V(\omega)+A(\omega,a_{\mathrm{good}})
$$

$$
Q(\omega,a_{\mathrm{bad}})
=
V(\omega)+A(\omega,a_{\mathrm{bad}})
$$

相减后：

$$
Q(\omega,a_{\mathrm{good}})
-
Q(\omega,a_{\mathrm{bad}})
=
A(\omega,a_{\mathrm{good}})
-
A(\omega,a_{\mathrm{bad}})
$$

因为两个动作来自同一个状态，状态价值 $V(\omega)$ 完全抵消。

这正是论文专门使用 advantage 差值的原因：它只比较动作，不让“这个状态本身容易还是困难”干扰动作排序。

---

## 5. 代码怎样把 advantage 差转换成 loss

代码为每个 critic 计算：

```python
target_margin = critic_intervention_pref_margin
violation = relu(target_margin - margin_advantages)
per_sample_loss = mean(square(violation), axis=0)
```

对应数学形式：

$$
e_{t,k}
=
\max\left(
m_c-\Delta A_{t,k}^{\mathrm{pair}},
0
\right)
$$

$$
\ell_t
=
\frac{1}{K}\sum_k e_{t,k}^2
$$

当前默认配置在 [schema.py](/data/VLA-Precision/src/vla_precision/config/schema.py:323) 中是：

```python
critic_intervention_preference_margin = 0.05
critic_intervention_preference_weight = 50.0
```

也就是说，默认希望每个 critic 都满足：

$$
A_{\mathrm{good}}-A_{\mathrm{bad}}\geq 0.05
$$

### 一个具体例子

假设两个 critic 的输出为：

| critic   | 好动作 advantage | 坏动作 advantage | 差值 |
| -------- | ---------------: | ---------------: | ---: |
| critic 1 |             0.30 |             0.20 | 0.10 |
| critic 2 |             0.12 |             0.10 | 0.02 |

margin 为 $0.05$。

对于 critic 1：

$$
\max(0.05-0.10,0)^2=0
$$

它已经认为好动作领先了足够多，不产生惩罚。

对于 critic 2：

$$
\max(0.05-0.02,0)^2=0.0009
$$

两个 critic 取平均：

$$
\ell_t=(0+0.0009)/2=0.00045
$$

训练会继续调整 critic，使第二个 critic 也把好动作排在坏动作之上，并至少拉开 0.05。

---

## 6. 哪些样本真的参与公式（9）

不是所有 replay 样本都会进入 ranking loss。代码构造了：

```python
mask = batch["intervened"]
action_l2 = norm(good_actions - bad_actions)
valid_mask = mask * (action_l2 > 1e-6)
```

因此一个样本必须同时满足：

1. `intervened == 1`：确实发生过人工干预；
2. 最终执行动作和原始提议动作确实不同；
3. 如果启用了额外的数据来源屏蔽，还必须具有有效的 RL sample weight。

这对应论文中的 $c_t=1$。

代码用 `1e-6` 作为动作差异阈值。也就是说，如果只是数值浮点误差、两个动作实际上没有区别，就不会把它当作有效偏好对。

最终只在有效样本上平均：

```python
loss = sum(per_sample_loss * valid_mask) / max(pair_count, 1)
```

若一个 batch 中没有有效干预对，分母被限制为至少 1，loss 自然为 0，不会出现除零。

---

## 7. advantage 自己是怎么学会“好坏”的

advantage 没有单独的人工标签，例如不会直接告诉网络：

```text
好动作 advantage = 1
坏动作 advantage = 0
```

它通过两个目标联合学习。

### TD loss 提供绝对回报信息

在 [agent.py](/data/VLA-Precision/src/vla_precision/acob/agent.py:500) 中，代码计算：

$$
y_t
=
R_t^{(H)}
+
\gamma^H(1-d_t)
\min_k Q_{\bar\phi_k}(s_{t+1},a_{t+1})
$$

然后训练：

$$
Q_{\phi_k}(s_t,a_t)
=
V_{\phi_k}(s_t)+A_{\phi_k}(s_t,a_t)
$$

去逼近这个 TD target。

这让 critic 知道一个实际执行动作最终大致能带来多少回报。

### 公式（9）提供局部动作排序

对于被人工纠正的时刻，原提议动作没有真正执行，因此没有对应的真实下一状态，无法正常做 TD 学习。

公式（9）补充了一个局部监督：

> 虽然不知道原提议动作最终会得到多少 reward，但人类在同一状态下主动把它换掉，说明纠正动作应当优于原动作。

最终 critic loss 是：

$$
\mathcal L_{\mathrm{critic}}
=
\mathcal L_{\mathrm{TD}}
+
\lambda_{\mathrm{rank}}\mathcal L_{\mathrm{rank}}
$$

默认 $\lambda_{\mathrm{rank}}=50$。

---

## 8. ensemble 是怎样处理的

公式（9）的训练 loss 对所有 critic 的违约量取平均：

```python
per_sample_loss = mean(square(violation), axis=0)
```

因此每个 critic 都会单独学习：

$$
A_k(\mathrm{good})-A_k(\mathrm{bad})\geq m_c
$$

代码还计算了一个只用于监控的悲观 margin：

```python
min_advantage_margin = (
    predicted_good_advantages - bad_advantages
).min(axis=0)
```

即：

$$
\min_k
\left[
A_k(\mathrm{good})-A_k(\mathrm{bad})
\right]
$$

只有最悲观的 critic 也认为纠正动作更好，这个监控值才会大于 0。

需要区分：

- 公式（9）的 critic ranking loss：每个 critic 的 hinge-square loss取平均。
- 后续 actor 相对优势优化：先在每个 critic 内比较，再取所有 critic 中最小的差值，做悲观策略更新。

---

## 9. 最简洁的代码等价表达

忽略 batch、mask 和动作预处理后，公式（9）的代码本质上就是：

```python
good_adv = critic.advantage(state, executed_action)
bad_adv = critic.advantage(state, proposed_action)

delta_adv = good_adv - bad_adv
loss = mean(relu(margin - delta_adv) ** 2)
```

一句话总结：

> 代码中的公式（9）不是根据 reward 直接算 advantage，而是让 critic 的 advantage 网络分别评价同一状态下的“最终执行动作”和“被人工否决的原提议动作”，两者相减得到成对 advantage；如果好动作领先不足默认的 0.05，就通过平方 hinge loss 训练 critic 把它们拉开。

本次只进行了代码和论文的只读核对，没有修改任何文件。任务核验耗时约 1 分 41 秒。





严格来说，critic 的动作相关分数 $A_{\phi_k}(\omega,a)$ 没有一个直接的数值标签。代码通过两类监督信号间接训练它：

1. 公式（6）和（7）提供 Q 值的数值标签；
2. 公式（9）提供 advantage 的相对排序标签。



# Q:在训练时critic输出的动作相关分数的标签是如何得到的？对应paper 中的公式几？

## 1. 第一类标签：TD target，对应公式（6）、（7）

论文公式（6）构造 TD 目标：

$$
y_t
=
R_t^{(H)}
+
\bar\gamma(1-d_t)
\min_k Q_{\bar\phi_k}
(\omega_{t+1},\hat a_{t+1}^{\theta_n})
$$

代码在 [agent.py](/data/VLA-Precision/src/vla_precision/acob/agent.py:500) 中对应地计算：

```python
target_next_qs = self.forward_target_critic(
    batch["next_observations"],
    next_actions,
)

target_next_min_q = target_next_qs.min(axis=0)

target_q = (
    chunk_return
    + bootstrap_discount * batch["masks"] * target_next_min_q
)
```

这里的 `target_q` 就是 critic 的回归标签 $y_t$。

其中：

$$
R_t^{(H)}
=
r_t+\gamma r_{t+1}+\cdots+\gamma^{H-1}r_{t+H-1}
$$

代码为：

```python
discount_powers = discount ** arange(H)
chunk_return = sum(rewards * discount_powers)
bootstrap_discount = discount ** H
```

随后通过论文公式（7）训练 critic：

$$
\mathcal L_{\mathrm{TD}}
=
\frac1K\sum_k
\left(
Q_{\phi_k}(\omega_t,\hat a_t^{\mathrm{exec}})
-
y_t
\right)^2
$$

代码中是：

```python
predicted_outputs = self.forward_critic_components(
    observations,
    actions,
)

predicted_qs = predicted_outputs["q"]

td_loss = mean(
    (predicted_qs - target_qs) ** 2
)
```

而网络内部：

$$
Q_{\phi_k}(\omega,a)
=
V_{\phi_k}(\omega)+A_{\phi_k}(\omega,a)
$$

因此 TD 标签虽然监督的是 $Q$，梯度会同时更新：

- 状态分支 $V_{\phi_k}(\omega)$
- 动作分支 $A_{\phi_k}(\omega,a)$

也就是说，动作相关分数 $A$ 的第一部分学习信号来自长期回报，但并没有一个单独的 `target_advantage`。

---

## 2. 第二类标签：人类干预产生的偏好关系，对应公式（9）

公式（9）并不给 advantage 一个确定数值，而是提供一个排序标签：

$$
a_t^{\mathrm{exec}} \succ a_t^{\mathrm{prop}}
$$

即：

> 人类纠正后最终执行的动作，应当优于机器人原本提出、后来被覆盖的动作。

代码取出：

```python
good_actions = batch["actions"]
bad_actions = batch["intervention_bad_actions"]
```

分别计算：

$$
A_{t,k}^{\mathrm{good}}
=
A_{\phi_k}(\omega_t,\hat a_t^{\mathrm{exec}})
$$

$$
A_{t,k}^{\mathrm{bad}}
=
A_{\phi_k}(\omega_t,\hat a_t^{\mathrm{prop}})
$$

然后相减：

```python
margin_advantages = (
    predicted_good_advantages
    - bad_advantages
)
```

得到：

$$
\Delta A_{t,k}^{\mathrm{pair}}
=
A_{t,k}^{\mathrm{good}}
-
A_{t,k}^{\mathrm{bad}}
$$

公式（9）要求：

$$
\Delta A_{t,k}^{\mathrm{pair}}\ge m_c
$$

代码使用平方 hinge loss：

```python
violation = relu(target_margin - margin_advantages)
loss = mean(violation ** 2)
```

默认：

$$
m_c=0.05
$$

因此它相当于给 critic 一个相对标签：

```text
good advantage 至少应比 bad advantage 高 0.05
```

但不会规定：

```text
good advantage 必须等于 1
bad advantage 必须等于 0
```

---

## 3. 两种监督如何配合

可以把训练理解成：

| 信号                             | 对应公式   | 给出的标签/约束                                 |
| -------------------------------- | ---------- | ----------------------------------------------- |
| 环境 reward + 下一状态 bootstrap | 公式（6）  | Q 的数值目标 $y_t$                              |
| TD 回归                          | 公式（7）  | 让 $V+A$ 接近 $y_t$                             |
| 人类干预                         | 公式（9）  | 让 $A_{\mathrm{exec}}-A_{\mathrm{prop}}\ge m_c$ |
| 总 critic loss                   | 公式（10） | 将 TD 与局部排序联合训练                        |

总损失为：

$$
\mathcal L_{\mathrm{critic}}
=
\mathcal L_{\mathrm{TD}}
+
\lambda_{\mathrm{rank}}\mathcal L_{\mathrm{rank}}
$$

默认代码中：

$$
m_c=0.05,\qquad \lambda_{\mathrm{rank}}=50
$$

---

## 4. 为什么被覆盖动作没有 TD 标签

发生人工干预时：

- `exec action` 真正被机器人执行了；
- `proposal action` 被人类覆盖，没有实际执行。

因此，最终执行动作有：

- reward；
- next observation；
- terminal 状态；
- TD target。

但是原提议动作没有真实的后继状态：

$$
(s_t,a_t^{\mathrm{prop}},r_t,s_{t+1})
$$

这条 transition 实际上并不存在，所以不能可靠地给它构造公式（6）的 TD 标签。

论文因此明确让原提议动作：

- 不参与 TD 回归；
- 只参与公式（9）的同状态偏好排序。

这是公式（9）存在的核心原因。

## 结论

critic 的动作相关分数没有直接的标量标签：

$$
\boxed{\text{不存在单独的 }A_{\text{target}}}
$$

它通过两种方式学出来：

$$
\boxed{
\text{公式（6）、（7）：用回报产生的 TD target 监督 }Q=V+A
}
$$

以及：

$$
\boxed{
\text{公式（9）：用人类干预监督两个动作的 advantage 排序}
}
$$

所以如果你问“动作分数的绝对数值标签来自哪里”，答案是公式（6）的 TD target；如果问“好动作应该比坏动作高多少”，答案是公式（9）的 margin-ranking 约束。



# Q:critic 网络的结构是怎么样的？

critic 是一个“双头、双 critic ensemble”的网络：共享状态编码器，之后分成 Value 分支和 Advantage 分支，最终相加得到 Q。

$$
Q_{\phi_k}(\omega,a)
=
V_{\phi_k}(\omega)
+
A_{\phi_k}(\omega,a),
\qquad k=1,2
$$
整体结构可以概括为：

```text
多路图像 ──→ 每路 ResNet-10 ──→ Spatial Learned Embedding ──→ 256维特征 ─┐
                                                                       ├─→ 状态特征 ω_enc
机器人状态 ──→ Dense(64) → LayerNorm → tanh ─────────────────────────────┘
                                      │
                   ┌──────────────────┴──────────────────┐
                   │                                     │
                   ▼                                     ▼
          Value MLP，只看状态                    拼接完整动作 chunk
          [256, 256]                             [ω_enc, flatten(a)]
                   │                                     │
                Dense(1)                         Advantage MLP [256,256]
                   │                                     │
                 V(ω)                                 Dense(1)
                                                         │
                                                      A(ω,a)
                   │                                     │
                   └──────────────────┬──────────────────┘
                                      ▼
                              Q(ω,a) = V(ω) + A(ω,a)
```

以上结构独立初始化两份，构成 $K=2$ 的 critic ensemble。

## 1. 输入是什么

critic 的状态输入不是 VLA 的语言 token 或 KV context，而是论文中的：

$$
\omega_t=(o_t,q_t)
$$

其中：

- $o_t$：一幅或多幅相机图像；
- $q_t$：机器人本体状态，例如关节状态、末端位姿等；
- 不直接包含任务语言 $\ell$。

实现入口是 [vision.py](/data/VLA-Precision/src/vla_precision/acob/vision.py:34) 中的 `EncodingWrapper`。

### 图像输入

对于配置中的每个 `image_key`，都会建立一个独立的图像 encoder：

```python
encoders = {
    image_key: PreTrainedResNetEncoder(...)
    for image_key in image_keys
}
```

默认配置为：

```python
critic_encoder = "resnet-pretrained"
```

即使用预训练的 ResNet-10。每个相机不是简单地提前拼在一起，而是：

1. 每路图像分别经过自己的 encoder；
2. 得到每路图像特征；
3. 最后在特征维度拼接。

如果 observation 带有时间堆叠，`EncodingWrapper` 当前只保留最后一个图像帧：

```python
image = image[:, -1:]
```

然后把时间维和通道维合并。

---

## 2. 图像编码器

默认图像编码器创建于 [agent.py](/data/VLA-Precision/src/vla_precision/acob/agent.py:1009)：

```python
pretrained_encoder = ResNet10Frozen(
    pre_pooling=True
)

PreTrainedResNetEncoder(
    pretrained_encoder=pretrained_encoder,
    pooling_method="spatial_learned_embeddings",
    num_spatial_blocks=8,
    bottleneck_dim=256,
)
```

每路图像大致经过：

```text
图像
 ↓
冻结的预训练 ResNet-10 卷积主干
 ↓
H' × W' × C 的空间 feature map
 ↓
SpatialLearnedEmbeddings，8 个空间权重模板
 ↓
展平特征
 ↓
Dense(256)
 ↓
LayerNorm
 ↓
tanh
 ↓
256 维图像特征
```

### “冻结”具体指什么

预训练 ResNet 的 feature map 经过了：

```python
jax.lax.stop_gradient(x)
```

因此卷积主干不会被 critic loss 更新。

但是后面的这些部分仍然可以训练：

- `SpatialLearnedEmbeddings`
- 256 维 bottleneck Dense
- LayerNorm
- 后续 V/A MLP

这样既利用预训练视觉特征，又降低在线 RL 训练视觉表示时的不稳定性和计算成本。

### Spatial Learned Embedding

它不是普通的 global average pooling。

代码为每个空间位置和通道学习 8 个权重模板，然后对空间维度加权求和：

$$
e_{c,j}
=
\sum_{h,w}
F_{h,w,c}W_{h,w,c,j},
\qquad j=1,\ldots,8
$$

因此网络可以学习：

> 图像中哪些空间位置对判断动作好坏更重要。

例如在机械臂精密操作中，它可能更关注夹爪、容器口、吸头等区域。

---

## 3. 机器人状态编码

机器人本体状态 `observations["state"]` 经过：

```python
Dense(64)
LayerNorm
tanh
```

得到一个 64 维状态特征。

然后与所有相机的图像特征拼接：

$$
\omega_{\mathrm{enc}}
=
[
e_{\mathrm{cam1}},
e_{\mathrm{cam2}},
\ldots,
e_{\mathrm{proprio}}
]
$$

如果有 $N$ 路相机，并且每路输出 256 维，那么状态编码总维度大致是：

$$
D_{\mathrm{state}}=256N+64
$$

例如两路相机时：

$$
D_{\mathrm{state}}=256\times2+64=576
$$

---

## 4. 动作如何输入 critic

动作不是单步动作，而是完整的 action chunk：

$$
a_t=(u_{t,0},\ldots,u_{t,H-1})
$$

环境动作首先通过和 OpenPI 一致的变换进行归一化：

```python
normalized_chunk = self.adapter.action_chunk(
    actions,
    observations,
)
```

然后把 chunk 展平：

$$
\hat a_t
=
\operatorname{vec}
\left(
[u_{t,0},\ldots,u_{t,H-1}]
\right)
$$

例如：

- action horizon $H=16$
- 单步动作维度 $d=7$

那么输入 advantage 网络的动作向量维度就是：

$$
16\times7=112
$$

处理代码位于 [agent.py](/data/VLA-Precision/src/vla_precision/acob/agent.py:216)。

对于固定夹爪任务，critic 会去掉不活动的夹爪维度；其他情况下通常保留所有环境动作维度。这对应论文中的：

$$
\hat a=C(\tilde a)=\operatorname{vec}(P_c\tilde a_{0:H})
$$

---

## 5. Value 分支

Value 分支只接收状态特征：

$$
V_{\phi_k}(\omega_t)
=
f^V_{\phi_k}(\omega_{\mathrm{enc}})
$$

具体网络是：

```text
状态编码
  ↓
Dense(256)
  ↓
LayerNorm
  ↓
tanh
  ↓
Dense(256)
  ↓
LayerNorm
  ↓
tanh
  ↓
Dense(1)
  ↓
标量 V
```

构造代码位于 [agent.py](/data/VLA-Precision/src/vla_precision/acob/agent.py:1027)：

```python
MLP(
    hidden_dims=[256, 256],
    activations=tanh,
    use_layer_norm=True,
    activate_final=True,
)
```

这里 `activate_final=True` 表示第二个 256 维隐藏层后面也使用 LayerNorm 和 tanh。最后再额外接一个线性的 `Dense(1)` value head。

Value 分支回答：

> 不考虑具体采取哪个动作，当前状态整体有多好？

---

## 6. Advantage 分支

Advantage 分支同时接收状态特征和展平动作：

$$
x_A=[\omega_{\mathrm{enc}},\hat a_t]
$$

网络结构同样是：

```text
[状态编码, 展平动作 chunk]
  ↓
Dense(256)
  ↓
LayerNorm
  ↓
tanh
  ↓
Dense(256)
  ↓
LayerNorm
  ↓
tanh
  ↓
Dense(1)
  ↓
标量 A(ω,a)
```

实现位于 [networks.py](/data/VLA-Precision/src/vla_precision/acob/networks.py:63)：

```python
advantage_inputs = jnp.concatenate(
    [obs_enc, actions],
    axis=-1,
)

advantage_outputs = self.advantage_network(
    advantage_inputs
)

advantage = Dense(1)(advantage_outputs)
```

它回答：

> 在当前状态下，这一个完整动作序列相对怎么样？

这里不是对 chunk 内每一步分别输出一个分数，而是对完整 action chunk 输出一个标量。

---

## 7. 最终 Q 值

两个分支直接相加：

```python
q = value + advantage
```

即：

$$
Q_{\phi_k}(\omega,a)
=
V_{\phi_k}(\omega)
+
A_{\phi_k}(\omega,a)
$$

调用方可以一次拿到三个结果：

```python
{
    "q": q,
    "value": value,
    "advantage": advantage,
}
```

其中：

- TD loss 使用 `q`；
- 论文公式（9）的局部排序使用 `advantage`；
- actor 的相对优势优化也使用 `advantage`。

---

## 8. 为什么要拆成 V 和 A 两个分支

假设同一个困难状态下有两个动作：

$$
Q(\omega,a_{\mathrm{good}})=-2.0
$$

$$
Q(\omega,a_{\mathrm{bad}})=-2.5
$$

直接看 Q，两个值都很低；但它们低可能主要是因为这个状态本身很困难。

拆解后可能是：

$$
V(\omega)=-3.0
$$

$$
A(\omega,a_{\mathrm{good}})=1.0
$$

$$
A(\omega,a_{\mathrm{bad}})=0.5
$$

此时 advantage 差清楚地表示：

$$
A_{\mathrm{good}}-A_{\mathrm{bad}}=0.5
$$

因为比较的是同一状态，状态项 $V(\omega)$ 被消掉。这使公式（9）可以专注于“哪个动作更好”，而不是“当前状态有多难”。

---

## 9. 双 critic ensemble

V 分支和 A 分支都通过 `ensemblize(..., 2)` 建立两份独立参数：

$$
Q_{\phi_1}=V_{\phi_1}+A_{\phi_1}
$$

$$
Q_{\phi_2}=V_{\phi_2}+A_{\phi_2}
$$

这里不是：

- 一个共享 V 加两个 A；
- 也不是共享 MLP 后接两个输出头。

而是两个 ensemble member 分别拥有自己的：

- Value MLP 参数；
- Advantage MLP 参数；
- `Dense(1)` 输出头。

状态图像 encoder 位于 ensemble 外部，因此视觉/本体编码器是共享的。

做 TD bootstrap 时取：

$$
\min(Q_{\bar\phi_1},Q_{\bar\phi_2})
$$

做相对 advantage actor 更新时则先分别计算：

$$
A_{\phi_k}(\omega,a_{\mathrm{current}})
-
A_{\phi_k}(\omega,a_{\mathrm{baseline}})
$$

再取两个 critic 中较小的差值。这是一种悲观估计，用来减少 critic 偶然高估某个动作带来的策略漂移。

---

## 10. 还有一套 target critic

除了在线 critic，训练状态中还保存相同结构的 target critic：

```python
critic_state.params
critic_state.target_params
```

target critic 使用软更新：

$$
\bar\phi
\leftarrow
\tau\phi+(1-\tau)\bar\phi
$$

代码默认：

$$
\tau=0.005
$$

在线 critic 用于当前预测和反向传播；target critic 用于公式（6）里的下一状态 bootstrap，使 TD target 不会随着当前 critic 每次更新而剧烈变化。

简而言之，这个 critic 可以理解为：

> 共享的多模态状态编码器，加上两个独立的双层 MLP ensemble；每个 ensemble member 都用一个只看状态的 V 分支和一个同时看状态及完整动作 chunk 的 A 分支，二者相加得到 Q。