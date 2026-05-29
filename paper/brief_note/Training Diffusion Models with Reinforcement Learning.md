#强化学习 #扩散模型 

# Training Diffusion Models with Reinforcement Learning
- 论文：[[2311.01223v4] Diffusion Models for Reinforcement Learning: A Survey](https://arxiv.org/abs/2311.01223v4)

## 其他参考
- [使用 DDPO 在 TRL 中微调 Stable Diffusion 模型](https://huggingface.co/blog/zh/trl-ddpo)
- [Training Diffusion Models with Reinforcement Learning – The Berkeley Artificial Intelligence Research Blog](https://bair.berkeley.edu/blog/2023/07/14/ddpo/)
- [jannerm/ddpo: Code for the paper "Training Diffusion Models with Reinforcement Learning"](https://github.com/jannerm/ddpo)
- [[2311.01223v4] Diffusion Models for Reinforcement Learning: A Survey](https://arxiv.org/abs/2311.01223v4)
- [[2505.18876] DiffusionRL: Efficient Training of Diffusion Policies for Robotic Grasping Using RL-Adapted Large-Scale Datasets](https://arxiv.org/abs/2505.18876)

## 使用 DDPO 在 TRL 中微调 Stable Diffusion 模型记录

RLHF 的一般训练步骤如下: 

1. 有监督微调“基础”模型，以学习新数据的分布。
2. 收集偏好数据并用它训练奖励模型。
3. 使用奖励模型作为信号，通过强化学习对模型进行微调。              

需要指出的是，在 RLHF 中偏好数据是获取人类反馈的主要来源。 

DDPO 加进来后，整个工作流就变成了: 

1. 从预训练的扩散模型开始。
2. 收集偏好数据并用它训练奖励模型。
3. 使用奖励模型作为信号，通过 DDPO 微调模型   

请注意，DDPO 工作流把原始 RLHF 工作流中的第 3 步省略了，这是因为经验表明 (后面你也会亲眼见证) 这是不需要的。

下面我们实战一下，训练一个扩散模型来输出更符合人类审美的图像，我们分以下几步来走: 

1. 从预训练的 Stable Diffusion (SD) 模型开始。
2. 在美学视觉分析 (Aesthetic Visual Analysis，AVA) 数据集上训练一个带有可训回归头的冻结 CLIP 模型，用于预测人们对输入图像的平均喜爱程度。
3. 使用美学预测模型作为奖励信号，通过 DDPO 微调 SD 模型。
