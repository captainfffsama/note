#损失函数 

Dice Loss 是一种常用于医学图像分割任务中的损失函数，尤其适合处理类别不平衡问题，如背景像素远多于前景像素的情况。它源自 Sørensen-Dice 系数（也称为 Dice 相似性系数），该系数是用来衡量两个样本集之间相似度的一种统计量。在深度学习中，Dice Loss 被用来优化模型输出与真实标签之间的重叠度，促进模型学习到更精确的边界信息。

Dice Loss 的公式通常定义为：

$$
\text{Dice Loss} = 1 - \text{Dice Coefficient} 
$$

而 Dice Coefficient（Dice 相似性系数）的计算公式为：

$$
\text{Dice Coefficient} = \frac{2 \times \text{TP}}{2 \times \text{TP} + \text{FN} + \text{FP}}
$$

其中：

- TP（True Positives）是真正例，即模型预测为正类且实际也是正类的像素数量。
- FP（False Positives）是假正例，即模型预测为正类但实际上为负类的像素数量。
- FN（False Negatives）是假反例，即模型预测为负类但实际上为正类的像素数量。

在深度学习的上下文中，为了方便计算和梯度传播，Dice Loss 通常会被转换为连续形式，适用于神经网络的输出（通常是预测概率）。对于二分类问题，假设\(p\) 是模型对某个像素属于正类的概率，\(g\) 是该像素的真实标签（0 或 1），那么 Dice Loss 的连续形式可以表示为：

$$
\text{Dice Loss} = 1 - \frac{2 \times \sum_{i} p_i \cdot g_i + \epsilon}{\sum_{i} p_i + \sum_{i} g_i + \epsilon}
$$

这里， $\sum_{i}$ 是对所有像素求和， $\epsilon$ 是一个很小的正值（如 $1e^{-6}$ ），用于防止除以零的情况发生。

通过最小化 Dice Loss，模型被鼓励提高预测概率与真实标签之间的重叠度，从而在图像分割任务中实现更准确的边界定位和类别区分。