#具身智能 #VLA #实验记录

# 模型参数情况

模型参数情况参见 [smolvla训练记录](smolvla训练记录.md), 区别在于将最终 loss 中的 padding 进行了截断，原始 loss 中将动作维数从 7 扩展到 32，然后 32 都贡献了 loss，这里将 padding 的 25 维去掉

## Loss

![](../../Attachments/smolvla-loss-remove-padding-tfboard.png)

# 真机测试现象

模型臂晃了半天还是还没准备下探