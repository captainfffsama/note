#大模型 

原始视频地址：<https://www.bilibili.com/video/BV1pf421z757?spm_id_from=333.788.videopod.episodes&vd_source=c6acf7e2d08361599bddd176f227d590&p=2>

<https://www.bilibili.com/video/BV1pf421z757?spm_id_from=333.788.videopod.episodes&vd_source=c6acf7e2d08361599bddd176f227d590&p=2>

# 重点记录
## 神经网络基础
**为何一定要有激活函数：**
没有激活函数，多层网络之间的矩阵运算就可以合并，导致网络退化成单层网络。
**为何一定要多层：**
可以增加模型的表达能力。

## Transfer Learning, Self-supervised Learning and Pre-training

迁移学习有基于特征的迁移，基于参数的迁移。大模型在某种程度上是一种基于参数的迁移。

## Word 2 Vec

该方法就是以小窗口的形式在整篇文章上滑动，然后在小窗口上进行完形填空。

存在的挑战：

1. 人类语言有高度二义性，同一个词在不同语境下有不同意思。
2. 个别词被扣出的时候，小窗口填空其实可以填多个答案。