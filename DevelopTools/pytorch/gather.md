#pytorch 

[toc]
[原文](https://zhuanlan.zhihu.com/p/352877584)
# 1 背景

去年我理解了**torch.gather()**用法，今年看到又给忘了，索性把自己的理解梳理出来，方便今后遗忘后快速上手。

**官方文档：** 

![](https://pic4.zhimg.com/v2-f8a19c1c3d3e4da167f9517615bf6857_b.jpg)

官方文档对torch.gather()的定义非常简洁

> 定义：从原tensor中获取指定dim和指定index的数据

看到这个核心定义，我们很容易想到`gather()`的**基本想法**其实就类似**从完整数据中按索引取值**般简单，比如下面从列表中按索引取值

```python3
lst = [1, 2, 3, 4, 5]
value = lst[2]  # value = 3
value = lst[2:4]  # value = [3, 4]
```

上面的取值例子是**取单个值**或具**有逻辑顺序序列**的例子，而对于深度学习常用的**批量tensor**数据来说，我们的需求可能是选取其中**多个且乱序**的值，此时`gather()`就是一个很好的tool，它可以帮助我们从批量tensor中取出指定乱序索引下的数据，因此其用途如下

> 用途：方便从批量tensor中获取指定索引下的数据，该索引是**高度自定义化**的，可乱序的

# 2 实战


我们找个3x3的二维矩阵做个实验

```python3
import torch

tensor_0 = torch.arange(3, 12).view(3, 3)
print(tensor_0)
```

输出结果

```python3
tensor([[ 3,  4,  5],
        [ 6,  7,  8],
        [ 9, 10, 11]])
```

## 2.1 输入行向量index，并替换行索引(dim=0)

```python3
index = torch.tensor([[2, 1, 0]])
tensor_1 = tensor_0.gather(0, index)
print(tensor_1)
```

输出结果
```python3
tensor([[9, 7, 5]])
```

过程如图所示

![](https://pic4.zhimg.com/v2-1aed61fbaa97775e816c23d8d907bea3_b.jpg)

## 2.2 输入行向量index，并替换列索引(dim=1)

```python3
index = torch.tensor([[2, 1, 0]])
tensor_1 = tensor_0.gather(1, index)
print(tensor_1)
```

输出结果
```python3
tensor([[5, 4, 3]])
```

过程如图所示

![](https://pic2.zhimg.com/v2-a437754fed6b29f5af13927ee06e13f9_b.jpg)

## 2.3 输入列向量index，并替换列索引(dim=1)

```python3
index = torch.tensor([[2, 1, 0]]).t()
tensor_1 = tensor_0.gather(1, index)
print(tensor_1)
```

输出结果
```python3
tensor([[5],
        [7],
        [9]])
```

过程如图所示

![](https://pic4.zhimg.com/v2-75bcf4697138165941cb2ed083475a07_b.jpg)

## 2.4 输入二维矩阵index，并替换列索引(dim=1)

```python3
index = torch.tensor([[0, 2], 
                      [1, 2]])
tensor_1 = tensor_0.gather(1, index)
print(tensor_1)
```

输出结果
```python3
tensor([[3, 5],
        [7, 8]])
```

过程同上

# 3 在强化学习DQN中的使用


在PyTorch官网DQN页面的代码中，是这样获取![](https://www.zhihu.com/equation?tex=Q%28S_t%2Ca%29)
的

```python3
# Compute Q(s_t, a) - the model computes Q(s_t), then we select the
# columns of actions taken. These are the actions which would've been taken
# for each batch state according to policy_net
state_action_values = policy_net(state_batch).gather(1, action_batch)
```

其中![](https://www.zhihu.com/equation?tex=Q%28S_t%29)
，即policy\_net(state\_batch)为shape=(128, 2)的二维表，动作数为2

![](https://pic3.zhimg.com/v2-1c17f6ed7a009f8b0b177a0a88ab00ce_b.jpg)

而我们通过神经网络输出的对应**批量动作**为

![](https://pic4.zhimg.com/v2-aca086b8c902ae30e4b3e0dc0fba9be7_b.jpg)

此时，使用gather()函数即可轻松获取**批量状态**对应**批量动作**的![](https://www.zhihu.com/equation?tex=Q%28S_t%2Ca%29)

# 4 总结


从以上典型案例，我们可以归纳出torch.gather()的使用要点

*   **输入index的shape等于输出value的shape**
*   **输入index的索引值仅替换该index中对应dim的index值**
*   **最终输出为替换index后在原tensor中的值**