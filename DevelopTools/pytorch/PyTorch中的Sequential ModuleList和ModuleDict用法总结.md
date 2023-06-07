#pytorch #sequential

[toc]

[原文](https://blog.csdn.net/QLeelq/article/details/115208866)

# 1. 区别于联系

首先来一张图，总体概括一下它们的区别：  
![](https://img-blog.csdnimg.cn/e7b327518a134d07994d9610a7ea855c.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBA6Z2e5pma6Z2e5pma,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)

* 区别

> 1.  `nn.Sequential` 内部实现了 `forward` 函数，因此可以不用写 `forward` 函数。而 `nn.ModuleList` 和 `nn.ModuleDict` 则没有实现内部 forward 函数。
> 2.  `nn.Sequential` 需要严格按照顺序执行，而其它两个模块则可以任意调用。

下面分别进行介绍。

## 1 .1 nn.Sequential

*   `nn.Sequential` 里面的模块按照 `顺序进行排列` 的，所以必须确保前一个模块的输出大小和下一个模块的输入大小是一致的。

*   `nn.Sequential` 中可以使用 `OrderedDict` 来指定每个 `module` 的名字。

## 1 .2 nn.ModuleList

*   `nn.ModuleList` 里面储存了不同 `module`，并自动将每个 `module` 的 `parameters` 添加到网络之中的容器 (`注册`)，里面的 module 是按照 List 的形式 `顺序存储` 的，但是在 forward 中调用的时候可以随意组合。

* 可以任意将 `nn.Module` 的子类 (比如 `nn.Conv2d`, `nn.Linear` 之类的) 加到这个 list 里面，方法和 Python 自带的 list 一样，也就是说它可以使用 `extend，append` 等操作。

## 1 .3 nn.ModuleDict

*   `ModuleDict` 可以像常规 Python 字典一样索引，同样自动将每个 `module` 的 `parameters` 添加到网络之中的容器 (`注册`)。

* 同样的它可以使用 OrderedDict、dict 或者 ModuleDict 对它进行 update，也就是追加。

# 2. nn.sequential

这里举两个例子来说明 `nn.sequential`，一个是直接通过 `nn.Sequential` 添加子模块，另一个方法是使用 `OrderedDict` 来指定每个模块的名字。

下面两种方法可以达到同样的效果。

```python
import torch.nn as nn

model = nn.Sequential(
          nn.Conv2d(1,20,5),
          nn.ReLU(),
          nn.Conv2d(20,64,5),
          nn.ReLU()
        )
print(model)
print('='*50)
from collections import OrderedDict


model = nn.Sequential(OrderedDict([
          ('conv1', nn.Conv2d(1,20,5)),
          ('relu1', nn.ReLU()),
          ('conv2', nn.Conv2d(20,64,5)),
          ('relu2', nn.ReLU())
        ]))

print(model)

```

输出：

```shell
Sequential(
  (0): Conv2d(1, 20, kernel_size=(5, 5), stride=(1, 1))
  (1): ReLU()
  (2): Conv2d(20, 64, kernel_size=(5, 5), stride=(1, 1))
  (3): ReLU()
)
==================================================
Sequential(
  (conv1): Conv2d(1, 20, kernel_size=(5, 5), stride=(1, 1))
  (relu1): ReLU()
  (conv2): Conv2d(20, 64, kernel_size=(5, 5), stride=(1, 1))
  (relu2): ReLU()
)

```

# 3. nn. ModuleList

一些常用方法：

*   append()：在 ModuleList 后面添加网络层

*   extend()：拼接两个 ModuleList

*   insert()：指定 ModuleList 中位置插入网络层

```python
import torch.nn as nn
class MyNet(nn.Module):
    def __init__(self):
        super(MyModule, self).__init__()
        self.linears = nn.ModuleList([nn.Linear(10, 10) for i in range(10)])

    def forward(self, x):
        
        for i, l in enumerate(self.linears):
            x = self.linears[i // 2](x) + l(x)
        return x

myNet = MyNet()
print(myNet)

```

输出：

```shell
MyModule(
  (linears): ModuleList(
    (0): Linear(in_features=10, out_features=10, bias=True)
    (1): Linear(in_features=10, out_features=10, bias=True)
    (2): Linear(in_features=10, out_features=10, bias=True)
    (3): Linear(in_features=10, out_features=10, bias=True)
    (4): Linear(in_features=10, out_features=10, bias=True)
    (5): Linear(in_features=10, out_features=10, bias=True)
    (6): Linear(in_features=10, out_features=10, bias=True)
    (7): Linear(in_features=10, out_features=10, bias=True)
    (8): Linear(in_features=10, out_features=10, bias=True)
    (9): Linear(in_features=10, out_features=10, bias=True)
  )
)

```

# 4. nn. ModuleDict

一些常用方法：

*   clear(): 清空 ModuleDict

*   items(): 返回可迭代的键值对 (key-value pairs)

*   keys(): 返回字典的键（key）

*   values(): 返回字典的值 (value)

*   pop(): 返回一对键值，并从字典中删除

*   update(): 添加 dict、OrderedDict 或者 ModuleDict 结构。

```python
import torch.nn as nn
class MyNet(nn.Module):
    def __init__(self):
        super(MyModule, self).__init__()
        self.choices = nn.ModuleDict({
                'conv': nn.Conv2d(10, 10, 3),
                'pool': nn.MaxPool2d(3)
        })
        self.activations = nn.ModuleDict([
                ['lrelu', nn.LeakyReLU()],
                ['prelu', nn.PReLU()]
        ])

    def forward(self, x, choice, act):
        
        
        return x
    
my_net = MyNet()
print(my_net)

```

输出如下，forward 不管怎么设计，都不能很好的打印出 my_net 网络。

```python
MyModule(
  (choices): ModuleDict(
    (conv): Conv2d(10, 10, kernel_size=(3, 3), stride=(1, 1))
    (pool): MaxPool2d(kernel_size=3, stride=3, padding=0, dilation=1, ceil_mode=False)
  )
  (activations): ModuleDict(
    (lrelu): LeakyReLU(negative_slope=0.01)
    (prelu): PReLU(num_parameters=1)
  )
)

```

# 5. 自行设计网络

## （1）使用 python 的 list 添加 (不可行)

```python
import torch.nn as nn
class MyNet(nn.Module):
    def __init__(self):
        super(net_modlist, self).__init__()
        self.modlist = [
                       nn.Conv2d(1, 20, 5),
                       nn.ReLU(),
                        nn.Conv2d(20, 64, 5),
                        nn.ReLU()
                        ]

    def forward(self, x):
        for m in self.modlist:
            x = m(x)
        return x

my_net = MyNet()
print(my_net)

print('====================================')
for param in my_net.parameters():
    print(type(param.data), param.size())
```

输出如下，可以看到使用 Python 中的 list 形式添加卷积层和它们的 parameters 并没有自动注册到我们的网络中。

```shell
net_modlist()
====================================

```

## （2）手动添加 (可行)

也可以手工挨个的添加网络。

```python
import torch.nn as nn
class MyNet(nn.Module):
    def __init__(self):
        super(net_modlist, self).__init__()
        self.conv2d_1 = nn.Conv2d(1, 20, 5)
        self.ReLU = nn.ReLU()
        self.conv2d_2 = nn.Conv2d(20, 64, 5),

    def forward(self, x):
        x = self.conv2d_1(x)
        x = self.ReLU(x)
        x = self.conv2d_2(x)
        x = self.ReLU(x)
        return x

my_net = MyNet()
print(my_net)

print('====================================')
for param in my_net.parameters():
    print(type(param.data), param.size())

```

输出如下，可以看到手动设置也可以把添加的卷积层和它们的 parameters 注册到我们的网络中。

```python
net_modlist(
  (conv2d_1): Conv2d(1, 20, kernel_size=(5, 5), stride=(1, 1))
  (ReLU): ReLU()
)
====================================
<class 'torch.Tensor'> torch.Size([20, 1, 5, 5])
<class 'torch.Tensor'> torch.Size([20])

```

* * *

参考：  
[nn.Sequential官网](https://pytorch.org/docs/stable/generated/torch.nn.Sequential.html?highlight=nn%20sequential#torch.nn.Sequential)  
[nn.ModuleList官网](https://pytorch.org/docs/stable/generated/torch.nn.ModuleList.html?highlight=nn%20module#torch.nn.ModuleList)  
[nn.ModuleDict官网](https://pytorch.org/docs/stable/generated/torch.nn.ModuleDict.html?highlight=nn%20module#torch.nn.ModuleDict)