[原文](https://zhuanlan.zhihu.com/p/393041305)

写在前面：

感觉这部分内容正确理解之后其实一点都不难，但如果一知半解，看代码的时候还是会挺纠结的。

**目录：** 
--------

1.  **clone() 与 copy_() 分析**
2.  **detach() 与 .data 分析**
3.  **应用举例**

1.  **创建新 tensor**
2.  **tensor 值的复制**

* * *

1\. clone() 与 copy_() 分析
------------------------

### (1) clone():

创建一个 tensor 与源 tensor 有相同的 shape，dtype 和 device，不共享内存地址，但新 tensor 的梯度会叠加在源 tensor 上。下面例子可以说明上述特点：

```python3
import torch

a = torch.tensor([1.,2.,3.],requires_grad=True)
b = a.clone()

print(a.data_ptr()) # 3004830353216
print(b.data_ptr()) # 3004830353344 内存地址不同

print(a) # tensor([1., 2., 3.], requires_grad=True)
print(b) # tensor([1., 2., 3.], grad_fn=<CloneBackward>)  复制成功
print('-'*30)

c = a * 2
d = b * 3

c.sum().backward() 
print(a.grad) # tensor([2., 2., 2.])

d.sum().backward() 
print(a.grad) # tensor([5., 5., 5.]) # 源tensor的梯度叠加了新tensor的梯度
print(b.grad) # None # 此时复制出来的节点已经不属于叶子节点，因此不能直接得到其梯度
```

需要注意的是，b = a.clone() 之后，b 并非叶子节点（最后输出 b 的梯度时 warning 会提醒我们）。b 节点更像是 a 经过一个恒等函数之后得到的输出。

![](https://pic2.zhimg.com/v2-3b63e1fa9d37521b2b9c713568dc4a91_b.jpg)

计算图，蓝色箭头为正向计算，红色箭头为梯度流向

如图所示，我们需要留意从 a 到 b 的变换。我们应该吧 clone() 理解成一个“函数”，而不能把 b 理解成一个完全独立的节点。

总结一句话：由于不共享内存，因此 a 与 b 的值是非关联的，但梯度是有联系的。

### (2) copy_():

copy_() 函数完成与 clone() 函数类似的功能，但也存在区别。调用 copy_() 的对象是目标 tensor，参数是复制操作 from 的 tensor，最后会返回目标 tensor；而 clone() 的调用对象为源 tensor，返回一个新 tensor。当然 clone() 函数也可以采用 torch.clone() 调用，将源 tensor 作为参数。

copy_() 函数的调用对象既然是目标 tensor，那么就需要我们预先已有一个目标 tensor(clone() 就不需要)，源 tensor 的尺度需要可以广播到目标 tensor 的尺度。

```python3
import torch

a = torch.tensor([1., 2., 3.],requires_grad=True)
b = torch.empty_like(a).copy_(a)

print(a.data_ptr()) # 1597834661312
print(b.data_ptr()) # 1597834659712 # 内存地址不同

print(a) # tensor([1., 2., 3.], requires_grad=True)
print(b) # tensor([1., 2., 3.], grad_fn=<CopyBackwards>) # 复制成功
print('-'*30)

c = a * 2
d = b * 3

c.sum().backward()
print(a.grad) # tensor([2., 2., 2.])

d.sum().backward()
print(a.grad) # tensor([5., 5., 5.]) # 源tensor梯度累加了
print(b.grad) # None # 复制得到的节点依然不是叶子节点
```

由此可以看出，这个 copy_() 和 clone() 真的很像。

A.copy(B) 这个操作将 A 变成了 B 经过恒等函数后的输出。就好像 A = torch.clone(B) 一样。因此，上面例子中，B 梯度回传回 A 也就不难理解了。

总结一句话：由于不共享内存，因此 a 与 b 的值是非关联的，但梯度是有联系的。

### (3)clone() 与 copy_() 总结：

这两个函数都复制源 tensor 到新 tensor。两 tensor 不共享内存空间，新 tensor 是基于源 tensor 恒等变换出的 tensor，因此不是叶子节点，新 tensor 的梯度信息因此会累加到源 tensor 上。

2\. detach() 与 .data 分析
-----------------------

### (1) detach():

detach() 函数返回与调用对象 tensor 相关的一个 tensor，此新 tensor 与源 tensor**共享数据内存**（那么 tensor 的数据必然是相同的），但其 requires_grad 为 False，并且不包含源 tensor 的计算图信息。

```python
import torch

a = torch.tensor([1., 2., 3.],requires_grad=True)
b = a.detach()

print(a.data_ptr()) # 2432102290752
print(b.data_ptr()) # 2432102290752 # 内存位置相同

print(a) # tensor([1., 2., 3.], requires_grad=True)
print(b) # tensor([1., 2., 3.]) # 这里为False，就省略了
print('-'*30)

c = a * 2
d = b * 3

c.sum().backward()
print(a.grad) # tensor([2., 2., 2.])

d.sum().backward()
print(a.grad) # 报错了！ 由于b不记录计算图，因此无法计算b的相关梯度信息
print(b.grad)
```

如果绘制此时的计算图，是这样的：

![](https://pic4.zhimg.com/v2-6d4f9d6efab5e3091673a5208b7adc77_b.jpg)

b 脱离计算图，pytorch 不再记录其后续操作，自然也没有梯度信息了

由于 b 已经从计算图脱离出来，pytorch 自然也不跟踪其后续计算过程了。

如果我在 b = a.detach() 后面，紧接着加一句 b.requires\_grad\_()。那么，b 又需要梯度了，pytorch 会继续跟踪其有关计算。这种情况下，得到的计算图如下：

![](https://pic1.zhimg.com/v2-3899a863226e8885ccd8e156ec0c8014_b.jpg)

可以继续跟踪 b 的计算，但梯度不会从 b 流回 a，梯度被截断

可以看到，pytorch 可以继续跟踪 b 的计算，但梯度不会从 b 流回 a，梯度被截断。但由于 b 与 a 共享内存，a 与 b 的值会一直相等。

总结一句话，detach() 操作产生的节点与原节点共享内存，但梯度截断。

### (2) .data:

torch.tensor.data, 可以猜测其内涵便是得到一个 tensor 的数据信息，实际也是如此，其返回的信息与上面提到的 detach() 返回的信息是相同的。也具有内存相同，不保存梯度信息的特点。

至于两者的区别，detach() 是一种更加安全的做法，具体原因可以参考：

### (3) detach() 与 .data 总结：

相当于取了数值信息，数值相同，梯度截断。

关于这部分，再次强烈推荐上面的链接，讲的非常清楚，内容也非常重要。

3\. 应用举例
--------

### （1）创建新 tensor

clone() 与 copy_() 可以在新的内存空间复制源 tensor，但梯度信息不独立；

detach() 与.data 可以独立出梯度信息，但与源 tensor 具有相同内存。

因此**联合使用二者**可以创建出数据相同，完全独立的新 tensor。

常见的手段便是 b = a.clone().detach() 或是 b = a.detach().clone()

下面的链接介绍了 5 种建立新 tensor 的方式并进行了速度比较

### （2）tensor 值的改变

在深度学习中，如果我们创建了一个 model，那么 model 中可以训练的参数都是叶子节点。pytorch 规定 requires_grad=True 的叶子节点是不能做 in-place operation 的。那么他们是如何被初始化的呢？

一种曲线救国的方式是通过对 XX.data 进行操作。例如 XX.data.zero_ 便可以达到初始化效果了。毕竟 XX.data 与 XX 共享内存。

在 MoCo 中，需要用一个网络的参数初始化另一个网络的参数。由于 copy_() 也是一种 inplace operation，因此，只能采用 A.data.copy_(B.data) 的方式。

类似的例子应该还有很多，在阅读代码的时候可以多思考，多尝试。

* * *

参考链接：
-----