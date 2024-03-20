[原文](https://blog.csdn.net/Brikie/article/details/114255743)

[toc]

为了节省显存（内存），pytorch 在计算过程中不保存中间变量，包括中间层的特征图和非叶子张量的梯度等。有时对网络进行分析时需要查看或修改这些中间变量，此时就需要注册一个钩子（hook）来导出需要的中间变量。网上介绍这个的有不少，但我看了一圈，多少都有不准确或不易懂的地方，我这里再总结一下，给出实际用法和注意点。  
hook 方法有四种:  

- torch.Tensor.register_hook()  
- torch.nn.Module.register\_forward\_hook()  
- torch.nn.Module.register\_backward\_hook()  
- torch.nn.Module.register\_forward\_pre_hook().

# torch.Tensor.register_hook(hook)

用来导出指定张量的梯度，或修改这个梯度值。

```python
import torch
def grad_hook(grad):
    grad *= 2
x = torch.tensor([2., 2., 2., 2.], requires_grad=True)
y = torch.pow(x, 2)
z = torch.mean(y)
h = x.register_hook(grad_hook)
z.backward()
print(x.grad)
h.remove()    # removes the hook
```

>tensor([2., 2., 2., 2.])

注意：（1）上述代码是有效的，但如果写成 grad = grad * 2 就失效了，因为此时没有对 grad 进行本地操作，新的 grad 值没有传递给指定的梯度。保险起见，最好在 def 语句中写明 return grad。即：

```python
def grad_hook(grad):
    grad = grad * 2
    return grad
```

（2）可以用 remove() 方法取消 hook。注意 remove() 必须在 backward() 之后，因为只有在执行 backward() 语句时，pytorch 才开始计算梯度，而在 x.register\_hook(grad\_hook) 时它仅仅是 " 注册 " 了一个 grad 的钩子，此时并没有计算，而执行 remove 就取消了这个钩子，然后再 backward() 时钩子就不起作用了。  
（3）如果在类中定义钩子函数，输入参数必须先加上 self，即

```python
def grad_hook(self, grad):
    ...
```

# torch .nn.Module.register\_forward\_hook(module, in, out)

  用来导出指定子模块（可以是层、模块等 nn.Module 类型）的输入输出张量，但只可修改输出，常用来导出或修改卷积特征图。

```python
inps, outs = [],[]
def layer_hook(module, inp, out):
    inps.append(inp[0].data.cpu().numpy())
    outs.append(out.data.cpu().numpy())

hook = net.layer1.register_forward_hook(layer_hook)
output = net(input)
hook.remove()

```

注意：（1）因为模块可以是多输入的，所以输入是 tuple 型的，需要先提取其中的 Tensor 再操作；输出是 Tensor 型的可直接用。  
   （2）导出后不要放到显存上，除非你有 A100。  
   （3）只能修改输出 out 的值，不能修改输入 inp 的值（不能返回，本地修改也无效），修改时最好用 return 形式返回，如：

```python
def layer_hook(self, module, inp, out):
    out = self.lam * out + (1 - self.lam) * out[self.indices]
    return out

```

  这段代码用在 manifold mixup 中，用来对中间层特征进行混合来实现数据增强，其中 self.lam 是一个\[0,1\] 概率值，self.indices 是 shuffle 后的序号。

# torch.nn.Module.register\_forward\_pre_hook(module, in)

用来导出或修改指定子模块的输入张量。

```python
def pre_hook(module, inp):
    inp0 = inp[0]
    inp0 = inp0 * 2
    inp = tuple([inp0])
    return inp

hook = net.layer1.register_forward_pre_hook(pre_hook)
output = net(input)
hook.remove()

```

注意：

（1）inp 值是个 tuple 类型，所以需要先把其中的张量提取出来，再做其他操作，然后还要再转化为 tuple 返回。  
（2）在执行 output = net(input) 时才会调用此句，remove() 可放在调用后用来取消钩子。

# torch .nn.Module.register\_backward\_hook(module, grad\_in, grad\_out)

  用来导出指定子模块的输入输出张量的梯度，但只可修改输入张量的梯度（即只能返回 gin），输出张量梯度不可修改。

```python
gouts = []
def backward_hook(module, gin, gout):
    print(len(gin),len(gout))
    gouts.append(gout[0].data.cpu().numpy())
    gin0,gin1,gin2 = gin
    gin1 = gin1*2
    gin2 = gin2*3
    gin = tuple([gin0,gin1,gin2])
    return gin

hook = net.layer1.register_backward_hook(backward_hook)
loss.backward()
hook.remove()

```

注意：  
（1）其中的 grad\_in 和 grad\_out 都是 tuple，必须要先解开，修改时执行操作后再重新放回 tuple 返回。  
（2）这个钩子函数在 backward() 语句中被调用，所以 remove() 要放在 backward() 之后用来取消钩子。