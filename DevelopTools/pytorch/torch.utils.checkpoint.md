#pytorch 

[原文](https://blog.csdn.net/weixin_43002433/article/details/105322846?spm=1001.2101.3001.6650.1&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7Edefault-1.highlightwordscore&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7Edefault-1.highlightwordscore)

今天这篇文章主要是想记录一下在复现DenseNet时，看到PyTorch源码中有个memory\_efficient的参数及其详细使用，其中主要是应用**torch.utils.checkpoint**这个包，在训练的前向传播中不保留中间激活值，从而节省下内存，并在反向传播中重新计算相关值，以此来执行一个高效的内存管理。  
**需要注意的是**，PyTorch中的checkpoint与TensorFlow中的checkpoint，这两者的语义是不同的。后者通常表示训练过程中保存的网络模型，即可能是每隔多少个epoch或者准确率变得更好时保存模型为一个checkpoint，方便中途暂停训练修改超参数，并在修改完后从checkpoint继续开始训练。而前者的checkpoint主要用于节省训练模型过程中使用的内存，将模型或其部分的激活值的计算方法保存为一个checkpoint，在前向传播中不保留激活值，而在反向传播中根据checkpoint重新计算一次获得激活值用于反向传播。

* * *

torch.utils.checkpoint包内有两个api，**torch.utils.checkpoint.checkpoint**与**torch.utils.checkpoint.checkpoint\_sequential**，这两个函数的功能是几乎相同的，只是使用对象不同，前者用于模型或者模型的一部分，后者用于序列的模型。因此，在这篇文章中，我将以torch.utils.checkpoint.checkpoint为例说明(后文中简称checkpoint)。

1\. PyTorch文档中的说明
-----------------

PyTorch中的检查点(checkpoint)是通过在向后传播过程中重新运行每个检查段的前向传播计算来实现的。这可能导致像RNG状态这样的连续态比没有检查点的状态更高级。默认情况下，检查点包括处理RNG状态的逻辑，这样通过使用RNG(例如通过dropout)进行的检查点传递与非检查点传递相比具有确定的输出。存储和还原RNG状态的逻辑可能会导致性能下降，具体取决于检查点操作的运行时间。如果不需要与非检查点传递相比确定的输出，可以设置preserve\_rng\_state=False，来忽略在每个检查点期间隐藏和恢复RNG状态。

(简单理解应该是，像dropout这样，每次运行可能结果是不同的，前一次结果可能是要丢弃的，后一次的结果可能又是保留的，而设置preserve\_rng\_state=True，可以保证在checkpoint里保存dropout这样的RNG状态的逻辑，即前一次丢弃后一次就用丢弃的逻辑，前一次保留后一次就保留。)

隐藏逻辑将当前设备以及所有cuda张量参数的器件备的RNG状态保存并恢复到run\_fn。但是，该逻辑无法预料用户是否会在run\_fn本身内将张量移动到新器件里，因此，如果在run\_fn内将张量移动到新的设备里(新器件指不属于集合\[当前器件+张量参数的器件\]的器件备)，则与非检查点传递相比确定的输出将不再确保是确定的。

(简单理解就是，在run\_fn内不要随意修改张量的device，否则preserve\_rng\_state参数将可能失效，结果是无法事先确定的。)

2\. checkpoint函数的框架
-------------------

```python
torch.utils.checkpoint.checkpoint(function, *args, **kwargs)

""" Parameters
function – describes what to run in the forward pass of the model or part of the model. 
		It should also know how to handle the inputs passed as the tuple. 
		For example, in LSTM, if user passes (activation, hidden), function should correctly use the first input as activation and the second input as hidden

preserve_rng_state (bool, optional, default=True) – Omit stashing and restoring the RNG state during each checkpoint.

args – tuple containing inputs to the function
"""

```

3\. checkpoint函数解析
------------------

checkpoint操作是**通过将计算交换为内存**而起作用的。不同于常规地将整个计算图的所有中间激活值保存下来用于计算反向传播，**作为检查点的部分不再保存中间激活值，而是在反向传播中重新计算一次中间激活值，即重新运行一次检查点部分的前向传播**。由此可知，checkpoint操作在训练过程中可以节省下作为检查点部分所需占的内存，但是要付出在反向传播中重新计算的时间代价。(checkpoint操作可用于网络的任意部分。)

**具体地来说，在前向传递中，传入的function将以torch.no\_grad的方式运行，即不保存中间激活值。取而代之的是，前向传递保存了输入元组以及function参数。在反向传递中，保存下来的输入元组与function参数将会被重新取回，并且前向传递将会在function上重新计算，此时会追踪中间激活值，然后梯度将会根据这些中间激活值计算得到。** 

4\. 实例解读
--------

下方的代码截取了DenseNet实现中与checkpoint相关的部分，只为理解checkpoint的使用。

```python
import torch.utils.checkpoint as cp

def _btnk_func(self, inp):
    """This function calculates the output of bottleneck layer. Also created for checkpoint."""
	cat_features = torch.cat(inp, 1)
    btnk_out = self.conv_1x1(cat_features)
    return btnk_out

@torch.jit.unused
def _call_checkpoint_bottleneck(self, inp):
    def closure(*inps):
        return self._btnk_func(*inps)
    return cp.checkpoint(closure, inp)

if self.memory_efficient and self._any_requires_grad(prev_features):
    if torch.jit.is_scripting():
        raise Exception("Memory Efficient is not supported in JIT")
    btnk_out = self._call_checkpoint_bottleneck(prev_features)
else:
    btnk_out = self._btnk_func(prev_features)

```

从上方的代码片段可以看出，\_btnk\_func是计算瓶颈层前向传递的函数，\_call\_checkpoint\_bottleneck是用于装饰\_btnk\_func函数的checkpoint函数。故可知，相应的瓶颈层就是模型的检查点部分(checkpointed part)，\_btnk\_func就是传入checkpoint的function参数，在前向传递中保存下来，并在反向传递中用于重新计算瓶颈层的激活值。

使用checkpoint来进行的内存管理是非常容易实现的，一般选取参数量较大的模型部分作为检查点，将其前向传递的计算封装为函数，可用checkpoint函数对其装饰，在类定义的前向传递方法中，通过添加的memory\_efficient参数判断是否设置检查点，如果是保存输入元组与checkpoint函数用于反向传播，如果不是则运行原来的前向传递函数即可。

5\. 总结
------

checkpoint操作虽然能节省运行时占用的内存，但是会相应地增加运行时间，具体是否需要checkpoint操作可依据实际的计算资源来选择。

6\. 相关警告
--------

*   检查点不适用于torch.autograd.gard()，仅适用于torch.autograd.backward()。
*   如果在反向传递过程中的函数调用与在前向传递过程中的函数调用有任何的不同，例如，由于某些全局变量的原因，检查点版本将不相等，它将无法被检测到。

* * *

*   [TORCH.UTILS.CHECKPOINT](https://pytorch.org/docs/stable/checkpoint.html)