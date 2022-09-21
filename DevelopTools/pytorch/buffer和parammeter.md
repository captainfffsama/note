#pytorch 

[toc]

pytorch 中模型参数分为两种:
- 需要反传且需要被优化器更新的,称为 parameter
- 会参与反传且产生梯度,但永远都不需要被优化器更新的,称为 buffer

在模型保存中`torch.save(model.state_dict(),path)`方法中,这两种参数都会被保存到`OrderDict`

---
**parameter**  

使用方法`model.parameters()`可以从模型中返回得到,创建方式有两种
- 将类型为`nn.Module`的类的成员(`self.xxx`),使用 `nn.Parameter()` 创建,会被注册到 parameters 中
- 直接使用 `nn.Parameter()` 创建,然后使用 `register_parameter()` 进行注册

**buffer**
创建方法是直接创建 tensor,然后使用 `register_buffer()` 注册.

例子:  

```python
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        buffer = torch.randn(2, 3)  # tensor
        self.register_buffer('my_buffer', buffer)
        self.param = nn.Parameter(torch.randn(3, 3))  # 模型的成员变量

    def forward(self, x):
        # 可以通过 self.param 和 self.my_buffer 访问
        pass
model = MyModel()
for param in model.parameters():
    print(param)
print("----------------")
for buffer in model.buffers():
    print(buffer)
print("----------------")
print(model.state_dict())
```

输出:
```shell
Parameter containing:
tensor([[-0.9, 0.7,  0.11],
        [0.12, 0.41, 0.69],
        [0.52, 0.44, 0.88]],requires_grad=True)
------------
tensor([[-1.1, 1.2, 0.3],
        [0.6,  0.7, 0.8]])
------------
OrderDict([('param',tensor([[-0.9, 0.7,  0.11],
        [0.12, 0.41, 0.69],
        [0.52, 0.44, 0.88]])),('my_buffer',tensor([[-1.1, 1.2, 0.3],
        [0.6,  0.7, 0.8]]))])
```

# 注意
某个 `nn.Module` 类的成员若不用更新,不进行注册,是无法保存到 `OrderDict`,且在进行设备之间移动时,注册的参数是会自动进行移动的,但是没有注册的不行.


# 参考
- https://zhuanlan.zhihu.com/p/89442276
