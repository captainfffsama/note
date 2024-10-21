 #pytorch 

[官方文档](https://pytorch.org/docs/stable/tensors.html#torch.Tensor.scatter_)

`scatter` 的参数有:

- dim(int): 用于指示沿着哪个轴进行填充
- index(LongTensor): 轴数和 `src` 相同,里面每一个元素指示了其在 `src` 上对应位置的值应该放在 `target` 的第 `dim` 轴上的第几个位置
- src(Tensor or float): 要填充到 `target` 的值
- reduce(str): `add` 或者是 `multiply` 指示了填充的方式.

例子:

```python
>>> src = torch.arange(1, 11).reshape((2, 5))
>>> src
tensor([[ 1,  2,  3,  4,  5],
        [ 6,  7,  8,  9, 10]])
>>> index = torch.tensor([[0, 1, 2, 0]])
>>> torch.zeros(3, 5, dtype=src.dtype).scatter_(0, index, src)
tensor([[1, 0, 0, 4, 0],
        [0, 2, 0, 0, 0],
        [0, 0, 3, 0, 0]])
```

即对于 3 D 张量:

```python
self[index[i][j][k]][j][k] = src[i][j][k]  # if dim == 0
self[i][index[i][j][k]][k] = src[i][j][k]  # if dim == 1
self[i][j][index[i][j][k]] = src[i][j][k]  # if dim == 2
```

#   参考
- <https://blog.csdn.net/guofei_fly/article/details/104308528>
- <https://pytorch.org/docs/stable/tensors.html#torch.Tensor.scatter_>)