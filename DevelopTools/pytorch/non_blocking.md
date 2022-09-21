# 何时设置`non_blocking=True`
当你在 `dataloader` 中使用 [`pin_memory=True`](pin_memory.md) 时,意味着你将强制生成的 tensor 使用锁页内存,这意味着即使内存不足,也不会将 tensor 放到虚拟内存上.但是内存够的情况下是推荐使用 [`pin_memory=True`](pin_memory.md) 的,这样可以加快数据的交换.

当设置 [`pin_memory=True`](pin_memory.md) 时,还推荐设置`non_blocking=True`,这样意味着数据的传输是异步的,数据在 GPU 中计算的同时,还可以异步的将 CPU 数据传输到 GPU 上?

**但是一定要注意的是,当你接下来的操作是将 GPU 上张量回传到 CPU,就千万不要使用 non_blocking=True,否则会导致当 CPU 开始计算的时候, 其实数据还没有传输完**

可以参见底下一个例子:
```python
import torch
action_gpu = torch.tensor([1.0], device=torch.device('cuda'), pin_memory=True)
print(action_gpu)
action_cpu = action_gpu.to(torch.device('cpu'), non_blocking=True)
print(action_cpu)
```
输出:
>tensor([1.], device='cuda:0')
tensor([0.])
Process finished with exit code 0
# 参考资料
- https://discuss.pytorch.org/t/should-we-set-non-blocking-to-true/38234/20
- https://pytorch.org/docs/stable/notes/cuda.html?highlight=non_blocking