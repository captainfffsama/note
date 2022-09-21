#pytorch 

torch 中所有张量底层实现都是一个一维数组,然后通过附加元信息通过不同偏置来等到多维张量.

# view
[view](https://pytorch.org/docs/stable/generated/torch.Tensor.view.html#torch.Tensor.view) 会返回一个张量的视图,不会改变底层数据的排列方式,仅仅使用新的形状来显示数据.
当数据不连续时,即数据实际内存排列相邻关系和数据张量语义上排列关系不一致时,使用 view 会报错,无法得到正确的结果.

诸如[`narrow(), view(), expand() and transpose()`](https://stackoverflow.com/questions/48915810/pytorch-what-does-contiguous-do) ,都不会改变底层张量内容,仅仅是新建了一份元信息来记录新的张量步进等信息,这些操作之后,张量将不再连续.

使用 [`contiguous()`](https://stackoverflow.com/questions/48915810/pytorch-what-does-contiguous-do) 将在内存中新建一个数据副本,保证数据排列的连续.

# reshape
[reshape](https://pytorch.org/docs/stable/generated/torch.reshape.html) 实际上就是先执行了 [`contiguous()`](https://stackoverflow.com/questions/48915810/pytorch-what-does-contiguous-do)  然后执行 `view()`.它会尽量返回一个张量的 view,若不成功,才会返回一个经过排序之后的数据副本.

# 参考
1. https://zhuanlan.zhihu.com/p/64551412
2. https://www.zhihu.com/question/60321866
3. https://stackoverflow.com/questions/48915810/pytorch-what-does-contiguous-do
