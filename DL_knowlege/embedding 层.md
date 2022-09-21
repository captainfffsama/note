#pytorch 

[toc]

# 主要作用
最早是使用在 NLP 中.由于单词数目众多,使用 one-hot 会导致维度过高且 one-hot 编码空间无法保证相似的单词距离相近,不同的单词距离远.

具体可以参见李宏毅视频: [B站](https://www.bilibili.com/video/av10590361/?p=25)  [油管有字幕](https://www.youtube.com/watch?v=X7PH3NuYW0Q&list=PLJV_el3uVTsPy9oCRY30oBPNLCo89yu49&index=24)

参考: https://blueschang.github.io/2018/12/25/%E3%80%8C%E6%9D%8E%E5%AE%8F%E6%AF%85%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%E3%80%8D%E5%AD%A6%E4%B9%A0%E7%AC%94%E8%AE%B0-Unsupervised%20Learning%20-%20Word%20Embedding/

# pytorch 中 embedding 的用法
具体API参见: https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html?highlight=embedding#torch.nn.Embedding  

说明可以参见:
- https://www.jianshu.com/p/63e7acc5e890
- https://www.zhihu.com/question/32275069
- https://www.cnblogs.com/sunupo/p/12815567.html
- https://zhuanlan.zhihu.com/p/341176854

`embedding` 函数前面如下:
```python
torch.nn.Embedding(num_embeddings, embedding_dim, padding_idx=None, max_norm=None, norm_type=2.0, scale_grad_by_freq=False, sparse=False, _weight=None, device=None, dtype=None)
```

其中
- `num_embeddings(int)` 表示字典的大小,最小不能小于输入张量元素中的最大值
- `embedding_dim(int)` 表示待查字典中每个元素的维度,可以自己合理设定. 
- `padding_idx(int,optional)` 当输入的数据长度不一致时,我们需要在输入向量上填充特定值来补齐长度,这个补齐的值就是这个参数.这个参数会将字典中对应位置的元素全部写为0,在训练过程中也不会更新梯度.
- `max_norm(float,optional)` 若指定,那么当有嵌入张量大于这个值时候,便会再次归一化
- `norm_type(float,optional)` 使用何种范数来进行归一化,默认2
- `scale_grad_by_freq(bool,optional)` 根据 mini-batch 中的词频对梯度进行缩放,默认 `False`
- `sparse(bool,optional)` `weight`是否稀疏.  


pytorch 中 embedding 层权重初始化的方式是一个符合 $N(0,1)$ 的随机矩阵,但是在实际任务中是先训练一个 embedding 来用,这个 embedding 一般是可以表征各个维度之间相似关系或者语义关系的. 

下面展示了一个例子:
```python
import torch
import torch.nn as nn
embedding=nn.Embedding(10,3)
embedding
```
```
Embedding(10, 3)
```
```python
embedding.weight
```

```
Parameter containing:
tensor([[-0.3523, -0.4629, -0.5387],
        [-0.0723,  0.9476, -0.4386],
        [-0.4533,  0.3748,  0.8845],
        [-1.0195,  0.6322,  1.1587],
        [ 0.6859,  1.7911, -0.2218],
        [ 1.5013,  0.4605,  0.3281],
        [ 0.9778, -0.0298, -0.5207],
        [ 0.9994, -0.8369,  0.8848],
        [ 1.2411,  2.0432, -0.4907],
        [-1.2613,  1.2164,  0.0430]], requires_grad=True)

```

```python
i=torch.LongTensor([[1,2],[3,4]])
embedding(i)
```
```
tensor([[[-0.0723,  0.9476, -0.4386],
         [-0.4533,  0.3748,  0.8845]],

        [[-1.0195,  0.6322,  1.1587],
         [ 0.6859,  1.7911, -0.2218]]], grad_fn=<EmbeddingBackward>)
```

这里将 `i` 中的元素视为是 `embedding.weight` 行的索引.
