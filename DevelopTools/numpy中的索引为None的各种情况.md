#numpy 
[原文](https://zhuanlan.zhihu.com/p/486373530)
[toc]

pytorch 的 tensor 同理。

先说结论：

**None 实际上是增加了一个维度，它不是原维度的索引。** 

#  为什么引入 None

说一下我找原文档的心路历程：

在**numpy 的官方文档**里搜索“slice”，也就是切片，很容易就能找到 [关于slice的介绍](https://link.zhihu.com/?target=https%3A//numpy.org/doc/1.22/user/basics.indexing.html%3Fhighlight%3Dslice%23slicing-and-striding)[\[1\]](#ref_1)：

Basic slicing extends Python’s basic concept of slicing to N dimensions. Basic slicing occurs when obj is asliceobject (constructed bystart:stop:stepnotation inside of brackets), an integer, or a tuple ofsliceobjects and integers.Ellipsisandnewaxisobjects can be interspersed with these as well.

简单来说就是 numpy 的切片扩展了 python 的切片。当索引是切片对象（由括号内的 start:stop:step 语法构造）、整数、切片对象的元组或整数的元组，切片操作就会发生。后面一句话特别重要：**省略号和 newaxis 对象也可以穿插其中**。

省略号就是 python 语法的“…”，那么 newaxis 是什么呢？直觉告诉我它和 None 有关。找到 [newaxis的文档](https://link.zhihu.com/?target=https%3A//numpy.org/doc/1.22/reference/constants.html%23numpy.newaxis)[\[2\]](#ref_2)，里面第一句话就是：

> A convenient alias for None, useful for indexing arrays.

也就是说，**numpy.newaxis 是 None 的别名**，在索引数组时有用。而文档紧接着给的例子也特别直接：

![](https://pic4.zhimg.com/v2-bcdf62241b43bd20331ba938d8ca4633_b.jpg)

第一句 newaxis is None , is None 。。。

官方这么直白，这下不用我多说，你也知道 None 是什么意思了吧？None 就是 newaxis，也就是建立一个新的索引维度。其实就是为了写法上的方便，本来新建一个维度用 reshape 函数、unsqueeze 函数也可以做到。其实文档后面也给出了解释 [\[3\]](#ref_3)：

> This can be handy to combine two arrays in a way that otherwise would require explicit reshaping operations.

这种写法很方便地把两个数组结合起来，否则，还需要明确的 reshape 操作。

那么，怎么用呢？

# 以一维为例


```python
x = np.arange(3) # array([0, 1, 2])
```

_（ 注意，这个一维数组的 shape 是 (3,)，而不是 (1,3)，初学者很容易犯错。）_

如果想把 x 的 shape 变成 (1,3)，只需要把 None 放在第一个维度的位置，以下两种写法等价：

结果如下：

如果想把 x 的 shape 变成 (3,1)，只需要把 None 放在第二个维度的位置：

结果如下：

其实，None 可以任意添加和组合，例如下面的写法：

结果如下：

```shell
array([[[[0]],
        [[1]],
        [[2]]]])
```

这个数组的 shape 是 (1,3,1,1)。

# 以二维为例


```python
x = np.arange(6).reshape((2,3))
```

x 如下：

```shell
array([[0, 1, 2],
       [3, 4, 5]])
```

在第一个维度插入，以下三种写法等价：

```python
x[None]
x[None,:]
x[None,:,:]
```

输出结果如下，shape 为 (1, 2, 3)：

```shell
array([[[0, 1, 2],
        [3, 4, 5]]])
```

在第二个维度插入，以下两种写法等价：

输出结果如下，shape 为 (2, 1, 3)：

```shell
array([[[0, 1, 2]],
       [[3, 4, 5]]])
```

在第三个维度插入：

输出结果如下，shape 为 (2, 3, 1)：

```shell
array([[[0],
        [1],
        [2]],

       [[3],
        [4],
        [5]]])
```

更高维的情况以此类推。

这种写法一般在进行矩阵运算的时候会用到。比如：

```python
x = np.arange(5)
x[:, None] + x[None, :]
```

这样可以很**优雅**地获得 _列向量 + 行向量_ 的结果 (划重点：优雅～）：

```shell
array([[0, 1, 2, 3, 4],
      [1, 2, 3, 4, 5],
      [2, 3, 4, 5, 6],
      [3, 4, 5, 6, 7],
      [4, 5, 6, 7, 8]])
```

# 参考


1.  [^](#ref_1_0)1 [https://numpy.org/doc/1.22/user/basics.indexing.html?highlight=slice#slicing-and-striding](https://numpy.org/doc/1.22/user/basics.indexing.html?highlight=slice#slicing-and-striding)
2.  [^](#ref_2_0)2 [https://numpy.org/doc/1.22/reference/constants.html#numpy.newaxis](https://numpy.org/doc/1.22/reference/constants.html#numpy.newaxis)
3.  [^](#ref_3_0)3 [https://numpy.org/doc/1.22/user/basics.indexing.html?highlight=slice#dimensional-indexing-tools](https://numpy.org/doc/1.22/user/basics.indexing.html?highlight=slice#dimensional-indexing-tools)