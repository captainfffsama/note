#python 

NewType，我们可以借助于它来声明一些具有特殊含义的类型，例如像 Tuple 的例子一样，我们需要将它表示为 Person，即一个人的含义，但但从表面上声明为 Tuple 并不直观，所以我们可以使用 NewType 为其声明一个类型，如：
```python
Person = NewType('Person', Tuple[str, int, float])
person = Person(('Mike', 22, 1.75))
```

这里实际上 person 就是一个 tuple 类型，我们可以对其像 tuple 一样正常操作。

# 参考
- <https://docs.python.org/zh-cn/3/library/typing.html#typing.NewType>
- <https://cuiqingcai.com/7071.html>
- <https://www.45ma.com/post-74719.html>
- <https://stackoverflow.com/questions/58755948/what-is-the-difference-between-typevar-and-newtype>