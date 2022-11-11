#python 

TypeVar，我们可以借助它来自定义兼容特定类型的变量，比如有的变量声明为 int、float、None 都是符合要求的，实际就是代表任意的数字或者空内容都可以，其他的类型则不可以，比如列表 list、字典 dict 等等，像这样的情况，我们可以使用 TypeVar 来表示。 例如一个人的身高，便可以使用 int 或 float 或 None 来表示，但不能用 dict 来表示，所以可以这么声明：
```python
height = 1.75
Height = TypeVar('Height', int, float, None)
def get_height() -> Height:
    return height
```
这里我们使用 TypeVar 声明了一个 Height 类型，然后将其用于注解方法的返回结果。

# 参考
- <https://cuiqingcai.com/7071.html>
- <https://www.45ma.com/post-74719.html>
- <https://stackoverflow.com/questions/58755948/what-is-the-difference-between-typevar-and-newtype>