#python 

# get
 `get`方法可以返回一个默认值,但是默认值是不会写到 dict 中的

# setdefault
`setdefault`方法与`get`的最大区别在于`setdefault`会在 dict 以当前的默认值为值,将被查的键值对插入到 dict 中

# defaultdict
对于`defaultdict`,字典找键的时候,先使用`__getitem__`方法尝试找,找不到调用`__missing__`再找.
**因此: `defaultdict`使用`get`方法时,是不会调用预设好的默认工厂函数的!!!因为`get`方法跳过`__getitem`方法**