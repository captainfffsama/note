#python 

# 定义
返回模块、类、实例或任何其它具有 `__dict__` 属性的对象的 `__dict__` 属性。  
模块和实例这样的对象具有可更新的 `__dict__` 属性；但是，其它对象的 `__dict__` 属性可能会设为限制写入（例如，类会使用 `types.MappingProxyType` 来防止直接更新字典）。  
不带参数时，`vars()` 的行为类似 `locals()`。 请注意，locals 字典仅对于读取起作用，因为对 locals 字典的更新会被忽略。  
如果指定了一个对象但它没有 `__dict__` 属性（例如，当它所属的类定义了 `__slots__` 属性时）则会引发 `TypeError` 异常。  

# 一个用法
在[BasesHomo](https://github.com/megvii-research/BasesHomo/blob/af2c5527e17ee1d23abd8b60673c0940e586385c/evaluate.py#L147) 中见到`vars(args)`可以将`parser.parse_args()`返回的参数直接变成字典.

# 参考
- <https://www.cjavapy.com/article/1256/>
- <https://docs.python.org/zh-cn/3/library/functions.html#vars>

