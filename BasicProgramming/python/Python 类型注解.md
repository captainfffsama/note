#python #类型标注 

[原文](https://www.gairuo.com/p/python-type-annotations)

[toc]

简单说，Python 类型注解功能可以让我们的代码更加易读，从而达到编写更加健壮的代码目标。类型注解又叫类型暗示，将函数、变量声明为一种特定类型。当然，它并不是严格的类型绑定，所以这个机制并不能阻止调用者传入不应该传入的参数。

# 背景
由于 Python 是一个动态语言，定义和使用变量时不需要申明变量的数据类型即可使用，但这样也会带来一定的问题，如阅读代码时不知道数据是什么类型，调用时不小心会传入错误的数据类型。因此，Python 的类型注解功能就显得比较重要了。

一个字符和数字类型是无法相加的，否则会报以下错误：

```python
a = 1
b = '2'
a + b
# TypeError: unsupported operand type(s) for +: 'int' and 'str'
```

调用者在使用函数时如果没有完善的文档，不知道要传入的数据类型分别是什么，同时文档也很难表达复杂的数据类型。有了类型注解可以让 IDE 知道了数据类型后，更加准确地进行自动补全。有了类型注解可以提供给第三方工具，做代码分析，发现隐形bug。函数注解的信息，保存在 `__annotations__` 属性中可以来调用。

在 Python 3.5 中，Python PEP 484 引入了类型注解（type hints），在 Python 3.6 中，PEP 526 又进一步引入了变量注解（Variable Annotations）。

# 基本方法

以下是一个使用示例：

```python
# 定义一个变量
x: int = 2
x: int | float = 2 # 表示 or Python 3.10 开始支持
x + 1
# 3

# 定义一个除法函数
def div(a: int, b: int) -> float:
    return a/b

# 查看类型注解信息
div.__annotations__
{'a': int, 'b': int, 'return': float}
```

语法注意的点：

-   变量类型：在变量名后加一个冒号，冒号后写变量的数据类型，如 int、dict 等
-   函数返回类型：方法参数中如变量类型，在参数括号后加一个箭头，箭头后返回值的类型
-   格式要求（PEP 8，非强制）：变量名和冒号无空格，冒号和后边类型间加一个空格，箭头左右均有一个空格

# 提示性

但值得注意的是，这种类型和变量注解实际上只是一种类型提示，对运行实际上是没有影响的。

```python
def add(a: int, b: int) -> str:
    return a + b

# 查看返回类型
type(add(1, 2))
# int 
```
比如上例，调用 add 方法的时候，我们注解返回一个字符串，但它返回的仍然是 int，也不会报错，也不会对参数进行类型转换。

不过有了类型注解，一些 IDE 是可以识别出来并提示的，比如 PyCharm 就可以识别出来在调用某个方法的时候参数类型不一致，会提示 WARNING。如类似以下字典嵌套列表类型的提示：

```python
'''
Expected type 'Mapping[str, str]',
got 'Dict[str, Union[List[str], List[Union[bool, str]]]]' instead
'''
```

# 类型定义方法


以下几种类型的定义方法：

*   None
*   内置的类（int, str, float）
*   标准库（typing 模块中的类型对象）
*   第三方扩展库
*   抽象出的基类
*   用户定义的类
*   类别名 type aliases（`s = str`，然后用 s）
*   使用 NewType
*   使用 own generic types

更多见 typing 中的构建方法。

# 复杂结构

对于一些结构的数据类型就不能用上述简单的方法表示了，比如：

```python
names: list = ['lily', 'tom']
version: tuple = (6, 6, 6)
operations: dict = {'sad': False, 'happy': True}
```

以上虽然定义了最外层的数据结构，如列表、元组、字典，但他们每个元素的类型是什么也没有定义清楚，因此需要在 typing 库的帮助下来定义准确内部的结构：

```python
from typing import List, Tuple, Dict

names: List[str] = ['lily', 'tom']
version: Tuple[int, int, int] = (6, 6, 6)
operations: Dict[str, bool] = {'sad': False, 'happy': True}
```

再如：

```python
from typing import Union, List, Tuple, Dict

config: Dict[str, Union[List[str], Tuple[bool, str]]]= {
            'width': ['100%', 'Width of img'],
            'height': ['auto', 'Height of img'],
            'fluid': (True, '外层包含一个div')
        } 
```

config 变量存在着复杂的嵌套结构，以上注解就声明了它的结构，方便调用者来使用。

# GenericAlias 类型

GenericAlias 对象通常通过标类来创建。它们最常用于容器类，如 list 或 dict。例如，list\[int\] 是一个 GenericAlias 对象，通过使用参数 int 标注 list 类创建。GenericAlias 对象主要用于类型注释。

```python
# 也可以直接用内置类型定义
d: dict[str, int] = {}
type(dict[str, int])
# types.GenericAlias
```

GenericAlias 对象充当泛型类型的代理，实现参数化泛型。对于容器类，提供给该类订阅的参数可能指示对象包含的元素的类型。例如，set\[bytes\] 可以在类型注释中用于表示所有元素都是 bytes 类型的集合。

对于定义 `_class_getitem__` 特殊方法但不是容器的类，提供给该类订阅的参数通常会指示对象上定义的一个或多个方法的返回类型。例如，正则表达式可用于 str 数据类型和 bytes 数据类型：

*   如果 `x = re.search('foo', 'foo')`，x 将是 re.Match 对象，其中 x.group(0) 和 x\[0\] 的返回值都将是 str 类型。我们可以在类型注释中用 GenericAlias re.Match\[str\] 表示这种对象。
*   如果 `y = re.search(b'bar', b'bar')`（b代表字节），y 也将是 re.Match 的一个实例。但 y.group(0) 和 y\[0\] 的返回值都将是字节类型。在类型注释中，我们将用 re.Match\[bytes\] 表示 re.Match 对象。

GenericAlias 对象 `types.GenericAlias` 类的实例，也可用于直接创建 GenericAlias 对象。

### T\[X, Y, ...\] 形式

用 `T[X, Y, ...]` 形式创建表示类型 T 的泛型，该类型由类型 X、Y 和更多参数化，具体取决于所使用的 T。例如：

```python
# 一个函数需要一个包含浮点元素的列表
def average(values: list[float]) -> float:
    return sum(values) / len(values)

# mapping 对象，dict类型，泛型的两个类型参数分别代表了键类型和值类型
# 下例中的函数需要一个 dict，其键的类型为 str，值的类型为 int
def send_post_request(url: str, body: dict[str, int]) -> None:
    ...
```

注：内置函数 isinstance() 和 issubclass() 不接受第二个参数为`GenericAlias` 类型。

### 应用影响

Python 运行时不强制执行类型注释，这扩展到泛型类型及其类型参数。从 GenericAlias 创建容器对象时，不会根据其类型检查容器中的元素。例如，不鼓励使用以下代码，但将在运行时不会出现错误：

```python
t = list[str]
t([1, 2, 3])
# [1, 2, 3]
```
不仅如此，在创建对象的过程中，应用了参数后的泛型还会抹除类型参数：

```python
t = list[str]
type(t)
# <class 'types.GenericAlias'>

l = t()
type(l)
# <class 'list'>
```

### 标准泛型类

以下标准库类支持参数化泛型不完全列表：

*   tuple
*   list
*   dict
*   set
*   frozenset
*   type
*   collections.deque
*   collections.defaultdict
*   collections.OrderedDict
*   collections.Counter
*   collections.ChainMap
*   collections.abc.Awaitable
*   collections.abc.Coroutine
*   collections.abc.AsyncIterable
*   collections.abc.AsyncIterable
*   collections.abc.AsyncGenerator
*   collections.abc.Iterable
*   collections.abc.Iterator
*   collections.abc.Generator
*   collections.abc.Reversible
*   collections.abc.Container
*   collections.abc.Collection
*   collections.abc.Callable
*   collections.abc.Set
*   collections.abc.MutableSet
*   collections.abc.Mapping
*   collections.abc.MutableMapping
*   collections.abc.Sequence
*   collections.abc.MutableSequence
*   collections.abc.ByteString
*   collections.abc.MappingView
*   collections.abc.KeysView
*   collections.abc.ItemsView
*   collections.abc.ValuesView
*   contextlib.AbstractContextManager
*   contextlib.AbstractAsyncContextManager
*   dataclasses.Field
*   functools.cached\_property
*   functools.partialmethod
*   os.PathLike
*   queue.LifoQueue
*   queue.Queue
*   queue.PriorityQueue
*   queue.SimpleQueue
*   re.Pattern
*   re.Match
*   shelve.BsdDbShelf
*   shelve.DbfilenameShelf
*   shelve.Shelf
*   types.MappingProxyType
*   weakref.WeakKeyDictionary
*   weakref.WeakMethod
*   weakref.WeakSet
*   weakref.WeakValueDictionary

### GenericAlias 特殊属性

`genericalias.__origin__` 本属性指向未应用参数之前的泛型类：

```python
list[int].__origin__
# <class 'list'>
```

`genericalias.__args__`：此属性是传递给泛型类的原始 `_class_getitem__()` 的泛型类型的元组（可能长度为1）：

```python
dict[str, list[int]].__args__
# (<class 'str'>, list[int])
```

`genericalias.__parameters__`：该属性是延迟计算出来的一个元组（可能为空），包含了 `__args__` 中的类型变量。

```python
from typing import TypeVar

T = TypeVar('T')
list[T].__parameters__
# (~T,)
```

注：带有参数 typing.ParamSpec 的 GenericAlias 对象，在类型替换后其 `__parameters__` 可能会不准确，因为 `typing.ParamSpec` 主要用于静态类型检查。

# union 联合类型

union 联合类型（合并类型），为一个联合对象，它包含了在多个 类型对象 上执行 | (按位或) 运算后的值。 这些类型主要用于 类型标注。与 typing.Union 相比，联合类型表达式可以实现更简洁的类型提示语法。

### X | Y | ...

Python 3.10 开始引入了 X | Y 语法的类型联合运算符表示 '类型 X 或类型 Y' ，相比使用 `typing.Union[X, Y]` （union 对象）更清晰。以下是前后的写法对比：

```python
def square(number: Union[int, float]) -> Union[int, float]:
    return number ** 2

# 简洁的写法
def square(number: int | float) -> int | float:
    return number ** 2

type(int | str)
# types.UnionType
```

### union 对象比较

操作 `union_object == other` 将 union 对象可与其他 union 对象进行比较。详细结果如下（以下均返回 True）：

```python
# 多次组合的结果会平推：
(int | str) | float == int | str | float

# 冗余的类型会被删除：
int | str | int == int | str

# 在相互比较时，会忽略顺序：
int | str == str | int

# 与 typing.union 兼容：
int | str == typing.Union[int, str]

# Optional 类型可表示为与 None 的组合。
str | None == typing.Optional[str]
```

### 实例和子类检测

这也被接受作为 isinstance() 和 issubclass() 的第二个参数：

```python
isinstance(1, int | str)
isinstance("", int | str)
# True

from collections import OrderedDict
issubclass(OrderedDict, dict | tuple)
# True
```

但不能使用包含参数化泛型（parameterized generics）的 union 对象：

```python
# union 对象的第二个是参数化泛型
isinstance(1, int | list[int])
'''
Traceback (most recent call last):
 File "<stdin>", line 1, in <module>
TypeError: isinstance() argument 2 cannot contain a parameterized generic
'''
```

### types.UnionType 访问

由于上文中说的 isinstance() 的第二个参数不能使用包含参数化泛型（parameterized generics）的 union 对象，用户类型可以经由 types.UnionType 访问，它不能由类型直接实例化为对象：

```python
import types
isinstance(int | str, types.UnionType)
# True

types.UnionType()
'''
Traceback (most recent call last):
 File "<stdin>", line 1, in <module>
TypeError: cannot create 'types.UnionType' instances
'''
```

### OR 特殊方法

为了支持 X | Y 语法，类型对象加入了 `__or__()` 方法。若是元类已实现了 `__or__()`，union 也可以覆盖掉：

```python
class M(type):
    def __or__(self, other):
        return "Hello"

class C(metaclass=M):
    pass

C | int
# 'Hello'
int | C
# int | __main__.C
```

# lambda 的类型标注

由于类型注解的语法和 lambda 的语法冲突，因此不能直接对 lambda 做类型注解，但我们可以将 lambda 传给一个变量，通过对这个变量做 lambda，达到相同的目的。以下对 lambda 的几个例子：

```python
from typing import Callable

# is_even 传入 int 返回布尔
is_even: Callable[[int], bool] = lambda x: (x % 2 == 0)
# func 传入两个字符串，返回 int
func: Callable[[str, str], int] = lambda var1, var2: var1.index(var2)
```

# 类型检测

inspect 模块是 Python 内置的类型检查标准库，它提供了一些有用的函数帮助获取对象的信息，例如模块、类、方法、函数、回溯、帧对象以及代码对象。例如它可以帮助你检查类的内容，获取某个方法的源代码，取得并格式化某个函数的参数列表，或者获取你需要显示的回溯的详细信息。

查看 [inspect 教程](https://www.gairuo.com/p/python-library-inspect)。

可以通过 mypy 库来检验最终代码是否符合注解：

```shell
#安装 mypy
pip install mypy

# 执行代码
mypy test.py
```

如果不符合标注的类型要求会报错。

# 内置库

### types 动态类型创建和内置类型名称

此模块定义了一些工具函数，用于协助动态创建新的类型。它还为某些对象类型定义了名称，这些名称由标准 Python 解释器所使用，但并不像内置的 int 或 str 那样对外公开。

详情可以查看 [Python 标准库 types](https://www.gairuo.com/p/python-library-types) 。

### typing 类型注解标注

typing 库是 Python 内置的类型标注标准库，用它可以来解决这些问题，它可以实现复杂的类型注解工作，可以查看 [typing 教程](https://www.gairuo.com/p/python-library-typing)。

# 参考

*   https://docs.python.org/zh-cn/3/library/stdtypes.html#type-annotation-types-generic-alias-union
*   https://www.python.org/dev/peps/pep-3107/
*   https://www.python.org/dev/peps/pep-0484/
*   https://www.python.org/dev/peps/pep-0526/
*   https://www.python.org/dev/peps/pep-0604

