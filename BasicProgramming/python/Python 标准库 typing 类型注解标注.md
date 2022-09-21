#python
#类型标注 



[原文](https://www.gairuo.com/p/python-library-typing)

[toc]



更新时间：2021-11-10 09:33:53


在[Python 类型注解](Python%20类型注解.md)中我们介绍过，通过类型注解可以提高代码的可读性和易用性，但对于复杂的数据结构就需要借助 typing 模块来表达这些数据结构。

# typing 的作用

[Python 类型注解](Python%20类型注解.md)是用来对变量和函数的参数返回值类型做注解（暗示），帮助开发者写出更加严谨的代码，让调用方减少类型方面的错误。

但是，类型注解语法传入的类型表述能力有限，不能说明复杂的类型组成情况，因此引入了 typing 模块，来实现复杂的类型表达。

# 基础用法

以下是典型的用法：

```python
from typing import List, Tuple, Dict

names: List[str] = ['lily', 'tom']
version: Tuple[int, int, int] = (6, 6, 6)
operations: Dict[str, bool] = {'sad': False, 'happy': True}
```

安装 mypy 库运行脚本，会强制按类型检测，不符合类型注解要求的会报错：

```python
# 安装
pip install mypy -U
# 运行脚本
mypy program.py
```

# 类型

这些是一些最常见的内置类型的示例：

| Type | Description |
| --- | --- |
| int | 整型 integer |
| float | 浮点数字 |
| bool | 布尔（int 的子类） |
| str | 字符 (unicode) |
| bytes | 8 位字符 |
| object | 任意对象（公共基类） |
| List\[str\] | 字符组成的列表 |
| Tuple\[int, int\] | 两个int对象的元组 |
| Tuple\[int, ...\] | 任意数量的 int 对象的元组 |
| Dict\[str, int\] | 键是 str 值是 int 的字典 |
| Iterable\[int\] | 包含 int 的可迭代对象 |
| Sequence\[bool\] | 布尔值序列（只读） |
| Mapping\[str, int\] | 从 str 键到 int 值的映射（只读） |
| Any | 具有任意类型的动态类型值 |
| Union | 联合类型 |
| Optional | 参数可以为空或已经声明的类型 |
| Mapping | 映射，是 collections.abc.Mapping 的泛型 |
| MutableMapping | Mapping 对象的子类，可变 |
| Generator | 生成器类型, Generator\[YieldType、SendType、ReturnType\] |
| NoReturn | 函数没有返回结果 |
| Set | 集合 set 的泛型, 推荐用于注解返回类型 |
| AbstractSet | collections.abc.Set 的泛型，推荐用于注解参数 |
| Sequence | collections.abc.Sequence 的泛型，list、tuple 等的泛化类型 |
| TypeVar | 自定义兼容特定类型的变量 |
| Generic | 自定义泛型类型 |
| NewType | 声明一些具有特殊含义的类型 |
| Callable | 可调用类型, Callable\[\[参数类型\], 返回类型\] |

Any 类型和类型构造函数如 List，Dict，Iterable 和 Sequence 定义了类型模型。

Dict 类型是一个通用类，由 \[...\] 中的类型参数表示。 如 Dict\[int，str\] 是从整数到字符串的字典，而 Dict\[Any, Any\] 是动态键入（任意）值和键的字典。 List 是另一个通用类。 Dict 和 List 分别是内置 dict 和 list 的别名。

Iterable、Sequence 和 Mapping 是与 Python 协议相对应的通用类型。 例如，当期望 Iterable\[str\] 或 Sequence\[str\] 时，str 对象或 List\[str\] 对象有效。 请注意，尽管它们类似于 collections.abc（以前的collections）中定义的抽象基类，但它们也不相同，因为内置的collection 类型对象不支持索引。

# 变量

Python 3.6 在 PEP 526 中引入了用于注释变量的语法，我们在大多数示例中都使用了它。

```python
# 声明变量类型的类型的方式，python 3.6 +
age: int = 1
# 在 Python 3.5 及更低版本中，您可以改用类型注释
# 同上效果
age = 1  # type: int
# 无需初始化变量即可对其进行注释
a: int  # ok（但不能调用，name 'a' is not defined，直到被赋值）
# 在条件分支中很有用
child: bool
if age < 18:
    child = True
else:
    child = False
```

# 内置类型

typing 内置的一些类型的用法：

```python
from typing import List, Set, Dict, Tuple, Optional

# 对于简单的 Python 内置类型，只需使用类型的名称
x: int = 1
x: float = 1.0
x: bool = True
x: str = "test"
x: bytes = b"test"
# 对于 collections ，类型名称用大写字母表示，并且
# collections 内类型的名称在方括号中
x: List[int] = [1]
x: Set[int] = {6, 7}
# 与上述相同，但具有类型注释语法
x = [1]  # type: List[int]
# 对于映射，需要键和值的类型
x: Dict[str, float] = {'field': 2.0}
# 对于固定大小的元组，指定所有元素的类型
x: Tuple[int, str, float] = (3, "yes", 7.5)
# 对于可变大小的元组，使用一种类型和省略号
x: Tuple[int, ...] = (1, 2, 3)
# 使用 Optional[] 表示可能为 None 的值
x: Optional[str] = some_function()
# Mypy 理解 if 语句中的值不能为 None
if x is not None:
    print(x.upper())
# 如果由于某些不变量而使值永远不能为 None，请使用断言
assert x is not None
print(x.upper())
```

# 函数

Python 3 支持函数声明的注释语法。

```python
from typing import Callable, Iterator, Union, Optional, List

# 注释函数定义的方式
def stringify(num: int) -> str:
    return str(num)

# 指定多个参数的方式
def plus(num1: int, num2: int) -> int:
    return num1 + num2

# 在类型注释后为参数添加默认值
def f(num1: int, my_float: float = 3.5) -> float:
    return num1 + my_float

# 注释可调用（函数）值的方式, lambda 可以此方法
x: Callable[[int, float], float] = f

# 产生整数的生成器函数安全地返回只是一个
# 整数迭代器的函数，因此这就是我们对其进行注释的方式
def g(n: int) -> Iterator[int]:
    i = 0
    while i < n:
        yield i
        i += 1

# 可以将功能注释分成多行
def send_email(address: Union[str, List[str]],
               sender: str,
               cc: Optional[List[str]],
               bcc: Optional[List[str]],
               subject='',
               body: Optional[List[str]] = None
               ) -> bool:
    ...
```

# 混杂结构

以下是一些复杂结构的用法：

```python
from typing import Union, Any, List, Optional, cast

# Union 表示可能是以下几种类型
x: List[Union[int, str]] = [3, 5, "test", "fun"]

# 不知道类型或它太动态而无法为它编写类型，请使用 Any
x: Any = mystery_function()

# 如果使用空容器或“无”初始化变量
# 类型注解帮助 mypy 获知类型信息
x: List[str] = []
x: Optional[str] = None

# 每个位置 arg 和每个关键字 arg 均为 str
def call(self, *args: str, **kwargs: str) -> str:
    request = make_request(*args, **kwargs)
    return self.do_api_query(request)

# cast 可以转换类型
a = [4]
b = cast(List[int], a)  # 正常通过
c = cast(List[str], a)  # 正常通过 (运行是不做检查，无影响)

# 如果要在类上使用动态属性，请使其覆盖 “ __setattr__”
# 或 “ __getattr__”。
#
# "__setattr__" 允许动态分配名称
# "__getattr__" 允许动态访问名称
class A:
    # 如果 x 与“值”属于同一类型，则这将允许分配给任何 A.x
    # （使用“value: Any”以允许任意类型）
    def __setattr__(self, name: str, value: int) -> None: ...

    # 如果 x 与返回类型兼容，则将允许访问任何 A.x
    def __getattr__(self, name: str) -> int: ...

a.foo = 42  # Works
a.bar = 'Ex-parrot'  # Fails type checking
```

# 用户定义的泛型类型

用户定义的类可以定义为泛型类。

```python
from typing import TypeVar, Generic
from logging import Logger

T = TypeVar('T')

class LoggedVar(Generic[T]):
    def __init__(self, value: T, name: str, logger: Logger) -> None:
        self.name = name
        self.logger = logger
        self.value = value

    def set(self, new: T) -> None:
        self.log('Set ' + repr(self.value))
        self.value = new
    
    def get(self) -> T:
        self.log('Get ' + repr(self.value))
        return self.value
    
    def log(self, message: str) -> None:
        self.logger.info('%s: %s', self.name, message)
```

Generic\[T\] 是定义类 LoggedVar 的基类，该类使用单类型参数 T。在该类体内，T 是有效的类型。

# lambda 的类型标注

由于类型注解的语法和 lambda 的语法冲突，因此不能直接对 lambda 做类型注解，但我们可以将 lambda 传给一个变量，通过对这个变量做 lambda，达到相同的目的。以下对 lambda 的几个例子：

```python
from typing import Callable

# is_even 传入 int 返回布尔
is_even: Callable[[int], bool] = lambda x: (x % 2 == 0)
# func 传入两个字符串，返回 int
func: Callable[[str, str], int] = lambda var1, var2: var1.index(var2)
```

# 鸭子类型

在程序设计中，鸭子类型（英语：duck typing）是动态类型的一种风格。在这种风格中，一个对象有效的语义，不是由继承自特定的类或实现特定的接口，而是由"当前方法和属性的集合"决定。

在典型的 Python 代码中，许多可以将列表或 dict 作为参数的函数只需要将其参数设为“类似于列表”（list-like）或“类似于 dict”（dict-like）即可。 “类似列表”或“类似字典”（或类似其他的东西）的特定含义被称为「鸭子类型」，并且标准化了在惯用Python中常见的几种鸭子类型。

来源

这个概念的名字来源于由 James Whitcomb Riley 提出的鸭子测试，“鸭子测试” 可以这样表述：「当看到一只鸟走起来像鸭子、游泳起来像鸭子、叫起来也像鸭子，那么这只鸟就可以被称为鸭子。」在鸭子类型中，关注点在于对象的行为，能作什么；而不是关注对象所属的类型。例如，在不使用鸭子类型的语言中，我们可以编写一个函数，它接受一个类型为"鸭子"的对象，并调用它的"走"和"叫"方法。Alex Martelli 很早（2000年）就在发布到 comp.lang.python 新闻组上的一则消息中使用了这一术语。他同时对鸭子测试的错误的字面理解提出了提醒，以避免人们错误认为这个术语已经被使用。

> “换言之，不要检查它是不是一个鸭子：检查它像不像一个鸭子地叫，等等。取决于你需要哪个像鸭子的行为的子集来使用语言。”

用例如下：

```python
from typing import Mapping, MutableMapping, Sequence, Iterable, List, Set

# 将 Iterable 用于一般可迭代对象（for 中可用的任何东西）
# 以及需要序列（支持 len 和  __getitem__ 的序列）
def f(ints: Iterable[int]) -> List[str]:
    return [str(x) for x in ints]

f(range(1, 3))

# Mapping 映射描述了一个我们不会经常变化的
# 类似 dict 的对象（带有  __getitem__）
# 而 MutableMapping 则描述了一个对象（带有 __setitem__）
def f(my_mapping: Mapping[int, str]) -> List[int]:
    my_mapping[5] = 'maybe'  # mypy 会引发错误
    return list(my_mapping.keys())

f({3: 'yes', 4: 'no'})

def f(my_mapping: MutableMapping[int, str]) -> Set[str]:
    my_mapping[5] = 'maybe'  # mypy 正常执行
    return set(my_mapping.values())

f({3: 'yes', 4: 'no'})
```

# 类中的应用

```python
class MyClass:
    # 在类主体中声明实例变量
    attr: int
    # 具有默认值的实例变量
    charge_percent: int = 100

    # __init__ 方法不返回任何内容，因此返回 None
    def __init__(self) -> None:
        ...
    
    # 对于实例方法，省略 self 的类型
    def my_method(self, num: int, str1: str) -> str:
        return num * str1

# 用户定义的类作为注释中的类型有效
x: MyClass = MyClass()

# 可以使用 ClassVar 批注来声明类变量
class Car:
    seats: ClassVar[int] = 4
    passengers: ClassVar[List[str]]

# 可以在  __init__ 中声明属性的类型
class Box:
    def __init__(self) -> None:
        self.items: List[str] = []
```

# 协程和异步

```python
import asyncio

# 类似于正常函数
async def countdown35(tag: str, count: int) -> str:
    while count > 0:
        print('T-minus {} ({})'.format(count, tag))
        await asyncio.sleep(0.1)
        count -= 1
    return "Blastoff!"
```

# 其他

```python
import sys
import re
from typing import Match, AnyStr, IO

# "typing.Match" 正则匹配对象类型
x: Match[str] = re.match(r'[0-9]+', "15")

#使用 IO[] 来接受或返回来自 open() 调用的任何对象的函数
#（ IO[] 不区分读取，写入或其他方式）
def get_sys_IO(mode: str = 'w') -> IO[str]:
    if mode == 'w':
        return sys.stdout
    elif mode == 'r':
        return sys.stdin
    else:
        return sys.stdout

# 要先定义再引用，以下会报错
def f(foo: A) -> int:  # This will fail
    ...

class A:
    ...

# 可以使用字符串文字 'A' 解决，只要文件中稍后有该名称的类
def f(foo: 'A') -> int:  # Ok
    ...
```

# 装饰器


装饰器功能可以通过泛型表示。

```python
from typing import Any, Callable, TypeVar

F = TypeVar('F', bound=Callable[..., Any])

def bare_decorator(func: F) -> F:
    ...

def decorator_args(url: str) -> Callable[[F], F]:
    ...
```

# NoReturn

NoReturn，当一个方法没有返回结果时，为了注解它的返回类型，我们可以将其注解为 NoReturn，例如：

```python
def fun() -> NoReturn:
    print('Hi!')
```

# 参考

*   https://docs.python.org/zh-cn/3/library/typing.html
*   https://www.python.org/dev/peps/pep-0484/
*   https://github.com/python/typing
*   https://mypy.readthedocs.io/en/stable/cheat\_sheet\_py3.html
*   https://realpython.com/python-type-checking/
*   https://cuiqingcai.com/7071.html

* * *
