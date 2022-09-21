#python #类型标注

# Python 中的类型标注

静态语言与动态语言孰优孰劣一直是网络上争论不休的话题。在这篇论文中（[英文原文](https://cacm.acm.org/magazines/2017/10/221326-a-large-scale-study-of-programming-languages-and-code-quality-in-github/abstract)，[中文翻译](https://www.zcfy.cc/article/a-large-scale-study-of-programming-languages-and-code-quality-in-github)），研究者通过统计 GitHub 上的不同语言的热门项目，的确得出了静态语言比动态语言更好维护的结论。


如果我们想同时拥有静态语言的严谨和动态语言的自由，一方面就是让静态语言更动态，比如 Java 10 中新的 `var` 关键字；另一方面当然就是让动态语言变得更加静态，这篇文章的主角：Python type hints。


## Python type hints 的进化


### Python 3.0: function annotation


在 Python 中，我们可以写这么一个函数


如果我想要声明这个函数参数和返回值的类型，可以使用下面的写法：


这就是被称为 **function annotation** 的写法。使用冒号 `:` 加类型名来代表参数的类型，使用箭头 `->` 加类型表示返回值的类型。理解这种写法的关键就是，把高亮的部分都忽略，这些高亮的部分都不会被 Python 解析器所解析。你只需要把搞两部分忽略，就能看到熟悉的 Python 函数语法。


![](https://i.loli.net/2019/05/06/5ccfe8af55408.jpg#align=left&display=inline&height=76&margin=%5Bobject%20Object%5D&originHeight=76&originWidth=372&status=done&style=none&width=372)


Python 解释器在运行时并不会检查类型，所以哪怕参数的类型不对，Python 解释器也不会因此抛出任何异常。


为了能够检查出类型的错误，我们还需要一些额外的静态检查工具。比如 Python 官方维护的 [mypy](https://github.com/python/mypy) 和 facebook 维护的 [pyre](https://github.com/facebook/pyre-check)。在开发的过程中，我们可以使用这类工具扫描代码，提前发现代码中的 bug，和其他语言的编译过程异曲同工。这篇文章会使用 mypy 作为例子，毕竟 Python type hints 的很多语法都继承自 mypy。


首先安装 mypy:


然后使用 mypy 对上面的文件进行检查：


除了内置的类型之外，Python 语法也支持用户创建的类


不管你们相不相信，这种写法在 python3.0 中就已经被支持了（[PEP 3107](https://www.python.org/dev/peps/pep-3107/)）。


但是 Pyhton3 诞生这么多年，我印象中使用 Python type hints 语法的项目寥寥无几，除了对 Python 2.7 的兼容性外，另一个可能的原因是这套语法的功能还不够强大。在后来的 Python 版本中其语法也进行了不断的增强。


### Python 3.5: typing

---

在 python3.5 中，引入了 `typing` 模块，现在我们可以表示嵌套结构了


`Dict[str, int]` 表示一个 keys 的类型为 str，values 的类型为 int 的字典，比如 `{"a": 1, "b": 2}`


`List[int]` 表示由整型组成的列表，比如`[0, 1, 1, 2, 3]`


基于 typing 提供的类型，我们可以写出相当复杂的嵌套结构：


`Dict[str, Dict[str, List[str]]]`


由于 `typing` 模块并没有对 python 本身的语法作出修改，所以低于 3.5 的 python 版本也可以通过安装 pip 库 [typing](https://pypi.org/project/typing/) 来获得这个功能。


另外一个有趣的事情是，`typing` 使用方括号 `List[str]` 而不是圆括号 `List(str)`。如果你使用了后面的方法的话，mypy 会提醒你 `Suggestion: use List[...] instead of List(...)`。


## typing 高级用法


### Union


有时候我们的参数、返回值（以及下面会谈到的变量）并不只有一种类型，这种情况下我们就可以使用 `Union` 对不同的类型进行或操作：


上面这个函数的返回值可能是 `None`，也可能是一个字符串


### Callable


Python 中万物皆是对象，函数也是对象。`Callable` 就可以表示函数类型。


在上面的例子中，`get_regex` 函数返回一个 `Callable` 对象。这个对象接受两个位置参数，类型分别是 `str` 和 `str`。它的返回值的类型是 `bool`。


不过 Callable 不能表示位置参数，在下面我还会详细地谈到这个问题。


### Any


无论如何，Python 的类型注解系统总是无法表达所有的类型


Python 总有一些我们很难表达的形式或者类型。这种情况下我就能使用 `typing.Any`，它代表任何东西。比如说


### Python 3.6: variable annotations


在 python3.6 中，除了函数的参数和返回值外，变量也可以表示类型了（[PEP 526](https://www.python.org/dev/peps/pep-0526/)）


通过这种形式，我们可以实现 “先声明，后赋值” 的写法。而这种特性就是 Python 3.7 的 dataclass 的基础。


### Python 3.7: dataclass


我在上面说过，python 解析器并不会在意类型注解，严格来说这是不对的，Python 会把类型信息放在 `__annotations__` 属性中：


所以在 python 中，类型信息是可以被获取到并被加以利用的。在过去，我们会这么写一个类：


但是在 python3.7 中，我们可以利用新的 dataclass 大幅简化这类语法：


dataclass 作为一个例子，展示了 Python 的 type hints 的潜力。


## 格式


## 其他好处：IDE


## python typing 不能做什么


相比于 typescript 之于 javascript，python typing 的语法改动非常有限。这也导致了 Python 的类型检查并不能够覆盖所有的情况，比如说关键字参数：


> There is no syntax to indicate optional or keyword arguments; such function types are rarely used as callback types. `Callable[..., ReturnType]` (literal ellipsis) can be used to type hint a callable taking any number of arguments and returning `ReturnType`.
> ——[python typing 官方文档](https://docs.python.org/3.7/library/typing.html#typing.Callable)



下面这个


对于这个错误，mypy 可以检查出来：


但是只要加上一个装饰器，哪怕这个装饰器并没有改变参数和返回值的类型，由于 Python 语法的限制，我们无法精确地声明被装饰后的函数的关键字参数类型，于是 mypy 就检查不出这个错误了：


我可不是在钻牛角尖，我在某个项目中大量使用到了装饰器配合关键字参数的写法，结果 mypy 对这种情况的静态检查根本无能为力，让我又回到了以前只有在 runtime 时才能暴露问题的时代。


同样是对动态语言的类型检查，相比于 typescript 对 javascript 语法的重新设计，Python 选择了尽可能兼容原来的写法，方便上手的同时，距离真正的静态语言还有着相当多的进步空间。
[https://ocavue.com/python_typing.html#python-3-5-typing](https://ocavue.com/python_typing.html#python-3-5-typing)
