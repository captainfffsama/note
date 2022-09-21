#python #描述器
[toc]

原文:https://pyzh.readthedocs.io/en/latest/Descriptor-HOW-TO-Guide.html

## [1.1. 摘要](https://pyzh.readthedocs.io/en/latest/Descriptor-HOW-TO-Guide.html#id9)

定义描述器, 总结描述器协议，并展示描述器是怎么被调用的。展示一个自定义的描述器和包括函数，属性(property), 静态方法(static method), 类方法在内的几个Python内置描述器。通过给出一个纯Python的实现和示例应用来展示每个描述器是怎么工作的。

学习描述器不仅让你接触到更多的工具，还可以让你更深入地了解Python，让你体会到Python设计的优雅之处。

## [1.2. 定义和介绍](https://pyzh.readthedocs.io/en/latest/Descriptor-HOW-TO-Guide.html#id10)

一般来说，一个描述器是一个有“绑定行为”的对象属性(object attribute)，它的访问控制被描述器协议方法重写。这些方法是 `__get__()`, `__set__()`, 和 `__delete__()` 。有这些方法的对象叫做描述器。

默认对属性的访问控制是从对象的字典里面(__dict__)中获取(get), 设置(set)和删除(delete)它。举例来说， `a.x` 的查找顺序是, `a.__dict__['x']` , 然后 `type(a).__dict__['x']` , 然后找 `type(a)` 的父类(不包括元类(metaclass)).如果查找到的值是一个描述器, Python就会调用描述器的方法来重写默认的控制行为。这个重写发生在这个查找环节的哪里取决于定义了哪个描述器方法。注意, 只有在新式类中时描述器才会起作用。(新式类是继承自 `type` 或者 `object` 的类)

描述器是强大的，应用广泛的。描述器正是属性, 实例方法, 静态方法, 类方法和 `super` 的背后的实现机制。描述器在Python自身中广泛使用，以实现Python 2.2中引入的新式类。描述器简化了底层的C代码，并为Python的日常编程提供了一套灵活的新工具。

## [1.3. 描述器协议](https://pyzh.readthedocs.io/en/latest/Descriptor-HOW-TO-Guide.html#id11)

```python
descr.__get__(self, obj, type=None) --> value
descr.__set__(self, obj, value) --> None
descr.__delete__(self, obj) --> None
```

这是所有描述器方法。一个对象具有其中任一个方法就会成为描述器，从而在被当作对象属性时重写默认的查找行为。

如果一个对象同时定义了 `__get__()` 和 `__set__()`,它叫做资料描述器(data descriptor)。仅定义了 `__get__()` 的描述器叫非资料描述器(常用于方法，当然其他用途也是可以的)

资料描述器和非资料描述器的区别在于：相对于实例的字典的优先级。如果实例字典中有与描述器同名的属性，如果描述器是资料描述器，优先使用资料描述器，如果是非资料描述器，优先使用字典中的属性。(译者注：这就是为何实例 `a` 的方法和属性重名时，比如都叫 `foo` Python会在访问 `a.foo` 的时候优先访问实例字典中的属性，因为实例函数的实现是个非资料描述器)

要想制作一个只读的资料描述器，需要同时定义 `__set__` 和 `__get__`,并在 `__set__` 中引发一个 `AttributeError` 异常。定义一个引发异常的 `__set__` 方法就足够让一个描述器成为资料描述器。

## [1.4. 描述器的调用](https://pyzh.readthedocs.io/en/latest/Descriptor-HOW-TO-Guide.html#id12)

描述器可以直接这么调用： `d.__get__(obj)`

然而更常见的情况是描述器在属性访问时被自动调用。举例来说， `obj.d` 会在 `obj` 的字典中找 `d` ,如果 `d` 定义了 `__get__` 方法，那么 `d.__get__(obj)` 会依据下面的优先规则被调用。

调用的细节取决于 `obj` 是一个类还是一个实例。另外，描述器只对于新式对象和新式类才起作用。继承于 `object` 的类叫做新式类。

对于对象来讲，方法 `object.__getattribute__()` 把 `b.x` 变成 `type(b).__dict__['x'].__get__(b, type(b))` 。具体实现是依据这样的优先顺序：资料描述器优先于实例变量，实例变量优先于非资料描述器，__getattr__()方法(如果对象中包含的话)具有最低的优先级。完整的C语言实现可以在 [Objects/object.c](https://hg.python.org/cpython/file/2.7/Objects/object.c) 中 [PyObject_GenericGetAttr()](https://docs.python.org/2/c-api/object.html#c.PyObject_GenericGetAttr) 查看。

对于类来讲，方法 `type.__getattribute__()` 把 `B.x` 变成 `B.__dict__['x'].__get__(None, B)` 。用Python来描述就是:

```python
def __getattribute__(self, key):
    "Emulate type_getattro() in Objects/typeobject.c"
    v = object.__getattribute__(self, key)
    if hasattr(v, '__get__'):
       return v.__get__(None, self)
    return v
```

其中重要的几点：

- 描述器的调用是因为 `__getattribute__()`
- 重写 `__getattribute__()` 方法会阻止正常的描述器调用
- `__getattribute__()` 只对新式类的实例可用
- `object.__getattribute__()` 和 `type.__getattribute__()` 对 `__get__()` 的调用不一样
- 资料描述器总是比实例字典优先。
- 非资料描述器可能被实例字典重写。(非资料描述器不如实例字典优先)

`super()` 返回的对象同样有一个定制的 `__getattribute__()` 方法用来调用描述器。调用 `super(B, obj).m()` 时会先在 `obj.__class__.__mro__` 中查找与B紧邻的基类A，然后返回 `A.__dict__['m'].__get__(obj, A)` 。如果不是描述器，原样返回 `m` 。如果实例字典中找不到 `m` ，会回溯继续调用 `object.__getattribute__()` 查找。(译者注：即在 `__mro__` 中的下一个基类中查找)

注意:在Python 2.2中，如果 `m` 是一个描述器, `super(B, obj).m()` 只会调用方法 `__get__()` 。在Python 2.3中，非资料描述器(除非是个旧式类)也会被调用。 `super_getattro()` 的实现细节在： [Objects/typeobject.c](http://svn.python.org/view/python/trunk/Objects/typeobject.c?view=markup) ，[del] 一个等价的Python实现在 [Guido’s Tutorial](http://www.python.org/2.2.3/descrintro.html#cooperation) [/del] (译者注：原文此句已删除，保留供大家参考)。

以上展示了描述器的机理是在 `object`, `type`, 和 `super` 的 `__getattribute__()` 方法中实现的。由 `object` 派生出的类自动的继承这个机理，或者它们有个有类似机理的元类。同样，可以重写类的 `__getattribute__()` 方法来关闭这个类的描述器行为。

## [1.5. 描述器例子](https://pyzh.readthedocs.io/en/latest/Descriptor-HOW-TO-Guide.html#id13)

下面的代码中定义了一个资料描述器，每次 `get` 和 `set` 都会打印一条消息。重写 `__getattribute__()` 是另一个可以使所有属性拥有这个行为的方法。但是，描述器在监视特定属性的时候是很有用的。

```python
class RevealAccess(object):
    """A data descriptor that sets and returns values
       normally and prints a message logging their access.
    """

    def __init__(self, initval=None, name='var'):
        self.val = initval
        self.name = name

    def __get__(self, obj, objtype):
        print 'Retrieving', self.name
        return self.val

    def __set__(self, obj, val):
        print 'Updating' , self.name
        self.val = val

>>> class MyClass(object):
    x = RevealAccess(10, 'var "x"')
    y = 5

>>> m = MyClass()
>>> m.x
Retrieving var "x"
10
>>> m.x = 20
Updating var "x"
>>> m.x
Retrieving var "x"
20
>>> m.y
5
```

这个协议非常简单，并且提供了令人激动的可能。一些用途实在是太普遍以致于它们被打包成独立的函数。像属性(property), 方法(bound和unbound method), 静态方法和类方法都是基于描述器协议的。

## [1.6. 属性(properties)](https://pyzh.readthedocs.io/en/latest/Descriptor-HOW-TO-Guide.html#id14)

调用 `property()` 是建立资料描述器的一种简洁方式，从而可以在访问属性时触发相应的方法调用。这个函数的原型:

```python
property(fget=None, fset=None, fdel=None, doc=None) -> property attribute
```

下面展示了一个典型应用：定义一个托管属性(Managed Attribute) `x` 。

```python
class C(object):
    def getx(self): return self.__x
    def setx(self, value): self.__x = value
    def delx(self): del self.__x
    x = property(getx, setx, delx, "I'm the 'x' property.")
```

想要看看 `property()` 是怎么用描述器实现的？ 这里有一个纯Python的等价实现:

```python
class Property(object):
    "Emulate PyProperty_Type() in Objects/descrobject.c"

    def __init__(self, fget=None, fset=None, fdel=None, doc=None):
        self.fget = fget
        self.fset = fset
        self.fdel = fdel
        self.__doc__ = doc

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        if self.fget is None:
            raise AttributeError, "unreadable attribute"
        return self.fget(obj)

    def __set__(self, obj, value):
        if self.fset is None:
            raise AttributeError, "can't set attribute"
        self.fset(obj, value)

    def __delete__(self, obj):
        if self.fdel is None:
            raise AttributeError, "can't delete attribute"
        self.fdel(obj)

    def getter(self, fget):
        return type(self)(fget, self.fset, self.fdel, self.__doc__)

    def setter(self, fset):
        return type(self)(self.fget, fset, self.fdel, self.__doc__)

    def deleter(self, fdel):
        return type(self)(self.fget, self.fset, fdel, self.__doc__)
```

当用户接口已经被授权访问属性之后，需求发生一些变化，属性需要进一步处理才能返回给用户。这时 `property()` 能够提供很大帮助。

例如，一个电子表格类提供了访问单元格的方式: `Cell('b10').value` 。 之后，对这个程序的改善要求在每次访问单元格时重新计算单元格的值。然而，程序员并不想影响那些客户端中直接访问属性的代码。那么解决方案是将属性访问包装在一个属性资料描述器中:

```python
class Cell(object):
    . . .
    def getvalue(self, obj):
        "Recalculate cell before returning value"
        self.recalc()
        return obj._value
    value = property(getvalue)
```

## [1.7. 函数和方法](https://pyzh.readthedocs.io/en/latest/Descriptor-HOW-TO-Guide.html#id15)

Python的面向对象特征是建立在基于函数的环境之上的。非资料描述器把两者无缝地连接起来。

类的字典把方法当做函数存储。在定义类的时候，方法通常用关键字 `def` 和 `lambda` 来声明。这和创建函数是一样的。唯一的不同之处是类方法的第一个参数用来表示对象实例。Python约定，这个参数通常是 *self*, 但也可以叫 *this* 或者其它任何名字。

为了支持方法调用，函数包含一个 `__get__()` 方法以便在属性访问时绑定方法。这就是说所有的函数都是非资料描述器，它们返回绑定(bound)还是非绑定(unbound)的方法取决于他们是被实例调用还是被类调用。用Python代码来描述就是:

```python
class Function(object):
    . . .
    def __get__(self, obj, objtype=None):
        "Simulate func_descr_get() in Objects/funcobject.c"
        return types.MethodType(self, obj, objtype)
```

下面运行解释器来展示实际情况下函数描述器是如何工作的:

```python
>>> class D(object):
     def f(self, x):
          return x

>>> d = D()
>>> D.__dict__['f'] # 存储成一个function
<function f at 0x00C45070>
>>> D.f             # 从类来方法，返回unbound method
<unbound method D.f>
>>> d.f             # 从实例来访问，返回bound method
<bound method D.f of <__main__.D object at 0x00B18C90>>
```

从输出来看，绑定方法和非绑定方法是两个不同的类型。它们是在文件 Objects/classobject.c(http://svn.python.org/view/python/trunk/Objects/classobject.c?view=markup) 中用C实现的， `PyMethod_Type` 是一个对象，但是根据 `im_self` 是否是 *NULL* (在C中等价于 *None* ) 而表现不同。

同样，一个方法的表现依赖于 `im_self` 。如果设置了(意味着bound), 原来的函数(保存在 `im_func` 中)被调用，并且第一个参数设置成实例。如果unbound, 所有参数原封不动地传给原来的函数。函数 `instancemethod_call()` 的实际C语言实现只是比这个稍微复杂些(有一些类型检查)。

## [1.8. 静态方法和类方法](https://pyzh.readthedocs.io/en/latest/Descriptor-HOW-TO-Guide.html#id16)

非资料描述器为将函数绑定成方法这种常见模式提供了一个简单的实现机制。

简而言之，函数有个方法 `__get__()` ，当函数被当作属性访问时，它就会把函数变成一个实例方法。非资料描述器把 `obj.f(*args)` 的调用转换成 `f(obj, *args)` 。 调用 `klass.f(*args)` 就变成调用 `f(*args)` 。

下面的表格总结了绑定和它最有用的两个变种:

> | Transformation | Called from an Object | Called from a Class |
> | :------------- | :-------------------- | :------------------ |
> | function       | f(obj, *args)         | f(*args)            |
> | staticmethod   | f(*args)              | f(*args)            |
> | classmethod    | f(type(obj), *args)   | f(klass, *args)     |

静态方法原样返回函数，调用 `c.f` 或者 `C.f` 分别等价于 `object.__getattribute__(c, "f")` 或者 `object.__getattribute__(C, "f")` 。也就是说，无论是从一个对象还是一个类中，这个函数都会同样地访问到。

那些不需要 `self` 变量的方法适合用做静态方法。

例如, 一个统计包可能包含一个用来做实验数据容器的类。这个类提供了一般的方法，来计算平均数，中位数，以及其他基于数据的描述性统计指标。然而，这个类可能包含一些概念上与统计相关但不依赖具体数据的函数。比如 `erf(x)` 就是一个统计工作中经常用到的，但却不依赖于特定数据的函数。它可以从类或者实例调用: `s.erf(1.5) --> .9332` 或者 `Sample.erf(1.5) --> .9332`.

既然staticmethod将函数原封不动的返回，那下面的代码看上去就很正常了:

```python
>>> class E(object):
     def f(x):
          print x
     f = staticmethod(f)

>>> print E.f(3)
3
>>> print E().f(3)
3
```

利用非资料描述器， `staticmethod()` 的纯Python版本看起来像这样:

```python
class StaticMethod(object):
 "Emulate PyStaticMethod_Type() in Objects/funcobject.c"

 def __init__(self, f):
      self.f = f

 def __get__(self, obj, objtype=None):
      return self.f
```

不像静态方法，类方法需要在调用函数之前会在参数列表前添上class的引用作为第一个参数。不管调用者是对象还是类，这个格式是一样的:

```python
>>> class E(object):
     def f(klass, x):
          return klass.__name__, x
     f = classmethod(f)

>>> print E.f(3)
('E', 3)
>>> print E().f(3)
('E', 3)
```

当一个函数不需要相关的数据做参数而只需要一个类的引用的时候，这个特征就显得很有用了。类方法的一个用途是用来创建不同的类构造器。在Python 2.3中, `dict.fromkeys()` 可以依据一个key列表来创建一个新的字典。等价的Python实现就是:

```python
class Dict:
    . . .
    def fromkeys(klass, iterable, value=None):
        "Emulate dict_fromkeys() in Objects/dictobject.c"
        d = klass()
        for key in iterable:
            d[key] = value
        return d
    fromkeys = classmethod(fromkeys)
```

现在，一个新的字典就可以这么创建:

```python
>>> Dict.fromkeys('abracadabra')
{'a': None, 'r': None, 'b': None, 'c': None, 'd': None}
```

用非资料描述器协议， `classmethod()` 的纯Python版本实现看起来像这样:

```python
class ClassMethod(object):
     "Emulate PyClassMethod_Type() in Objects/funcobject.c"

     def __init__(self, f):
          self.f = f

     def __get__(self, obj, klass=None):
          if klass is None:
               klass = type(obj)
          def newfunc(*args):
               return self.f(klass, *args)
          return newfunc
```