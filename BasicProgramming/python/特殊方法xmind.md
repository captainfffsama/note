#python 

# 特殊方法

## 基本制定

### `object.__new__(cls[,...])`

### `object.__init__(self[,...])`

### `object.__del__(self)`

### `object.__repr__(sellf)`

### `object.__str__(self)`

### `object.___bytes__(self)`

### `object.__format__(self,format_spec)`

### 富比较方法

如果两个操作数的类型不同，且右操作数类型是左操作数类型的直接或间接子类，则优先选择右操作数的反射方法，否则优先选择左操作数的方法。虚拟子类不会被考虑。



- `object.__lt__(self,other)` <
- `object.__le__(self,other)` <=
- `object.__ne__(self,other)` !=默认情况下委托`__eq__`方法并将结果取反
- `object.__gt__(self,other)` >
- `object.__ge__(self,other)` >=
- `object.__eq__(self,other)` ==

### `object.__hash__(self)`

如果一个类没有定义 `__eq__()`方法，那么也不应该定义 `__hash__()`操作；如果它定义了 `__eq__()` 但没有定义 `__hash__()`，则其实例将不可被用作可哈希集的项。如果一个类定义了可变对象并实现了 `__eq__()` 方法，则不应该实现 `__hash__()`，因为可哈希集的实现要求键的哈希集是不可变的（如果对象的哈希值发生改变，它将处于错误的哈希桶中）。
用户定义的类默认带有 `__eq__()` 和 `__hash__()` 方法；使用它们与任何对象（自己除外）比较必定不相等，并且 `x.__hash__()` 会返回一个恰当的值以确保 x == y 同时意味着 x is y 且 `hash(x) == hash(y)`。
一个类如果重载了 `__eq__()` 且没有定义 `__hash__()` 则会将其 `__hash__()` 隐式地设为 None。当一个类的 `__hash__()` 方法为 None 时，该类的实例将在一个程序尝试获取其哈希值时正确地引发 `TypeError`，并会在检测 `isinstance(obj, collections.abc.Hashable)` 时被正确地识别为不可哈希对象。
如果一个重载了 `__eq__()` 的类需要保留来自父类的 `__hash__()` 实现，则必须通过设置 `__hash__ = <ParentClass>.__hash__` 来显式地告知解释器。
如果一个没有重载 `__eq__()` 的类需要去掉哈希支持，则应该在类定义中包含 `__hash__ = None`。一个自定义了 `__hash__()` 以显式地引发 TypeError 的类会被 `isinstance(obj, collections.abc.Hashable)` 调用错误地识别为可哈希对象。

### `object.__bool__(self)`

如果未定义此方法，则会查找并调用 __len__() 并在其返回非零值时视对象的逻辑值为真。如果一个类既未定义 __len__() 也未定义 __bool__() 则视其所有实例的逻辑值为真。

## 自定义属性访问

### `object.__getattr__(self,name)`

注意这个方法 是当尝试访问属性引发AttributeError 时才调用,和__setattr()__是不对等的

### `object.__getattribute__(self,name)`

为了避免此方法中的无限递归，其实现应该总是调用具有相同名称的基类方法来访问它所需要的任何属性，例如 object.__getattribute__(self, name)。

### `object.__setattr__(self,name,value)`

### `object.__delattr__(self,name)`

此方法应该仅在 del obj.name 对于该对象有意义时才被实现。

### `object.__dir__(self)`

## 自定义模块属性访问

### `__getattr__`

### `__dir__`

### 

## 实现描述器

### `object.__get__(self,instance,owner=None`

### `object.__set__(self,instance,value)`

### `object.__delete__(self,instance)`

### `object.__set_name__(self,owner,name)`

