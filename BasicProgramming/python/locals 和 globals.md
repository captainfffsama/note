#python 

[toc]  

# Python 两个内置函数: locals 和 globals


这两个函数主要提供，基于字典的访问局部和全局变量的方式。

在理解这两个函数时，首先来理解一下 Python 中的名字空间概念。Python 使用叫做名字空间的东西来记录变量的轨迹。名字空间只是一个字典，它的键字就是变量名，字典的值就是那些变量的值。

实际上，名字空间可以像 Python 的字典一样进行访问。

每个函数都有着自已的名字空间，叫做局部名字空间，它记录了函数的变量，包括函数的参数和局部定义的变量。每个模块拥有它自已的名字空间，叫做全局名字空间，它记录了模块的变量，包括函数、类、其它导入的模块、模块级的变量和常量。还有就是内置名字空间，任何模块均可访问它，它存放着内置的函数和异常。

当一行代码要使用变量 x 的值时，Python 会到所有可用的名字空间去查找变量，按照如下顺序：

*   **1、局部名字空间** - 特指当前函数或类的方法。如果函数定义了一个局部变量 x，Python将使用这个变量，然后停止搜索。
*   **2、全局名字空间** - 特指当前的模块。如果模块定义了一个名为 x 的变量，函数或类，Python将使用这个变量然后停止搜索。
*   **3、内置名字空间** - 对每个模块都是全局的。作为最后的尝试，Python 将假设 x 是内置函数或变量。

如果 Python 在这些名字空间找不到 x，它将放弃查找并引发一个 NameError 的异常，同时传递 **There is no variable named 'x'** 这样一条信息。

局部变量函数 locals 例子（locals 返回一个名字/值对的字典）：

**实例**

```python
def foo(arg, a):  
    x = 1  
    y = 'xxxxxx'  
    for i in range(10):  
        j = 1  
        k = i  
    print(locals())  
#调用函数的打印结果      
foo(1,2)  
#{'k': 9, 'j': 1, 'i': 9, 'y': 'xxxxxx', 'x': 1, 'a': 2, 'arg': 1}  
```

**from module import** 和 **import module** 之间的不同。使用 import module，模块自身被导入，但是它保持着自已的名字空间，这就是为什么你需要使用模块名来访问它的函数或属性（module.function）的原因。但是使用 from module import，实际上是从另一个模块中将指定的函数和属性导入到你自己的名字空间，这就是为什么你可以直接访问它们却不需要引用它们所来源的模块的原因。

locals 是只读的，globals 不是。

locals 不可修改，globals 可以修改，原因是：

*   **locals()** 实际上没有返回局部名字空间，它返回的是一个拷贝。所以对它进行修改，修改的是拷贝，而对实际的局部名字空间中的变量值并无影响。
*   globals() 返回的是实际的全局名字空间，而不是一个拷贝与 locals 的行为完全相反。

所以对 globals 所返回的 dictionary 的任何的改动都会直接影响到全局变量的取值。

**实例**

```python
#!/usr/bin/env python    

z = 7 #定义全局变量    
def foo(arg):    
    x = 1    
    print( locals() )    
    print('x=',x)  
    locals()['x'] = 2 #修改的是局部名字空间的拷贝，而实际的局部名字空间中的变量值并无影响。    
    print( locals() )  
    print( "x=",x )

  foo(3)    
print( globals() )  
print( 'z=',z )  
globals()["z"] = 8 #globals（）返回的是实际的全局名字空间，修改变量z的值    
print( globals() )  
print( "z=",z )
```
输出结果为：

```python
{'x': 1, 'arg': 3} 
x= 1 
{'x': 1, 'arg': 3} 
x= 1 
{'__name__': '__main__', '__doc__': None, '__package__': None, '__loader__': <_frozen_importlib_external.SourceFileLoader object at 0x10b099358>, '__spec__': None, '__annotations__': {}, '__builtins__': <module 'builtins' (built-in)>, '__file__': 'test.py', '__cached__': None, 'z': 7, 'foo': <function foo at 0x10ae48e18>} 
z= 7 
{'__name__': '__main__', '__doc__': None, '__package__': None, '__loader__': <_frozen_importlib_external.SourceFileLoader object at 0x10b099358>, '__spec__': None, '__annotations__': {}, '__builtins__': <module 'builtins' (built-in)>, '__file__': 'test.py', '__cached__': None, 'z': 8, 'foo': <function foo at 0x10ae48e18>} 
z= 8
```
