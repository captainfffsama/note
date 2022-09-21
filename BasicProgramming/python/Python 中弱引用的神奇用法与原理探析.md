#python 
#弱引用

[toc]
#  背景

开始讨论弱引用（ weakref ）之前，我们先来看看什么是弱引用？它到底有什么作用？

假设我们有一个多线程程序，并发处理应用数据： 

```python
# 占用大量资源，创建销毁成本很高 
class Data: 
    def __init__(self, key): 
        pass
```

应用数据 Data 由一个 key 唯一标识，同一个数据可能被多个线程同时访问。由于 Data 需要占用很多系统资源，创建和消费的成本很高。我们希望 Data 在程序中只维护一个副本，就算被多个线程同时访问，也不想重复创建。

为此，我们尝试设计一个缓存中间件 Cacher ： 

```python
import threading 
# 数据缓存 
class Cacher: 
    def __init__(self): 
        self.pool = {} 
        self.lock = threading.Lock() 
    def get(self, key): 
        with self.lock: 
            data = self.pool.get(key) 
            if data: 
                return data 
            self.pool[key] = data = Data(key) 
            return data
```

Cacher 内部用一个 dict 对象来缓存已创建的 Data 副本，并提供 get 方法用于获取应用数据 Data 。get 方法获取数据时先查缓存字典，如果数据已存在，便直接将其返回；如果数据不存在，则创建一个并保存到字典中。因此，数据首次被创建后就进入缓存字典，后续如有其它线程同时访问，使用的都是缓存中的同一个副本。

感觉非常不错！但美中不足的是：Cacher 有资源泄露的风险！

因为 Data 一旦被创建后，就保存在缓存字典中，永远都不会释放！换句话讲，程序的资源比如内存，会不断地增长，最终很有可能会爆掉。因此，我们希望一个数据等所有线程都不再访问后，能够自动释放。

我们可以在 Cacher 中维护数据的引用次数， get 方法自动累加这个计数。于此同时提供一个 remove 新方法用于释放数据，它先自减引用次数，并在引用次数降为零时将数据从缓存字段中删除。

线程调用 get 方法获取数据，数据用完后需要调用 remove 方法将其释放。Cacher 相当于自己也实现了一遍引用计数法，这也太麻烦了吧！Python 不是内置了垃圾回收机制吗？为什么应用程序还需要自行实现呢？

冲突的主要症结在于 Cacher 的缓存字典：它作为一个中间件，本身并不使用数据对象，因此理论上不应该对数据产生引用。那有什么黑科技能够在不产生引用的前提下，找到目标对象吗？我们知道，赋值都是会产生引用的！

# 典型用法

这时，弱引用（ weakref ）隆重登场了！弱引用是一种特殊的对象，能够在不产生引用的前提下，关联目标对象。 

```python
# 创建一个数据 
>>>d = Data('fasionchan.com') 
>>> d 
<__main__.Data object at 0x1018571f0>
# 创建一个指向该数据的弱引用 
>>> import weakref 
>>>r = weakref.ref(d) 
# 调用弱引用对象，即可找到指向的对象 
>>> r() 
<__main__.Data object at 0x1018571f0>
>>> r() is d 
True 
# 删除临时变量d，Data对象就没有其他引用了，它将被回收 
>>> del d 
# 再次调用弱引用对象，发现目标Data对象已经不在了（返回None） 
>>> r()

```

![](https://s3.51cto.com/oss/202112/09/9732ee24310627705f10c3681c30a954.jpg)

这样一来，我们只需将 Cacher 缓存字典改成保存弱引用，问题便迎刃而解！ 

```python
import threading 
import weakref 
# 数据缓存 
class Cacher: 
    def __init__(self): 
        self.pool = {} 
        self.lock = threading.Lock() 
    def get(self, key): 
        with self.lock: 
            r = self.pool.get(key) 
            if r: 
                data = r() 
                if data: 
                    return data 
            data = Data(key) 
            self.pool[key] = weakref.ref(data) 
            return data
```

由于缓存字典只保存 Data 对象的弱引用，因此 Cacher 不会影响 Data 对象的引用计数。当所有线程都用完数据后，引用计数就降为零因而被释放。

实际上，用字典缓存数据对象的做法很常用，为此 weakref 模块还提供了两种只保存弱引用的字典对象：

*    weakref.WeakKeyDictionary ，键只保存弱引用的映射类（一旦键不再有强引用，键值对条目将自动消失）；
*    weakref.WeakValueDictionary ，值只保存弱引用的映射类（一旦值不再有强引用，键值对条目将自动消失）；

因此，我们的数据缓存字典可以采用 weakref.WeakValueDictionary 来实现，它的接口跟普通字典完全一样。这样我们不用再自行维护弱引用对象，代码逻辑更加简洁明了： 

```python
import threading 
import weakref 
# 数据缓存 
class Cacher: 
    def __init__(self): 
        self.pool = weakref.WeakValueDictionary() 
        self.lock = threading.Lock() 
    def get(self, key): 
        with self.lock: 
            data = self.pool.get(key) 
            if data: 
                return data 
            self.pool[key] = data = Data(key) 
            return data` 
```

weakref 模块还有很多好用的工具类和工具函数，具体细节请参考官方文档，这里不再赘述。

# 工作原理

那么，弱引用到底是何方神圣，为什么会有如此神奇的魔力呢？接下来，我们一起揭下它的面纱，一睹真容！ 

```python
>>>d = Data('fasionchan.com') 
# weakref.ref 是一个内置类型对象 
>>> from weakref import ref 
>>> ref 
<class 'weakref'>
# 调用weakref.ref类型对象，创建了一个弱引用实例对象 
>>>r = ref(d) 
>>> r 
<weakref at 0x1008d5b80; to 'Data' at 0x100873d60>` 
```

经过前面章节，我们对阅读内建对象源码已经轻车熟路了，相关源码文件如下：

*    Include/weakrefobject.h 头文件包含对象结构体和一些宏定义；
*    Objects/weakrefobject.c 源文件包含弱引用类型对象及其方法定义；

我们先扒一扒弱引用对象的字段结构，定义于 Include/weakrefobject.h 头文件中的第 10-41 行： 

```cpp
typedef struct _PyWeakReference PyWeakReference; 
/* PyWeakReference is the base struct for the Python ReferenceType, ProxyType, 
 * and CallableProxyType. 
 */ 
#ifndef Py_LIMITED_API 
struct _PyWeakReference { 
    PyObject_HEAD 
    /* The object to which this is a weak reference, or Py_None if none. 
     * Note that this is a stealth reference:  wr_object's refcount is 
     * not incremented to reflect this pointer. 
     */ 
    PyObject *wr_object; 
    /* A callable to invoke when wr_object dies, or NULL if none. */ 
    PyObject *wr_callback; 
    /* A cache for wr_object's hash code.  As usual for hashes, this is -1 
     * if the hash code isn't known yet. 
     */ 
    Py_hash_t hash; 
    /* If wr_object is weakly referenced, wr_object has a doubly-linked NULL- 
     * terminated list of weak references to it.  These are the list pointers. 
     * If wr_object goes away, wr_object is set to Py_None, and these pointers 
     * have no meaning then. 
     */ 
    PyWeakReference *wr_prev; 
    PyWeakReference *wr_next; 
}; 
#endif` 
```

由此可见，PyWeakReference 结构体便是弱引用对象的肉身。它是一个定长对象，除固定头部外还有 5 个字段：

![](https://s5.51cto.com/oss/202112/09/b85ca0027ba941bf26bf27cc0fef480c.jpg)

*    wr_object ，对象指针，指向被引用对象，弱引用根据该字段可以找到被引用对象，但不会产生引用；
*    wr_callback ，指向一个可调用对象，当被引用的对象销毁时将被调用；
*    hash ，缓存被引用对象的哈希值；
*    wr_prev 和 wr_next 分别是前后向指针，用于将弱引用对象组织成双向链表；

结合代码中的注释，我们知道：

![](https://s6.51cto.com/oss/202112/09/d0cb131b0b9adbf9c3083139d11b36f7.jpg)

*    弱引用对象通过 wr_object 字段关联被引用的对象，如上图虚线箭头所示；
*    一个对象可以同时被多个弱引用对象关联，图中的 Data 实例对象被两个弱引用对象关联；
*    所有关联同一个对象的弱引用，被组织成一个双向链表，链表头保存在被引用对象中，如上图实线箭头所示；
*    当一个对象被销毁后，Python 将遍历它的弱引用链表，逐一处理：
    *     将 wr_object 字段设为 None ，弱引用对象再被调用将返回 None ，调用者便知道对象已经被销毁了；
    *     执行回调函数 wr_callback （如有）；

由此可见，弱引用的工作原理其实就是设计模式中的 观察者模式（ Observer ）。当对象被销毁，它的所有弱引用对象都得到通知，并被妥善处理。

# 实现细节

掌握弱引用的基本原理，足以让我们将其用好。如果您对源码感兴趣，还可以再深入研究它的一些实现细节。

前面我们提到，对同一对象的所有弱引用，被组织成一个双向链表，链表头保存在对象中。由于能够创建弱引用的对象类型是多种多样的，很难由一个固定的结构体来表示。因此，Python 在类型对象中提供一个字段 tp_weaklistoffset ，记录弱引用链表头指针在实例对象中的偏移量。

![](https://s3.51cto.com/oss/202112/09/7465627be3399423e2bc6bb4d56ffc67.jpg)

由此一来，对于任意对象 o ，我们只需通过 ob_type 字段找到它的类型对象 t ，再根据 t 中的 tp_weaklistoffset 字段即可找到对象 o 的弱引用链表头。

Python 在 Include/objimpl.h 头文件中提供了两个宏定义： 

```cpp
/* Test if a type supports weak references */ 
#define PyType_SUPPORTS_WEAKREFS(t) ((t)->tp_weaklistoffset > 0) 
#define PyObject_GET_WEAKREFS_LISTPTR(o) \ 
    ((PyObject **) (((char *) (o)) + Py_TYPE(o)->tp_weaklistoffset))` 

```

*    PyType_SUPPORTS_WEAKREFS 用于判断类型对象是否支持弱引用，仅当 tp_weaklistoffset 大于零才支持弱引用，内置对象 list 等都不支持弱引用；
*    PyObject_GET_WEAKREFS_LISTPTR 用于取出一个对象的弱引用链表头，它先通过 Py_TYPE 宏找到类型对象 t ，再找通过 tp_weaklistoffset 字段确定偏移量，最后与对象地址相加即可得到链表头字段的地址；

我们创建弱引用时，需要调用弱引用类型对象 weakref 并将被引用对象 d 作为参数传进去。弱引用类型对象 weakref 是所有弱引用实例对象的类型，是一个全局唯一的类型对象，定义在 Objects/weakrefobject.c 中，即：_PyWeakref_RefType（第 350 行）。

![](https://s3.51cto.com/oss/202112/09/2ee93226509e6d380d0451fb803063fd.jpg)

根据对象模型中学到的知识，Python 调用一个对象时，执行的是其类型对象中的 tp_call 函数。因此，调用弱引用类型对象 weakref 时，执行的是 weakref 的类型对象，也就是 type 的 tp_call 函数。tp_call 函数则回过头来调用 weakref 的 tp_new 和 tp_init 函数，其中 tp_new 为实例对象分配内存，而 tp_init 则负责初始化实例对象。

回到 Objects/weakrefobject.c 源文件，可以看到 _PyWeakref_RefType 的 tp_new 字段被初始化成 weakref___new__ （第 276 行）。该函数的主要处理逻辑如下：

1.   解析参数，得到被引用的对象（第 282 行）；
2.   调用 PyType_SUPPORTS_WEAKREFS 宏判断被引用的对象是否支持弱引用，不支持就抛异常（第 286 行）；
3.   调用 GET_WEAKREFS_LISTPTR 行取出对象的弱引用链表头字段，为方便插入返回的是一个二级指针（第 294 行）；
4.   调用 get_basic_refs 取出链表最前那个 callback 为空 基础弱引用对象（如有，第 295 行）；
5.   如果 callback 为空，而且对象存在 callback 为空的基础弱引用，则复用该实例直接将其返回（第 296 行）；
6.   如果不能复用，调用 tp_alloc 函数分配内存、完成字段初始化，并插到对象的弱引用链表（第 309 行）；
    *     如果 callback 为空，直接将其插入到链表最前面，方便后续复用（见第 4 点）；
    *     如果 callback 非空，将其插到基础弱引用对象（如有）之后，保证基础弱引用位于链表头，方便获取；

当一个对象被回收后，tp_dealloc 函数将调用 PyObject_ClearWeakRefs 函数对它的弱引用进行清理。该函数取出对象的弱引用链表，然后逐个遍历，清理 wr_object 字段并执行 wr_callback 回调函数（如有）。具体细节不再展开，有兴趣的话可以自行查阅 Objects/weakrefobject.c 中的源码，位于 880 行。

好了，经过本节学习，我们彻底掌握了弱引用相关知识。弱引用可以在不产生引用计数的前提下，对目标对象进行管理，常用于框架和中间件中。弱引用看起来很神奇，其实设计原理是非常简单的观察者模式。弱引用对象创建后便插到一个由目标对象维护的链表中，观察（订阅）对象的销毁事件。
