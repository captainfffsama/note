#C11 
#CPP 

[TOC]
C++11 引入了一种新方法来表示空指针 `nullptr`,它是一种特殊类型的字面值,可以被转换成任意其他的指针类型.

常见的几个生成空指针的方法:
```cpp
int *p1 = nullptr; //等价于int *p1 = 0;
int *p2 = 0;       //直接将p2初始化为字面常量0
// 需要首先 #include <cstdlib>
int *p3 = NULL;   //等价于int *p3 = 0;
```

其中`int *p3 = NULL; ` 是过去的做法,`NULL` 是一个预处理变量,在`cstdlib`中定义,值就是0.在进去`delete`前,需要检查:
```cpp
if (p3)
{delete p3;}
```

**但对于包含 `nullptr` 的指针变量直接应用 `delete` 是安全的,没有必要在进行检查判断!**
>C++17入门经典 P135

# 参考
>C++ Primer 第5版 P48
