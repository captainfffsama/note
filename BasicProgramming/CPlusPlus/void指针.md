#CPP 

`void*` 是一种特殊的指针类型,可用于存放任意对象的地址.从 `void*`视角来看内存空间也仅仅就是内存空间,无法访问内存空间中所存的对象.   
它能做的事情也比较有限,比如和别的指针比较,作为函数的输入或输出,或者赋值给另外的一个 `void*` 指针.**注意我们不能直接操作 void* 指针所指的对象.***

# 参考
> C++ primer 5th P50