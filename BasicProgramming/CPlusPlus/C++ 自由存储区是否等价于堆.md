#CPP 

当我问你C++的内存布局时，你大概会回答：

>[“在C++中，内存区分为5个区，分别是堆、栈、自由存储区、全局/静态存储区、常量存储区”。](内存分区.md) 

如果我接着问你自由存储区与[堆](内存分区.md#堆)有什么区别，你或许这样回答：

> “malloc在堆上分配的内存块，使用free释放内存，而new所申请的内存则是在自由存储区上，使用delete来释放。”

这样听起来似乎也没错，但如果我接着问：

> 自由存储区与堆是两块不同的内存区域吗？它们有可能相同吗？

你可能就懵了。

事实上，我在网上看的很多博客，划分自由存储区与堆的分界线就是new/delete与malloc/free。然而，尽管C++标准没有要求，但很多编译器的new/delete都是以malloc/free为基础来实现的。那么请问：借以malloc实现的new，所申请的内存是在堆上还是在自由存储区上？

从技术上来说，堆（heap）是C语言和操作系统的术语。堆是操作系统所维护的一块特殊内存，它提供了动态分配的功能，当运行程序调用malloc()时就会从中分配，稍后调用free可把内存交还。而自由存储是C++中通过new和delete动态分配和释放对象的抽象概念，通过new来申请的内存区域可称为自由存储区。基本上，所有的C++编译器默认使用堆来实现自由存储，也即是缺省的全局运算符new和delete也许会按照malloc和free的方式来被实现，这时藉由new运算符分配的对象，说它在堆上也对，说它在自由存储区上也正确。但程序员也可以通过重载操作符，改用其他内存来实现自由存储，例如全局变量做的对象池，这时自由存储区就区别于堆了。我们所需要记住的就是：

> 堆是操作系统维护的一块内存，而自由存储是C++中通过new与delete动态分配和释放对象的抽象概念。堆与自由存储区并不等价。

再回过头来来看看这个问题的起源在哪里。最先我们使用C语言的时候，并没有这样的争议，很明确地知道malloc/free是在堆上进行内存操作。直到我们在Bjarne Stroustrup的书籍中数次看到free store （自由存储区）,说实话，我一直把自由存储区等价于堆。而在Herb Sutter的《exceptional C++》中，明确指出了free store（自由存储区） 与 heap（堆） 是有区别的。关于自由存储区与堆是否等价的问题讨论，大概就是从这里开始的：

> Free Store The free store is one of the two dynamic memory areas, allocated/freed by new/delete. Object lifetime can be less than the time the storage is allocated; that is, free store objects can have memory allocated without being immediately initialized, and can be destroyed without the memory being immediately deallocated. During the period when the storage is allocated but outside the object's lifetime, the storage may be accessed and manipulated through a void\* but none of the proto-object's nonstatic members or member functions may be accessed, have their addresses taken, or be otherwise manipulated.
> 
> Heap The heap is the other dynamic memory area, allocated/freed by malloc/free and their variants. Note that while the default global new and delete might be implemented in terms of malloc and free by a particular compiler, the heap is not the same as free store and memory allocated in one area cannot be safely deallocated in the other. Memory allocated from the heap can be used for objects of class type by placement-new construction and explicit destruction. If so used, the notes about free store object lifetime apply similarly here.

来源：[http://www.gotw.ca/gotw/009.htm](http://www.gotw.ca/gotw/009.htm)

作者也指出，之所以把堆与自由存储区要分开来，是因为在C++标准草案中关于这两种区域是否有联系的问题一直很谨慎地没有给予详细说明，而且特定情况下new和delete是按照malloc和free来实现，或者说是放过来malloc和free是按照new和delete来实现的也没有定论。这两种内存区域的运作方式不同、访问方式不同，所以应该被当成不一样的东西来使用。

*   自由存储是C++中通过new与delete动态分配和释放对象的**抽象概念**，而堆（heap）是C语言和操作系统的术语，是操作系统维护的一块动态分配内存。
*   new所申请的内存区域在C++中称为自由存储区。藉由堆实现的自由存储，可以说new所申请的内存区域在堆上。
*   堆与自由存储区还是有区别的，它们并非等价。

假如你来自C语言，从没接触过C++；或者说你一开始就熟悉C++的自由储存概念，而从没听说过C语言的malloc，可能你就不会陷入“自由存储区与堆好像一样，好像又不同”这样的迷惑之中。这就像Bjarne Stroustrup所说的：

> usually because they come from a different language background.


