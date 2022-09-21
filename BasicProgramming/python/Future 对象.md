#python #协程 #并发 

**这里提及的 Future 对象是协程中的 Future 对象,不是 concurrent 模块中的**

一个 Future 代表一个异步运算的最终结果。线程不安全。

Future 是一个 [awaitable](https://docs.python.org/zh-cn/3.8/glossary.html#term-awaitable) 对象。协程可以等待 Future 对象直到它们有结果或异常集合或被取消。

通常 Future 用于支持底层回调式代码(例如在协议实现中使用asyncio [transports](https://docs.python.org/zh-cn/3.8/library/asyncio-protocol.html#asyncio-transports-protocols)) 与高层异步/等待式代码交互。

经验告诉我们永远不要面向用户的接口暴露 Future 对象，同时建议使用 [`loop.create_future()`](https://docs.python.org/zh-cn/3.8/library/asyncio-eventloop.html#asyncio.loop.create_future) 来创建 Future 对象。这种方法可以让 Future 对象使用其它的事件循环实现，它可以注入自己的优化实现。