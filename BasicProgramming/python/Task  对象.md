#python #协程 #并发   

一个与 [`Future 类似`](https://docs.python.org/zh-cn/3.8/library/asyncio-future.html#asyncio.Future) 的对象，可运行 Python [协程](https://docs.python.org/zh-cn/3.8/library/asyncio-task.html#coroutine)。非线程安全。

Task 对象被用来在事件循环中运行协程。如果一个协程在等待一个 Future 对象，Task 对象会挂起该协程的执行并等待该 Future 对象完成。当该 Future 对象 *完成*，被打包的协程将恢复执行。

事件循环使用协同日程调度: 一个事件循环每次运行一个 Task 对象。而一个 Task 对象会等待一个 Future 对象完成，该事件循环会运行其他 Task、回调或执行 IO 操作。

使用高层级的 [`asyncio.create_task()`](https://docs.python.org/zh-cn/3.8/library/asyncio-task.html#asyncio.create_task) 函数来创建 Task 对象，也可用低层级的 [`loop.create_task()`](https://docs.python.org/zh-cn/3.8/library/asyncio-eventloop.html#asyncio.loop.create_task) 或 [`ensure_future()`](https://docs.python.org/zh-cn/3.8/library/asyncio-future.html#asyncio.ensure_future) 函数。不建议手动实例化 Task 对象。

要取消一个正在运行的 Task 对象可使用 [`cancel()`](https://docs.python.org/zh-cn/3.8/library/asyncio-task.html#asyncio.Task.cancel) 方法。调用此方法将使该 Task 对象抛出一个 [`CancelledError`](https://docs.python.org/zh-cn/3.8/library/asyncio-exceptions.html#asyncio.CancelledError) 异常给打包的协程。如果取消期间一个协程正在等待一个 Future 对象，该 Future 对象也将被取消。

[`cancelled()`](https://docs.python.org/zh-cn/3.8/library/asyncio-task.html#asyncio.Task.cancelled) 可被用来检测 Task 对象是否被取消。如果打包的协程没有抑制 [`CancelledError`](https://docs.python.org/zh-cn/3.8/library/asyncio-exceptions.html#asyncio.CancelledError) 异常并且确实被取消，该方法将返回 `True`。

[`asyncio.Task`](https://docs.python.org/zh-cn/3.8/library/asyncio-task.html#asyncio.Task) 从 [`Future`](https://docs.python.org/zh-cn/3.8/library/asyncio-future.html#asyncio.Future) 继承了其除 [`Future.set_result()`](https://docs.python.org/zh-cn/3.8/library/asyncio-future.html#asyncio.Future.set_result) 和 [`Future.set_exception()`](https://docs.python.org/zh-cn/3.8/library/asyncio-future.html#asyncio.Future.set_exception) 以外的所有 API。

Task 对象支持 [`contextvars`](https://docs.python.org/zh-cn/3.8/library/contextvars.html#module-contextvars) 模块。当一个 Task 对象被创建，它将复制当前上下文，然后在复制的上下文中运行其协程。