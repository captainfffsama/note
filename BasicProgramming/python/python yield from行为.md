#yield_from #python #协程


- 子生成器产出的值都直接传给委派生成器的调用方(即客户端代码).
- 使用`send()`方法发给委派生成器的值都是直接传给字子生成器.若发送的值时None,那么调用子生成器的`__next__()`方法.(其实使用`next`方法预激活协程和使用`send(None)`是一样的).若调用方法抛出`StopIteration`异常,那么委派生成器恢复运行.任何其他异常都会向上冒泡,传给委派生成器.
- 生成器退出时,生成器(或子生成器)中的`return expr`表达式会触发`StopIteration(expr)`异常抛出.
- `yield from`表达式的值是子生成器终止时传给`StopIteration`异常的第一个参数.
- 传入委派生成器的异常,除了`GeneratorExit`之外都传给子生成器的`throw()`方法.若调用`throw()`方法抛出`StopIteration`异常,委派生成器恢复运行.`StopIteration`之外的异常会向上冒泡,传给委派生成器.
- 若把`GeneratorExit`异常传入委派生成器,或在委派生成器上调用`close()`方法,那么在子生成器上调用`close()`方法,若它有的话.若调用`close()`方法导致异常抛出,那么异常会向上冒泡,传给委派生成器;否则,委派生成器抛出`GeneratorExit`异常.

> 参见 流畅的python 16.8节