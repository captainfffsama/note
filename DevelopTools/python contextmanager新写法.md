在 `contextlib.contextmanager` 装饰的上下文中，官方类型规范推荐将返回值类型从 `Iterator[T]` 替换为 ** `Generator[YieldType, SendType, ReturnType]` **（对于上下文管理器，通常简写或明确为 `Generator[Foo, None, None]`）。

### 正确的修改方法

为了继续支持 `with` 语句并消除 Pylance 警告，只需将返回值的类型注解修改为 ** `Generator[Foo, None, None]` ** 即可。

#### 示例代码对比

**修改前的代码（报警告）：**

```Python
from contextlib import contextmanager
from typing import Iterator

class Foo:
    pass

@contextmanager
def my_context() -> Iterator[Foo]:  # ⚠️ Pylance 警告
    foo = Foo()
    try:
        yield foo
    finally:
        pass
```

**修改后的代码（推荐用法）：**

```Python
from contextlib import contextmanager
from typing import Generator

class Foo:
    pass

@contextmanager
def my_context() -> Generator[Foo, None, None]:  # ✅ 正确写法
    foo = Foo()
    try:
        yield foo
    finally:
        pass

# 使用方式完全保持不变：
with my_context() as foo:
    # 这里的 foo 同样会被推导为 Foo 类型
    pass
```

### 为什么会报这个警告？

1. **底层机制**：`@contextmanager` 装饰的函数本质上是一个**生成器函数**（内部包含 `yield`）。

2. **类型规范**：`Generator[YieldType, SendType, ReturnType]` 的三个泛型参数分别代表：

    - `YieldType`（`Foo`）：`yield` 抛出的值，也就是 `with … as target` 中 `target` 拿到的类型。

    - `SendType`（`None`）：向生成器 `send()` 发送的值，上下文管理器中不使用，填 `None`。

    - `ReturnType`（`None`）：生成器 `return` 返回的值，上下文管理器中通常填 `None`。

虽然 `Iterator[Foo]` 在逻辑上能满足“迭代”出 `Foo` 的需求，但它丢失了生成器“可被发送值”和“可有返回值”的完整类型约束，因此最新的规范和 Pylance 更推荐用明确的 `Generator` 类型。