#CPP 

在C程序中，如果定义了一个静态函数，而没有去使用，编译时会有一个告警：
```cpp
#include <stdio.h>

int main(void)
{
    printf("main\n");
}

static void a(void)
{
    printf("a\n");
}
```

```bash
$ gcc a.c -Wall
a.c:8:13: warning: 'a' defined but not used [-Wunused-function]
 static void a(void)
             ^
```

而使用`attribute((unused))`可以告诉编译器忽略此告警：
```cpp
#include <stdio.h>

int main(void)
{
    printf("main\n");
}

__attribute__((unused)) static void a(void)
{
    printf("a\n");
}
```

```bash
$ gcc a.c -Wall
$
```

#  参考
1. https://www.jianshu.com/p/21aef14340a8
2. https://stackoverflow.com/questions/31909631/c11-style-unused-attribute-in-gcc/31909713