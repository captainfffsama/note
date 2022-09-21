#cmake   


```cmake
set_property(<GLOBAL                      |
              DIRECTORY [<dir>]           |
              TARGET    [<target1> ...]   |
              SOURCE    [<src1> ...]
                        [DIRECTORY <dirs> ...] |
                        [TARGET_DIRECTORY <targets> ...]
              INSTALL   [<file1> ...]     |
              TEST      [<test1> ...]     |
              CACHE     [<entry1> ...]    >
             [APPEND] [APPEND_STRING]
             PROPERTY <name> [<value1> ...])
```
在给定的作用域内给零个或者多个对象设置[属性](https://cmake.biofan.org/properties.html),
第一个参数确定了作用域,是以下值中之一:
- GLOBAL
- DIRECTORY
- TARGET
- SOURCE
- INSTALL
- TEST
- CACHE
