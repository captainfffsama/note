#cmake

[toc]

# 变量
分为普通变量,环境变量,缓存变量,具体参考[cmake-language翻译](./cmake-language翻译.md)和[modern cmake](https://cmake.biofan.org/variables.html)
使用`set()`和`unset()`命令定义.

# 属性 property
CMake 存储信息的另一种方式.可以附加在诸如目录 directory 或是目标 target 上.语法如下:
```cmake
set_property(TARGET TargetName
             PROPERTY CXX_STANDARD 11)

set_target_properties(TargetName PROPERTIES
                      CXX_STANDARD 11)
```
使用`get_property(ResultVariable TARGET TargetName PROPERTY CXX_STANDARD)`可以获取属性.

# 参考
- https://moevis.github.io/cheatsheet/2018/09/12/Modern-CMake-%E7%AC%94%E8%AE%B0.html
