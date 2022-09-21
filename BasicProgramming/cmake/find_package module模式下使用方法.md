#cmake 
#CPP 

[原文](https://cloud.tencent.com/developer/ask/87956)


命令`find_package`有两种模式：`Module`模式和`Config`模式。你正在尝试`Module`在实际需要`Config`模式时使用模式。

### Module mode

`Find<package>.cmake`文件位于项目内。像这样的东西：

```
CMakeLists.txt
cmake/FindFoo.cmake
cmake/FindBoo.cmake

```

`CMakeLists.txt` 内容：

```
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")
find_package(Foo REQUIRED) # FOO_INCLUDE_DIR, FOO_LIBRARIES
find_package(Boo REQUIRED) # BOO_INCLUDE_DIR, BOO_LIBRARIES

include_directories("${FOO_INCLUDE_DIR}")
include_directories("${BOO_INCLUDE_DIR}")
add_executable(Bar Bar.hpp Bar.cpp)
target_link_libraries(Bar ${FOO_LIBRARIES} ${BOO_LIBRARIES})

```

请注意，`CMAKE_MODULE_PATH`当你需要重写标准`Find<package>.cmake`文件时，它具有高优先级并且可能有用。

### Config mode（install）

`<package>Config.cmake`文件位于**外部**并由`install` 其他项目的命令生成（`Foo`例如）。

`foo`library：

```
> cat CMakeLists.txt 
cmake_minimum_required(VERSION 2.8)
project(Foo)

add_library(foo Foo.hpp Foo.cpp)
install(FILES Foo.hpp DESTINATION include)
install(TARGETS foo DESTINATION lib)
install(FILES FooConfig.cmake DESTINATION lib/cmake/Foo)

```

配置文件的简化版本：

```
> cat FooConfig.cmake 
add_library(foo STATIC IMPORTED)
find_library(FOO_LIBRARY_PATH foo HINTS "${CMAKE_CURRENT_LIST_DIR}/../../")
set_target_properties(foo PROPERTIES IMPORTED_LOCATION "${FOO_LIBRARY_PATH}")

```

默认项目安装在`CMAKE_INSTALL_PREFIX`目录中：

```
> cmake -H. -B_builds
> cmake --build _builds --target install
-- Install configuration: ""
-- Installing: /usr/local/include/Foo.hpp
-- Installing: /usr/local/lib/libfoo.a
-- Installing: /usr/local/lib/cmake/Foo/FooConfig.cmake

```

### Config mode（use）

使用`find_package(... CONFIG)`包括`FooConfig.cmake`进口的目标`foo`：

```
> cat CMakeLists.txt 
cmake_minimum_required(VERSION 2.8)
project(Boo)

# import library target \`foo\`
find_package(Foo CONFIG REQUIRED)

add_executable(boo Boo.cpp Boo.hpp)
target_link_libraries(boo foo)
> cmake -H. -B_builds -DCMAKE_VERBOSE_MAKEFILE\=ON
> cmake --build _builds
Linking CXX executable Boo
/usr/bin/c++ ... -o Boo /usr/local/lib/libfoo.a

```

请注意，导入的目标是**高度可**配置的。
