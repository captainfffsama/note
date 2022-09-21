#cmake   

原始地址:https://cmake.org/cmake/help/v3.20/guide/tutorial/index.html#id1

[toc]

# 引言  
学习本教程,您将以一个循序渐进的方式了解 CMake 是如何解决各种常见的构建系统的问题. 通过示例来了解一个关键操作点是如何协同操作的将对您的学习大有裨益.教程文档和示例源码可以在 CMake 源码中的 `Help/guide/tutorial` 目录中找到.每个知识点都有它自己的子目录,里面包含了可以直接运行的代码.教程示例是循序渐进的,我们将在每个步骤中都为上一步提供了更加完整的解决方案.  

# 基础开头 (Step 1) 
从源码构建可执行文件是最基本的步骤.对于这个简单的项目,我们只须一个三行内容的 `CMakeLists.txt`.这将是本教程的起点.下面我们在 `Step1` 目录创建一个如下内容的 `CMakeLists.txt`:
```cmake
cmake_minimum_required(VERSION 3.10)

# 设置项目名称
project(Tutorial)

# 添加可执行文件
add_executable(Tutorial tutorial.cxx)
```  
注意在 `CMakeLists.txt` 中我们仅使用了小写字母的命令.其实无论是大小写还是混合大小写, CMake 都是支持的. `tutorial.cxx` 的源码位于 `Step1` 目录中,它可以计算一个数的平方根.

`tutorial.cxx` 内容:  
```cmake
// A simple program that computes the square root of a number
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>

int main(int argc, char* argv[])
{
  if (argc < 2) {
    std::cout << "Usage: " << argv[0] << " number" << std::endl;
    return 1;
  }

  // convert input to double
  const double inputValue = atof(argv[1]);

  // calculate square root
  const double outputValue = sqrt(inputValue);
  std::cout << "The square root of " << inputValue << " is " << outputValue
            << std::endl;
  return 0;
}
```  

## 添加版本号和配置头文件 
下面我们将为我们的可执行文件和项目添加版本号.当然我们可以只在源码管理时做,但是使用 `CMakeLists.txt` 可以更加灵活.  
首先,修改 `CMakeLists.txt` ,使用 [project()](https://cmake.org/cmake/help/v3.20/command/project.html#command:project) 来设置项目名和版本号.  
```cmake
cmake_minimum_required(VERSION 3.10)  

# 设置项目名和版本号
project(Tutorial VERSION 1.0)
```
配置一个配置头文件 [`TutorialConfig.h.in`](https://github.com/captainfffsama/CMake3.20TutorialExample/blob/master/Step2/TutorialConfig.h.in) 将版本号传递给源码:   
```cmake
configure_file(TutorialConfig.h.in TutorialConfig.h)
```
由于配置文件将被写入到二进制树中,因此我们需要在搜索目录中包含配置文件所在的目录.添加以下内容到 `CMakeLists.txt` 末尾:  
```cmake
target_include_directories(Tutorial PUBLIC
                            "${PROJECT_BINARY_DIR}"
                            )
``` 

在源目录中创建 `TutorialConfig.h.in` 文件并添加以下内容  
```cpp
// the configured options and settings for Tutorial
#define Tutorial_VERSION_MAJOR @Tutorial_VERSION_MAJOR@
#define Tutorial_VERSION_MINOR @Tutorial_VERSION_MINOR@
```
当 CMake 配置此配置头文件是,其中的 `@Tutorial_VERSION_MINOR@` 和 `@Tutorial_VERSION_MINJOR@` 将被替换.  
接着,修改 [`tutorial.cxx`](https://github.com/captainfffsama/CMake3.20TutorialExample/blob/master/Step2/tutorial.cxx) 来包含配置头文件, `TutorialConfig.h`.  
最后,让我们在 `tutorial.cxx` 中添加以下内容来打印可执行文件的名称和版本号.
```cmake
  if (argc < 2) {
    // report version
    std::cout << argv[0] << " Version " << Tutorial_VERSION_MAJOR << "."
              << Tutorial_VERSION_MINOR << std::endl;
    std::cout << "Usage: " << argv[0] << " number" << std::endl;
    return 1;
  }
```

## 指定 C++ 标准
接下来，通过在 [`tutorial.cxx`](https://github.com/captainfffsama/CMake3.20TutorialExample/blob/master/Step3/tutorial.cxx) 中用 `std::stod` 替换 `atof`，将一些 C++ 11功能添加到我们的项目中。 同时，删除 `#include <cstdlib>`。  
```cpp
const double inputValue = std::stod(argv[1]);
```
我们需要在 CMake 代码中显式声明出正确的标志.在 CMake 中使用 [`CMAKE_CXX_STANDARD`](https://cmake.org/cmake/help/v3.20/variable/CMAKE_CXX_STANDARD.html#variable:CMAKE_CXX_STANDARD) 变量可以指定 C++ 标准.在本教程中,在 `CMakeLists.txt` 中指定 [`CMAKE_CXX_STANDARD`](https://cmake.org/cmake/help/v3.20/variable/CMAKE_CXX_STANDARD.html#variable:CMAKE_CXX_STANDARD) 为 11, [`CMAKE_CXX_STANDARD_REQUIRED`](https://cmake.org/cmake/help/v3.20/variable/CMAKE_CXX_STANDARD_REQUIRED.html#variable:CMAKE_CXX_STANDARD_REQUIRED) 为 True.注意确保 `CMAKE_CXX_STANDARD` 的声明添加在 `add_executable` 之前.   
```cmake
cmake_minimum_required(VERSION 3.10)

# 设置项目名和版本号
project(Tutorial VERSION 1.0)

# 指定 C++ 标准
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)
```

## 构建并测试
使用 [cmake](https://cmake.org/cmake/help/v3.20/manual/cmake.1.html#manual:cmake(1)) 或者 [cmake-gui](https://cmake.org/cmake/help/v3.20/manual/cmake-gui.1.html#manual:cmake-gui(1)) 可以配置项目并使用你选择的构建工具构建.  

例如,对于命令行工具 cmake,我们跳转到 CMake 源码目录中的 `Help/guide/tutorial` 目录并创建以下构建目录:  
```bash
mkdir Step1_build
```
然后,进入构建目录并执行 CMake 来配置项目,生成一个原始的构架系统:
```bash
cd Step1_build  
cmake ../Step1
```
然后调用构建系统来编译/链接项目:
```bash
cmake --build .
```
最后,对新构建的 `Tutorial` 可执行文件使用以下命令:
```bash
Tutorial 4294967296
Tutorial 10
Tutorial
```

# 添加库(Step 2)
现在我们将为项目添加库.这个库包含来计算数字平方根的实现.可执行文件可以使用这个库而非编译器提供的标准平方根函数.   
我们将库放到名为 `MathFunctions` 目录下.这个目录将包含头文件 `MathFunctions.h` 和一个源文件 `mysqrt.cxx`.源文件中有一个名为 `mysqrt` 的函数,其功能和原来编译器的 `sqrt` 函数类似.   

在 `MathFunctions` 目录中的 `CMakeLists.txt` 添加以下内容:
```cmake
add_library(MathFunctions mysqrt.cxx)
```
在顶层的 `CMakeLists.txt` 中使用 [`add_subdirectory()`](https://cmake.org/cmake/help/v3.20/command/add_subdirectory.html#command:add_subdirectory) 来确保库将被构建.我们给可执行文件(executable)添加了一个新库,并将 `MathFunctions` 目录添加到 include 目录中来保证 [`MathFunctions.h`]() 头文件可以被找到.在顶层 `CMakeLists.txt` 中加入以下几行:

```cmake
# add the MathFunctions library
add_subdirectory(MathFunctions)

# add the executable
add_executable(Tutorial tutorial.cxx)

target_link_libraries(Tutorial PUBLIC MathFunctions)

# add the binary tree to the search path for include files
# so that we will find TutorialConfig.h
target_include_directories(Tutorial PUBLIC
                          "${PROJECT_BINARY_DIR}"
                          "${PROJECT_SOURCE_DIR}/MathFunctions"
                          )
```

现在我们将使 MatchFunctions 库变为可选项.对于本教程来说这样做是没有必要的,但是对于大型项目,这种做法非常常见.首先,我们在顶层的 `CMakeLists.txt` 文件中添加一个选项:   
```cmake
option(USE_MYMATH "Use tutorial provided math implementation" ON)

# configure a header file to pass some of the CMake settings
# to the source code
configure_file(TutorialConfig.h.in TutorialConfig.h)
```

这个选项在 cmake-gui 或者 ccmake 中以 ON 作为默认值进行显示,当然这个默认值是可以更改的.这个配置将被存在 CMake 缓存中,因此用户不必每次在构建目录中执行 CMake 时都设定这个值.  

接下来设定构建和链接 MatchFunciotns 库的条件.在顶层 `CMakeLists.txt` 末尾添加类似以下的内容:
```cmake
if(USE_MYMATH)
    add_subdirectory(MathFunctions)
    list(APPEND EXTRA_LIBS MathFunctions)
    list(APPEND EXTRA_INCLUDES "${PROJECT_SOURCE_DIR}/MathFunctions")
endif()

# add the executable
add_executable(Tutorial tutorial.cxx)

target_link_libraries(Tutorial PUBLIC ${EXTRA_LIBS})

# add the binary tree to the search path for include files
# so that we will find TutorialConfig.h
target_include_directories(Tutorial PUBLIC
                           "${PROJECT_BINARY_DIR}"
                           ${EXTRA_INCLUDES}
                           )
```

注意这里使用了 `EXTRA_LIBS` 变量来汇集所有的可选库,以便之后链接到可执行文件. `EXTRA_INCLUDES` 变量作用与之类似,只不过是作用于头文件的.这是处理很多可选组件时的一个经典方法,等下我们将使用现代方法来完成这件事.

对源文件的修改则是非常容易的,在 `tutorial.cxx` 使用以下语句包含 `MathFunctions.h` 头文件:
```cpp
#ifdef USE_MYMATH
#  include "MathFunctions.h"
#endif
```
同样,使用 `USE_MYMATH` 来控制使用哪个平方根函数:
```cpp
#ifdef USE_MYMATH
  const double outputValue = mysqrt(inputValue);
#else
  const double outputValue = sqrt(inputValue);
#endif
```  
由于源代码要求使用 `USE_MYMATH`,我们可以在 `TutorialConfig.h.in` 中添加以下内容:  
```cpp
#cmakedefine USE_MYMATH
```

**练习:**为何在 `USE_MYMATH` 之后配置 `TutorialConfig.h.in` 很重要?如果倒过来呢?   

执行 cmake 或者 cmake-gui 来配置项目并构建.然后执行构建好的可执行文件.  

现在,我们来更新 `USE_MYMATH` 的值.使用 cmake-gui 或 ccmake 是最简单的方法.当然,改变命令行参数也可以,比如:
```bash
cmake ../Step2 -DUSE_MYMATH=OFF
```
重新构建和执行教程.   
哪一个函数给出来更好的结果,是 sqrt 还是 mysqrt?

# 添加库的使用要求 (Step 3)
使用要求可以对库或者可执行文件的链接和 include 进行更好的控制,对于 CMake 内部目标的属性传递也可以进行更多的控制.其涉及的命令有:
-   [`target_compile_definitions()`](https://cmake.org/cmake/help/v3.20/command/target_compile_definitions.html#command:target_compile_definitions)  
-   [`target_compile_options()`](https://cmake.org/cmake/help/v3.20/command/target_compile_options.html#command:target_compile_options)  
-   [`target_include_directories()`](https://cmake.org/cmake/help/v3.20/command/target_include_directories.html#command:target_include_directories)  
-   [`target_link_libraries()`](https://cmake.org/cmake/help/v3.20/command/target_link_libraries.html#command:target_link_libraries)  

接下来,我们将重构 [添加库(Step 2)]() 这一节的代码,使用现代 CMake 的方法来满足需求.我们首先声明任何链接到 MathFunctions 的人都需要包含当前源码目录,而 MathFunctions 自己则不需要.这将变成一个 `INTERFACE` 使用要求.   
`INTERFACE` 表示这件事消费者,使用者需要但是生产者不需要.在 `MathFunctions/CMAKELists.txt` 中添加以下内容:
```cmake
target_include_directories(MathFunctions
            INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}
            )
```
由于我们指定了 MathFunctions 的使用要求,因此我们现在可以放心的移除之前在顶层 `CMakeLists.txt` 中使用的 `EXTRA_INCLUDES` 变量:
```cmake 
if(USE_MYMATH)
  add_subdirectory(MathFunctions)
  list(APPEND EXTRA_LIBS MathFunctions)
endif()
```
和
```cmake
target_include_directories(Tutorial PUBLIC
                           "${PROJECT_BINARY_DIR}"
                           )
```
可以删去.   

完成此操作之后,执行 cmake 或 cmake-gui 来配置项目,在构建目录中使用 `cmake --build .` 来构建项目.   

# 安装和测试(Step 4)
现在我们可以开始给项目添加安装规则和测试支持.

## 安装规则
安装规则非常简单:对于 MathFunctions,我们希望安装库和头文件,对于应用程序,我们希望安装可执行文件和配置头文件.

在 `MathFunctions/CMakeLists.txt` 末尾添加:
```cmake
install(TARGETS MathFunctions DESTINATION lib)
install(FILES MathFunctions.h DESTINATION include)         
```
在顶层 `CMakeLists.txt` 末尾添加:
```cmake
install(TARGETS Tutorial DESTINATION bin)
install(FILES "${PROJECT_BINARY_DIR}/TutorialConfig.h"
  DESTINATION include
  )
```
以上便是本教程创建一个基本本地安装的全部要求.   

接下来配置和构建项目.

使用 cmake 命令中的 `install` 选项可以执行安装步骤(3.15以下的 CMake 而可以使用 `make install`).对于多套配置的工具,不要忘了使用 `--config` 参数来指定配置.若是使用 IDE,只需要构建 `INSTALL` 目标.这一步将安装头文件,库和可执行文件,例如:
```cmake
cmake --install .
```
变量 [`DCMAKE_INSTALL_PREFIX`](https://cmake.org/cmake/help/v3.20/variable/CMAKE_INSTALL_PREFIX.html#variable:CMAKE_INSTALL_PREFIX) 被用来确定文件安装的根目录.若是使用 `cmake --install` 命令,使用 `--prefix` 参数可以覆盖安装路径前缀.例如:
```cmake
cmake --install . --prefix "/home/myuser/installdir"
```
跳转到安装目录并验证教程是否安装成功.  

## 测试支持
现在我们来测试我们的应用.在顶层 `CMakeLists.txt` 末尾,我们可以启用测试,然后添加一些基本测试,以验证应用程序是否可以正常工作.
```cmake
enable_testing()

# does the application run
add_test(NAME Runs COMMAND Tutorial 25)

# does the usage message work?
add_test(NAME Usage COMMAND Tutorial)
set_tests_properties(Usage
  PROPERTIES PASS_REGULAR_EXPRESSION "Usage:.*number"
  )

# define a function to simplify adding tests
function(do_test target arg result)
  add_test(NAME Comp${arg} COMMAND ${target} ${arg})
  set_tests_properties(Comp${arg}
    PROPERTIES PASS_REGULAR_EXPRESSION ${result}
    )
endfunction(do_test)

# do a bunch of result based tests
do_test(Tutorial 4 "4 is 2")
do_test(Tutorial 9 "9 is 3")
do_test(Tutorial 5 "5 is 2.236")
do_test(Tutorial 7 "7 is 2.645")
do_test(Tutorial 25 "25 is 5")
do_test(Tutorial -25 "-25 is [-nan|nan|0]")
do_test(Tutorial 0.0001 "0.0001 is 0.01")
```
第一个测试简单验证了应用是否可以运行,没有 segfault 或者其他的崩溃,返回值为0.这是 CTest 的基本形式.  

下面一个测试使用 [`PASS_REGULAR_EXPRESSION`](https://cmake.org/cmake/help/v3.20/prop_test/PASS_REGULAR_EXPRESSION.html#prop_test:PASS_REGULAR_EXPRESSION) 测试属性来验证输出是否包含某些字符.在这种情况下,验证提供的参数数量不正确时,是否打印用法消息.  

最后,我们定义来一个名为 `do_test` 的函数来运行应用并验证给定输入的平方根是否可以计算正确.
每次调用 `do_test`,都会像项目中添加一个测试,包括名称,输入和应有的结果.
重新构建应用,进入到二进制目录并执行 ctest:`ctest -N` 和 `ctest -VV`.对于多套配置生成器(比如 Visual Studio),必须指定配置类型.在 Debug 模式下执行测试,比如在构建目录使用 `ctest -C Debug -VV`(注意不是 Debug 子目录).在 IDE 中可以构建 `RUN_TESTS` 目标. 

# 添加系统自检(Step 5)

