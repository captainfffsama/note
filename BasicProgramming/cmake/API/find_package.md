#cmake  

[toc]  


相关参考:
- https://zhuanlan.zhihu.com/p/97369704
- https://www.cnblogs.com/narjaja/p/9533199.html  

官方:
- https://cmake.org/cmake/help/latest/command/find_package.html
- https://gitlab.kitware.com/cmake/community/-/wikis/doc/tutorials/How-To-Find-Libraries  


# 基本使用和模块模式
本函数主要用于从外部项目查找包和加载设置. 
CMake 内置了一些包,[具体列表](https://cmake.org/cmake/help/latest/manual/cmake-modules.7.html),若CMake 是使用 apt 安装,在 ubuntu下,这些包可以在`/usr/share/cmake-3.20/Modules/`下找到.
函数命令首先在模块路径中寻找实现编译好的`Find<包名>.cmake`的文件.模块路径的顺序依次是:
- `${CMAKE_MODULE_PATH}`中所有目录
- 刚刚提到的`/usr/share/cmake-3.20/Modules`目录

 包名通常是全部大写,也有用帕斯卡命名法的.

通常在找到`Find<包名>.cmake`文件之后,这个文件里通常会定义以下变量:
```cmake
<包名>_FOUND # 指示包是否找到
<包名>_INCLUDE_DIRS # 指示包的头文件目录
# 或
<包名>_INCLUDES

<包名>_LIBRARIES # 指示包的链接库目录
#或
<包名>_LIBRARIES
#或
<包名>_LIBS

<包名>_DEFINITIONS
```
之后使用`target_include_directories()`添加头文件,`target_link_libraries()`添加链接库即可.

若函数在模块模式下没有找到包,则会进入配置模式寻找`<包名>Config.cmake`或者`<包名>-Config.cmake`   

将`CMAKE_FIND_PACKAGE_PREFER_CONFIG`设置为True,那么会优先使用配置模式.
## 函数形式和参数详解
```
find_package(<PackageName> [version] [EXACT] [QUIET] [MODULE]
             [REQUIRED] [[COMPONENTS] [components...]]
             [OPTIONAL_COMPONENTS components...]
             [NO_POLICY_SCOPE])
```

**version**:  
version 用来指示需要的包的版本要求,通常有两种形式:
1. 指定确切版本
2. 通过指定一个范围.比如`3.2.1...4.2.5`,这种方式仅在cmake 3.19后支持,[详细见](https://gitlab.kitware.com/cmake/cmake/-/issues/21131).
注意CMake 不会对版本数进行任何转换,包的版本号来自于包自带的版本文件.一般包会有一个配置文件`<包名>-Config.cmake`, 版本文件名称一般是`<包名>-Config-version.cmake`或是`<包名>-Config-Version.cmake`
若没有指明这个参数,那么`version`和`EXACT`会自动从外部调用前向继承,注意这个继承仅仅是针对包与包之间的.若设定了`version`范围,但是包被设计成只接受单一版本的,那么 CMake 会取范围的下限,即最老的包版本.
关于版本文件的说明和写法,详细见:https://cmake.org/cmake/help/latest/command/find_package.html?highlight=find_package#id4

**EXACT**:  
这个参数要求版本必须完全匹配,和 `version` 的第二种形式不兼容.

**QUIET**:  
静默模式,会禁用所有信息性提示,包括包没有找到的消息.若是没有 `REQUIRED` 参数下,没找到包不会停止处理和报错,只会有一条信息性提示.

**REQUIRED**:  
若包没找到,会给出一个错误消息,并停止构建进程.

**MODULE**:  
若模块模式下没有找到包,也不会进入到配置模式

**COMPONENTS**:
之后列出必要的组件,可以查找这个组件,但是是否查到的信息由被查的包控制

**OPTIONAL_COMPONENTS**:
用来列举可选组件.


# 完整参数和配置模式
一般情况下使用以上基本配置参数即可,但是这里我们也列出全部参数
```
find_package(<PackageName> [version] [EXACT] [QUIET]
             [REQUIRED] [[COMPONENTS] [components...]]
             [OPTIONAL_COMPONENTS components...]
             [CONFIG|NO_MODULE]
             [NO_POLICY_SCOPE]
             [NAMES name1 [name2 ...]]
             [CONFIGS config1 [config2 ...]]
             [HINTS path1 [path2 ... ]]
             [PATHS path1 [path2 ... ]]
             [PATH_SUFFIXES suffix1 [suffix2 ...]]
             [NO_DEFAULT_PATH]
             [NO_PACKAGE_ROOT_PATH]
             [NO_CMAKE_PATH]
             [NO_CMAKE_ENVIRONMENT_PATH]
             [NO_SYSTEM_ENVIRONMENT_PATH]
             [NO_CMAKE_PACKAGE_REGISTRY]
             [NO_CMAKE_BUILDS_PATH] # Deprecated; does nothing.
             [NO_CMAKE_SYSTEM_PATH]
             [NO_CMAKE_SYSTEM_PACKAGE_REGISTRY]
             [CMAKE_FIND_ROOT_PATH_BOTH |
              ONLY_CMAKE_FIND_ROOT_PATH |
              NO_CMAKE_FIND_ROOT_PATH])
```

**CONFIG**:
作用同`NO_MODULE`,仅使用配置模式.

配置模式下,会创建`<PackageName>_DIR`的变量来保存被引包的配置文件所在目录.在找到包并处理完配置文件之后,配置文件的完整路径将存在变量`<PackageName>_CONFIG`中.  
所有配置文件可能的名称放在`<PackageName>_CONSIDERED_CONFIGS`中,其相关的版本信息放在`<PackageName>_CONSIDERED_VERSIONS`中.  

关于函数处理完成之后得到的参数可以见:https://cmake.org/cmake/help/latest/command/find_package.html?highlight=find_package#id6  
关于函数处理是搜索配置文件的详细顺序和可能地址见:https://cmake.org/cmake/help/latest/command/find_package.html?highlight=find_package#id5  

