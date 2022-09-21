#cmake
#CPP 
#待处理 

# 第三方依赖使用指北
原文地址:https://cmake.org/cmake/help/latest/guide/using-dependencies/index.html?highlight=package_dir#using-dependencies-guide

[toc]

#  引言

使用 CMake 来管理第三方二进制包的最佳方式有好几种方式,其实现方式取决于 CMake 是如何找到这些第三方二进制包的.

第三方包随附的 CMake 文件里包含了所有构建依赖的说明.对于可选的依赖,使用不同的构建特性是可以构建成功的.而有些依赖的必须的.针对每条构建依赖, CMake 都会搜索默认的地址和第三方提供的额外地址来定位每条依赖.

若必要的依赖没有被找到,那么 CMake 会将 `NOTFOUND` 缓存起来.当然,这个值是可以更改的,具体参考[`User Interaction Guide`](../user-interaction/index.html#guide:User Interaction Guide "User Interaction Guide")

# 提供配置文件包的链接库

若第三方二进制包提供了[配置文件包](https://cmake.org/cmake/help/latest/manual/cmake-packages.7.html#config-file-packages),那么使用依赖是最为便利的.这些配置文件是一些文本文件,指导着 CMake 如何使用二进制链接库以及如何关联链接,帮助工具和 CMake 宏.

这些文件通常在类似`lib/cmake/<PackageName>`的目录中,当然,可能在其他其他地方.这里 `<PackageName>` 和 [`find_package()`](CMAKE%20%20api.md)中参数是一致的,比如:`[find_package](PackageName REQUIRED)`.

`lib/cmake/<PackageName>`目录中通常包含名为`<PackageName>Config.cmake` 或 `<PackageName>-config.cmake`的文件.这是 CMake 关键的包的进入点.一般还有个名为`<PackageName>ConfigVersion.cmake`的可选文件,此文件常被`find_package()`用来验证第三方包的版本是否满足构建的要求.当然,`find_package()`验证版本是可选的.

一旦`Config.cmake`文件被找到,且版本满足要求.那么`find_package`命令就会认为包已被找到,且整个链接库的包是完整可用的.

也许还会有一些额外文件提供了一些 CMake 宏或者是[Imported Targets](https://cmake.org/cmake/help/latest/manual/cmake-buildsystem.7.html#imported-targets)供使用.这些文件没有什么强制的命名约定,使用[`include()`](https://cmake.org/cmake/help/latest/command/include.html#command:include)即可将这些文件引入到主配置文件中.

使用第三方二进制包[`调用CMake`](https://cmake.org/cmake/help/latest/guide/user-interaction/index.html#guide:User%20Interaction%20Guide)要求`find_package()`能成功找到包.若包在 CMake 已知的目录,那么就可以调用成功.这些已知目录是和平台有关的,在 Linux 上在 `/usr` 开头的目录可以找到.windows 则大概在`Program Files`.

针对包在诸如`/opt/mylib` 或 `$HOME/dev/prefix` 这种没法自动找到的目录这种情况,CMake 提供了一些方法让用户来指示如何找到这些包.

比如可以在和[CMake交互时设定](https://cmake.org/cmake/help/latest/guide/user-interaction/index.html#setting-build-variables)[`CMAKE_PREFIX_PATH`](https://cmake.org/cmake/help/latest/variable/CMAKE_PREFIX_PATH.html#variable:CMAKE_PREFIX_PATH)变量.该变量可以指示一个包含了[配置文件包](https://cmake.org/cmake/help/latest/manual/cmake-packages.7.html#config-file-packages)的列表.比如一个安装在`/opt/somepackage`的包,其配置文件可能是`/opt/somepackage/lib/cmake/somePackage/SomePackageConfig.cmake`.此时,`/opt/somepackage`将被添加进`CMAKE_PREFIX_PATH`.

使用前缀目录填充的环境变量`CMAKE_PREFIX_PATH`也可以用来指示包的搜索.类似`PATH`环境变量,该环境变量是平台特定的,分隔符在Unix上是`:`,windows上是`;`.

`CMAKE_PREFIX_PATH`变量在有多个前缀目录需要指定或是在同一目录中有多个不同的包可用的情况下提供了便利.使用`<PackageName>_DIR`同样可以指定包的路径,比如`SomePackage_DIR`.但此变量必须使用包含配置文件目录的完整路径.

译者注:这里意思就是 `CMAKE_PREFIX_PATH` 之后 CMake 会拼接上包名,在这个目录下找包配置文件,但是`<PackageName>_DIR`就不行,比如完全指定配置文件所在的目录.  
比如一个名为`Torch`的包的配置文件位于`/opt/Torch`下.在`CMAKE_PREFIX_PATH`中追加`/opt`或者`/opt/Torch`都是可以找到包配置文件的(一般就写`/opt`),但是使用`Torch_DIR`就必须指定`set(Torch "/opt/Torch")`

# 从包中导入目标(Imported Targets)

一些包含配置文件包的第三方包可能提供[导入目标Imported Targets](https://cmake.org/cmake/help/latest/manual/cmake-buildsystem.7.html#imported-targets).Imported Targets 通常在包含有于第三方包相关的特定配置路径的文件中指定,比如指定库的debug和release版本.

Often the third-party package documentation will point out the names of imported targets available after a successful `find_package` for a library. Those imported target names can be used with the [`target_link_libraries()`](../../command/target_link_libraries.html#command:target_link_libraries "target_link_libraries") command.

A complete example which makes a simple use of a third party library might look like:

cmake\_minimum\_required(VERSION  3.10)
project(MyExeProject  VERSION  1.0.0)

find\_package(SomePackage  REQUIRED)
add\_executable(MyExe  main.cpp)
target\_link\_libraries(MyExe  PRIVATE  SomePrefix::LibName)

See [`cmake-buildsystem(7)`](../../manual/cmake-buildsystem.7.html#manual:cmake-buildsystem(7) "cmake-buildsystem(7)") for further information about developing a CMake buildsystem.

### [Libraries not Providing Config-file Packages](#id5)[¶](#libraries-not-providing-config-file-packages "Permalink to this headline")

Third-party libraries which do not provide config-file packages can still be found with the [`find_package()`](../../command/find_package.html#command:find_package "find_package") command, if a `FindSomePackage.cmake` file is available.

These module-file packages are different to config-file packages in that:

1.  They should not be provided by the third party, except perhaps in the form of documentation
    
2.  The availability of a `Find<PackageName>.cmake` file does not indicate the availability of the binaries themselves.
    
3.  CMake does not search the [`CMAKE_PREFIX_PATH`](../../variable/CMAKE_PREFIX_PATH.html#variable:CMAKE_PREFIX_PATH "CMAKE_PREFIX_PATH") for `Find<PackageName>.cmake` files. Instead CMake searches for such files in the [`CMAKE_MODULE_PATH`](../../variable/CMAKE_MODULE_PATH.html#variable:CMAKE_MODULE_PATH "CMAKE_MODULE_PATH") variable. It is common for users to set the [`CMAKE_MODULE_PATH`](../../variable/CMAKE_MODULE_PATH.html#variable:CMAKE_MODULE_PATH "CMAKE_MODULE_PATH") when running CMake, and it is common for CMake projects to append to [`CMAKE_MODULE_PATH`](../../variable/CMAKE_MODULE_PATH.html#variable:CMAKE_MODULE_PATH "CMAKE_MODULE_PATH") to allow use of local module-file packages.
    
4.  CMake ships `Find<PackageName>.cmake` files for some [`third party packages`](../../manual/cmake-modules.7.html#manual:cmake-modules(7) "cmake-modules(7)") for convenience in cases where the third party does not provide config-file packages directly. These files are a maintenance burden for CMake, so new Find modules are generally not added to CMake anymore. Third-parties should provide config file packages instead of relying on a Find module to be provided by CMake.
    

Module-file packages may also provide [Imported Targets](../../manual/cmake-buildsystem.7.html#imported-targets). A complete example which finds such a package might look like:

cmake\_minimum\_required(VERSION  3.10)
project(MyExeProject  VERSION  1.0.0)

find\_package(PNG  REQUIRED)

\# Add path to a FindSomePackage.cmake file
list(APPEND  CMAKE\_MODULE\_PATH  "${CMAKE\_SOURCE\_DIR}/cmake")
find\_package(SomePackage  REQUIRED)

add\_executable(MyExe  main.cpp)
target\_link\_libraries(MyExe  PRIVATE
  PNG::PNG
  SomePrefix::LibName
)

The [`<PackageName>_ROOT`](../../variable/PackageName_ROOT.html#variable:<PackageName>_ROOT "<PackageName>_ROOT") variable is also searched as a prefix for [`find_package()`](../../command/find_package.html#command:find_package "find_package") calls using module-file packages such as `FindSomePackage`.
