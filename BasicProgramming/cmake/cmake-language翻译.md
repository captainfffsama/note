#cmake
#翻译

[toc]

原文地址:https://cmake.org/cmake/help/latest/manual/cmake-language.7.html

# Organization 架构
CMake 文件指的是使用"CMake 语言"写的 `CMakeLists.txt` 或是以 `.cmake` 为后缀名的文件.
一个项目中的 CMake 语言源文件可以分为以下三类:
- 目录(Directories) `CMakeLists.txt`
- 脚本(Scripts) `<script>.cmake`
- 模块(Modules) `<module>.cmake`

## Directories
处于顶级源目录中的 `CMakeLists.txt` 是 CMake 处理一个项目树时的入口点.该文件既可以包含整个项目的构建规范,也可以使用 `add_subdirectory()` 命令来添加子目录进行构建.被 `add_subdirectory()` 添加的子目录也应在自己目录下包含一个 `CMakeLists.txt` 作为目录入口.对于每一个包含了 `CMakeLists.txt` 的目录, CMake 都会在构建树中生成一个对应的目录,并将之设置为默认的工作和输出目录.

## Scripts
在[CMake 命令行工具](https://cmake.org/cmake/help/latest/manual/cmake.1.html)中使用 `-P` 参数可以执行单个的 `<scripts>.cmake`.在脚本模式下, CMake 仅仅执行给定的 CMake 语言源文件中的命令,但不生成整个构建系统,也不允许 CMake 命令构建目标.

## Modules
在 Directories 或者 Scripts 中都可以使用 [`include()`](https://cmake.org/cmake/help/latest/command/include.html#command:include) 命令来加载 `<modules>.cmake` 源文件.具体可以参见 [cmake-modules](https://cmake.org/cmake/help/latest/manual/cmake-modules.7.html#manual:cmake-modules(7)). 若项目包含自己设置的模块,则可以在 `CMAKE_MODULE_PATH` 变量中指定它们的位置.

# 语法
## 编码
CMake 语言源文件可以使用 7-bit ASCII 文本编写,以在所有支持的平台上实现最大的可移植性.换行符可以是 `\n` 或 `\r\n`,但是在输入文件被读取时统一转换为 `\n`.  
注意这个实现是纯8位的,所以在支持此编码的系统上,源文件可以编码为 UTF-8.此外,CMake 3.2 以上将支持在 windows 将源文件编码为 UTF-8(使用 UTF-16 将调用系统 API).此外，CMake 3.0及以上版本允许在源文件中使用UTF-8字节顺序标记.

## 源文件
每个 CMake 源文件包含0或多条[命令](),命令之间可以使用换行符,空格和[注释]()作为分隔:
```cmake
file         ::=  file_element*
file_element ::=  command_invocation line_ending |
                  (bracket_comment|space)* line_ending
line_ending  ::=  line_comment? newline
space        ::=  <match '[ \t]+'>
newline      ::=  <match '\n'>
```

注意，任何不在[命令参数]()或[括号注释]()内的源文件行都可以以[行注释]()结束.

## 命令
命令的结构是命令名称后跟使用括号括起来的参数,括号中参数以空格分隔:
```cmake
command_invocation  ::=  space* identifier space* '(' arguments ')'
identifier          ::=  <match '[A-Za-z_][A-Za-z0-9_]*'>
arguments           ::=  argument? separated_arguments*
separated_arguments ::=  separation+ argument? |
                         separation* '(' arguments ')'
separation          ::=  space | line_ending
```
例:
```cmake
add_executable(hello world.c)
```
命令是大小写无关的.在命令调用时,每个`(` 或 `)` 都是作为一个[无引号参数Unquoted Argument]()传给命令的.因此在使用 `if()` 命令时,可以使用括号来表达条件嵌套关系
```cmake
if(FALSE AND (FALSE OR TRUE)) # 结果为 FALSE
```

> **注意:** CMake 3.0 之前的版本要求命令名至少是2字符.  
CMake 2.8.12 之前的版本默认接受一个无引号参数或是一个中括号参数之后紧跟一个中括号参数,且不用空格分隔.为了兼容,之后的版本也接受这种写法,但是会抛出警告.

## 命令参数
三种类型:
- 中括号参数 bracket_argument
- 引号参数 quoted_argument
- 无引号参数 unquoted_argument

### 中括号参数
受 Lua 长括号的语法启发,括号参数将内容包括在一对闭合中括号中,这对括号可以是多个括号叠加起来,中间可以加=,只需长度相同,类似 Lua 中括号的解析.   


译者注:  
Lua中括号机制:
```lua
string3 = [[this is string3\n]] -- 0 级正的长括号
string4 = [=[this is string4\n]=] -- 1 级正的长括号
string5 = [==[this is string5\n]==] -- 2 级正的长括号
string6 = [====[ this is string6\n[===[]===] ]====] -- 4 级正的长括号，可以包含除了本级别的反长括号外的所有内容
print(string3)
print(string4)
print(string5)
print(string6)
```
输出:
```bash
this is string3\n
this is string4\n
this is string5\n
 this is string6\n[===[]===]
```

中括号中参数内容是纯文本,不对其中的[转义字符]()或是[变量引用]()进行转换.中括号参数总是作为一个参数,提供给命令的.
例:
```cmake
message([=[
This is the first line in a bracket argument with bracket length 1.
No \-escape sequences or ${variable} references are evaluated.
This is always one argument even though it contains a ; character.
The text does not end on a closing bracket of length 0 like ]].
It does end in a closing bracket of length 1.
]=])
```

>注意: 3.0 版本前不接受中括号参数,它们将左括号解释为一个无引号参数的开始.

### 引号参数
引号参数将参数包裹在双引号中.  
其内容包含了左右双引号之间所有的内容,中间的转义字符和变量引用将被转换.

例:
```cmake
message("This is a quoted argument containing multiple lines.
This is always one argument even though it contains a ; character.
Both \\-escape sequences and ${variable} references are evaluated.
The text does not end on an escaped double-quote like \".
It does end in an unescaped double quote.
")
```
每行末尾的`\`个数为奇数,那么最后一个`\`将被视为是连续符,其后紧随的换行符将被忽略.例:
```cmake
message("\
This is the first line of a quoted argument. \
In fact it is the only line but since it is long \
the source code uses line continuation.\
")
```
上面将被解释为一行.
>注意:3.0之前的不支持使用`\`续行.这种语法将导致异常报错.

### 无引号参数
无引号参数不被任何括号包裹.中间不包含空格,`(`,`)`,`#`,`"`,`\`,除非这些符号被反斜杠转义.  

无引号参数中的转义字符或是变量引用将被转换.其值的分隔结果和 [List] 分隔元素的结果类似.每个非空的元素都将被作为一个参数送给命令.因此,无引号参数提供给命令时,可以被视为零个或者多个参数.  

例:
```cmake
foreach(arg
    NoSpace
    Escaped\ Space
    This;Divides;Into;Five;Arguments
    Escaped\;Semicolon
    )
  message("${arg}")
endforeach()
```
>**注意:**为了支持原始的 CMake 代码,无引号参数可以包含双引号字符串,双引号字符串中可能含有空格,并支持生成式变量引用`$(MAKEVAR)`.  
无转义的双引号必须左右都有,可以不在无引号参数的开头,并被视为是内容的一部分.例如,无引号参数`-Da="b c"`,`-Da=$(v)`,`a" "b"c"d`,它们都将被解释为字面意思.它们可以被写为引号参数`"-Da=\"b c\""`,`"-Da=$(v)"`,`"a\" \"b\"c\"d"`.  
生成式引用将按照字面意思作为内容的一部分,而不进行变量的扩展.它们将被整体视为是一个参数,而非分隔开的`$`,`(`,`MAKEVAR`,`)`.  
我们不建议在新代码中使用无引号参数.在必要时应使用引号参数或者中括号参数来表示内容.

## 转义字符
转义字符是`\`后紧跟一个字符.  

`\`后紧跟一个非字母数字字符将被视为是对文字字符进行转义而不被解释为 CMake 语法.`\t`编码为 Tab,`\r` 编码为回车,`\n`编码是换行.变量引用外侧的`\;`将被编码为它自己,但在无引号参数中可能被编码为`;`而不是对参数值进行分隔.变量引用内部的`\;`将被编码为`;`.(参考[CMP0053](https://cmake.org/cmake/help/latest/policy/CMP0053.html#policy:CMP0053)文档)

## 变量引用
变量引用的形式是`${<variable>}`,在引号参数和无引号参数中将被展开,使用变量的值替代变量引用,若变量值没有设定,则被展开为一个空字符串.变量引用是可以嵌套的,并由内而外的展开,比如:`${outer_${inner_variable}_variable}`.  

变量应用的字面上可由数字字面,`/_.+-`和转义字符组成.具体参考[CMP0053](https://cmake.org/cmake/help/latest/policy/CMP0053.html#policy:CMP0053)  
变量一节文档展示了变量名的范围和如何设置值.  

环境变量(environment variable)引用的形式是`$ENV{<variable>}`.具体参考[环境变量]()一节.  

缓存变量(cache variable)形式是`$CACHE{<variable>}`.参考[缓存]()一节.   

`if()`命令有一个特殊的语法,它允许变量引用使用简短形式 `<variable>` 来替代 `${<variable>}`.但是环境变量引用和缓存变量引用是不适用这一条的.  

## 注释 
注释以`#`开头,`#`不可在中括号参数和引号参数内,也不被`\`转义成无引号参数一部分.它有括号注释和行注释两种形式.

### 括号注释
`#`之后跟着一个方括号括起来的注释.  
例:  
```cmake
#[[This is a bracket comment.
It runs until the close bracket.]]
message("First Argument\n" #[[Bracket Comment]] "Second Argument")
```
>注意:3.0之前版本不支持括号注释.括号注释将被解释为以`#`开头的行注释  

### 行注释
`#`若没有紧跟`[`,那么该行将被视为行注释,直到行末.  
例:  
```cmake
# This is a line comment.
message("First Argument\n" # This is a line comment :)
        "Second Argument") # This is a line comment.
``` 
# 控制流 
## 条件控制块
以 `if()` / `elseif()` / `else()` / `endif()`命令界定的代码块.

## 循环
以 `foreach()` / `endforeach()` 和 `while()` / `endwhile()` 命令界定的代码块将被循环.在内部使用 `break()` 命令可以中断循环.使用 `continue()` 命令可以跳到下一个迭代.  

## 命令定义
以 `macro()` / `endmacro()` 和 `function()` / `endfunction()` 命令界定的代码将被记录下来供后面调用,前者是宏,后者是函数.  

# 变量 
变量是 CMake 语言中最基础的存储单元.它们的值都是字符串类型,尽管有些命令会将字符串解释为其它类型的值.`set()` 和 `unset()` 命令将显示的设置和取消一个变量,但其他命令则有可能修改变量.变量名是大小写敏感的,且几乎可以包含任何文本,但是我们建议变量名仅使用数字字母,`_`和`-`.  

变量有动态的作用域.每个变量"设定"或者"取消",都会在当前作用域中创建一个绑定:  

函数作用域  
    即在 `function()` 命令创建的函数的代码块中, 变量set或是unset都将在这个作用域中绑定,对当前函数和其嵌套函数是可见的,但是在函数返回之后不可见.

目录作用域
    源树下的每个目录`CMakeLists.txt`都有自己的变量绑定.在进入目录前, CMake 将复制定义在父目录中的所有变量绑定,并将之初始化新的目录作用域.使用`cmake -P`运行的CMake 脚本中的变量绑定作用域仅限定于当前脚本.   

持久化缓存
    CMake 存储了一组单独的"缓存"变量或是"缓存条目".它们的值在项目构建树的多次运行时可以持续存在.缓存条目的绑定作用域只能被显示的修改,比如在`set()`或者`unset()`命令中添加`CACHE`选项.  

当展开变量引用时,CMake 将首先搜索函数调用栈,若存在,退回到当前目录作用域进行绑定.若发现"set"绑定,将使用它的值,若发现"unset"绑定或者没发现绑定,CMake 将搜索缓存条目.若在缓存条目中找到,便使用它的值.否则,变量引用将被展开为一个空字符串.而`$CACHE{VAR}`将直接搜索整个缓存条目.   

[CMake-variables](https://cmake.org/cmake/help/latest/manual/cmake-variables.7.html#manual:cmake-variables(7))文档展示了很多CMake提供的变量.  

>注意:CMake保留以下标识符:
以`CMAKE_`,`_CMAKE_`或是以`_`加上任何CMake命令名称开头的,其中`_`可以是`_`,`-`

# 环境变量
环境变量和普通变量的不同主要在以下几点:  

作用域:  
    环境变量在整个CMake进程中具有全局作用域,且永不被缓存.

引用:
    引用形式是`$ENV{variable}`  

初始化
    CMake 环境变量的初始值是调用进程的初始值.使用`set()`或者`unset()`可以修改其值,但仅影响CMake进程的运行,并不影响整个系统的环境变量.修改的值不会写回到调用进程,对于后续的测试或者是构建进程也是不可见的.

具体参见[cmake-env-varibales](https://cmake.org/cmake/help/latest/manual/cmake-env-variables.7.html#manual:cmake-env-variables(7))

# 列表
尽管 CMake 中所有值都被存储为字符串,但在诸如展开无引号参数语境中,字符串将被视为是列表.在该语境下,`;`将被视为是分隔符,将字符串分隔成列表,注意此时`;`后不可紧跟`[`或是`]`或被`\`转义.`\;`将不会分隔值,并在结果元素中被替换成`;`.  
列表元素最终将被解释成使用`;`分隔的字符串,比如以下使用`set()`命令存储多个值到目标变量中:
```cmake
set(srcs a.c b.c c.c) #将 "srcs" 设置成 "a.c;b.c;c.c"
```

列表应用于诸如源文件列表等简单情况,而不应用于复制的数据处理任务.大多数命令构造列表时不会转义列表元素中的`;`,仅会扁平化列表:
```cmake
set(srcs a.c b.c c.c) # 将"srcs" 设置为 "a;b;c",而不是 "a;b|;c"
```

