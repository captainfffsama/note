#qt 
#qmake
#待处理

[原文](https://www.ljjyy.com/archives/2021/02/100642.html)

[toc]

本系列文章翻译[qmake](https://doc.qt.io/qt-5/qmake-manual.html)的用户手册，该版本qmake在Qt 5.15中使用。

# 简介

qmake 工具有助于简化跨平台的开发项目的构建过程。它用于自动生成 Makefile，借助 qmake 我们创建每个 Makefile 时可能只需要简单地编写几行信息即可。不管我们的项目是不是用 Qt 编写的，我们都可以使用 qmake 来处理项目的构建流程。

qmake 根据 project 文件中的信息生成 Makefile。project 文件由开发人员创建，通常很简单，当然我们也可以为复杂的项目创建复杂的 project 文件。

qmake 包含了支持 Qt 开发的附加功能，自动包含了 moc 和 uic 的构建规则。

qmake 还可以为开发人员不需要任何修改地生成 Microsoft Visual studio 项目文件。

# 概述

qmake 工具为我们提供了一个面向项目的构建系统，用于管理应用程序、库和其他组件的构建过程。利用 qmake 使我们能够控制所使用的源文件，并可以在单个文件中简要地描述流程中的每个步骤。qmake 将每个 project 文件中的信息解析并生成对应的 Makefile 文件，之后就可以使用这个 Makefile 文件执行编译和链接。

## 描述一个项目

项目由 project (以 .pro 为后缀) 文件的内容描述。qmake 使用 project 文件中的信息来生成包含构建每个项目所需的所有命令的 Makefile。项目文件通常包含源文件和头文件的列表、一般配置信息和任何特定于应用程序的详细信息，例如要链接的额外库的列表或要使用的额外路径的列表。

项目文件可以包含许多不同的元素，包括注释，变量声明，内置函数和一些简单的控件结构。在大多数简单项目中，仅需使用一些基本配置选项声明用于构建项目的源文件和头文件。有关如何创建简单项目文件的更多信息，请参见[简单入门](#简单入门)

我们可以为复杂项目创建更复杂的项目文件。有关创建项目文件的更多信息，可以查看 创建项目文件。有关可以在项目文件中使用的变量和函数的详细信息，可以查看 参考文档。

我们可以使用应用程序或库项目模板来指定专门的配置选项，以微调构建过程。相关的更多内容，可以查阅 构建常用项目类型。

我们可以使用 Qt Creator 的新建项目向导来创建项目文件。我们选择某个项目模板后 Qt Creator 将创建一个具有默认值的项目文件，使我们能够构建和运行该项目。我们可以通过修改这个默认创建的项目文件的内容来满足我们自己项目的构建需求。

我们还可以使用 qmake 来生成项目文件。有关 qmake 命令行选项的更多内容，可以查阅 运行 qmake

qmake 的基本配置特性可以处理大多数跨平台项目的需要。然而，有时使用一些特定于平台的变量可能是有用的，甚至是必要的。有关更多这方面的信息，可以查阅平台相关事项

## 构建一个项目

对于简单的项目，我们只需要在项目的顶层（包含project文件的）目录中运行 qmake 来生成 Makefile。然后根据 Makefile 运行平台的 make 工具来构建项目即可。

有关 qmake 在配置构建流程时使用的环境变量的更多信息，可以查阅配置 qmake

## 使用三方库

三方库指南向大家展示了如何在 Qt 项目中使用简单的三方库。

## 预编译头文件

在大型项目中，可以利用预编译的头文件来加快构建过程。有关更多信息，可以查阅 使用预编译头文件。

# 简单入门

本节内容将向大家介绍 qmake 的一些基础知识。本手册中的其他主题将包含有关使用 qmake 的更详细内容。

## 简单起始

假设我们已经完成了应用程序的基本实现，并创建了下列文件：

```bash
hello.cpp
hello.h
main.cpp
```

我们可以在 Qt 发行版的 `examples/qmake/tutorial` 目录中找到这些文件。 关于应用程序的设置功能，我们所需知道的另一件事是它本身是用 Qt 编写的。首先，使用我们喜欢的纯文本编辑器，在 `examples/qmake/tutorial` 中创建一个名为 hello.pro 的文件。 我们需要做的第一件事是添加一些脚本，这些脚本将告诉 qmake 有关开发项目的源文件和头文件信息。

我们首先将源文件添加到项目文件中。 为此，我们需要使用 SOURCES变量。我们可以新起一行，并使用 SOURCES += hello.cpp 的方式即可添加 hello.cpp 作为源文件。

```pro
SOURCES +=  hello.cpp
```

我们对项目中的每个源文件重复这个步骤：

```pro
SOURCES += hello.cpp
SOURCES += main.cpp
```

如果我们更喜欢使用 Make-like 的语法：把所有的文件都在一行中列出，我们可以像下面这样使用换行转义的方式：

```pro
SOURCES = hello.cpp \
          main.cpp
```

现在源文件已列在项目文件中，接下来我们添加头文件。头文件的添加方式与源文件完全相同，只是我们使用的变量名是 HEADERS。完成此操作后，我们的项目文件类似下面所示：

```pro
HEADERS = hello.h

SOURCES = hello.cpp \
          main.cpp
```

目标文件名称是自动设置的。它一般与项目文件名相同，并带有适合平台的后缀。例如，如果项目文件被称为 hello.pro，目标文件的名称在 Windows 平台上将是 hello.exe，而在 Unix 平台上将是 hello。如果我们想使用一个自定义的名字，我们可以在项目文件中，使用 TARGET变量，并添加类似下面的脚本进行设置（建议将这个设置放在项目文件的起始处）：

```pro
TARGET = helloworld
```

之后完成的项目文件类似下面这样：

```pro
TARGET = helloworld

HEADERS = hello.h

SOURCES = hello.cpp \
          main.cpp
```

现在可以使用 qmake 为这个应用程序生成 Makefile 文件了。启动命令行，并进入项目文件所在的目录中，然后键入以下内容:

```bash
qmake -o Makefile hello.pro
```

然后我们可以键入 make（或 nmake）来完成编译。

对于 Visual Studio 用户，qmake 也可以用来生成 Visual Studio 项目文件。 其命令如下：

```bash
qmake -tp vc hello.pro
```

## 使程序可调式

应用程序的发行版一般不需要包含任何调试符号或其他调试信息。但是在开发过程中，使用应用程序带有相关信息的调试版本非常有用。 通过将 debug 添加给项目文件中的 CONFIG 变量，可以轻松实现此目的。例如：

```pro
TARGET = helloworld

HEADERS = hello.h

SOURCES = hello.cpp \
          main.cpp

CONFIG += debug
```

我们还是像之前一样使用 qmake 来生成 Makefile。现在，在调试环境中运行应用程序时，我们将获得有关应用程序有用的调试信息。

## 添加平台相关的源文件

现在假设我们有两个新文件要包含到项目文件中：hellowin.cpp 和 hellounix.cpp。我们不能仅仅将它们添加到 SOURCES 变量中就完事了，因为这会将两个文件都放在 Makefile 中。 因此，我们在这里需要做的是使用一个作用域，它将根据我们为哪个平台构建而被进入并处理内部的脚本。

为 Windows 添加平台相关文件的简单作用域示例如下：

```pro
win32 {
    SOURCES += hellowin.cpp
}
```

在为 Windows 平台构建时，qmake 将 hellowin.cpp 添加到源文件列表中。在为任何其他平台构建时，qmake 都会忽略它。现在剩下的工作就是为特定于 unix 平台的文件创建一个作用域。

完成之后，我们的项目文件将类似下面这样：

```pro
TARGET = helloworld

HEADERS = hello.h

SOURCES = hello.cpp \
          main.cpp

CONFIG += debug

win32 {
    SOURCES += hellowin.cpp
}
unix {
    SOURCES += hellounix.cpp
}
```

接下来就是像以前一样使用 qmake 来生成 Makefile。

## 如果某文件不存在则停止 qmake

如果某个文件不存在，我们可能就不想创建 Makefile 了。我们可以使用 exists() 函数检查文件是否存在。我们可以使用 error() 函数停止 qmake 的处理。这与作用域的工作方式相同。只需用函数替换作用域条件。对 main.cpp 文件的检查的示例如下:

```pro
!exists( main.cpp ) {
    error( "No main.cpp file found" )
}
```

! 符号用于对检测结果取反。也就是说，如果文件存在，exists( main.cpp ) 为真；如果文件不存在，!exists( main.cpp ) 为真。

```pro
TARGET = helloworld

HEADERS = hello.h

SOURCES = hello.cpp \
          main.cpp

CONFIG += debug

win32 {
    SOURCES += hellowin.cpp
}
unix {
    SOURCES += hellounix.cpp
}

!exists( main.cpp ) {
    error( "No main.cpp file found" )
}
```

像以前一样使用 qmake 来生成 makefile。如果我们临时重命名 main.cpp 文件，我们将看到错误提示消息，qmake 将停止继续处理。

## 检测多个条件

假设我们正在使用 Windows 平台，并且希望在命令行上运行应用程序时能够使用 qDebug() 查看语句输出。要查看输出，必须使用适当的控制台设置来构建应用程序。我们可以通过将 console 添加到 CONFIG 变量中很容易地实现这个设置，这样在 Windows 的 Makefile 中就会包含此设置。但是，我们假设只在运行 Windows 和已经在 CONFIG 变量上添加了 debug 时才添加 console 到 CONFIG 变量 。这时就需要使用两个相互嵌套的作用域。我们把要处理的设置放在第二个左右域内，像下面这样:

```pro
win32 {
    debug {
        CONFIG += console
    }
}
```

嵌套的作用域可以用冒号连接在一起，所以最终的项目文件会类似下面这样:

```pro
TARGET = helloworld

HEADERS = hello.h

SOURCES = hello.cpp \
          main.cpp

CONFIG += debug

win32 {
    SOURCES += hellowin.cpp
}
unix {
    SOURCES += hellounix.cpp
}

!exists( main.cpp ) {
    error( "No main.cpp file found" )
}

win32:debug {
    CONFIG += console
}

显示详细信息
```

如大家所见，这还是挺容易的！现在，我们已经完成了qmake 的入门教程，并可以开始为开发项目编写项目文件了。

# 创建项目文件

项目文件包含 qmake 构建应用程序、库或插件所需的所有信息。通常，我们使用一系列声明来指定项目中的资源，但是，对简单编程构造的支持使我们能够描述不同平台和环境的不同构建过程。

## 1. 项目文件元素

qmake 使用的项目文件格式既可以用于支持简单的构建系统，也可以用于支持相当复杂的构建系统。简单的项目文件使用简单的声明式风格，定义标准变量来指示项目中使用的源文件和头文件。复杂的项目可以使用控制流结构来微调构建过程。

以下各节介绍项目文件中使用的不同类型的元素。

### 1.1 变量

在项目文件中，变量用于保存字符串列表。在最简单的项目中，这些变量通知 qmake 有关要使用的配置选项，或者提供要在构建过程中使用的文件名和路径。

qmake 查找每个项目文件中的某些变量，并且它使用这些变量的内容来确定它应该写入 Makefile 的内容。例如，HEADERS 和 SOURCES 变量中的值列表用于告诉 qmake 关于与项目文件位于同一目录中的头和源文件。

#### 1.1.1 扩展变量值

变量还可用于内部存储临时值列表，并且可以用新值覆盖或扩展现有的值列表。

以下代码段说明了如何将值列表分配给变量：

```pro
HEADERS = mainwindow.h paintwidget.h
```

变量中的值列表以下列方式扩展：

```pro
SOURCES = main.cpp mainwindow.cpp \
          paintwidget.cpp
CONFIG += console
```

**注意**：第一个赋值仅包括与 HEADERS 变量在同一行上指定的值。第二个赋值使用反斜杠 （\） 跨行拆分 SOURCES 变量中的值。

CONFIG 变量是 qmake 生成 Makefile 时使用的另一个特殊变量。我们将在常规配置中讨论它。在上面的代码段中，console 将添加到 CONFIG 中包含的现有值列表中。

#### 1.1.2 常用变量

下表列出了一些常用的变量并简单描述了它们的内容。有关变量及其描述的完整列表，请参见 Variables 。

| 变量      | 内容                                                         |
| --------- | ------------------------------------------------------------ |
| CONFIG    | 常规项目配置选项                                             |
| DESTDIR   | 可执行文件或二进制文件将存放的目录                           |
| FORMS     | 用户界面编译器 (uic)要处理的 UI 文件的列表                   |
| HEADERS   | 构建项目时使用的头文件(.h)文件名列表                         |
| QT        | 项目中使用的 Qt 模块的列表                                   |
| RESOURCES | 要包含在最终项目中的资源 （.qrc） 文件的列表。有关这些文件的详细信息，请参阅 Qt 资源系统 |
| SOURCES   | 生成项目时要使用的源代码文件的列表                           |
| TEMPLATE  | 用于项目的模板。这决定了构建过程的输出是应用程序、库还是插件 |

#### 1.1.3 读取变量值

变量的内容可以通过在变量名前面加上 \$\$ 来读取。这可以用来将一个变量的内容分配给另一个变量:

```pro
TEMP_SOURCES = $$SOURCES
```

\$\$ 操作符与对字符串和值列表进行操作的内置函数一起被广泛使用。有关更多信息，请参见 qmake 语言

### 1.2 空格

通常，空格用于分隔为变量分配的值。要指定包含空格的值，必须用双引号将包含空格的值括起来：

```pro
DEST = "Program Files"
```

引号引用的文本在变量持有的值列表中被视为单个项。类似的方法也可用于处理包含空格的路径，尤其是在为 Windows 平台定义 INCLUDEPATH 变量时：

```pro
win32:INCLUDEPATH += "C:/mylibs/extra headers"
unix:INCLUDEPATH += "/home/user/extra headers"
```

### 1.3 注释

可以在项目文件添加注释。注释以 # 字符开始，并一直延续到其所在行的末尾。例如:

```pro
# Comments usually start at the beginning of a line, but they
# can also follow other content on the same line.
```

要在变量赋值中包含 # 字符，需要使用内置的 LITERAL_HASH 变量的内容。

### 1.4 内置功能和控制流程

qmake 提供了许多内置函数，以便处理变量的内容。在简单的项目文件中最常用的函数是 include() 函数，该函数将文件名作为参数。给定文件的内容包含在项目文件中使用include函数的位置。include 函数最常用来包含其他项目文件:

```pro
include(other.pro)
```

对条件结构的支持是通过作用域提供的，作用域的行为类似于编程语言中的 if 语句:

```pro
win32 {
    SOURCES += paintwidget_win.cpp
}
```

仅当条件为 true 时，才在大括号内分配。在这种情况下，必须设置 win32 CONFIG 选项。这在 Windows 上会自动发生。开口支架必须与条件位于同一行上。

只有在条件为真时，大括号内的赋值才会执行。在这种情况下，必须添加 win32 的 CONFIG 设置选项。这在 Windows 上是自动发生的，无需主动添加。**注意左大括号必须与条件位于同一行上**。

对通常需要循环的变量或更复杂的操作由内置函数，如 find()、unique() 和 count() 提供。这些函数以及其他许多函数用于操作字符串和路径、支持用户输入和调用外部工具。有关使用函数的更多信息，请参见 qmake 语言。有关所有函数及其描述的列表，请参见 替换功能 和 测试功能。

## 2. 项目模板

TEMPLATE 变量用于定义将要构建的项目的类型。如果在项目文件中没有声明它，qmake 假设应该构建一个应用程序，并将生成一个适当的 Makefile（或等效文件）。

下表总结了可用的项目类型，并描述了 qmake 将为每个项目生成的文件：

| Template      | qmake 输出                                                   |
| ------------- | ------------------------------------------------------------ |
| app (default) | 构建一个应用程序的 Makefile                                  |
| lib           | 构建一个库的 Makefile                                        |
| aux           | 什么都不做。如果不需要调用编译器来创建目标，则使用此方法，例如，我们的项目可能是用解释语言编写的。 **注意**：此模板类型仅适用于基于 Makefile 的生成器。特别是，它不适用于 vcxproj 和 Xcode 生成器。 |
| subdirs       | 包含使用 SUBDIRS 变量指定的子目录的规则的 Makefile。每个子目录必须包含自己的项目文件 |
| vcapp         | 用于构建一个应用程序的 Visual Studio 项目文件                |
| vclib         | 用于构建一个库的 Visual Studio 项目文件                      |
| vcsubdirs     | 在子目录中构建项目的 Visual Studio 解决方案文件              |


有关为使用 app 和 lib 模板的项目编写项目文件的建议，请参阅构建常见项目类型。

当使用 subdirs 模板时，qmake 生成一个 Makefile 来检查每个指定的子目录，处理它在其中找到的任何项目文件，并在新创建的 Makefile 上运行平台的 make 工具。SUBDIRS 变量可用于包含要处理的所有子目录的列表。

## 3. 常规配置

CONFIG 变量指定项目应该配置的选项和特性。

项目可以在 debug 或 release 模式下构建，也可以同时在 debug 和 release 模式下构建。如果同时指定了 debug 和 release，则最后一个将生效。如果您指定 debug_and_release 选项来构建项目的 debug 和 release 版本，那么 qmake 生成的 Makefile 包含生成两个版本的规则。这可以通过以下方式调用:

```bash
make all
```

将 build_all 选项添加到 CONFIG 变量会使该规则成为构建项目时的默认规则。

**注意**: CONFIG 变量中指定的每个选项也可以用作范围条件。我们可以使用内置的 CONFIG() 函数来测试某些配置选项是否存在。例如，下面几行代码将该函数显示为作用域中的条件，以测试是否使用了 opengl 这个选项:

```
CONFIG(opengl) {
    message(Building with OpenGL support.)
} else {
    message(OpenGL support is not available.)
}
```

这允许为 debug 和 release 构建定义不同的配置。有关更多信息，请参见使用条件域。

以下选项定义要构建的项目类型。

**注意**:其中一些选项只有在相关平台上使用时才生效。

| 选项 | 描述                                                         |
| ---- | ------------------------------------------------------------ |
| qt   | 该项目是一个 Qt 应用程序，应针对 Qt 库进行链接。我们可以使用 QT 变量来控制应用程序所需的任何其他 Qt 模块。默认情况下，此值将添加，但我们可以将其删除以用于非 Qt 项目 |
| x11  | 该项目是一个 X11 应用程序或库。如果目标使用 Qt，则不需要此值 |

应用程序和库项目模板 为我们提供了更专门的配置选项来优化构建过程。选项将在构建常见项目类型中详细说明。

例如，如果我们的应用程序使用 Qt 库，并且我们希望在调试模式下构建它，那么我们的项目文件可以包含以下行:

```pro
CONFIG += qt debug
```

**注意**: 我们必须使用“+=”，而不是“=”，否则 qmake 将无法使用 Qt 的配置来确定我们的项目所需的设置。

## 4. 声明 Qt 库

如果 CONFIG 变量包含 qt 值，则启用 qmake 对 qt 应用程序的支持。这样，这样就可以对应用程序使用的 Qt 模块进行微调。这是使用 QT 变量实现的，该变量可用于声明所需的扩展模块。例如，我们可以以下列方式启用 XML 和网络模块：

```pro
QT += network xml
```

**注意**：QT 默认包含 core 模块和 gui 模块，因此上面的声明将网络和 XML 模块添加到这个默认列表中。以下分配省略了默认模块，并且在编译应用程序的源代码时将导致错误：

```pro
QT = network xml # 省略了core 模块和 gui 模块
```

如果我们想要构建一个没有 gui 模块的项目，我们需要使用“-=”操作符来排除它。默认情况下，QT同时包含 core 模块和 gui 模块，所以下面这行代码将生成一个小一些的 QT 项目:

```pro
QT -= gui # 仅使用 core 模块
```

有关可以添加到 QT 变量的 Qt 模块列表，请参阅 QT。

## 5. 配置功能

qmake 可以使用 feature (.prf) 文件中指定的额外配置功能来设置。这些额外的特性通常为构建过程中使用的自定义工具提供支持。要在构建过程中添加要素，将要素名称（要素文件名的词干）追加到 CONFIG 变量中。

例如，qmake 可以配置生成过程，以利用 pkg-config 支持的外部库,如 D-Bus 和 ogg ，使用以下代码行:

```
CONFIG += link_pkgconfig
PKGCONFIG += ogg dbus-1
```

有关添加功能部件的更多信息，请参见添加新配置功能部件。

## 6. 声明其他库

如果除了使用 Qt 提供的库外，我们还使用了其他库，则需要在项目文件中指定它们。

qmake 搜索库的路径和要链接到的特定库可以添加到 LIBS 变量中的值列表中。我们可以指定库的路径，或使用 Unix 样式表示法来指定库和路径。

例如，下面几行显示了如何指定库：

```
LIBS += -L/usr/local/lib -lmath
```

可以使用 INCLUDEPATH 变量以类似方式指定包含头文件的路径。

例如，添加几个路径来搜索头文件：

```
INCLUDEPATH = c:/msdev/include d:/stl/include
```

# 构建常用项目类型

本章介绍如何为基于 Qt 的三种常见项目类型设置 qmake 项目文件：应用程序、库和插件。尽管所有项目类型都使用许多相同的变量，但每个变量都使用特定于项目的变量来自定义输出文件。

此处不介绍特定于平台的变量。有关详细信息，可以参阅 Qt for Windows - Deployment 和 Qt for macOS

## 1. 构建应用程序

app 模板告诉 qmake 生成一个 Makefile 来构建应用程序。使用此模板，可以通过向 CONFIG 变量定义添加以下选项之一来指定应用程序的类型：

| 选项     | 描述                                                |
| -------- | --------------------------------------------------- |
| windows  | 该应用程序是一个 GUI 应用程序。                     |
| console  | 仅限 app 模板使用：该应用程序是一个控制台应用程序。 |
| testcase | 该应用是一个自动化测试                              |

使用此模板时，将识别下列 qmake 系统变量。我们应该在 .pro 文件中使用它们来指定有关应用程序的信息。对于其他附加的平台相关的系统变量，我们可以查看平台相关事项来进一步了解。

- `HEADERS` - 应用程序使用的头文件列表
- `SOURCES`- 应用程序使用的C++源文件列表
- `FORMS` - 应用程序使用的 UI 文件列表 (由 Qt Designer) 创建
- `LEXSOURCES`- 应用使用的 Lex 资源文件列表
- `YACCSOURCES` - 应用使用的 Yacc 资源文件列表
- `TARGET` - 应用程序的可执行文件的名称。默认情况下为项目文件的名称。(如果有后缀名，例如 Windows 平台上的 .exe，会自动添加)
- `DESTDIR` - 目标可执行文件所在的目录
- `DEFINES` - 应用程序所需的任何其他预处理器定义的列表
- `INCLUDEPATH` - 应用所需的任何附加 include 路径列表
- `DEPENDPATH` - 应用程序的依赖项搜索路径
- `VPATH` - 告诉 qmake 在何处搜索无法打开的文件。例如，qmake 查找 SOURCES 并找到一个它无法打开的条目，它将遍历整个 VPATH 列表，以查看是否可以自行找到该文件
- `DEF_FILE` - 仅适用于 Windows 平台: 应用程序要链接的 .def 文件

对于上面列出的这些变量，我们只需要使用具有值的系统变量即可。例如，如果我们没有任何额外的包含路径，那么我们就不需要使用 INCLUDEPATH。qmake 会自动添加必要的默认值。一个项目文件的例子像下面这样：

```pro
TEMPLATE = app
DESTDIR  = c:/helloapp
HEADERS += hello.h
SOURCES += hello.cpp
SOURCES += main.cpp
DEFINES += USE_MY_STUFF
CONFIG  += release
```

对于只需要单一值的属性项，例如 TEMPLATE 或 DESTDIR，我们使用“=”；但是对于多值项，我们使用“+=”来将现有项添加到该类型。使用“=”将使值替换为新值。例如，我们写成 DEFINES = USE_MY_STUFF，则删除列表中所有其他的预处理器定义。

## 2. 构建一个测试用例

testcase 项目是一个 app 项目，它将作为一个自动测试来运行。任何应用程序都可以通过向 CONFIG 变量添加值 testcase 来标记为 testcase

对于 testcase 项目，qmake 将在生成的 Makefile 中插入一个 check target。这个 target 将运行应用程序。如果测试终止时退出码为零，则认为测试通过。

check target 可以通过 SUBDIRS 自动递归。这意味着可以在 SUBDIRS 项目中运行 make check 命令来运行整个测试套件。

check target 的执行行为可以由某些 Makefile 变量定制。这些变量是：

| 变量       | 描述                                                         |
| ---------- | ------------------------------------------------------------ |
| TESTRUNNER | 每个测试命令的前置命令或 shell 片段。一个示例的用例是“timeout”脚本，如果测试没有在指定的时间内完成，它将终止测试。 |
| TESTARGS   | 附加到每个测试命令的其他参数。例如，传递额外的参数来设置测试中输出的文件名和格式可能很有用(例如 QTestLib 支持的 -o filename,format 选项)。 |

**注意**：上面的变量是在调用 make 工具时设置，而不是在 .pro 文件中设置的。大多数 make 工具支持直接在命令行上设置 Makefile 变量：

```pro
# Run tests through test-wrapper and use xunitxml output format.
# In this example, test-wrapper is a fictional wrapper script which terminates
# a test if it does not complete within the amount of seconds set by "--timeout".
# The "-o result.xml,xunitxml" options are interpreted by QTestLib.
make check TESTRUNNER="test-wrapper --timeout 120" TESTARGS="-o result.xml,xunitxml"
```

Testcase 项目可以通过以下 CONFIG 选项进一步定制：

| 选项               | 描述                                         |
| ------------------ | -------------------------------------------- |
| insignificant_test | 在进行 make check 时，将忽略测试的退出代码。 |

测试用例通常是用 QTest 或 TestCase 编写的，但是使用 CONFIG += testcase 和 make check 并不是必需的。唯一必须的要求是测试程序在成功时退出码为零，失败时退出码为非零。

## 3. 构建库

lib 模板告诉 qmake 生成一个 Makefile 来构建一个库。在使用 lib 模板时，除了具有 app 模板支持的系统变量之外，它还支持 VERSION 变量。使用它可以在 .pro 文件中指定有关库的信息。

在使用 lib 模板时，可以在 CONFIG 变量中添加以下选项来确定要构建的库的类型：

| 选项      | 描述                  |
| --------- | ------------------- |
| dll       | 该库是一个共享库(dll) |
| staticlib | 该库是一个静态库      |
| plugin    | 该库是一个插件        |

还可以定义以下选项来提供关于库的其他信息。

```
win32:VERSION = 1.2.3.4 # major.minor.patch.build
else:VERSION = 1.2.3    # major.minor.patch
```

库的目标文件名是与平台相关的。例如，在 X11、macOS 和 iOS 上，库名将由 lib 作为前缀。在 Windows 上，文件名不添加前缀。

## 4. 构建插件

插件是使用 lib 模板构建的，如前一节所述。这告诉 qmake 为项目生成 Makefile，该 Makefile 将以适合每个平台的形式(通常是库的形式)构建插件。与普通库一样，VERSION 变量也可用于指定关于插件的信息。

```pro
win32:VERSION = 1.2.3.4 # major.minor.patch.build
else:VERSION = 1.2.3    # major.minor.patch
```

## 4.1 构建 Qt Designer 插件

Qt Designer 插件是使用一组特定的配置设置来构建的，这些设置依赖于 Qt 为我们的系统配置的方式。为了方便起见，可以通过将 designer添加到 QT 变量来启用这些设置。例如:

```pro
QT += widgets designer
```

有关基于插件的项目的更多示例，请参见 Qt Designer示例。

## 5. 在调试和发布模式下创建和安装

有时，以调试和发布模式构建项目是必要的。尽管 CONFIG 变量可以同时包含 debug 和 release 选项，但是只应用最后指定的选项。

## 5.1 在调试和发布模式下构建

要使项目能够在两种模式下构建，我们必须将 debug_and_release 选项添加到 CONFIG 变量：

```pro
CONFIG += debug_and_release

CONFIG(debug, debug|release) {
    TARGET = debug_binary
} else {
    TARGET = release_binary
}
```

上面代码段中的作用域在每种模式下修改构建目标，以确保生成的目标具有不同的名称。为目标提供不同的名称可以确保其中一个不会覆盖另一个。

当 qmake 处理项目文件时，它将生成一个 Makefile 规则，以允许在两种模式下构建项目。这可以通过以下方式调用：

```bash
make all
```

build_all 选项可以添加到项目文件中的 CONFIG 变量中，以确保项目在默认情况下以两种模式构建：

```pro
CONFIG += build_all
```

这允许使用默认规则处理 Makefile:

```bash
make
```

## 5.2 在调试和发布模式下安装

build_all 选项还确保在调用安装规则时将安装目标的两个版本：

```pro
make install
```

可以根据目标平台定制构建目标的名称。例如，一个库或插件可以在 Windows 上使用与 Unix 平台不同的约定来命名:

```pro
CONFIG(debug, debug|release) {
    mac: TARGET = $$join(TARGET,,,_debug)
    win32: TARGET = $$join(TARGET,,d)
}
```

上面代码段中的行为是在以调试模式构建时修改用于构建目标的名称。可以将 else 子句添加到其后，以对发布模式执行类似的操作。如果保持原样，发布模式下目标名称将保持不变。

# 运行 qmake

qmake 的行为可以通过在命令行上指定各种选项来定制。它们允许对构建过程进行微调，提供有用的诊断信息，并可用于指定项目的目标平台。

## 命令语法

用于运行 qmake 的语法采用以下简单形式：

```bash
qmake [mode] [options] files
```

## 操作模式

qmake 支持两种不同的操作模式。在默认模式下，qmake 使用项目文件中的信息来生成 Makefile，但是也可以使用 qmake 来生成项目文件。如果您想显式地设置模式，则必须在所有其他选项之前指定它。模式可以是以下两个值之一:

| 模式值    | 说明                             |
| --------- | -------------------------------- |
| -makefile | 此时 qmake 输出将是一个 Makefile |
| -project  | 此时 qmake 输出将是一个项目文件  |

**注意**: 创建的文件可能需要额外的编辑。例如，添加 QT 变量以适应项目所需的模块需求。

可以使用这些 options 指定常规设置和特定于模式的设置。只适用于 Makefile 模式的选项在 **Makefile 模式选项** 部分进行了描述，而影响项目文件创建的选项则在 **项目模式选项** 部分进行了描述。

## 文件

files 参数表示一个或多个项目文件（以空格分隔）的列表。

## 通用选项

可以在 qmake 的命令行上指定各种选项，以便定制构建过程，并覆盖平台的默认设置。下面的基本选项提供了使用 qmake 的帮助，指定 qmake 在哪里写入输出文件，并控制将写入控制台的调试信息的级别:

| 选项    | 说明                                                         |
| ------- | ------------------------------------------------------------ |
| -help   | 显示帮助                                                     |
| -o file | qmake 输出将定向到 file。 如果未指定此选项，qmake 将尝试 为其输出使用合适的文件名，具体取决于它所运行的模式。 如果指定了’-‘，输出将定向到 stdout。 |
| -d      | qmake 将输出调试信息。多次添加-d会增加冗余。                 |

用于项目的模板通常由项目文件中的 TEMPLATE 变量指定。我们可以使用以下选项覆盖或修改此设置：

| 选项       | 说明                                                         |
| ---------- | ------------------------------------------------------------ |
| -t tmpl    | qmake 将使用 tmpl 覆盖任何设置的 TEMPLATE 变量， 但仅在 .pro 文件被处理之后 |
| -tp prefix | qmake 将为 TEMPLATE 变量添加 prefix 前缀                     |

警告信息的级别可以进行微调，以帮助您发现项目文件中的问题:

| 选项     | 说明                                                         |
| -------- | ------------------------------------------------------------ |
| -Wall    | qmake 将报告所有已知的警告                                   |
| -Wnone   | qmake 不会生成任何警告信息                                   |
| -Wparser | qmake 只会生成解析器警告。这将提醒我们在解析项目文件时注意常见的陷阱和潜在的问题 |
| -Wlogic  | qmake 会对项目文件中的常见缺陷和潜在问题发出警告。例如，qmake 将报告列表中出现的多个文件和丢失的文件 |

## Makefile 模式选项

```
qmake -makefile [options] files
```

在 Makefile 模式下，qmake 将生成一个用于构建项目的 Makefile。此外，在这种模式下，可以使用以下选项来影响生成项目文件的方式:

| 选项        | 说明                                                         |
| ----------- | ------------------------------------------------------------ |
| -after      | qmake 将处理指定文件之后在命令行上给出的赋值                 |
| -nocache    | qmake 将忽略 .qmake.cache 缓存文件                           |
| -nodepend   | qmake 不会生成任何依赖信息                                   |
| -cache file | qmake 将使用指定的 file 作为缓存文件， 忽略任何其他 .qmake.cache 缓存文件 |
| -spec spec  | qmake 将使用 spec 作为平台和编译器信息的路径， 并忽略 QMAKESPEC 的值 |

我们还可以在命令行上传递 qmake 赋值。它们在所有指定的文件之前被处理。例如，下面的命令从 test.pro 生成 Makefile:

```
qmake -makefile -o Makefile "CONFIG+=test" test.pro
```

但是，某些指定的选项可以省略，因为它们是默认的 qmake 行为:

```
qmake "CONFIG+=test" test.pro
```

如果我们确定希望在指定文件之后处理变量，那么可以传递 -after 选项。当指定此值时，命令行中 -after 选项之后的所有赋值都将延迟到解析指定的文件之后。

## 项目模式选项

```
qmake -project [options] files
```

在项目模式下，qmake 将生成一个项目文件。此外，我们可以在此模式下提供以下选项:

| 选项   | 说明                                                         |
| ------ | ------------------------------------------------------------ |
| -r     | qmake 将递归地遍历提供的目录                                 |
| -nopwd | qmake 不会在当前工作目录中查找源代码。它将只使用指定的 files |

在此模式下，files 参数可以是文件或目录的列表。如果指定了一个目录，那么它将被包含在 DEPENDPATH 变量中，并且那里的相关代码将被包含在生成的项目文件中。如果给定一个文件，它将被追加到正确的变量，这取决于它的扩展名。例如，UI 文件被添加到 FORMS 中，c++ 文件被添加到 SOURCES 中。

还可以在此模式下在命令行上传递赋值。当这样做时，这些任务将放在生成的项目文件的最后。

# 平台相关事项

## 1. macOS、iOS、tvOS 和 watchOS 平台

这些平台特有的特性包括支持创建通用的二进制文件（binaries）、框架（frameworks）和包（bundles）。

### 1.1 源码与二进制包

源包中提供的 qmake 版本配置与二进制包中提供的 qmake 稍不同，因为它使用不同的功能规范。当源包通常使用 macx-g++ 规范时，二进制包通常配置为使用 macx-xcode 规范。

两种包的用户都可以通过使用 -spec 选项调用 qmake 来覆盖此配置(有关更多信息，请参见 运行 qmake)。例如，要使用来自二进制包中的 qmake 在项目目录中创建 Makefile，可以调用以下命令:

```
qmake -spec macx-g++
```

### 1.2 使用框架

qmake 能够自动生成生成规则，用于针对位于 /Library/Frameworks/ 的 macOS 标准框架目录中的框架进行链接

标准框架目录以外的目录需要指定到生成系统，这是通过将链接器选项追加到 LIBS 变量来实现的，如以下示例所示：

```
LIBS += -F/path/to/framework/directory/
```

框架本身是通过附加 -framework 选项和框架的名称到 LIBS 变量来链接的：

```
LIBS += -framework TheFramework
```

### 1.3 创建框架

可以对任何给定的库项目进行配置，以便将生成的库文件放置在 framework中，以便部署。为此，可以设置项目使用 `lib` template，并将 lib_bundle 选项添加到 CONFIG变量:

```
TEMPLATE = lib
CONFIG += lib_bundle
```

与库关联的数据是使用 QMAKE_BUNDLE_DATA 变量指定的。它包含将与库包一起安装的项，通常用于指定头文件集合，如下面的示例所示:

```
FRAMEWORK_HEADERS.version = Versions
FRAMEWORK_HEADERS.files = path/to/header_one.h path/to/header_two.h
FRAMEWORK_HEADERS.path = Headers
QMAKE_BUNDLE_DATA += FRAMEWORK_HEADERS
```

我们可以使用 FRAMEWORK_HEADERS 变量指定特定框架所需的头文件。将其追加到 QMAKE_BUNDLE_DATA 变量可确保将这些头文件的信息添加到将与库一起安装的资源集合中。此外，框架名称和版本由 QMAKE_FRAMEWORK_BUNDLE_NAME 和 QMAKE_FRAMEWORK_VERSION 变量指定。默认情况下，用于这些变量的值是从 TARGET 变量和 VERSION 变量获取的。

有关部署应用程序和库的详细信息，请参阅 Qt for macOS - Deployment

### 1.4 创建和移动 Xcode 项目

macOS上的开发人员可以利用 qmake 对 Xcode 项目文件的支持，正如 Qt for macOS 文档中所描述的那样。通过运行 qmake 从现有的 qmake 项目文件生成 Xcode 项目。例如:

```
qmake -spec macx-xcode project.pro
```

**注意**: 如果以后将项目移动到磁盘上，则必须再次运行 qmake 来处理项目文件并创建一个新的 Xcode 项目文件。

### 1.5 同时支持两个构建目标

实现这一点目前是不可行的，因为 Xcode 的活动构建配置概念与 qmake 的构建目标概念在概念上是不同的。

Xcode 活动构建配置设置用于修改 Xcode 配置、编译器标志和类似的构建选项。与 Visual Studio 不同，Xcode 不允许根据选择的是调试还是发布构建配置来选择特定的库文件。qmake 调试和发布设置控制哪些库文件链接到可执行文件。

目前无法从 qmake 生成的 Xcode 项目文件中设置 Xcode 配置设置中的文件。库在 Xcode 构建系统的框架和库阶段的链接方式。

此外，所选的活动构建配置存储在 .pbxuser 文件中，该文件是由 Xcode 在第一次加载时生成的，而不是由 qmake 创建的。

## 2. Windows 平台

该平台特有的特性包括支持 Windows 资源文件(提供的或自动生成的)、创建 Visual Studio 项目文件，以及在部署使用 Visual Studio 2005 或更高版本开发的 Qt 应用程序时处理清单文件。

### 2.1 添加 Windows 资源文件

本节介绍如何使用 qmake 处理 Windows 资源文件，使其链接到应用程序可执行文件(EXE)或动态链接库(DLL)。qmake 可以有选择地自动生成一个适当填充的 Windows 资源文件。

一个链接的 Windows 资源文件可以包含许多元素，这些元素可以通过它的 EXE 或 DLL 来访问。但是，Qt 资源系统 应该用于以独立于平台的方式访问链接的资源。但是链接的 Windows 资源文件的一些标准元素可以被 Windows 本身访问。例如，在 Windows 资源管理器中，文件属性的版本选项卡由资源元素填充。此外，EXE 的程序图标是从这些元素中读取的。因此，Qt 创建的 Windows EXE 或 DLL 同时使用这两种技术是一个很好的实践：通过 Qt 资源系统 链接平台无关的资源，并通过 Windows 资源文件添加 Windows 特定的资源。

通常，资源定义脚本(将 .rc 文件编译为 Windows 资源文件。在 Microsoft 工具链中，RC 工具生成一个 .res 文件，该文件可以通过 Microsoft 链接器链接到 EXE 或 MinGW工具链使用 windres 工具生成一个 .o 文件，该文件可以通过 MinGW 链接器链接到 EXE 或 DLL

通过设置至少一个系统变量 VERSION, qmake 可选地自动生成适当填充的 .rc 文件。生成的 .rc 文件被自动编译并链接。添加到 .rc 文件的元素由系统变量 QMAKE_TARGET_COMPANY , QMAKE_TARGET_DESCRIPTION, QMAKE_TARGET_COPYRIGHT, QMAKE_TARGET_PRODUCT, RC_CODEPAGE, RC_ICONS, RC_LANG 和 VERSION
定义

如果这些元素还不够，qmake 有两个系统变量 RC_FILE 和 RES_FILE，它们直接指向外部创建的 .rc 或 .res 文件。通过设置这些变量之一，指定的文件被链接到 EXE 或 DLL

**注意**：如果设置 RC_FILE 或 RES_FILE , qmake 生成 .rc 文件的过程将被阻塞。在这种情况下，qmake 不会对给定的 .rc 文件或 .res 或 .o 文件进行进一步的修改；与 .rc 文件生成相关的变量将不起作用

### 2.2 创建 Visual Studio 项目文件

本节介绍如何将现有的 qmake 项目导入 Visual Studio。qmake 能够获取一个项目文件并创建一个包含开发环境所需的所有必要信息的 Visual Studio 项目。这是通过将 qmake 项目模板 template设置为 `vcapp` (用于应用程序项目)或 `vclib` (用于库项目)来实现的。

这也可以通过命令行选项来设置，例如:

```
qmake -tp vc
```

可以递归地生成子目录中的 `.vcproj` 文件和主目录中的 `.sln` 文件，只需输入:

```
qmake -tp vc -r
```

每次更新项目文件时，我们都需要运行 qmake 来生成更新的 Visual Studio 项目

**注意**：如果我们正在使用 Visual Studio 附加程序，请选择 **Qt > Import from .pro file** 导入 `.pro` 文件

### 2.3 Visual Studio Manifest 文件

在部署使用 Visual Studio 2005 或其更高版本构建的 Qt 应用程序时，请确保正确处理了在链接应用程序时创建的清单文件。对于生成 dll 的项目，这是自动处理的。

删除嵌入到应用程序可执行文件的清单可以通过以下 CONFIG 变量的赋值来完成：

```
CONFIG -= embed_manifest_exe
```

此外，可以通过以下 CONFIG 变量的赋值来删除嵌入 dll 的清单:

```
CONFIG -= embed_manifest_dll
```

更详细的内容请查阅 Windows部署指南。

# qmake 语言

许多 qmake 项目文件使用 name = value 和 name += value 定义列表简单地描述了项目使用的源文件和头文件。qmake 还提供了其他操作符、函数和作用域，可用于处理变量声明中提供的信息。这些高级特性使得从单个项目文件为多个平台生成 makefile 变得简单高效。

## 1. 运算符

在大多数工程文件中，分配操作符（`=`）和添加操作符（`+=`）可被用来引入（include）有关于项目的几乎全部信息。典型的使用方式是分配给一个变量的值列表，并且我们可以依据各种测试的结果来添加更多的值。由于 qmake 有时候会使用默认值来初始化某些变量，因此此时使用删除（`-=`）操作符来过滤掉不需要的值就是相当必要的了。以下内容将会讲解用操作符来修改变量的内容的方法。

### 1.1 赋值

我们使用 `=` 操作符将值指定给一个变量：

```
TARGET = myapp
```

在上一行中，设定 TARGET 变量的值为 `myapp`，这样我们就可以使用一个 “myapp” 值来覆盖任何以前设置给 TARGET 的值了

### 1.2 附加

`+=` 操作符将在一个变量的值列表添加一个新值：

```
DEFINES += USE_MY_STUFF
```

在上面一行语句中我们附加 USE_MY_STUFF 到预定义列表，这样我们就可以在 Makefile 中使用 USE_MY_STUFF 这个预定义了

### 1.3 移除

`-=` 操作符用于在一个变量的值列表中删除一个值：

```
DEFINES -= USE_MY_STUFF
```

在上面一行语句中我们从预定义列表中移除 USE_MY_STUFF 的预定义，这样在 Makefile 中的有关 USE_MY_STUFF 的预定义将会失效

### 1.4 避免重复的附加

`*=` 操作符也被用于在一个变量的值列表中添加一个值，但只有当它不是已存在变量的时候才有效。这可以防止变量值被多次的包含在一个变量中列表。例如：

```
DEFINES *= USE_MY_STUFF
```

上面的语句中，USE_MY_STUFF 将只有在预定义列表中不存在该定义时才会被添加，友情提示，unique() 函数也可以用来确保一个变量的每个值只包含一个实例

### 1.5 替换

`~=` 操作符用于用指定的值替换任何一个相匹配的正则表达式的值:

```
DEFINES ~= s/QT_[DT].+/QT
```

上面一行语句中，在预定义列表中的任何以 QT_D 或者 QT_T 开头的预定义都将被替换为 QT

### 1.6 变量扩展

`$$` 操作符被用于提取变量的内容，并且也能被用作在变量之间传值，或者传递这些值给函数

```
EVERYTHING = $$SOURCES $$HEADERS
message("The project contains the following files:")
message($$EVERYTHING)
```

变量可以用来存储环境变量的内容。这些可以在运行 qmake 时使用，或者在生成项目时生成的 Makefile 中使用。

要在运行 qmake 时获取环境值的内容，请使用 `$$(...)` 运算符：

```
DESTDIR = $$(PWD)
message(The project will be installed in $$DESTDIR)
```

在上面的分配中，当处理项目文件时读取 `PWD` 环境变量的值。

要在生成的 Makefile 文件被处理时获取环境值的内容，请使用 `$(...)` 运算符：

```
DESTDIR = $$(PWD)
message(The project will be installed in $$DESTDIR)

DESTDIR = $(PWD)
message(The project will be installed in the value of PWD)
message(when the Makefile is processed.)
```

在上面的代码中，处理项目文件时会立即读取 PWD 的值，但在生成的 Makefile 文件中将 `$(PWD)` 的值分配给 DESTDIR 变量发生在 Makefile 文件被处理时。这使得构建过程更加灵活，只要在处理 Makefile 时正确设置环境变量即可

### 1.7 访问 qmake 属性

特殊的 `$$[...]` 操作符可用于访问 qmake 属性：

```
message(Qt version: $$[QT_VERSION])
message(Qt is installed in $$[QT_INSTALL_PREFIX])
message(Qt resources can be found in the following locations:)
message(Documentation: $$[QT_INSTALL_DOCS])
message(Header files: $$[QT_INSTALL_HEADERS])
message(Libraries: $$[QT_INSTALL_LIBS])
message(Binary files (executables): $$[QT_INSTALL_BINS])
message(Plugins: $$[QT_INSTALL_PLUGINS])
message(Data files: $$[QT_INSTALL_DATA])
message(Translation files: $$[QT_INSTALL_TRANSLATIONS])
message(Settings: $$[QT_INSTALL_CONFIGURATION])
message(Examples: $$[QT_INSTALL_EXAMPLES])

显示详细信息
```

更多内容，大家可以查阅 Configuring qmake 文档

该操作符可访问的属性通常用于允许第三方插件和组件集成到 Qt 中。例如，如果在 Qt Designer 的项目文件中声明如下，则可以将 Qt Designer 插件与 Qt Designer 的内置插件一起安装：

```
target.path = $$[QT_INSTALL_PLUGINS]/designer
INSTALLS += target
```

## 2. 条件域

条件域类似于编程语言中的 if 语句。如果一个特定的条件是真的,在条件域内的声明将会被处理

### 2.1 条件域的语法

条件域包含一个条件后跟一个在同一行的左花括号，然后是一系列的命令和定义，最后是在新的一行的一个右花括号。就像下面这样：

```
<condition> {
     <command or definition>
     ...
}
```

左花括号必须要和条件写在同一行。条件域可以包扩不止一个条件;这些之后就要介绍到。

### 2.2 条件域和条件

一个条件域被写成一个条件后跟一系列声明包含在一对大括号中，例如：

```
win32 {
     SOURCES += paintwidget_win.cpp
}
```

如果 qmake 用于 Windows 平台，上面的代码将添加 paintwidget_win.cpp 文件到 Makefile 的资源列表。如果 qmake 用于其他的平台，该条语句将被忽略。

当然我们也可以逆向思维，达到同样的目的，例如我们使用下面的语句：

```
!win32 {
     SOURCES -= paintwidget_win.cpp
}
```

也可以达到一样的目的。

条件域可嵌套组合多个条件。例如，如果您想要为一个特定的平台中，在满足了调试的被启用后，包含（include）一个特定的文件，然后你就可以写如下代码：

```
macx {
     debug {
         HEADERS += debugging.h
     }
}
```

来满足你的需求。

为了简化嵌套条件域，我们可以使用 `:` 操作符，对于上一个例子中的功能，我们可以用如下代码来简化它：

```
macx:debug {
     HEADERS += debugging.h
}
```

我们也可以使用 `:` 操作符来执行单一线条件的操作，例如：

```
win32:DEFINES += USE_MY_STUFF
```

上面一行的作用是，仅在 windows 平台上添加 USE_MY_STUFF 定义到 DEFINES 列表。通常，`:` 操作符很像是逻辑与（`&&`）操作符，它会拼接一些条件，并且要求它们都为真。

我们也有 `|` 操作符，用来实现像逻辑或操作符（`||`）一样的功能，它用来连接一些条件，并且仅要求其中至少一个为真。例如：

```
win32|macx {
     HEADERS += debugging.h
}
```

如果需要混合使用这两个操作符，可以使用 `if` 函数来指定操作符优先级

```
if(win32|macos):CONFIG(debug, debug|release) {
    # Do something on Windows and macOS,
    # but only for the debug configuration.
}
win32|if(macos:CONFIG(debug, debug|release)) {
    # Do something on Windows (regardless of debug or release)
    # and on macOS (only for debug).
}
```

条件接受通配符来匹配一系列 CONFIG 值或 mkspec 名称

```
win32-* {
    # Matches every mkspec starting with "win32-"
    SOURCES += win32_specific.cpp
}
```

**注意**：使用上面的通配符检查 mkspec 名称是 qmake 检查平台的陈旧方法。现在，我们建议使用 QMAKE_PLATFORM 变量中由 mkspec 定义的值

我们也可以编写复杂的测试语句，对条件进行逐一的测试，这主要依靠 “`else`” 来完成，例如我们可以像下面这样写我们的代码：

```
win32:xml {
     message(Building for Windows)
     SOURCES += xmlhandler_win.cpp
} else:xml {
     SOURCES += xmlhandler.cpp
} else {
     message("Unknown configuration")
}
```

### 2.3 配置和条件域

在 CONFIG 变量中存储的值是由 qmake 特别处理的。每一个可能的值都可以用作条件域的条件。例如，CONFIG 保存的列表的值可以使用 opengl 来扩展：

```
CONFIG += opengl
```

如果我们像上面那样做的话，任何测试 `opengl` 的条件域都将是有效的，并且会被处理，我们可以使用这个功能给最后的可执行文件一个适当的名称：

```
opengl {
     TARGET = application-gl
} else {
     TARGET = application
}
```

该特性使更改项目的配置变得很容易，而不会丢失特定配置可能需要的所有自定义设置，而我们所要做的，可能只是一个特定的配置。在上面的代码中，在第一个条件域中声明的代 码将会被处理，因此最终的可执行文件将会被命名为 “`application-gl`”。然而，如果 `opengl`没有被指定，声明在第二个条件域内的代码会被处理，最终的可执行文件会被称为 “`application`”。

正因为我们可以把自定义的值附加给 CONFIG，我们就可以很方便的定制项目文件和调整 Makefile 文件。

### 2.4 平台条件域值

除了 `win32`，`macx` 和 `unix` 这样的常用于条件域条件的值，还有其他各种内置平台和编译器具体值也可以在条件域中用于测试。这些基于平台规范在 Qt 的 mkspecs 目录中被提供。例如，下面的代码用于显示当前使用的规范并且测试 `linux-g++` 规范。

```
message($$QMAKESPEC) 

linux-g++ {
     message(Linux)
}
```

我们可以测试任何其它平台的编译器组合，只要它的规范在 mkspecs 目录中存在。

## 3. 变量

项目文件中使用的许多变量是 qmake 在生成 Makefile 时使用的特殊变量，例如 DEFINES、SOURCES 和 HEADERS。此外，我们可以创建供自己使用的变量。qmake 在遇到给定名称的赋值时创建具有该名称的新变量。例如:

```
MY_VARIABLE = value
```

对于如何处理自己的变量没有限制，因为 qmake 会忽略它们，除非在处理作用域时需要对它们求值。

还可以通过在变量名前面加上 `$$` 来将当前变量的值赋给另一个变量。例如:

```
MY_DEFINES = $$DEFINES
```

现在 MY_DEFINES 变量包含项目文件中定义变量中的内容。这也等价于:

```
MY_DEFINES = $${DEFINES}
```

第二种表示法允许我们将变量的内容追加到另一个值，而不使用空格分隔这两个值。例如，下面的代码将确保最终的可执行文件有一个包含所使用的项目模板的名称:

```
TARGET = myproject_$${TEMPLATE}
```

## 4. 替换函数

qmake 提供了一个内置函数的选择，允许处理变量的内容。这些函数处理提供给它们的参数，并返回一个值或值列表。要将结果赋值给一个变量，可以使用 `$$` 操作符，就像将一个变量的内容赋值给另一个变量一样:

```
HEADERS = model.h
HEADERS += $$OTHER_HEADERS
HEADERS = $$unique(HEADERS)
```

这种类型的函数应该在赋值的右侧使用(即作为操作数)。

我们可以定义自己的函数来处理变量的内容如下:

```
defineReplace(functionName){
    #function code
}
```

下面的示例函数以 variable（变量名）作为惟一的参数，使用 eval() 内置函数从变量中提取值列表，并编译文件列表:

```
defineReplace(headersAndSources) {
    variable = $$1
    names = $$eval($$variable)
    headers =
    sources =

    for(name, names) {
        header = $${name}.h
        exists($$header) {
            headers += $$header
        }
        source = $${name}.cpp
        exists($$source) {
            sources += $$source
        }
    }
    return($$headers $$sources)
}

显示详细信息
```

## 5. 测试函数

qmake 提供了内置函数，可以在编写作用域时将其用作条件。这些函数不返回值，而是指示成功或失败:

```
count(options, 2) {
    message(Both release and debug specified.)
}
```

这种类型的函数应该只在条件表达式中使用。

可以定义自己的函数来为条件域提供条件。下面的示例测试列表中的每个文件是否存在，如果都存在则返回 true，如果有一个文件不存在则返回 false:

```
defineTest(allFiles) {
    files = $$ARGS

    for(file, files) {
        !exists($$file) {
            return(false)
        }
    }
    return(true)
}
```

# 高级应用

## 1. 添加新配置特性

qmake 允许我们通过将自己的名称添加到 CONFIG 变量指定的值列表中来创建自己的 `features`，这些特性可以包含在项目文件中。特性是 `.prf` 文件中的自定义函数和定义的集合，可以驻留在许多标准目录中的一个目录中。这些目录的位置在多个位置定义，并且 qmake 在查找 `.prf` 文件时按以下顺序检查每个标准目录：

1. 在 **QMAKEFEATURES** 环境变量中列出的目录中，其中包含由平台的路径列表分隔符分隔的目录列表(Unix 为冒号，Windows 为分号)
2. 在 **QMAKEFEATURES** 属性变量中列出的目录中，其中包含由平台的路径列表分隔符分隔的目录列表
3. 位于 mkspecs 目录中的 features 目录中。mkspecs 目录可以位于 **QMAKEPATH** 环境变量中列出的任何目录之下，**QMAKEPATH** 环境变量包含由平台的路径列表分隔符分隔的目录列表。例如: `$QMAKEPATH/mkspecs/<features>`
4. 位于 **QMAKESPEC** 环境变量提供的目录下的 features 目录中。例如: `$QMAKESPEC/<features>`
5. 位于 data_install/mkspecs 目录中的 features 目录中。例如: `data_install/mkspecs/<features>`
6. 在 features 目录中，该目录作为 **QMAKESPEC** 环境变量指定的目录的兄弟目录而存在。例如: `$QMAKESPEC/../<features>`

以下 features 目录被用于搜索 features 文件:

1. `features/unix`、`features/win32` 或 `features/macx`，取决于使用的平台
2. `features/`

例如，在项目文件中添加以下代码:

```
CONFIG += myfeatures
```

附加属性到 CONFIG 变量，qmake 将在完成项目文件的解析后，在上面列出的位置中搜索 myfeatures.prf 文件。以 Unix 系统为例，它将查找以下文件:

1. `$QMAKEFEATURES/myfeatures.prf` (对于 **QMAKEFEATURES** 环境变量中列出的每个目录)
2. `$$QMAKEFEATURES/myfeatures.prf`(对于 **QMAKEFEATURES** 属性变量中列出的每个目录)
3. `myfeatures.prf` (在项目的根目录中)。项目根由顶级 `.pro` 文件确定。但是，如果我们放置 .qmake.cache 在子目录或子项目目录下，然后项目根目录也会成为子目录。
4. `$QMAKEPATH/mkspecs/features/unix/myfeatures.prf` 和 `$QMAKEPATH/mkspecs/features/myfeatures.prf` (对于 **QMAKEPATH** 环境变量中列出的每个目录)
5. `$QMAKESPEC/features/unix/myfeatures.prf` 和 `$QMAKESPEC/features/myfeatures.prf`
6. `data_install/mkspecs/features/unix/myfeatures.prf` 和 `data_install/mkspecs/features/myfeatures.prf`
7. `$QMAKESPEC/../features/unix/myfeatures.prf` 和 `$QMAKESPEC/../features/myfeatures.prf`

**注意**: `.prf` 文件的名称必须小写

## 2. 安装文件

在 Unix 上使用构建工具来安装应用程序和库是很常见的；例如，通过调用 `make install`。由于这个原因，qmake 具有 `install set` （安装集）的概念，即包含有关如何安装项目的一部分的说明的对象。例如，一组文档文件可以用以下方式描述:

```
documentation.path = /usr/local/program/doc
documentation.files = docs/*
```

`path` 成员通知 qmake 文件应该安装在 `/usr/local/program/doc` (path 成员)中，而 `files` 成员指定应该复制到安装目录的文件。在本例中，`docs` 目录中的所有内容都将被复制到 `/usr/local/ program/doc` 中。

一旦一个安装集已经完全描述，我们可以通过如下方式附加它到安装列表:

```
INSTALLS += documentation
```

qmake 将确保将指定的文件复制到安装目录。如果需要对这个过程进行更多控制，还可以为对象的 `extra`（额外）成员提供定义。例如，下面的行告诉 qmake 为这个安装集执行一系列命令:

```
unix:documentation.extra = create_docs; mv master.doc toc.doc
```

unix 条件域 确保这些特定的命令只在 unix 平台上执行。可以使用其他范围规则为其他平台定义适当的命令。

在 `extra` (额外)成员中指定的命令将在对象的其他成员中的指令执行之前执行。

如果我们将内置安装集附加到 **INSTALLS** 变量，并且没有指定 `files` 或 `extra` 成员，qmake 将决定需要为我们复制什么。目前，支持 `target` 和 `dlltarget` 安装集。例如:

```
target.path = /usr/local/myprogram
INSTALLS += target
```

在上面的代码行中，qmake 知道需要复制什么，并将自动处理安装过程。

## 3. 添加自定义目标

qmake 尝试实现跨平台构建工具的所有功能。当我们真正需要运行与平台相关的特殊命令时，这通常不太理想。这可以通过对不同的 qmake 后端进行特定的指令来实现。

Makefile 输出的定制是通过对象样式的 API 执行的，在 qmake 的其他地方可以找到这种 API。对象是通过指定其成员自动定义的。例如:

```
mytarget.target = .buildfile
mytarget.commands = touch $$mytarget.target
mytarget.depends = mytarget2

mytarget2.commands = @echo Building $$mytarget.target
```

上面的代码定义了一个名为 `mytarget` 的 qmake 目标，其中包含一个名为 `.buildfile` 的 Makefile 目标，该目标由 `touch` 命令生成。最后，`.depends` 成员指定 mytarget 依赖于 mytarget2，这是之后定义的另一个目标。mytarget2 是一个虚拟目标。它只被定义为将一些文本打印到控制台。

最后一步是使用 **QMAKE_EXTRA_TARGETS** 变量来指示 qmake 这个对象是要构建的目标:

```
QMAKE_EXTRA_TARGETS += mytarget mytarget2
```

这就是实际构建自定义目标所需要做的全部工作。当然，我们可能希望将其中一个目标绑定到 qmake 构建目标。要做到这一点，我们只需要在 PRE_TARGETDEPS 列表中包含 Makefile 目标。

自定义目标规范支持以下成员:

| 成员           | 描述                                                         |
| -------------- | ------------------------------------------------------------ |
| commands       | 生成自定义构建目标（target）的命令                           |
| CONFIG         | 定制构建目标（target）的特定配置选项。可以设置为 recursive，以指示应该在 Makefile 中创建规则，以调用子目标特定的 Makefile 内部的相关目标。这个成员默认为每个子目标创建一个条目 |
| depends        | 自定义构建目标（target）所依赖的现有构建目标                 |
| recurse        | 指定在”Makefile”中创建要调用子目标（ sub-target）特定 Makefile 中的规则时应使用哪些子目标（ sub-target）。此成员仅在 CONFIG 中设置 recursive 时使用。典型的值是“Debug”和“Release” |
| recurse_target | 为 Makefile 中的规则指定应该通过子目标（ sub-target） Makefile 构建的目标（target）。这个成员添加了类似于 `$(MAKE) -f Makefile.[subtarget] [recurse_target]`。此成员仅在 CONFIG 中设置 recursive 时使用 |
| target         | 自定义构建目标（target）的名称                               |

显示详细信息

## 4. 添加编译器

可以自定义 qmake 来支持新的编译器和预处理程序:

```
new_moc.output  = moc_${QMAKE_FILE_BASE}.cpp
new_moc.commands = moc ${QMAKE_FILE_NAME} -o ${QMAKE_FILE_OUT}
new_moc.depend_command = g++ -E -M ${QMAKE_FILE_NAME} | sed "s,^.*: ,,"
new_moc.input = NEW_HEADERS
QMAKE_EXTRA_COMPILERS += new_moc
```

有了上述定义，如果 moc 可用，我们可以使用 drop-in 替换 moc。该命令对所有给定给 `NEW_HEADERS`变量的参数(来自 `input` 成员)执行，结果被写入由 `output` 成员定义的文件。这个文件被添加到项目中的其他源文件中。此外，qmake 将执行 `depend_command` 来生成依赖信息，并将这些信息放到项目中。

自定义编译器规范支持以下成员:

| 成员            | 描述                                                         |
| --------------- | ------------------------------------------------------------ |
| commands        | 用于从输入生成输出的命令                                     |
| CONFIG          | 自定义编译器的特定配置选项。有关详细信息，请参见 配置        |
| depend_command  | 指定用于为输出生成依赖项列表的命令                           |
| dependency_type | 指定输出文件的类型。如果它是一种已知的类型(例如 TYPE_C、TYPE_UI、TYPE_QRC)，则将其作为这些类型的文件之一处理 |
| depends         | 指定输出文件的依赖项                                         |
| input           | 该变量指定应使用自定义编译器处理的文件                       |
| name            | 自定义编译器正在做什么的描述。这只在一些后端使用             |
| output          | 从自定义编译器创建的文件名                                   |
| output_function | 指定用于指定要创建的文件名的自定义 qmake 函数                |
| variables       | 指示在 pro 文件中引用 (VARNAME) 时，此处指定的变量被替换为(QMAKE_COMP_VARNAME) |
| variable_out    | 应该将从输出创建的文件添加到的变量                           |

显示详细信息

此时 CONFIG 成员支持以下选项:

| 选项                  | 描述                                                         |
| --------------------- | ------------------------------------------------------------ |
| combine               | 指示将所有输入文件合并到单个输出文件中                       |
| target_predeps        | 指示输出应添加到 PRE_TARGETDEPS 列表中                       |
| explicit_dependencies | 输出的依赖项只能从 depends 成员生成，而不能从其他任何地方生成 |
| dep_existing_only     | 检查 .depend_command 的每个依赖项是否存在。不存在的依赖关系将被忽略。这个值是在Qt 5.13.2中引入的 |
| dep_lines             | .depend_command 的输出被解释为每行一个文件。默认情况下，是空格分开的，只有在向后兼容的情况下才会进行维护 |
| no_link               | 指示不应将输出添加到要链接的对象列表中                       |

## 5. 库依赖项

通常，当链接到一个库时，qmake 依赖于底层平台来了解这个库链接到哪些其他库，并让平台将它们拉进来。然而，在许多情况下，这是不够的。例如，当静态地链接一个库时，不会链接到其他库，因此不会创建对这些库的依赖关系。但是，以后链接到这个库的应用程序需要知道在哪里可以找到静态库需要的符号。如果显式启用跟踪，则 qmake 尝试在适当的情况下跟踪库的依赖项。

第一步是在库本身中启用依赖项跟踪。要做到这一点，你必须告诉 qmake 保存关于 library 的信息:

```
CONFIG += create_prl
```

这只与 `lib` 模板（template）相关，其他模板将忽略它。当此选项被启用时，qmake 将创建一个以 .prl 结尾的文件，该文件将保存关于库的一些元信息。这个元文件就像一个普通的项目文件，但是只包含内部变量声明。在安装这个库时，通过在 INSTALLS 变量中将其指定为目标，qmake 会自动将 .prl 文件复制到安装路径。

此过程的第二步是启用在使用静态库的应用程序中读取此元信息：

```
CONFIG += link_prl
```

启用此功能后，qmake 将处理应用程序链接的所有库并查找其元信息。qmake 将用它来确定相关的链接信息，特别是向应用程序项目文件的
DEFINES 列表以及 LIBS 添加值。一旦 qmake 处理了此文件，它将查看 LIBS 变量中新引入的库，并找到其依赖的 .prl 文件，一直持续到解析完所有库。此时，将像往常一样创建 Makefile，并针对应用程序显式地链接库。

.prl 文件应该只由 qmake 创建，并且不应该跨操作系统传递使用，因为它们可能包含平台相关的信息。

# 使用预编译头文件

预编译头文件(PCH)是一些编译器支持的一种性能特性，用于编译稳定的代码体，并将代码的编译状态存储在二进制文件中。在后续编译期间，编译器将加载存储的状态，并继续编译指定的文件。因为稳定的代码体不需要重新编译，所以后续的每次编译都会更快。

qmake 支持在一些平台和构建环境中使用预编译头文件，包括:

- Windows
  - nmake
  - Visual Studio projects (VS 2008 and later)
- macOS, iOS, tvOS, and watchOS
  - Makefile
  - Xcode
- Unix
  - GCC 3.4 and above
  - clang

## 1. 向项目添加预编译头文件

预编译头必须包含在整个项目中稳定且静态的代码。一个典型的预编译头可能是这样的:

```
// Add C includes here

#if defined __cplusplus
// Add C++ includes here
#include <stdlib>
#include <iostream>
#include <vector>
#include <QApplication> // Qt includes
#include <QPushButton>
#include <QLabel>
#include "thirdparty/include/libmain.h"
#include "my_stable_class.h"
...
#endif

显示详细信息
```

**注意**: 预编译头文件需要将 C 包含从 C++ 包含中分离出来，因为用于 C 文件的预编译头文件可能不包含 C++ 代码。

### 1.1 项目选项

要让我们的项目使用预编译头文件，我们只需要在我们的项目文件中定义 PRECOMPILED_HEADER 变量:

```
PRECOMPILED_HEADER = stable.h
```

qmake 将处理剩下的工作，以确保创建和使用预编译头文件。我们不需要在 HEADERS 中包含预编译头文件，因为如果配置支持预编译头文件，qmake 就会这样做。

默认情况下，MSVC 和 g++ 的设定目标窗口启用了 precompile_header

使用此选项，我们可以在使用预编译头文件时触发项目文件中的条件块来添加设置。例如:

```
precompile_header:!isEmpty(PRECOMPILED_HEADER) {
DEFINES += USING_PCH
}
```

要对 MSVC nmake 目标上的 C 文件也使用预编译头，请将 precompile_header_c 添加到 CONFIG 变量中。如果该 C 头文件也用于 C++，并且它包含 C++ 关键字/include，那么用 `#ifdef __cplusplus` 将它们括起来)。

## 2. 可能出现的问题

在某些平台上，预编译头文件的文件名后缀与其他目标文件的文件名后缀相同。例如，以下声明可能导致生成两个具有相同名称的不同对象文件：

```
PRECOMPILED_HEADER = window.h
SOURCES            = window.cpp
```

为了避免类似的潜在冲突，最好为将要预编译的头文件提供独特的名称。

## 3. 示例项目

我们可以在 Qt 发行版的 examples/qmake/precompile 目录中找到以下源代码:

### 3.1 `mydialog.ui`

下图在 Qt Creator 设计模式中显示了 mydialog.ui 文件。我们可以在编辑模式下查看其代码。

![img](https://www.ljjyy.com/img/qt/4011078-911bd21177398025.webp)

qmake-precompile-ui

### 3.2 `stable.h`

```
/* Add C includes here */

#if defined __cplusplus
/* Add C++ includes here */

# include <iostream>
# include <QApplication>
# include <QPushButton>
# include <QLabel>
#endif
```

### 3.3 `myobject.h`

```
#include <QObject>

class MyObject : public QObject
{
public:
    MyObject();
    ~MyObject();
};
```

### 3.4 `myobject.cpp`

```
#include <iostream>
#include <QDebug>
#include <QObject>
#include "myobject.h"

MyObject::MyObject()
    : QObject()
{
    std::cout << "MyObject::MyObject()\n";
}
```

### 3.5 `util.cpp`

```
void util_function_does_nothing()
{
    // Nothing here...
    int x = 0;
    ++x;
}
```

### 3.6 `main.cpp`

```
#include <QApplication>
#include <QPushButton>
#include <QLabel>
#include "myobject.h"
#include "mydialog.h"

int main(int argc, char **argv)
{
    QApplication app(argc, argv);

    MyObject obj;
    MyDialog dialog;

    dialog.connect(dialog.aButton, SIGNAL(clicked()), SLOT(close()));
    dialog.show();

    return app.exec();
}

显示详细信息
```

### 3.7 `precompile.pro`

```
TEMPLATE  = app
LANGUAGE  = C++
CONFIG   += cmdline precompile_header

# Use Precompiled headers (PCH)
PRECOMPILED_HEADER  = stable.h

HEADERS   = stable.h \
            mydialog.h \
            myobject.h
SOURCES   = main.cpp \
            mydialog.cpp \
            myobject.cpp \
            util.cpp
FORMS     = mydialog.ui

显示详细信息
```

# 配置 qmake

## 属性

qmake 拥有一个用于持久配置的系统，它允许我们在 qmake 中设置一个属性，并在每次调用 qmake 时查询它。我们可以使用如下方式在 qmake 中设置属性：

```
qmake -set PROPERTY VALUE
```

我们可以使用适当的属性名和值来代替上面命令中的 PROPERTY 和 VALUE 的内容

我们可以在 qmake 中通过如下两种方式查看属性信息:

```
qmake -query PROPERTY
qmake -query #请求展示全部的 PROPERTY/VALUE 对的列表
```

**注意**: qmake -query 除了使用 qmake -set 属性值设置的属性外，还列出了内置属性。这些信息将被保存到 QSettings 对象中(这意味着它将存储在不同平台的不同位置)。

下面的列表中列出了内置的属性:

| 属性                     | 说明                                                         |
| ------------------------ | ------------------------------------------------------------ |
| QMAKE_SPEC               | 主机构建期间解析并存储在 QMAKESPEC 变量中的主机 mkspec 的简称 |
| QMAKE_VERSION            | qmake 的版本                                                 |
| QMAKE_XSPEC              | 在目标构建期间解析并存储在 QMAKESPEC 变量中的目标 mkspec 的简称 |
| QT_HOST_BINS             | 主机可执行文件的位置                                         |
| QT_HOST_DATA             | qmake 使用的主机可执行程序的数据位置                         |
| QT_HOST_PREFIX           | 所有主机路径的默认前缀                                       |
| QT_INSTALL_ARCHDATA      | 与体系结构相关的通用 Qt 数据的位置                           |
| QT_INSTALL_BINS          | Qt 二进制文件的位置(工具和应用程序)                          |
| QT_INSTALL_CONFIGURATION | Qt 的设置保存的位置。不适用于 Windows                        |
| QT_INSTALL_DATA          | 与体系结构无关的常规 Qt 数据的位置                           |
| QT_INSTALL_DOCS          | 文档位置                                                     |
| QT_INSTALL_EXAMPLES      | 示例位置                                                     |
| QT_INSTALL_HEADERS       | 所有头文件的位置                                             |
| QT_INSTALL_IMPORTS       | QML 1.x 扩展的位置                                           |
| QT_INSTALL_LIBEXECS      | 库在运行时所需的可执行文件的位置                             |
| QT_INSTALL_LIBS          | 库的位置                                                     |
| QT_INSTALL_PLUGINS       | Qt 插件的位置                                                |
| QT_INSTALL_PREFIX        | 所有路径的默认前缀                                           |
| QT_INSTALL_QML           | QML 2.x 扩展的位置                                           |
| QT_INSTALL_TESTS         | Qt 测试用例的位置                                            |
| QT_INSTALL_TRANSLATIONS  | Qt 字符串翻译信息的位置                                      |
| QT_SYSROOT               | 目标构建环境使用的sysroot                                    |
| QT_VERSION               | Qt 版本。建议大家使用 $$QT.<module>.version 变量查询 Qt 模块特定的版本号。 |

显示详细信息

例如，我们可以使用 QT_INSTALL_PREFIX 属性查询这个版本的 qmake 对应的 Qt 的安装目录：

```
qmake -query "QT_INSTALL_PREFIX"
```

我们可以在 qmake 的 project 文件中通过如下方式使用上面所列出的属性值，具体示例如下所示:

```
QMAKE_VERS = $$[QMAKE_VERSION]
```

## QMAKESPEC

qmake 需要一个平台和编译器描述文件，该文件包含许多用于生成适当 Makefile 的默认值。标准 Qt 发行版附带了许多这样的文件，位于 Qt 安装目录中的 mkspecs 子目录中。

QMAKESPEC 环境变量通常包含以下内容：

- 包含 qmake.conf 文件的目录的完整路径。在这种情况下，qmake 将从该目录中打开 qmake.conf 文件。如果文件不存在，qmake 将退出并提示错误。
- 平台编译器组合的名称。在这种情况下，qmake 将在编译 Qt 时指定的数据路径的 mkspecs 子目录中指定的目录中进行搜索 (参见 QLibraryInfo::DataPath)。

**注意**: QMAKESPEC 路径将在 INCLUDEPATH 系统变量的内容之后自动添加到生成的 Makefile 中。

## 缓存文件

缓存文件是一个特殊的 qmake 读取文件，用于查找 qmake.conf 文件、项目文件或命令行中未指定的设置。运行 qmake 时，它查找一个名为 .qmake 的文件。缓存在当前目录的父目录中，除非指定 -nocache。如果 qmake 没有找到这个文件，它将静默地忽略这个处理步骤。

如果 qmake 找到这个 .qmake 文件。则它将在处理项目文件之前先处理这个文件。

## 文件扩展名

在正常情况下，qmake 将尝试为我们的平台使用适当的文件扩展名。但是，有时需要覆盖每个平台的默认选项，并显式定义 qmake 要使用的文件扩展名。这是通过重新定义某些内置变量来实现的。例如， moc 文件使用的扩展名可以在项目文件中通过以下方式重新定义:

```
QMAKE_EXT_MOC = .mymoc
```

以下变量用于重新定义可被 qmake 识别的常见文件扩展名:

| 变量           | 简介                                                        |
| -------------- | ----------------------------------------------------------- |
| QMAKE_EXT_MOC  | 修改用在 moc 文件上的扩展名                                 |
| QMAKE_EXT_UI   | 修改用于 Qt Designer UI 文件的扩展名（通常定义在 FORMS 中） |
| QMAKE_EXT_PRL  | 修改 库依赖文件 的扩展名                                    |
| QMAKE_EXT_LEX  | 更改 Lex 文件使用的后缀(通常定义在 LEXSOURCES 中)           |
| QMAKE_EXT_YACC | 更改 Yacc 文件使用的后缀通常定义在 YACCSOURCES 中）         |
| QMAKE_EXT_OBJ  | 更改生成的目标文件的后缀                                    |

上面所有的属性都只接受第一个值，因此我们必须为它分配一个值，该值将在整个项目文件中使用。另外，还有两个变量接受一个值列表:

| 变量          | 简介                                               |
| ------------- | -------------------------------------------------- |
| QMAKE_EXT_CPP | 让 qmake 把带有这些后缀的所有文件解析为 C++ 源文件 |
| QMAKE_EXT_H   | 让 qmake 把带有这些后缀的所有文件解析为 C++ 头文件 |

# 参考手册>变量

qmake 的基本行为受到定义于每个项目中的构建过程的变量声明的影响。其中一些声明资源(如头文件和源文件)对于每个平台都是通用的。其他的用于定制特定平台上的编译器和链接器的行为。

特定于平台的变量遵循它们扩展或修改的变量的命名模式，但是在它们的名称中包含相关平台的名称。例如，一个 makespec 可以使用 QMAKE_LIBS 来指定每个项目需要链接的库的列表，并使用 QMAKE_LIBS_X11 来扩展这个列表。

## 1. CONFIG

指定项目配置和编译器选项。这些值是 qmake 内部认可的，具有特殊的意义

下面的 CONFIG 选项值控制编译器和链接器标志:

| 选项                            | 描述                                                         |
| ------------------------------- | ------------------------------------------------------------ |
| release                         | 项目将以发布模式构建。如果还指定了debug，则最后一个设置的会生效 |
| debug                           | 项目将在调试模式下构建                                       |
| debug_and_release               | 项目将以调试和发布两种模式构建                               |
| debug_and_release_target        | 这个选项是默认设置的。如果还设置了debug_and_release，那么调试和发布将在单独的调试和发布目录中构建完成 |
| build_all                       | 如果指定了debug_and_release，则默认情况下以调试和发布模式构建项目 |
| autogen_precompile_source       | 自动生成一个 .cpp 文件，其中包含 .pro 文件中指定的预编译头文件 |
| ordered                         | 当使用 subdirs 模板时，该选项指定应该按照给定目录的顺序处理列出的目录 **注意**: 不建议使用此选项。指定 SUBDIRS 变量文档中描述的依赖项 |
| precompile_header               | 支持在项目中使用 预编译头文件                                |
| precompile_header_c (MSVC only) | 支持对C文件使用 预编译头文件                                 |
| warn_on                         | 编译器应该输出尽可能多的警告。如果还指定了warn_off，则最后指定的生效 |
| warn_off                        | 编译器应该输出尽可能少的警告                                 |
| exceptions                      | 启用异常支持。默认设置                                       |
| exceptions_off                  | 异常支持被禁用                                               |
| ltcg                            | 链接时间代码生成（Link time code generation）已启用。这个选项在默认情况下是关闭的 |
| rtti                            | RTTI 支持已启用。默认情况下，使用编译器默认值                |
| rtti_off                        | RTTI 支持被禁用。默认情况下，使用编译器默认值                |
| stl                             | STL 支持已启用。默认情况下，使用编译器默认值                 |
| stl_off                         | STL 支持被禁用。默认情况下，使用编译器默认值                 |
| thread                          | 启用线程支持。当 CONFIG 包含 qt (这是默认值) 时启用此功能    |
| utf8_source                     | 指定项目的源文件使用 UTF-8 编码。默认情况下，使用编译器默认值 |
| hide_symbols                    | 将二进制文件中符号的默认可见性设置为隐藏。默认情况下，使用编译器默认值 |
| c99                             | 启用 C99 支持。如果编译器不支持 C99，或者不能选择 C 标准，这个选项没有任何作用。默认情况下，使用编译器默认值 |
| c11                             | 启用 C11 支持。如果编译器不支持 C11，或者不能选择 C 标准，则此选项无效。默认情况下，使用编译器默认值 |
| strict_c                        | 禁用对 C 编译器扩展的支持。默认情况下，它们是启用的          |
| c++11                           | 启用 C++ 11 支持。如果编译器不支持 C++ 11，或者不能选择 C++ 标准，则此选项无效。默认情况下，支持是启用的 |
| c++14                           | 启用 C++ 14 支持。如果编译器不支持 C++ 14，或者不能选择 C++ 标准，则此选项无效。默认情况下，支持是启用的 |
| c++1z                           | 启用 C++ 17 支持。如果编译器不支持 C++ 17，或者不能选择 C++ 标准，则此选项无效。默认情况下，支持是启用的 |
| c++17                           | 同 c++1z                                                     |
| c++2a                           | 启用 C++ 2a 支持。如果编译器不支持 C++ 2a，或者不能选择 C++ 标准，则此选项无效。默认情况下，支持是启用的 |
| c++latest                       | 支持由编译器支持的最新 C++ 语言标准。默认情况下，此选项是禁用的 |
| strict_c++                      | 禁用对c++编译器扩展的支持。默认情况下，它们是启用的          |
| depend_includepath              | 启用将 INCLUDEPATH 的值附加到 DEPENDPATH。默认设置           |
| lrelease                        | 为 TRANSLATIONS 和 EXTRA_TRANSLATIONS 中列出的所有文件运行 `lrelease`。如果没有设置 `embed_translations`，那么将生成的 .qm 文件安装到 QM_FILES_INSTALL_PATH 中。使用 QMAKE_LRELEASE_FLAGS 向 lrelease 调用添加选项。默认不设置 |
| embed_translations              | 将从 lrelease 生成的翻译嵌入到可执行文件中，位于 QM_FILES_RESOURCE_PREFIX 之下。也需要设置 lrelease。非默认设置 |
| create_libtool                  | 为当前构建的库创建一个 libtool .la 文件                      |
| create_pc                       | 为当前构建的库创建一个 pkg-config .pc 文件                   |
| no_batch                        | 仅 NMake 适用: 关闭 NMake 批处理规则或推理规则的生成         |
| skip_target_version_ext         | 禁止将自动版本号附加到 Windows 上的 DLL 文件名               |
| suppress_vcproj_warnings        | 抑制 VS 项目生成器的警告                                     |
| windeployqt                     | 链接后自动调用 windeployqt，并将输出作为部署项添加           |
| dont_recurse                    | 取消当前子项目的 qmake 递归                                  |
| no_include_pwd                  | 不要将当前目录添加到 INCLUDEPATHS                            |

显示详细信息

当我们使用 debug_and_release 选项(这是 Windows 下的默认选项)时，项目将被处理三次:一次生成“meta”Makefile，两次生成 Makefile.Debug 和 Makefile.Release

在后者的传递过程中，`build_pass` 和相应的 `debug` 或 `release` 选项被附加到 CONFIG 中。这使得执行特定于构建的任务成为可能。例如:

```
build_pass:CONFIG(debug, debug|release) {
    unix: TARGET = $$join(TARGET,,,_debug)
    else: TARGET = $$join(TARGET,,,d)
}
```

作为手动编写构建类型条件的替代方法，一些变量提供了特定于构建的变体，例如 QMAKE_LFLAGS_RELEASE 和一般的 QMAKE_LFLAGS。这些应该在可用时使用。

元 Makefile 通过 debug 和 release 目标来调用子构建，通过 all 目标来组合构建。当使用 build_all 作为 CONFIG 选项时，组合构建是默认的。否则，设置 (debug, release)中最后指定的 CONFIG 选项将确定默认值。在这种情况下，我们可以显式地调用 all 目标来一次性构建两个配置

```
make all
```

**注意**: 在生成 Visual Studio 和 Xcode 项目时，细节略有不同

当链接一个库时，qmake 依赖于底层平台来了解这个库所链接的其他库。但是，如果静态链接，qmake 将不会得到这个信息，除非我们使用以下配置选项:

| 选项           | 描述                                                         |
| -------------- | ------------------------------------------------------------ |
| create_prl     | 此选项使 qmake 能够跟踪这些依赖项。当启用此选项时，qmake 将创建一个扩展名为 .prl 的文件，该文件将保存有关库的元信息(有关更多信息，请参阅 库依赖项 的相关部分) |
| link_prl       | 启用此选项后，qmake 将处理应用程序链接到的所有库，并查找它们的元信息(有关更多信息，请参阅 库依赖项 的相关部分) |
| no_install_prl | 此选项禁用为生成的 .prl 文件生成安装规则                     |

**注意**: 在构建静态库时需要 `create_prl`选项，而在使用静态库时需要 `link_prl` 选项

以下选项定义应用程序或库类型:

| 选项               | 描述                                                         |
| ------------------ | ------------------------------------------------------------ |
| qt                 | 目标是一个 Qt 应用程序或库，需要 Qt 库和头文件。Qt 库的适当包含路径和库路径将自动添加到项目中。这是默认定义的，可以使用 QT 变量进行微调 |
| x11                | 目标是一个 X11 应用程序或库。适当的包含路径和库将自动添加到项目中 |
| testcase           | 目标是一个自动化测试。一个检查目标 将被添加到生成的 Makefile 中以运行测试。只在生成 Makefiles 时有效 |
| insignificant_test | 自动测试的退出代码将被忽略。仅当 testcase 也被设置时才有效   |
| windows            | 目标是一个 Win32 窗口应用程序(仅限应用程序)。适当的包含路径、编译器标志和库将自动添加到项目中 |
| console            | 目标是一个Win32控制台应用程序(仅限应用程序)。适当的包含路径、编译器标志和库将自动添加到项目中。考虑为跨平台应用程序使用选项cmdline |
| cmdline            | 目标是一个跨平台的命令行应用程序。在Windows上，这意味着 CONFIG += console。在macOS上，这意味着 CONFIG -= app_bundle |
| shared             | 目标是一个共享对象或 DLL。适当的包含路径、编译器标志和库将自动添加到项目中。注意，dll 也可以在所有平台上使用;将创建具有目标平台适当后缀(.dll或.so)的共享库文件 |
| dll                | 同上                                                         |
| static             | 目标是一个静态库(仅限 lib)。适当的编译器标记将自动添加到项目中 |
| staticlib          | 同上                                                         |
| plugin             | 目标是一个插件(仅限 lib)。这也启用了 dll                     |
| designer           | 目标是 Qt Designer 的插件                                    |
| no_lflags_merge    | 确保存储在 LIBS 变量中的库列表在使用之前不会被缩减为惟一值列表 |

显示详细信息

下面这些选项是只在 Windows 上定义特定的功能:

| 选项               | 描述                                                         |
| ------------------ | ------------------------------------------------------------ |
| flat               | 当使用 vcapp 模板时，这将把所有源文件放到源文件组中，而把头文件放到头文件组中，不管它们位于哪个目录中。关闭此选项将根据驻留的目录将文件分组到源/头文件组中。这是默认打开的 |
| embed_manifest_dll | 在作为库项目的一部分创建的 DLL 中嵌入清单文件                |
| embed_manifest_exe | 在作为应用程序项目的一部分创建的 EXE 中嵌入清单文件          |

有关嵌入清单文件的选项的更多信息，请参见 平台相关事项

以下选项仅对 macOS 有效:

| 选项          | 描述                                           |
| ------------- | ---------------------------------------------- |
| app_bundle    | 将可执行文件放入一个包中(这是缺省值)           |
| lib_bundle    | 将库放入库包中                                 |
| plugin_bundle | 将插件放入插件包中。Xcode 项目生成器不支持此值 |

bundle 的构建过程还受到 QMAKE_BUNDLE_DATA 变量内容的影响

以下选项仅在 Linux/Unix 平台上有效:

| 选项                | 描述                           |
| ------------------- | ------------------------------ |
| largefile           | 包括对大文件的支持             |
| separate_debug_info | 将库的调试信息放在单独的文件中 |

解析作用域时，还将检查 CONFIG 变量。我们可以为此变量分配任何内容。

例如：

```
CONFIG += console newstuff
...
newstuff {
    SOURCES += new.cpp
    HEADERS += new.h
}
```

## 2. DEFINES

qmake 将此变量的值添加为编译器 C 预处理器宏（-D 选项）

例如：

```
DEFINES += USE_MY_STUFF
```

## 3. DEFINES_DEBUG

指定调试配置的预处理器定义。这个变量的值在项目加载后被添加到 DEFINES 中。这个变量通常在 qmake.conf 中设置，很少需要修改。

这个变量是在 Qt 5.13.2 中引入的。

## 4. DEFINES_RELEASE

指定发布配置的预处理器定义。这个变量的值在项目加载后被添加到 DEFINES 中。这个变量通常在 qmake.conf 中设置，很少需要修改。

**注意**: 对于 `MSVC mkspecs`，这个变量在默认情况下包含值 `NDEBUG`。

这个变量是在 Qt 5.13.2 中引入的。

## 5. DEF_FILE

**注意**: 这个变量只在 Windows 平台上使用 app 模板时使用。

指定要包含在项目中的 .def 文件。

## 6. DEPENDPATH

指定 qmake 要扫描的目录列表，以解析依赖项。当 qmake 遍历源代码中 #include 的头文件时，将使用此变量。

## 7. DESTDIR

指定将目标文件放在何处
例如：

```
DESTDIR = ../../lib
```

**注意**：支持的字符列表可以依赖于使用的生成工具。特别是，括号不适用于 make。

## 8. DISTFILES

指定要包含在 dist 目标中的文件列表。该特性只受 UnixMake 规范支持
例如：

```
DISTFILES += ../program.txt
```

## 9. DLLDESTDIR

**注意**: 这个变量只适用于 Windows 平台目标

指定复制目标（target） dll 的位置

## 10. EXTRA_TRANSLATIONS

指定翻译(.ts)文件列表，其中包含将用户界面文本翻译为非本机语言的内容。

与 TRANSLATIONS 不同，EXTRA_TRANSLATIONS 中的翻译文件仅由 lrelease 处理，而不是 lupdate

我们可以使用 CONFIG += lrelease 在构建过程中自动编译文件，并使用 CONFIG += lrelease embed_translations 使它们在 Qt 资源系统 中可用

有关 Qt 国际化(i18n)和本地化(l10n)的更多信息，请参阅 Qt Linguist 手册。

## 11. FORMS

指定 UI 文件(参见 Qt 设计器手册)在编译前由 uic 处理。构建这些 UI 文件所需的所有依赖项、头文件和源文件将自动添加到项目中
示例：

```
FORMS = mydialog.ui \
    mywidget.ui \
    myconfig.ui
```

## 12. GUID

指定在 .vcproj 文件中设置的 GUID。GUID 通常是随机确定的。但是，如果需要固定 GUID，可以使用此变量设置它。

此变量仅特定于 .vcproj 文件；否则将被忽略

## 13. HEADERS

定义项目的头文件列表。

qmake 自动检测头文件中的类是否需要 moc，并将适当的依赖项和文件添加到项目中，以生成和链接 moc 文件。

例如:

```
HEADERS = myclass.h \
          login.h \
          mainwindow.h
```

另请参阅 SOURCES

## 14. ICON

此变量仅在 Mac OS 上用于设置应用程序图标。有关更多信息，请参见应用程序图标文档

## 15. IDLSOURCES

此变量仅用于在 Windows 上生成 Visual Studio 项目，以将指定的文件放在生成的文件夹中

## 16. INCLUDEPATH

指定编译项目时应该搜索的 #include 目录。

例如:

```
INCLUDEPATH = c:/msdev/include d:/stl/include
```

要指定包含空格的路径，请使用 空格 中描述的双引号方式引用该路径：

```
win32:INCLUDEPATH += "C:/mylibs/extra headers"
unix:INCLUDEPATH += "/home/user/extra headers"
```

## 17. INSTALLS

指定在执行 make install 或类似的安装过程时将安装的资源列表。列表中的每个项通常都定义了一些属性，这些属性提供了关于将在何处安装它的信息。

例如，下面的 `target.path` 定义描述安装构建目标的位置，INSTALLS 任务将构建目标添加到要安装的现有资源列表中:

```
target.path += $$[QT_INSTALL_PLUGINS]/imageformats
INSTALLS += target
```

INSTALLS 有一个 .CONFIG 成员，它可以接受几个值:

| 值                 | 描述                                                         |
| ------------------ | ------------------------------------------------------------ |
| no_check_exist     | 如果没有设置，qmake 将查看要安装的文件是否确实存在。如果这些文件不存在，qmake 不会创建安装规则。如果需要安装作为构建过程的一部分生成的文件(如qdoc创建的HTML文件)，请使用此配置值 |
| nostrip            | 如果设置，典型的 Unix 条带功能将被关闭，调试信息将保留在二进制文件中 |
| executable         | 在 Unix 上，这将设置可执行标志                               |
| no_build           | 在进行 `make install` 时，如果还没有项目的生成版本，则首先生成项目，然后安装项目。如果不希望出现这种行为，可以设置此配置值，以确保构建目标没有作为依赖项添加到安装目标 |
| no_default_install | 项目有一个顶级项目目标，在该目标中，当我们进行 `make install` 时，将安装所有内容。但是，如果我们有一个设置了此配置值的安装目标，则默认情况下不会安装它。然后必须显式地输入 `make install_<file>`。 |

显示详细信息

有关更多信息，请参见 Installing Files

此变量还用于指定将部署到嵌入式设备的其他文件

## 18. LEXIMPLS

指定 Lex 实现文件的列表。这个变量的值通常由 qmake 或 qmake.conf 处理，很少需要修改。

## 19. LEXOBJECTS

指定中间 Lex 对象文件的名称。这个变量的值通常由 qmake 处理，很少需要修改。

## 20. LEXSOURCES

指定 Lex 源文件列表。所有依赖项、头文件和源文件将自动添加到项目中，以构建这些 lex 文件。例如：

```
LEXSOURCES = lexer.l
```

## 21. LIBS

指定要链接到项目中的库的列表。如果使用 Unix 的 -l(库) 和 -L(库路径)标志，qmake 将在 Windows 上正确地处理库(即将库的完整路径传递给链接器)。库必须存在，以便 qmake 找到位于 -l lib 的目录

例如：

```
unix:LIBS += -L/usr/local/lib -lmath
win32:LIBS += c:/mylibs/math.lib
```

要指定包含空格的路径，请使用 空格 中描述的双引号方式引用该路径：

```
win32:LIBS += "C:/mylibs/extra libs/extra.lib"
unix:LIBS += "-L/home/user/extra libs" -lextra
```

默认情况下，存储在 LIBS 中的库的列表在使用之前被缩减为唯一名称的列表。要更改此行为，请将 no_lflags_merge 选项添加到 CONFIG 变量:

```
CONFIG += no_lflags_merge
```

## 22. LIBS_PRIVATE

指定要私有链接到项目中的库的列表。这个变量的行为与 LIBS 相同，不同之处在于，为 Unix 构建的共享库项目不会在其链接接口中公开这些依赖项。

这样做的效果是，如果项目 C 依赖于库 B，而库 B 又私有依赖于库 A，但是 C 也想直接使用来自 A 的符号，它需要显式地链接到库 A。换句话说，私有链接的库不会在构建时被传递。

## 23. LITERAL_HASH

每当变量声明中需要一个文本哈希字符 (#) 时，就会使用这个变量，可能是作为文件名的一部分，也可能是传递给某个外部应用程序的字符串。

例如:

```
# To include a literal hash character, use the $$LITERAL_HASH variable:
urlPieces = http://doc.qt.io/qt-5/qtextdocument.html pageCount
message($$join(urlPieces, $$LITERAL_HASH))
```

通过以这种方式使用 LITERAL_HASH，可以使用 # 字符为要打印到控制台的 message() 函数构造 URL:

```
Project MESSAGE: http://doc.qt.io/qt-5/qtextdocument.html#pageCount
```

## 24. MAKEFILE

指定生成的 Makefile 的名称。这个变量的值通常由 qmake 或 qmake.conf 处理，很少需要修改。

## 25. MAKEFILE_GENERATOR

指定生成 Makefile 时要使用的 Makefile 生成器的名称。这个变量的值通常由 qmake 在内部处理，很少需要修改。

## 26. MSVCPROJ_*

这些变量由 qmake 在内部处理，不应该修改或使用。

## 27. MOC_DIR

指定应该放置所有中间 moc 文件的目录。

例如：

```
unix:MOC_DIR = ../myproject/tmp
win32:MOC_DIR = c:/myproject/tmp
```

## 28. OBJECTIVE_HEADERS

定义项目的 Objective-C++ 头文件。

qmake 自动检测头文件中的类是否需要 moc，并将适当的依赖项和文件添加到项目中，以生成和链接 moc 文件。

这类似于 HEADERS 变量，但是允许使用 Objective-C++ 编译器编译生成的 moc 文件。

另请参阅 **OBJECTIVE_SOURCES**

## 29. OBJECTIVE_SOURCES

指定项目中所有 Objective-C/C++ 源文件的名称。

这个变量现在已经废弃了，Objective-C/ C++ 文件(.m 和 .mm )可以添加到 SOURCES 变量中。

另请参阅 **OBJECTIVE_HEADERS**

## 30. OBJECTS

这个变量是由 SOURCES 变量自动填充的。每个源文件的扩展名都被 .o (Unix) 或. obj (Win32) 所代替。我们可以将对象添加到列表中

## 31. OBJECTS_DIR

指定应该放置所有中间对象的目录。

例如:

```
unix:OBJECTS_DIR = ../myproject/tmp
win32:OBJECTS_DIR = c:/myproject/tmp
```

## 32. POST_TARGETDEPS

列出目标(target)所依赖的库。一些后端，例如 Visual Studio 和 Xcode 项目文件的生成器，不支持这个变量。通常，这些构建工具在内部支持这个变量，它对于显式列出依赖的静态库非常有用。

这个列表位于所有内建(和 $$PRE_TARGETDEPS )依赖项之后。

## 33. PRE_TARGETDEPS

列出目标(target)所依赖的库。一些后端，例如 Visual Studio 和 Xcode 项目文件的生成器，不支持这个变量。通常，这些构建工具在内部支持这个变量，它对于显式列出依赖的静态库非常有用。

此列表放在所有内置依赖项之前

## 34. PRECOMPILED_HEADER

指示用于创建预编译头文件的头文件，以提高项目的编译速度。目前仅在某些平台上支持预编译头文件(Windows -所有 MSVC 项目类型、Apple - Xcode、Makefile、Unix - gcc 3.3 和更高版本)。

## 35. PWD

指向包含正在解析的当前文件的目录的完整路径。在编写项目文件以支持影子构建时，引用源树中的文件可能很有用。

另请参阅 _PRO_FILE_PWD_

**注意**: 不要试图覆盖此变量的值。

## 36. OUT_PWD

指向 qmake 放置生成的 Makefile 的目录的完整路径。

**注意**: 不要试图覆盖此变量的值。

## 37. QM_FILES_RESOURCE_PREFIX

指定资源系统中的目录，在该目录中 CONFIG += embed_translations 将使 .qm 文件可用。

默认值是 `:/i18n/`

## 38. QM_FILES_INSTALL_PATH

指定 CONFIG += lrelease 生成的目标目录 .qm 文件将被安装到的位置。如果设置 CONFIG += embed_translations 则没有任何效果

## 39. QML_IMPORT_PATH

此变量仅供 Qt Creator 使用。有关详细信息，请参见 Qt Creator: 使用带有插件的QML模块。

## 40. QMAKE

指定 qmake 程序本身的名称，并将其放置在生成的 Makefile 中。这个变量的值通常由qmake 或 qmake.conf 处理，很少需要修改。

## 41. QMAKESPEC

包含生成 Makefile 时使用的 qmake 配置的完整路径的系统变量。这个变量的值是自动计算的。

**注意**: 不要试图覆盖此变量的值。

## 42. QMAKE_AR_CMD

**注意**: 此变量仅在 Unix 平台上使用。

指定创建共享库时要执行的命令。这个变量的值通常由 qmake 或 qmake.conf 处理，很少需要修改。

## 43. QMAKE_BUNDLE_DATA

**注意**: 这个变量只在 macOS、iOS、tvOS 和 watchOS 上使用。

指定将与库包一起安装的数据，通常用于指定头文件集合。

例如，以下几行代码将 path/to/header_one.h 和 path/to/header_two.h 添加到一个包含框架提供的头信息的组:

```
FRAMEWORK_HEADERS.version = Versions
FRAMEWORK_HEADERS.files = path/to/header_one.h path/to/header_two.h
FRAMEWORK_HEADERS.path = Headers
QMAKE_BUNDLE_DATA += FRAMEWORK_HEADERS
```

最后一行将有关头文件的信息添加到将与库包一起安装的资源集合中。

在 CONFIG 变量中添加 lib_bundle 选项时创建库包。

有关创建库包的更多信息，请参见 平台相关事项。

## 44. QMAKE_BUNDLE_EXTENSION

**注意**: 这个变量只在 macOS、iOS、tvOS 和 watchOS 上使用。

指定要用于库包的扩展名。这允许使用自定义扩展来创建框架，而不是使用标准的 .framework 目录扩展名。

例如，下面的定义将产生一个扩展名为 .myframework 的框架:

```
QMAKE_BUNDLE_EXTENSION = .myframework
```

## 45. QMAKE_CC

指定在构建包含 C 源代码的项目时将使用的 C 编译器。在处理 Makefile 时，只要编译器可执行文件位于 PATH 变量中包含的路径上，就只需要指定该文件的文件名。

## 46. QMAKE_CFLAGS

指定用于构建项目的 C 编译器标志。这个变量的值通常由 qmake 或 qmake.conf 处理，很少需要修改。可以通过分别修改 QMAKE_CFLAGS_DEBUG 和 QMAKE_CFLAGS_RELEASE 变量来调整特定于调试和发布模式的标志。

## 47. QMAKE_CFLAGS_DEBUG

为调试构建指定 C 编译器标志。这个变量的值通常由 qmake 或 qmake.conf 处理，很少需要修改。

## 48. QMAKE_CFLAGS_RELEASE

为发布构建指定 C 编译器标志。这个变量的值通常由 qmake 或 qmake.conf 处理，很少需要修改。

## 49. QMAKE_CFLAGS_RELEASE_WITH_DEBUGINFO

指定 CONFIG 中设置 force_debug_info 的 C 编译器版本标志。这个变量的值通常由 qmake 或 qmake.conf 处理，很少需要修改

## 50. QMAKE_CFLAGS_SHLIB

**注意**: 此变量仅在 Unix 平台上使用。

指定用于创建共享库的编译器标志。这个变量的值通常由 qmake 或 qmake.conf 处理，很少需要修改。

## 51. QMAKE_CFLAGS_THREAD

指定用于创建多线程应用程序的编译器标志。这个变量的值通常由 qmake 或 qmake.conf 处理，很少需要修改。

## 52. QMAKE_CFLAGS_WARN_OFF

此变量仅在设置 `warn_off` CONFIG 选项时使用。这个变量的值通常由 qmake 或 qmake.con 处理，很少需要修改。

## 53. QMAKE_CFLAGS_WARN_ON

此变量仅在设置 `warn_on` CONFIG 选项时使用。这个变量的值通常由 qmake 或 qmake.conf 处理，很少需要修改。

## 54. QMAKE_CLEAN

指定要通过 `make clean` 删除的，生成文件(例如，通过 moc 和 uic 生成的文件)和对象文件的列表。

## 55. QMAKE_CXX

指定在构建包含 C++ 源代码的项目时将使用的 C++ 编译器。在处理 Makefile 时，只要编译器可执行文件位于 PATH 变量中包含的路径上，就只需要指定该文件的文件名。

## 56. QMAKE_CXXFLAGS

指定用于构建项目的 C++ 编译器标志。这个变量的值通常由 qmake 或 qmake.conf 处理，很少需要修改。可以通过分别修改 QMAKE_CXXFLAGS_DEBUG 和 QMAKE_CXXFLAGS_RELEASE 变量来调整特定于调试和发布模式的标志。

## 57. QMAKE_CXXFLAGS_DEBUG

指定用于调试构建的 C++ 编译器标志。这个变量的值通常由 qmake 或 qmake.conf 处理，很少需要修改。

## 58. QMAKE_CXXFLAGS_RELEASE

指定用于发布构建的 C++ 编译器标志。这个变量的值通常由 qmake 或 qmake.conf 处理，很少需要修改。

## 59. QMAKE_CXXFLAGS_RELEASE_WITH_DEBUGINFO

指定 CONFIG 中设置 force_debug_info 的 C++ 编译器版本标志。这个变量的值通常由 qmake 或 qmake.conf 处理，很少需要修改

## 60. QMAKE_CXXFLAGS_SHLIB

指定用于创建共享库的 C++ 编译器标志。这个变量的值通常由 qmake 或 qmake.conf 处理，很少需要修改。

## 61. QMAKE_CXXFLAGS_THREAD

指定用于创建多线程应用程序的 C++ 编译器标志。这个变量的值通常由 qmake 或 qmake.conf 处理，很少需要修改。

## 62. QMAKE_CXXFLAGS_WARN_OFF

指定用于抑制编译器警告的 C++ 编译器标志。这个变量的值通常由 qmake 或 qmake.conf 处理，很少需要修改。

## 63. QMAKE_CXXFLAGS_WARN_ON

指定用于生成编译器警告的 C++ 编译器标志。这个变量的值通常由 qmake 或 qmake.conf 处理，很少需要修改。

## 64. QMAKE_DEVELOPMENT_TEAM

**注意**: 这个变量只在 macOS、iOS、tvOS 和 watchOS 上使用。

开发团队用于签名证书和配置配置文件的标识符。

## 65. QMAKE_DISTCLEAN

指定要通过 `make distclean` 删除的文件列表。

## 66. QMAKE_EXTENSION_SHLIB

包含共享库的扩展。这个变量的值通常由 qmake 或 qmake.conf 处理，很少需要修改。

**注意**: 更改扩展的特定于平台的变量会覆盖此变量的内容。

## 67. QMAKE_EXTENSION_STATICLIB

包含共享静态库的扩展。这个变量的值通常由 qmake 或 qmake.conf 处理，很少需要修改。

## 68. QMAKE_EXT_MOC

修改放置在包含的 moc 文件上的扩展名。另请参阅 File Extensions

## 69. QMAKE_EXT_UI

修改用于Qt Designer UI文件的扩展名。另请参阅 File Extensions

## 70. QMAKE_EXT_PRL

修改用于创建的 PRL 文件的扩展名。另请参阅 File Extensions、Library Dependencies

## 71. QMAKE_EXT_LEX

修改对给定给 Lex 的文件使用的扩展名。另请参阅 File Extensions、LEXSOURCES

## 72. QMAKE_EXT_YACC

修改对给定给 Yacc 的文件使用的扩展名。另请参阅 File Extensions、YACCSOURCES

## 73. QMAKE_EXT_OBJ

修改用于生成的对象文件的扩展名。另请参阅 File Extensions

## 74. QMAKE_EXT_CPP

修改文件的后缀，这些后缀应该被解释为 C++ 源代码。另请参阅 File Extensions

## 75. QMAKE_EXT_H

修改文件的后缀，这些后缀应该被解释为 C 头文件。另请参阅 File Extensions

## 76. QMAKE_EXTRA_COMPILERS

指定附加编译器或预处理器的列表。另请参阅 Adding Compilers

## 77. QMAKE_EXTRA_TARGETS

指定其他 qmake 目标的列表。另请参阅 Adding Custom Targets

## 78. QMAKE_FAILED_REQUIREMENTS

包含失败需求的列表。这个变量的值是由 qmake 设置的，不能修改。另请参阅 requires() 和 REQUIRES

## 79. QMAKE_FRAMEWORK_BUNDLE_NAME

**注意**: 这个变量只在 macOS、iOS、tvOS 和 watchOS 上使用。

在框架项目中，此变量包含要用于构建框架的名称。

默认情况下，此变量包含与 TARGET 变量相同的值。

有关创建框架和库包的更多信息，请参见 Creating Frameworks

## 80. QMAKE_FRAMEWORK_VERSION

**注意**: 这个变量只在 macOS、iOS、tvOS 和 watchOS 上使用。

对于构建目标为 macOS、iOS、tvOS 或 watchOS 框架的项目，此变量用于指定将应用于所构建框架的版本号。

默认情况下，此变量包含与 VERSION 变量相同的值。

有关创建框架的更多信息，请参见 Creating Frameworks

## 81. QMAKE_HOST

提供有关运行 qmake 的主机的信息。例如，您可以从 `QMAKE_HOST.arch` 检索主机机器架构。

| 关键字          | 返回值             |
| --------------- | ------------------ |
| .arch           | 主机架构           |
| .os             | 主机 OS            |
| .cpu_count      | 可用 cpu 的数量    |
| .name           | 主机名称           |
| .version        | 主机 OS 版本号     |
| .version_string | 主机 OS 版本字符串 |

```
win32-g++:contains(QMAKE_HOST.arch, x86_64):{
    message("Host is 64bit")
    ...
}
```

## 82. QMAKE_INCDIR

指定附加到 INCLUDEPATH 的系统标题路径列表。这个变量的值通常由 qmake 或 qmake.conf 处理，很少需要修改。

## 83. QMAKE_INCDIR_EGL

指定在使用 OpenGL/ES 或 OpenVG 支持构建目标时要添加到 INCLUDEPATH 的 EGL 头文件的位置。这个变量的值通常由 qmake 或 qmake.conf 处理，很少需要修改。

## 84. QMAKE_INCDIR_OPENGL

指定在构建支持 OpenGL 的目标时要添加到 INCLUDEPATH 的 OpenGL 头文件的位置。这个变量的值通常由 qmake 或 qmake.conf处理，很少需要修改。

如果 OpenGL 实现使用 EGL(大多数 OpenGL/ES 系统)，那么 QMAKE_INCDIR_EGL 可能也需要设置。

## 85. QMAKE_INCDIR_OPENGL_ES2

这个变量指定在构建支持 OpenGL ES 2 的目标时要添加到 INCLUDEPATH 的 OpenGL 头文件的位置。

这个变量的值通常由 qmake 或 qmake.conf 处理，很少需要修改。

如果 OpenGL 实现使用 EGL(大多数 OpenGL/ES 系统)，那么 QMAKE_INCDIR_EGL 可能也需要设置。

## 86. QMAKE_INCDIR_OPENVG

指定在构建具有 OpenVG 支持的目标时要添加到 INCLUDEPATH 的 OpenVG 头文件的位置。这个变量的值通常由 qmake 或 qmake.conf 处理，很少需要修改。

如果 OpenVG 实现使用 EGL，那么可能还需要设置 QMAKE_INCDIR_EGL。

## 87. QMAKE_INCDIR_X11

**注意**: 此变量仅在 Unix 平台上使用。

指定在构建 X11 目标时要添加到 INCLUDEPATH 的 X11 头文件路径的位置。这个变量的值通常由 qmake 或 qmake.conf 处理，很少需要修改。

## 88. QMAKE_INFO_PLIST

**注意**: 这个变量只在 macOS、iOS、tvOS 和 watchOS 上使用。

指定要包含在 macOS、iOS、tvOS 和 watchOS 应用程序包中的属性列表文件 `.plist` 的名称。

在 `.plist`文件中，你可以定义一些变量，qmake 会用相关值替换它们:

| 占位符                                           | 效果                                                         |
| ------------------------------------------------ | ------------------------------------------------------------ |
| ${PRODUCT_BUNDLE_IDENTIFIER}, @BUNDLEIDENTIFIER@ | 扩展到目标 bundle 的 bundle 标识符字符串，例如: com.example.myapp。通过连接 QMAKE_TARGET_BUNDLE_PREFIX 和 QMAKE_BUNDLE的值(用句号(.)分隔)来确定。 |
| ${EXECUTABLE_NAME}, @EXECUTABLE@, @LIBRARY@      | 相当于 QMAKE_APPLICATION_BUNDLE_NAME、QMAKE_PLUGIN_BUNDLE_NAME 或 QMAKE_FRAMEWORK_BUNDLE_NAME (取决于创建的目标类型)的值，或者如果前面的值都没有设置，则等于 target 的值。 |
| ${ASSETCATALOG_COMPILER_APPICON_NAME}, @ICON@    | 展开为 ICON 的值。                                           |
| ${QMAKE_PKGINFO_TYPEINFO}, @TYPEINFO@            | 展开为 QMAKE_PKGINFO_TYPEINFO 的值。                         |
| ${QMAKE_FULL_VERSION}, @FULL_VERSION@            | 展开为用三个版本成份表示的 VERSION 值。                      |
| ${QMAKE_SHORT_VERSION}, @SHORT_VERSION@          | 展开为用两个版本成份表示的 VERSION 值。                      |
| ${MACOSX_DEPLOYMENT_TARGET}                      | 展开为 QMAKE_MACOSX_DEPLOYMENT_TARGET 的值。                 |
| ${IPHONEOS_DEPLOYMENT_TARGET}                    | 展开为 QMAKE_IPHONEOS_DEPLOYMENT_TARGET 的值。               |
| ${TVOS_DEPLOYMENT_TARGET}                        | 展开为 QMAKE_TVOS_DEPLOYMENT_TARGET 的值。                   |
| ${WATCHOS_DEPLOYMENT_TARGET}                     | 展开为 QMAKE_WATCHOS_DEPLOYMENT_TARGET 的值。                |

显示详细信息

**注意**:当使用 Xcode 生成器时，上面的 `${var}` 样式的占位符会被 Xcode 构建系统直接替换，qmake 不会处理这些占位符。`@var@` 样式的占位符只适用于 qmake Makefile 生成器，而不适用于 Xcode 生成器。

如果为 iOS 构建，`.plist`文件包含 NSPhotoLibraryUsageDescription 键，那么 qmake 将在构建中包含一个额外的插件，用于添加照片访问支持(例如，QFile/QFileDialog)。关于这个键的更多信息，可以查看自苹果的 `Info.plist`文档。

**注意**::大多数时候，默认 `Info.plist` 已经足够应付开发需要了。

## 89. QMAKE_IOS_DEPLOYMENT_TARGET

**注意**: 这个变量只在 iOS 上使用。

指定应用程序支持的 iOS 硬件的最小版本。

有关更多信息，请参见 Expressing Supported iOS Versions 。

## 90. QMAKE_LFLAGS

指定传递给链接器的通用标志集。如果需要更改用于特定平台或项目类型的标志，请使用其中一个专用变量，而不是此变量。

## 91. QMAKE_LFLAGS_CONSOLE

**注意**: 这个变量只适用于 Windows 平台目标

指定用于构建控制台程序的链接器标志。这个变量的值通常由 qmake 或 qmake.conf 处理，很少需要修改。

## 92. QMAKE_LFLAGS_DEBUG

指定调试生成的链接器标志。这个变量的值通常由 qmake 或 qmake.conf 处理，很少需要修改。

## 93. QMAKE_LFLAGS_PLUGIN

指定用于构建插件的链接器标志。这个变量的值通常由 qmake 或 qmake.conf 处理，很少需要修改。

## 94. QMAKE_LFLAGS_RPATH

**注意**: 此变量仅在 Unix 平台上使用。

指定使用 QMAKE_RPATHDIR 中的值所需的链接器标志。

这个变量的值通常由 qmake 或 qmake.conf 处理，很少需要修改。

## 95. QMAKE_LFLAGS_REL_RPATH

指定在 QMAKE_RPATHDIR 中启用相对路径所需的链接器标志。

这个变量的值通常由 qmake 或 qmake.conf 处理，很少需要修改。

## 96. QMAKE_REL_RPATH_BASE

指定动态链接器理解的字符串为引用可执行文件或库的位置。

这个变量的值通常由 qmake 或 qmake.conf 处理，很少需要修改。

## 97. QMAKE_LFLAGS_RPATHLINK

指定使用 QMAKE_RPATHLINKDIR 中的值所需的链接器标志。

这个变量的值通常由 qmake 或 qmake.conf 处理，很少需要修改。

## 98. QMAKE_LFLAGS_RELEASE

指定发布版本的链接器标志。这个变量的值通常由 qmake 或 qmake.conf 处理，很少需要修改。

## 99. QMAKE_LFLAGS_RELEASE_WITH_DEBUGINFO

指定在 `CONFIG` 中设置 `force_debug_info` 的版本构建的链接器标志。这个变量的值通常由 qmake 或 qmake.conf 处理，很少需要修改。

## 100. QMAKE_LFLAGS_APP

指定用于构建应用程序的链接器标志。这个变量的值通常由 qmake 或 qmake.conf 处理，很少需要修改。

## 101. QMAKE_LFLAGS_SHLIB

指定用于构建共享库的链接器标志。这个变量的值通常由 qmake 或 qmake.conf 处理，很少需要修改。

## 102. QMAKE_LFLAGS_SONAME

指定用于设置共享对象(如.so或.dll)名称的链接器标志。这个变量的值通常由 qmake 或 qmake.conf 处理，很少需要修改。

## 103. QMAKE_LFLAGS_THREAD

指定用于构建多线程项目的链接器标志。这个变量的值通常由 qmake 或 qmake.conf 处理，很少需要修改。

## 104. QMAKE_LFLAGS_WINDOWS

**注意**: 这个变量只适用于 Windows 平台目标

指定用于构建 Windows GUI 项目(即非控制台应用程序)的链接器标志。这个变量的值通常由 qmake 或 qmake.conf 处理，很少需要修改。

## 105. QMAKE_LIBDIR

指定所有项目的库搜索路径列表。这个变量的值通常由 qmake 或 qmake.conf 处理，很少需要修改。

要在项目文件中指定额外的搜索路径，请像下面这样使用 LIBS :

```
LIBS += -L/path/to/libraries
```

## 106. QMAKE_LIBDIR_POST

指定所有项目的系统库搜索路径列表。这个变量的值通常由 qmake 或 qmake.conf 处理，很少需要修改。

## 107. QMAKE_LIBDIR_FLAGS

**注意**: 此变量仅在 Unix 平台上使用。

指定所有带 `-L` 前缀的库目录的位置。这个变量的值通常由 qmake 或 qmake.conf 处理，很少需要修改。

## 108. QMAKE_LIBDIR_EGL

当 EGL 与 OpenGL/ES 或 OpenVG 一起使用时，指定 EGL 库目录的位置。这个变量的值通常由 qmake 或 qmake.conf 处理，很少需要修改。

## 109. QMAKE_LIBDIR_OPENGL

指定 OpenGL 库目录的位置。这个变量的值通常由 qmake 或 qmake.conf 处理，很少需要修改。

如果 OpenGL 实现使用 EGL(大多数 OpenGL/ES 系统)，那么 QMAKE_LIBDIR_EGL 可能也需要设置。

## 110. QMAKE_LIBDIR_OPENVG

指定 OpenVG 库目录的位置。这个变量的值通常由 qmake 或 qmake.conf 处理，很少需要修改。

如果 OpenVG 实现使用 EGL，那么 QMAKE_LIBDIR_EGL可能也需要设置。

## 111. QMAKE_LIBDIR_X11

**注意**: 此变量仅在 Unix 平台上使用。

指定 X11 库目录的位置。这个变量的值通常由 qmake 或 qmake.conf 处理，很少需要修改。

## 112. QMAKE_LIBS

指定每个项目需要链接到的其他库。这个变量的值通常由 qmake 或 qmake.conf 处理，很少需要修改。

要在项目文件中指定库，请使用 LIBS 。

## 113. QMAKE_LIBS_PRIVATE

指定每个项目需要链接到的其他私有库。这个变量的值通常由 qmake 或 qmake.conf 处理，很少需要修改。

要在库项目文件中指定私有库，请使用 LIBS_PRIVATE 。

## 114. QMAKE_LIBS_EGL

在使用 OpenGL/ES 或 OpenVG 构建 Qt 时指定所有 EGL 库。这个变量的值通常由 qmake 或 qmake.conf 处理，很少需要修改。

其通常的值是 `-lEGL`。

## 115. QMAKE_LIBS_OPENGL

指定所有 OpenGL 库。这个变量的值通常由 qmake 或 qmake.conf 处理，很少需要修改。

如果 OpenGL 实现使用 EGL(大多数 OpenGL/ES 系统)，那么 QMAKE_LIBDIR_EGL可能也需要设置。

## 116. QMAKE_LIBS_OPENGL_ES1, QMAKE_LIBS_OPENGL_ES2

这些变量指定了 OpenGL ES1 和 OpenGL ES2 的所有 OpenGL 库。这个变量的值通常由 qmake 或 qmake.conf 处理，很少需要修改。

如果 OpenGL 实现使用 EGL(大多数 OpenGL/ES 系统)，那么 QMAKE_LIBDIR_EGL可能也需要设置。

## 117. QMAKE_LIBS_OPENVG

指定所有 OpenVG 库。这个变量的值通常由 qmake 或 qmake.conf 处理，很少需要修改。通常的值是 `-lOpenVG` 。

一些 OpenVG 引擎是在 OpenGL 上实现的。这将在配置时检测到，QMAKE_LIBS_OPENGL 将隐式地添加到连接 OpenVG 库的 QMAKE_LIBS_OPENVG 中。

如果 OpenVG 实现使用 EGL，那么 QMAKE_LIBDIR_EGL可能也需要设置。

## 118. QMAKE_LIBS_THREAD

**注意**: 此变量仅在 Unix 平台上使用。

指定在构建多线程目标时需要链接的所有库。这个变量的值通常由 qmake 或 qmake.conf 处理，很少需要修改。

## 119. QMAKE_LIBS_X11

**注意**: 此变量仅在 Unix 平台上使用。

指定所有 X11 库。这个变量的值通常由 qmake 或 qmake.conf 处理，很少需要修改。

## 120. QMAKE_LIB_FLAG

如果指定了 `lib` 模板，则此变量不是空的。这个变量的值通常由 qmake 或 qmake.conf 处理，很少需要修改。

## 121. QMAKE_LINK

指定在构建基于应用程序的项目时将使用的链接器。当处理 Makefile 时，只要链接器可执行文件位于 PATH 变量中包含的路径上，就只需要指定链接器可执行文件的文件名。这个变量的值通常由 qmake 或 qmake.conf 处理，很少需要修改。

## 122. QMAKE_LINK_SHLIB_CMD

指定创建共享库时要执行的命令。这个变量的值通常由 qmake 或 qmake.conf 处理，很少需要修改。

## 123. QMAKE_LN_SHLIB

指定创建到共享库的链接时要执行的命令。这个变量的值通常由 qmake 或 qmake.conf 处理，很少需要修改。

## 124. QMAKE_LRELEASE_FLAGS

通过 CONFIG += lrelease 启用时传递给 lrelease 的附加选项列表。

## 125. QMAKE_OBJECTIVE_CFLAGS

指定用于生成项目的目标 C/C++ 编译器标记。除了 QMAKE_CFLAGS 和 QMAKE_CXXFLAGS 之外，还使用了这些标志。

## 126. QMAKE_POST_LINK

指定将 TARGET 连接在一起后要执行的命令。该变量通常为空，因此不执行任何操作。

**注意**:这个变量对 Xcode 项目没有影响。

## 127. QMAKE_PRE_LINK

指定在连接 TARGET 之前要执行的命令。该变量通常为空，因此不执行任何操作。

**注意**:这个变量对 Xcode 项目没有影响。

## 128. QMAKE_PROJECT_NAME

**注意**:此变量仅用于 Visual Studio 项目文件。

在为 IDE 生成项目文件时确定项目的名称。默认值是目标名称。这个变量的值通常由 qmake 处理，很少需要修改。

## 129. QMAKE_PROVISIONING_PROFILE

**注意**: 这个变量只在 macOS、iOS、tvOS 和 watchOS 上使用。

有效配置配置文件的 UUID。与 QMAKE_DEVELOPMENT_TEAM 一起使用，以指定配置配置文件。

**注意**:指定配置配置文件将禁用自动管理的签名。

## 130. QMAKE_MAC_SDK

在 macOS 上构建通用二进制文件时使用这个变量。

## 131. QMAKE_MACOSX_DEPLOYMENT_TARGET

**注意**: 这个变量只在 macOS 上使用。

指定应用程序支持的 macOS 的硬件最小版本。

更多信息，请参阅 macOS Version Dependencies

## 132. QMAKE_MAKEFILE

指定要创建的生成文件的名称。这个变量的值通常由 qmake 或 qmake.conf 处理，很少需要修改。

## 133. QMAKE_QMAKE

包含 qmake 可执行文件的绝对路径。

**注意**:不要尝试重写此变量的值。

## 134. QMAKE_RESOURCE_FLAGS

此变量用于自定义在使用它的每个构建规则中传递给 Resource Compiler 的选项列表。例如，下面的行确保 `-threshold` 和 `-compress` 选项在每次调用 `rcc` 时使用特定的值:

```
QMAKE_RESOURCE_FLAGS += -threshold 0 -compress 9
```

## 135. QMAKE_RPATHDIR

**注意**: 此变量仅在 Unix 平台上使用。

指定在链接时添加到可执行文件中的库路径列表，以便在运行时优先搜索这些路径。

当指定了相对路径时，qmake 将它们转换为动态链接器可以理解的相对于引用的可执行文件或库位置的形式。只有一些平台(目前是基于 Linux 和 Darwin-based 的平台)支持这一点，通过检查是否设置了 QMAKE_REL_RPATH_BASE 可以检测到。

## 136. QMAKE_RPATHLINKDIR

指定静态链接器的库路径列表，以搜索共享库的隐式依赖关系。更多信息，请参见 `ld(1)` 的手册页

## 137. QMAKE_RUN_CC

指定构建对象所需的单个规则。这个变量的值通常由 qmake 或 qmake.conf 处理，很少需要修改。

## 138. QMAKE_RUN_CC_IMP

指定构建对象所需的单个规则。这个变量的值通常由 qmake 或 qmake.conf 处理，很少需要修改。

## 139. QMAKE_RUN_CXX

指定构建对象所需的单个规则。这个变量的值通常由 qmake 或 qmake.conf 处理，很少需要修改。

## 140. QMAKE_RUN_CXX_IMP

指定构建对象所需的单个规则。这个变量的值通常由 qmake 或 qmake.conf 处理，很少需要修改。

## 141. QMAKE_SONAME_PREFIX

如果定义了该变量的值，则使用该变量的值作为一个路径，预先添加到构建的共享库的 SONAME 标识符中。SONAME 是动态链接器稍后用于引用库的标识符。通常，该引用可以是库名或完整库路径。在 macOS、iOS、tvOS 和 watchOS 上，可以使用以下占位符相对指定路径:

| 占位符           | 效果                                                         |
| ---------------- | ------------------------------------------------------------ |
| @rpath           | 展开为当前进程可执行文件或引用库中 LC_RPATH mach-o 命令定义的路径。 |
| @executable_path | 展开为当前进程可执行位置。                                   |
| @loader_path     | 展开为引用的可执行文件或库位置。                             |

在大多数情况下，使用 `@rpath` 就足够了，建议使用:

```
# <project root>/project.pro
QMAKE_SONAME_PREFIX = @rpath
```

但是，前缀也可以使用不同的占位符来指定，或者使用绝对路径，例如下面的路径之一：

```
# <project root>/project.pro
QMAKE_SONAME_PREFIX = @executable_path/../Frameworks
QMAKE_SONAME_PREFIX = @loader_path/Frameworks
QMAKE_SONAME_PREFIX = /Library/Frameworks
```

有关更多信息，请参见关于动态库安装名称的 dyld 文档。

## 142. QMAKE_TARGET

指定项目目标的名称。这个变量的值通常由 qmake 或 qmake.conf 处理，很少需要修改。

## 143. QMAKE_TARGET_COMPANY

仅 Windows 平台使用。指定公司为项目目标;这用于在适用的情况下将公司名称放入应用程序的属性中。只有在设置了 VERSION或 RC_ICONS 变量而没有设置 RC_FILE 和 RES_FILE 变量时，才会使用这个函数。

## 144. QMAKE_TARGET_DESCRIPTION

仅 Windows 平台使用。指定项目目标的说明;它用于在适用的情况下将描述放入应用程序的属性中。只有在设置了 VERSION或 RC_ICONS 变量而没有设置 RC_FILE 和 RES_FILE 变量时，才会使用这个函数。

## 145. QMAKE_TARGET_COPYRIGHT

仅 Windows 平台使用。指定项目目标的版权信息;它用于在适用的情况下将版权信息放入应用程序的属性中。只有在设置了 VERSION或 RC_ICONS 变量而没有设置 RC_FILE 和 RES_FILE 变量时，才会使用这个函数。

## 146. QMAKE_TARGET_PRODUCT

仅 Windows 平台使用。为项目目标指定产品;它用于将产品放入应用程序的属性中。只有在设置了 VERSION或 RC_ICONS 变量而没有设置 RC_FILE 和 RES_FILE 变量时，才会使用这个函数。

## 147. QMAKE_TVOS_DEPLOYMENT_TARGET

**注意**:此变量仅在 tvOS 平台上使用。

指定应用程序支持的 tvOS 的硬件最小版本。

有关更多信息，请参见 Expressing Supported iOS Versions 。

## 148. QMAKE_UIC_FLAGS

此变量用于在使用它的每个构建规则中定制传递给 User Interface Compiler 的选项列表。

## 149. QMAKE_WATCHOS_DEPLOYMENT_TARGET

**注意**:此变量仅在 watchOS 平台上使用。

指定应用程序支持的 watchOS 的硬件最小版本。

有关更多信息，请参见 Expressing Supported iOS Versions 。

## 150. QML_IMPORT_MAJOR_VERSION

指定用于自动生成的 QML 类型注册的主版本。有关更多信息，请参见 Defining QML Types from C++ 。

## 151. QML_IMPORT_MINOR_VERSION

当自动注册 C++ 中定义的 QML 类型时，使用这个次要版本注册模块的附加版本。通常，要注册的次要版本是从元对象推断出来的。

如果元对象没有改变，并且我们仍然想导入一个具有较新的副版本号的 QML 模块，那么我们可以使用这个变量。例如，MyModule 元对象是 1.1 的，但是我们希望导入的模块是 1.3 的。

## 152. QML_IMPORT_VERSION

将 QML_IMPORT_MAJOR_VERSION 和 QML_IMPORT_MINOR_VERSION 指定为 `<major>.<minor>` 版本字符串。

## 153. QML_IMPORT_NAME

指定用于自动生成的 QML 类型注册的模块名称。有关更多信息，请参见 Defining QML Types from C++ 。

## 154. QML_FOREIGN_METATYPES

指定在生成 qmltypes 文件时要考虑的元类型的 JSON 文件。当外部库提供向 QML 公开的类型(直接或作为其他类型的基类型或属性)时使用此属性。Qt 类型将被自动考虑，不需要在这里添加。

## 155. QT

指定项目使用的 Qt模块。每个模块的添加值，请参阅模块文档。

在 c++ 实现级别，使用 Qt 模块使其头文件可以被包含，并使其链接到二进制文件。

默认情况下，QT 包含 `core` 和 `gui`，确保无需进一步配置即可构建标准 GUI 应用程序。

如果我们想在不使用 Qt GUI 模块的情况下构建一个项目，则需要使用 “-=” 操作符排除 GUI 值。下面的代码行将使我们能创建最小的 Qt 项目:

```
QT -= gui # Only the core module is used.
```

如果我们的项目是一个 Qt Designer 插件，那么使用值 uiplugin 来指定要构建的项目是一个库，但是要有对 Qt Designer 的特定插件支持。有关更多信息，请参见 Building and Installing the Plugin 。

## 156. QTPLUGIN

指定要与应用程序链接的静态 Qt 插件的名称列表，以便它们作为内置资源可用。

qmake 会自动添加所使用的 Qt 模块通常需要的插件(参见 QT)。默认设置被调优为最佳的开箱即用体验。有关可用插件的列表和覆盖自动链接的方法，请参阅 Static Plugins 。

当前，当链接共享/动态的 Qt 构建时，或者在链接库时，这个变量没有影响。它可以在以后用于部署动态插件。

## 157. QT_VERSION

包含 Qt 的当前版本。

## 158. QT_MAJOR_VERSION

包含 Qt 的当前主要版本。

## 159. QT_MINOR_VERSION

包含 Qt 的当前次要版本。

## 160. QT_PATCH_VERSION

包含 Qt 的当前补丁版本。

## 161. RC_FILE

仅 Windows 平台使用。指定目标的 Windows 资源文件(.rc)的名称。参阅 Adding Windows Resource Files 。

## 162. RC_CODEPAGE

仅 Windows 平台使用。指定应该在生成的 .rc 文件中指定的代码页。只有在设置了 VERSION或 RC_ICONS 变量而没有设置 RC_FILE 和 RES_FILE 变量时，才会使用这个函数。

## 163. RC_DEFINES

仅 Windows 平台使用。qmake 将这个变量的值作为 RC 预处理器宏(/d 选项)添加。如果未设置此变量，则使用 DEFINES 变量。

```
RC_DEFINES += USE_MY_STUFF
```

## 164. RC_ICONS

仅 Windows 平台使用。指定应该包含在生成的 .rc 文件中的图标。只有当 RC_FILE 和 RES_FILE 变量未设置时使用。关于生成.rc文件的更多细节可以在 平台相关事项 中找到。

## 165. RC_LANG

仅 Windows 平台使用。指定应该在生成的.rc文件中指定的语言。只有在设置了 VERSION或 RC_ICONS 变量而没有设置 RC_FILE 和 RES_FILE 变量时，才会使用。

## 166. RC_INCLUDEPATH

指定传递给 Windows 资源编译器的包含路径。

## 167. RCC_DIR

指定 Qt 资源编译器输出文件的目录。

例如：

```
unix:RCC_DIR = ../myproject/resources
win32:RCC_DIR = c:/myproject/resources
```

## 168. REQUIRES

指定作为条件计算的值列表。如果任何条件为假，qmake 在构建时跳过这个项目(及其子目录)。

**注意**:如果我们想在构建时跳过项目或子项目，建议使用 requires() 函数。

## 169. RESOURCES

指定目标的资源集合文件(`qrc`)的名称。有关资源收集文件的更多信息，请参见Qt资源系统。

## 170. RES_FILE

仅 Windows 平台使用。指定此目标的 Windows 资源编译器输出文件的名称。请参阅 RC_FILE 和 添加 Windows 资源文件。

这个变量的值通常由 qmake 或 qmake.conf 处理，很少需要修改。

## 171. SOURCES

指定项目中所有源文件的名称。

例如：

```
SOURCES = myclass.cpp \
      login.cpp \
      mainwindow.cpp
```

也可参阅：HEADERS

## 172. SUBDIRS

当与 `subdirs` 模板一起使用时，此变量指定所有子目录或包含需要构建的项目部分的项目文件的名称。使用此变量指定的每个子目录必须包含其自己的项目文件。

建议每个子目录中的项目文件具有与子目录本身相同的基名称，因为这样可以省略文件名。例如，如果子目录名为 `myapp`，则该目录中的项目文件应称为 `myapp.pro`。

或者，我们可以在任何目录中指定 `.pro` 文件的相对路径。强烈建议只指定当前项目的父目录或子目录中的路径。

例如：

```
SUBDIRS = kernel \
          tools \
          myapp
```

如果我们需要确保子目录是以特定的顺序构建的，那么可以对相关的 `SUBDIRS` 元素使用 `.depends` 修饰符。

例如：

```
SUBDIRS += my_executable my_library tests doc
my_executable.depends = my_library
tests.depends = my_executable
```

上面的配置确保在 my_executable 之前构建 my_library，在 tests 之前构建 my_executable 。但是，doc 可以与其他子目录并行构建，从而加快构建过程。

**注意**:可以列出多个依赖项，它们都将在依赖于它们的目标之前构建。

**注意**:不建议使用 CONFIG += ordered，因为它会降低多核构建的速度。此设置下，与上面的示例不同的是，即使没有依赖关系，所有的构建也会按顺序进行。

除了定义构建顺序，还可以通过给 SUBDIRS 元素额外的修饰符来修改 SUBDIRS 的默认行为。支持修饰符是:

| 修饰符    | 效果                                                         |
| --------- | ------------------------------------------------------------ |
| .subdir   | 使用指定的子目录，而不是 SUBDIRS 值。                        |
| .file     | 明确指定子项目 `pro` 文件。不能与 `.subdir` 修饰符一起使用。 |
| .depends  | 此子项目依赖于指定的子项目。                                 |
| .makefile | 子项目的 makefile。仅在使用 makefile 的平台上可用。          |
| .target   | 用于与此子项目相关的 makefile 目标的基本字符串。仅在使用 makefile 的平台上可用。 |

例如，定义两个子目录，这两个都在不同的目录下的 SUBDIRS 值，其中一个子目录必须在另一个之前建立:

```
SUBDIRS += my_executable my_library
my_executable.subdir = app
my_executable.depends = my_library
my_library.subdir = lib
```

## 173. TARGET

指定目标文件的名称。默认情况下为项目文件去掉后缀的名称。

例如：

```
TEMPLATE = app
TARGET = myapp
SOURCES = main.cpp
```

上面的项目文件将在 unix 上生成名为 `myapp` 的可执行文件，在 Windows 上生成名为 `myapp.exe` 的可执行文件。

## 174. TARGET_EXT

指定 TARGET 的扩展名。这个变量的值通常由 qmake 或 qmake.conf 处理，很少需要修改。

## 175. TARGET_x

用主版本号指定 TARGET 的扩展名。这个变量的值通常由 qmake 或 qmake.conf 处理，很少需要修改。

## 176. TARGET_x.y.z

使用版本号指定 TARGET 的扩展名。这个变量的值通常由 qmake 或 qmake.conf 处理，很少需要修改。

## 177. TEMPLATE

指定生成项目时要使用的模板名称。允许的值是:

|选项|描述|
|app|创建用于构建应用程序的 Makefile(缺省值)。有关更多信息，请参见Building an Application。|
|lib|创建用于构建库的 Makefile。有关更多信息，请参见Building a Library。|
|subdirs|创建用于在子目录中构建目标的 Makefile。子目录是使用 SUBDIRS 变量指定的。|
|aux|创建一个不构建任何东西的 Makefile。如果不需要调用编译器来创建目标，请使用此选项;例如，因为我们的项目是用解释语言编写的。
**注意**:此模板类型仅适用于基于 makefile 的生成器。特别是，它不能与 vcxproj 和 Xcode 生成器一起工作。|
|vcapp|仅 Windows 使用。为 Visual Studio 创建一个应用程序项目。有关更多信息，请参见Creating Visual Studio Project Files。|
|vclib|仅 Windows 使用。为 Visual Studio 创建一个库项目。|

例如：

```
TEMPLATE = lib
SOURCES = main.cpp
TARGET = mylib
```

可以通过使用 `-t` 命令行选项指定新的模板类型来覆盖模板。这将在处理 `.pro` 文件后覆盖模板类型。对于使用模板类型来确定如何构建项目的 `.pro` 文件，有必要在命令行中声明 TEMPLATE，而不是使用 `-t` 选项。

## 178. TRANSLATIONS

指定翻译(.ts)文件列表，其中包含将用户界面文本翻译成非本地语言的内容。

TRANSLATIONS 中的翻译文件将由 lrelease 和 lupdate 工具处理。如果只希望 lrelease 处理文件，请使用 EXTRA_TRANSLATIONS 。

我们可以使用 CONFIG += lrelease 在构建过程中自动编译文件，并使用 CONFIG += lrelease embed_translations 使它们在 Qt 资源系统中可用。

有关 Qt 的国际化 (i18n) 和本地化 (l10n) 的更多信息，请参见 Qt Linguist Manual 。

## 179. UI_DIR

指定应该放置 uic 的所有中间文件的目录。

例如：

```
unix:UI_DIR = ../myproject/ui
win32:UI_DIR = c:/myproject/ui
```

## 180. VERSION

如果指定了 app template，则指定应用程序的版本号;如果指定了 lib 模板，则指定库的版本号。

在 Windows 上，如果没有设置 RC_FILE 和 RES_FILE 变量，则触发 `.rc` 文件的自动生成。生成的 .rc 文件将在 FILEVERSION 和 PRODUCTVERSION 条目中填充主、次、补丁级别和构建号。每个数字必须在 0 到 65535 之间。关于生成 .rc 文件的更多细节可以在 平台相关事项 中找到。

例如：

```
win32:VERSION = 1.2.3.4 # major.minor.patch.build
else:VERSION = 1.2.3    # major.minor.patch
```

## 181. VERSION_PE_HEADER

仅 Windows 平台使用。指定版本号，Windows 链接器通过 `/VERSION` 选项将其放入 .exe 或 .dll 文件的头文件中。只能指定主版本和次版本。如果没有设置 VERSION_PE_HEADER，则从 VERSION(如果设置了)退回到主版本和次版本。

```
VERSION_PE_HEADER = 1.2
```

## 182. VER_MAJ

如果指定了 `lib` template，则指定库的主版本号。

## 183. VER_MIN

如果指定了 `lib` template，则指定库的副版本号。

## 184. VER_PAT

如果指定了 `lib` template，则指定库的补丁版本号。

## 185. VPATH

告诉 qmake 在哪里搜索无法打开的文件。例如，如果 qmake 查找 SOURCES，并发现一个无法打开的条目，它会查看整个 VPATH 列表，看看是否能自己找到该文件。

也可以参阅 DEPENDPATH 。

## 186. WINDOWS_TARGET_PLATFORM_VERSION

指定目标 Windows 版本;这对应于 `vcxproj` 文件中的标签 `WindowsTargetPlatformVersion` 。

在桌面 Windows 中，默认值是环境变量 WindowsSDKVersion 的值。

在 Universal Windows Platform (UWP) 上，默认值是环境变量 `UCRTVERSION` 的值。

## 187. WINDOWS_TARGET_PLATFORM_MIN_VERSION

指定 Windows 目标平台的最小版本;这对应于 `vcxproj` 文件中的标签 `WindowsTargetPlatformMinVersion` 。

默认为 WINDOWS_TARGET_PLATFORM_VERSION。

## 188. WINRT_MANIFEST

指定要传递到 UWP 上的应用程序清单的参数。允许的值是:

| 成员                | 描述                                                         |
| ------------------- | ------------------------------------------------------------ |
| architecture        | 目标体系结构。默认为 VCPROJ_ARCH 。                          |
| background          | Tile 背景颜色。默认为绿色。                                  |
| capabilities        | 指定要添加到功能列表中的功能。                               |
| capabilities_device | 指定要添加到功能列表中的设备功能(位置、网络摄像头等)。       |
| CONFIG              | 指定处理输入清单文件的附加标志。目前，verbatim 是唯一可用的选项。 |
| default_language    | 应用程序的默认语言代码。默认为“en”。                         |
| dependencies        | 指定包所需的依赖关系。                                       |
| description         | 包的描述。默认为 `Default package description`。             |
| foreground          | 平铺前景(文本)颜色。默认为 light。                           |
| iconic_tile_icon    | 图像文件的 iconic tile 模板图标。默认由 mkspec 提供。        |
| iconic_tile_small   | 图像文件为小 iconic tile 模板图标。默认由 mkspec 提供。      |
| identity            | 应用程序的唯一 ID。默认情况下重用现有生成的清单的 UUID，如果没有，则生成一个新的 UUID。 |
| logo_30x30          | Logo 图像文件大小为 30 x 30 像素。                           |
| logo_41x41          | Logo 图像文件大小为 41 x 41 像素。此参数已废弃。             |
| logo_70x70          | Logo 图像文件大小为 70 x 70 像素。                           |
| logo_71x71          | Logo 图像文件大小为 71 x 71 像素。此参数已废弃。             |
| logo_150x150        | Logo 图像文件大小为 150 x 150 像素。所有 Windows Store 应用程序平台都支持这一功能。 |
| logo_310x150        | Logo 图像文件大小为 310 x 150 像素。所有 Windows Store 应用程序平台都支持这一功能。 |
| logo_310x310        | Logo 图像文件大小为 310 x 310 像素。所有 Windows Store 应用程序平台都支持这一功能。 |
| logo_620x300        | 启动画面的图像文件大小为 620 x 300 像素。                    |
| logo_480x800        | 启动画面的图像文件大小为 620 x 300 像素。此参数已废弃。      |
| logo_large          | 大型 logo 图像文件。这个必须是 150 x 150 像素。支持所有 Windows 商店应用平台。默认由 mkspec 提供。 |
| logo_medium         | 中等 logo 图像文件。这个必须是 70 x 70 像素。默认由 mkspec 提供。 |
| logo_small          | 小型 logo 图像文件。这个必须是 30 x 30 像素。默认由 mkspec 提供。 |
| logo_splash         | 启动屏幕图像文件。图像的像素大小必须为 620 x 300。默认由 mkspec 提供。 |
| logo_store          | 为 Windows 商店的标志图像文件。默认由 mkspec 提供。          |
| logo_wide           | 宽 logo 图像文件。这个必须是 310 x 150 像素。支持所有 Windows 商店应用平台。默认由 mkspec 提供。 |
| name                | 显示给用户的包的名称。默认为 TARGET 的值。                   |
| phone_product_id    | 该产品的 GUID。此参数已过时。                                |
| phone_publisher_id  | 发布者的GUID。此参数已过时。                                 |
| publisher           | 显示发布服务器的名称。默认为 `Default publisher display name`。 |
| publisher_id        | 发布者的专有名称(默认为: `CN=MyCN` )。                       |
| target              | 目标的名称(.exe)。默认为 TARGET 的值。                       |
| version             | 包的版本号。默认为1.0.0.0。                                  |
| minVersion          | 运行包所需的最低 Windows 版本。默认为 WINDOWS_TARGET_PLATFORM_VERSION 。 |
| maxVersionTested    | 软件包测试的最大 Windows 版本。默认为 WINDOWS_TARGET_PLATFORM_MIN_VERSION 。 |

显示详细信息

我们可以使用这些值的任何组合。

例如：

```
WINRT_MANIFEST.publisher = MyCompany
WINRT_MANIFEST.logo_store = someImage.png
WINRT_MANIFEST.capabilities += internetClient
WINRT_MANIFEST.capabilities_device += location
```

另外，可以使用 WINRT_MANIFEST 指定输入清单文件。

例如：

```
WINRT_MANIFEST = someManifest.xml.in
```

如果不处理输入清单文件，而只是将其复制到目标目录，则需要设置逐字配置。

```
WINRT_MANIFEST = someManifest.xml.in
WINRT_MANIFEST.CONFIG += verbatim
```

## 189. YACCSOURCES

指定要包含在项目中的 Yacc 源文件列表。所有依赖项、头文件和源文件将自动包含在项目中。

例如：

```
YACCSOURCES = moc.y
```

## 190. *PRO_FILE*

包含正在使用的项目文件的路径。

例如，下面一行将项目文件的位置打印到控制台:

```
message($$_PRO_FILE_)
```

**注意**:不要尝试重写此变量的值。

## 191. *PRO_FILE_PWD*

包含到包含正在使用的项目文件的目录的路径。

例如，下面一行将把包含项目文件的目录的位置打印到控制台:

```
message($$_PRO_FILE_PWD_)
```

**注意**:不要尝试重写此变量的值。

# 参考手册>替换函数

qmake 提供了在配置过程中处理变量内容的函数。这些函数称为替换函数。通常，它们返回可以分配给其他变量的值。可以通过在函数前面加上 `$$` 操作符来获得这些值。替换函数可以分为内置函数和函数库。

也可参阅 测试函数

## 1. 内置的替换函数

基本替换函数被实现为内置函数。

### 1.1. absolute_path(path[, base])

返回 path 的绝对路径。

如果没有指定 `base`，则使用当前目录作为起始目录。如果是相对路径，则在使用前相对于当前目录进行解析。

例如，下面的调用返回字符串 “`/home/toby/myproject/readme.txt`“:

```
message($$absolute_path("readme.txt", "/home/toby/myproject"))
```

这个函数是在 Qt 5.0 中引入的。

也可参阅 clean_path() 和 relative_path() 。

### 1.2. basename(variablename)

返回在 `variablename` 中指定的文件的基本名称。

例如：

```
FILE = /etc/passwd
FILENAME = $$basename(FILE) #passwd
```

### 1.3. cat(filename[, mode])

返回 `filename` 的内容。我们可以为 `mode` 指定以下选项:

| 选项  | 说明                                                         |
| ----- | ------------------------------------------------------------ |
| blob  | 以一个值的形式返回文件的全部内容                             |
| lines | 作为单独的值返回每一行(没有行结束)                           |
| true  | (默认值)和 `false` 返回文件内容作为单独的值，根据 qmake 值列表分割规则进行分割(如变量分配)。如果 `mode` 是 `false`，则将只包含换行字符的值插入到列表中，以指示换行符在文件中的位置。 |

### 1.4. clean_path(path)

返回目录分隔符规范化(转换为”`/`“)、删除冗余分隔符并且”`.`“和”`..`“都被(尽可能地)处理了的 `path`。这个函数是对 QDir::cleanPath 的封装。

这个函数是在 Qt 5.0 中引入的。

也可参阅 absolute_path(), relative_path(), shell_path(), system_path()

### 1.5. dirname(file)

返回指定 `file` 的目录名部分。例如:

```
FILE = /etc/X11R6/XF86Config
DIRNAME = $$dirname(FILE) #/etc/X11R6
```

### 1.6. enumerate_vars

返回所有已定义变量名的列表。

这个函数是在 Qt 5.0 中引入的。

### 1.7. escape_expand(arg1 [, arg2 …, argn])

接受任意数量的参数。它为每个参数展开转义序列 `\n`、`\r`、`\t`，并以列表的形式返回参数。

注意:如果按字面意思指定要展开的字符串，则需要对反斜杠进行转义，如下面的代码片段所示:

```
message("First line$$escape_expand(\\n)Second line")
```

### 1.8. find(variablename, substr)

返回 variablename 中与正则表达式 substr 匹配的所有值。

```
MY_VAR = one two three four
MY_VAR2 = $$join(MY_VAR, " -L", -L) -Lfive
MY_VAR3 = $$member(MY_VAR, 3) $$find(MY_VAR, t.*)
message($$MY_VAR2)
message($$MY_VAR3)
```

上面示例的输出：

```
Project MESSAGE: -Lone -Ltwo -Lthree -Lfour -Lfive
Project MESSAGE: four two three
```

### 1.9. files(pattern[, recursive=false])

展开指定的通配符模式并返回文件名列表。如果 `recursive` 为 `true`，则此函数递归执行子目录。

### 1.10. first(variablename)

返回 `variablename` 的第一个值。

例如，以下调用返回 `firstname`:

```
CONTACT = firstname middlename surname phone
message($$first(CONTACT))
```

也可参阅 take_first(), last().

### 1.11. format_number(number[, options…])

以 `options` 指定的格式返回 `number`。我们可以指定以下选项:

| 选项       | 说明                                                     |
| ---------- | -------------------------------------------------------- |
| ibase=n    | 设置输入的基数为 n                                       |
| obase=n    | 将输出的基数设置为 n                                     |
| width=n    | 设置输出的最小宽度为 n。如果输出小于 width，则用空格填充 |
| zeropad    | 用零代替空格填充输出                                     |
| padsign    | 在输出的正值前加一个空格                                 |
| alwayssign | 在输出的正值前加上一个加号                               |
| leftalign  | 将padding放在输出值的右侧                                |

目前不支持浮点数。

例如，下面的调用将十六进制数字 BAD 转换为 002989:

```
message($$format_number(BAD, ibase=16 width=6 zeropad))
```

这个函数是在 Qt 5.0 中引入的。

### 1.12. fromfile(filename, variablename)

将 filename 当做一个 qmake 项目文件，并返回分配给 `variablename` 的值。

也可参阅 infile()

### 1.13. getenv(variablename)

返回环境变量 `variablename` 的值。这基本上等同于 `$$(variablename)` 语法。但是，`getenv` 函数支持名称中带有圆括号的环境变量。

这个函数是在 Qt 5.0 中引入的。

### 1.14. join(variablename, glue, before, after)

将 `variablename` 和 `glue` 的值连接起来。如果这个值不是空的，这个函数会在这个值前面加上 `before` 和在后面加上 `after`。`variablename` 是唯一必须的字段，其他字段默认为空字符串。如果需要在 `glue`、`before` 或 `after` 中对空格进行编码，则必须引用它们。

### 1.15. last(variablename)

返回 `variablename` 的最后一个值。

例如，以下调用返回 `phone`:

```
CONTACT = firstname middlename surname phone
message($$last(CONTACT))
```

也可参阅 take_last(), first()

### 1.16. list(arg1 [, arg2 …, argn])

接受任意数量的参数。它创建一个唯一命名的变量，该变量包含参数列表，并返回该变量的名称。可以使用该变量编写循环，如下面的代码片段所示：

```
for(var, $$list(foo bar baz)) {
    ...
}
```

替换:

```
values = foo bar baz
for(var, values) {
    ...
}
```

### 1.17. lower(arg1 [, arg2 …, argn])

接受任意数量的参数并将它们转换为小写。

也可参阅 upper()

### 1.18. member(variablename [, start [, end]])

返回 `variablename` 列表中的值，可以使用 `start` 指定零开始的元素索引，使用 `end` 指定结束元素的索引（该函数会包含 `start` 和 `end` 元素）。

如果 `start` 未指定，则使用其默认值 0，此时该函数的功能相当于 `$$first(variablename)` 。

如果 `end` 未指定，则其默认值 将与 `start` 相等，此用法表示简单的数组索引，因为将只返回一个元素。

也可以在单个参数中指定开始和结束，数字由两个句点分隔。

负数表示从列表末尾开始的索引，而 `-1` 是最后一个元素。

如果任一索引范围越界，则返回空列表。

如果 `end` 小于 `start`，则按相反顺序返回元素。

**注意**：结束（`end`）索引是包含的和无序的，这意味着只有当索引无效时(即输入变量为空时)才会返回一个空列表。

也可参阅 str_member()

### 1.19. num_add(arg1 [, arg2 …, argn])

接受任意数量的数值参数并将它们相加，返回和。

可以隐式支持减法，因为可以简单地在数值前加一个减号来对其进行取负值 :

```
sum = $$num_add($$first, -$$second)
```

如果操作数可能已经是负数，则需要执行另一个额外的步骤来规范化数字:

```
second_neg = -$$second
second_neg ~= s/^--//
sum = $$num_add($$first, $$second_neg)
```

这个函数是在 Qt 5.8 中引入的。

### 1.20. prompt(question [, decorate])

显示指定的 `question`，并返回从 stdin 读取的值。

如果 `decorate` 为true(缺省值)，`question` 将获得一个通用的前缀和后缀，将其标识为提示符。

### 1.21. quote(string)

将整个 `string` 转换为单个实体并返回结果。这只是将字符串括在双引号中的一种特殊方式。

### 1.22. re_escape(string)

用反斜杠转义的每个特殊正则表达式字符返回 `string`。这个函数是 `QRegExp::escape`的包装器。

### 1.23. read_registry(tree, key[, flag])

返回目录树 `tree` 中的注册表项 `key` 的值。

仅支持目录 `HKEY_CURRENT_USER (HKCU)` 和 `HKEY_LOCAL_MACHINE (HKLM)`

```
flag` 的值可以是 `WOW64_32KEY (32)` 或 `WOW64_64KEY (64)
```

注意:此功能仅在 Windows 主机上可用。

这个函数是在 Qt 5.12.1 中引入的。

### 1.24. relative_path(filePath[, base])

返回 `filePath` 相对于 `base` 的路径。

如果未指定 `base`，则为当前项目目录。如果是相对的，则在使用前相对于当前项目目录进行解析。

如果 `filePath` 是相对的，它首先根据基本目录解析；在这种情况下，此函数实际上充当 `$$clean_path()` 。

这个函数是在 Qt 5.0 中引入的。

也可参见 absolute_path(), clean_path()

### 1.25. replace(string, old_string, new_string)

用 `string` 提供的变量内容中的 `new_string` 替换 `old_string` 的每个实例。例如，代码：

```
MESSAGE = This is a tent.
message($$replace(MESSAGE, tent, test))
```

输出：

```
This is a test.
```

### 1.26. resolve_depends(variablename, prefix)

这是一个我们通常不需要的内部功能。

这个函数是在 Qt 5.0 中引入的。

### 1.27. reverse(variablename)

以相反顺序返回 `variablename` 的值。

这个函数是在 Qt 5.0 中引入的。

### 1.28. section(variablename, separator, begin, end)

返回值 `variablename` 的一部分。这个函数是 `QString::section` 的包装器。

例如，下面的调用输出 surname:

```
CONTACT = firstname:middlename:surname:phone
message($$section(CONTACT, :, 2, 2))
```

### 1.29. shadowed(path)

将项目源目录的路径映射到构建目录。对于源代码内构建，该函数返回 `path` 。如果 `path` 指向源树之外，则返回一个空字符串。

这个函数是在 Qt 5.0 中引入的。

### 1.30. shell_path(path)

将 `path` 中的所有目录分隔符转换为与构建项目时使用的 shell(即由 make 工具调用的 shell)兼容的分隔符。例如，当使用 Windows shell 时，斜杠将转换为反斜杠。

这个函数是在 Qt 5.0 中引入的。

也可参阅 system_path()

### 1.31. shell_quote(arg)

为构建项目时使用的 shell 引用 `arg` 。

这个函数是在 Qt 5.0 中引入的。

也可参阅 system_quote()

### 1.32. size(variablename)

返回 `variablename` 的值的数目。

也可参阅 str_size()

### 1.33. sort_depends(variablename, prefix)

这是一个我们通常不需要的内部功能。

这个函数是在 Qt 5.0 中引入的。

### 1.34. sorted(variablename)

返回 `variablename` 中的值列表，其中项按 ASCII 升序排序。

在 format_number() 函数的帮助下，可以通过将值零填充到固定长度来完成数字排序。

这个函数是在 Qt 5.8 中引入的。

### 1.35. split(variablename, separator)

将 variablename 的值分割为单独的值，并以列表的形式返回它们。这个函数是 QString::split 的包装器。

例如:

```
CONTACT = firstname:middlename:surname:phone
message($$split(CONTACT, :))
```

### 1.36. sprintf(string, arguments…)

函数用 arguments 以逗号分隔的列表中的参数替换 string 中的%1-%9，并返回处理过的字符串。

### 1.37. str_member(arg [, start [, end]])

这个函数与 member() 相同，只是它操作的是字符串值而不是列表变量，因此索引引用字符位置。

这个函数可以用来实现许多常见的字符串切片操作:

```
# $$left(VAR, len)
left = $$str_member(VAR, 0, $$num_add($$len, -1))

# $$right(VAR, len)
right = $$str_member(VAR, -$$num, -1)

# $$mid(VAR, off, len)
mid = $$str_member(VAR, $$off, $$num_add($$off, $$len, -1))

# $$mid(VAR, off)
mid = $$str_member(VAR, $$off, -1)

# $$reverse(VAR)
reverse = $$str_member(VAR, -1, 0)

显示详细信息
```

**注意**: 在这些实现中，len 为 0 的参数需要单独处理。

也可参阅 member(), num_add().

这个函数是在 Qt 5.8 中引入的。

### 1.38. str_size(arg)

返回参数中的字符数。

也可参阅 size().

这个函数是在 Qt 5.8 中引入的。

### 1.39. system(command[, mode[, stsvar]])

我们可以使用 system 函数的这个变体从命令中获取 stdout，并将其分配给一个变量。

例如:

```
UNAME = $$system(uname -s)
contains( UNAME, [lL]inux ):message( This looks like Linux ($$UNAME) to me )
```

与 `$$cat()` 类似，mode 参数将 blob、lines、true 和 false 作为值。但是，传统的分词规则 (如empty 和 true 或 false) 略有不同。

如果传递 stsvar，命令的退出状态将存储在该变量中。如果命令崩溃，状态将为 -1，否则为该命令选择的非负退出代码。通常，将状态与零(成功)进行比较就足够了。

请参阅 system() 的测试变体。

### 1.40. system_path(path)

将 path 中的所有目录分隔符转换为与 system() 函数用于调用命令的 shell 兼容的分隔符。例如，在 Windows shell 中将斜杠转换为反斜杠。

这个函数是在 Qt 5.0 中引入的。

也可参阅 shell_path()

### 1.41. system_quote(arg)

为 system() 函数使用的 shell 引用 arg。

这个函数是在 Qt 5.0 中引入的。

也可参阅 shell_quote()

### 1.42. take_first(variablename)

返回 variablename 的第一个值，并将其从源变量中删除。

例如，这为实现队列提供了便利。

这个函数是在 Qt 5.8 中引入的。

也可参阅 take_last(), first().

### 1.43. take_last(variablename)

返回 variablename 的最后一个值，并将其从源变量中删除。

例如，这为实现堆栈提供了便利。

这个函数是在 Qt 5.8 中引入的。

也可参阅 take_first(), last().

### 1.44. unique(variablename)

返回 variablename 中删除重复条目的值列表。例如:

```
ARGS = 1 2 3 2 5 1
ARGS = $$unique(ARGS) #1 2 3 5
```

### 1.45. upper(arg1 [, arg2 …, argn])

接受任意数量的参数并将它们转换为大写。

也可参阅 lower()

### 1.46. val_escape(variablename)

转义 variablename 的值，使其能够解析为 qmake 代码。

这个函数是在 Qt 5.0 中引入的。

# 参考手册>测试函数

测试函数返回一个布尔值，我们可以在范围的条件部分测试该值。测试函数可以分为内置函数和函数库。

也可参阅 替换函数()

## 1. 内置的测试函数

基本测试函数被实现为内置函数。

```
cache(variablename, [set|add|sub] [transient] [super|stash], [source variablename])
```

这是一个我们通常不需要的内部功能。

这个函数是在 Qt 5.0 中引入的。

### 1.1. CONFIG(config)

这个函数可以用来测试 CONFIG() 变量中的变量。这与 scopes 相同，但是有一个额外的优点，即可以传递第二个参数来测试活动配置。由于值的顺序在 CONFIG 变量中很重要(也就是说，最后一组值将被认为是互斥值的活动配置)，第二个参数可以用来指定一组要考虑的值。例如:

```
CONFIG = debug
CONFIG += release
CONFIG(release, debug|release):message(Release build!) #will print
CONFIG(debug, debug|release):message(Debug build!) #no print
```

因为 release 被认为是活动设置(用于特性解析)，所以它将是用来生成构建文件的配置。在常见情况下，不需要第二个参数，但对于特定的互斥测试来说，它是很重要的。

### 1.2. contains(variablename, value)

如果变量 variablename 包含 value 值，则成功;否则失败。可以为参数 value 指定一个正则表达式。

我们可以使用范围检查此函数的返回值。

例如:

```
contains( drivers, network ) {
    # drivers contains 'network'
    message( "Configuring for network build..." )
    HEADERS += network.h
    SOURCES += network.cpp
}
```

只有当 drivers 变量包含值 network 时，范围的内容才会被处理。如果是这种情况，将向 HEADERS 和 SOURCES 变量添加适当的文件。

### 1.3. count(variablename, number)

如果变量 variablename 包含指定 number 值的列表，则成功;否则失败。

此函数用于确保只有在变量包含正确数量的值时才处理范围内的声明。例如:

```
options = $$find(CONFIG, "debug") $$find(CONFIG, "release")
count(options, 2) {
    message(Both release and debug specified.)
}
```

### 1.4. debug(level, message)

检查 qmake 是否在指定的调试级别上运行。如果是，则返回 true 并打印一条调试消息。

### 1.5. defined(name[, type])

测试是否定义了函数或变量 name。如果 type 被省略，则检查所有功能。若要只检查变量或特定类型的函数，请指定 type。它可以有以下值:

- test 只检查测试函数
- replace 只检查替换功能
- var 只检查变量

### 1.6. equals(variablename, value)

测试 variablename 是否等于字符串 value。

例如：

```
TARGET = helloworld
equals(TARGET, "helloworld") {
    message("The target assignment was successful.")
}
```

### 1.7. error(string)

这个函数从不返回值。qmake 将 string 作为错误消息显示给用户并退出。此函数仅用于不可恢复的错误。

例如：

```
error(An error has occurred in the configuration process.)
```

### 1.8. eval(string)

使用 qmake 语法规则计算字符串的内容并返回 true。可以在字符串中使用定义和赋值来修改现有变量的值或创建新的定义。

例如：

```
eval(TARGET = myapp) {
    message($$TARGET)
}
```

**注意**:可以使用引号来分隔字符串，如果不需要返回值，可以丢弃它。

### 1.9. exists(filename)

测试具有给定 filename 的文件是否存在。如果该文件存在，则函数成功;否则失败。如果为 filename 指定了正则表达式，如果有任何文件与指定的正则表达式匹配，则此函数成功。

例如：

```
exists( $(QTDIR)/lib/libqt-mt* ) {
      message( "Configuring for multi-threaded Qt..." )
      CONFIG += thread
}
```

**注意**:“/”应该用作目录分隔符，不管使用的是什么平台。

### 1.10. export(variablename)

将 variablename 的当前值从函数的本地上下文导出到全局上下文。

### 1.11. for(iterate, list)

开始循环，遍历 list 中的所有值，依次将 iterate 设置为每个列表中的值。方便起见，如果 list 为 1…10 然后 iterate 将遍历从 1 到 10 的值。

例如：

```
LIST = 1 2 3
for(a, LIST):exists(file.$${a}):message(I see a file.$${a}!)
```

### 1.12. greaterThan(variablename, value)

测试 variablename 的值大于 value。首先，这个函数尝试进行数值比较。如果至少有一个操作数不能转换，则此函数将执行字符串比较。

例如：

```
ANSWER = 42
greaterThan(ANSWER, 1) {
    message("The answer might be correct.")
}
```

不可能直接将两个数字作为字符串进行比较。作为一种解决方法，可以构造带有非数字前缀的临时值，并对它们进行比较。

例如：

```
VALUE = 123
TMP_VALUE = x$$VALUE
greaterThan(TMP_VALUE, x456): message("Condition may be true.")
```

也可参阅 lessThan()

### 1.13. if(condition)

评估 condition。它用于对布尔表达式进行分组。

例如:

```
if(linux-g++*|macx-g++*):CONFIG(debug, debug|release) {
    message("We are on Linux or Mac OS, and we are in debug mode.")
}
```

### 1.14. include(filename)

在当前项目中包含 filename 指定的文件内容。如果包含 filename，则此函数成功;否则失败。包含的文件会立即被处理。

我们可以使用此函数作为作用域的条件来检查是否包含该文件。例如:

```
include( shared.pri )
OPTIONS = standard custom
!include( options.pri ) {
    message( "No custom build options specified" )
OPTIONS -= custom
}
```

### 1.15. infile(filename, var, val)

如果文件 filename(当 qmake 自身解析时)包含值为 val 的变量 var，则成功;否则失败。如果未指定 val，该函数将测试文件中是否分配了 var。

### 1.16. isActiveConfig

这是 CONFIG 函数的别名。

### 1.17. isEmpty(variablename)

如果变量 variablename 为空，则成功;否则失败。这相当于 `count(variablename，0)`。

例如：

```
isEmpty( CONFIG ) {
CONFIG += warn_on debug
}
```

### 1.18. isEqual

这是 equals 函数的别名。

### 1.19. lessThan(variablename, value)

测试 variablename 的值小于 value。工作机制类似于 greaterThan() 。

例如：

```
ANSWER = 42
lessThan(ANSWER, 1) {
    message("The answer might be wrong.")
}
```

### 1.20. load(feature)

加载 feature 指定的特性文件(.prf)，除非特性已经加载。

### 1.21. log(message)

在控制台上打印一条消息。与 message 函数不同，它既不添加文本也不添加换行符。

这个函数是在 Qt 5.0 中引入的。

也可参阅 message().

### 1.22. message(string)

总是成功，并将 string 作为一条通用消息显示给用户。与 error() 函数不同，此函数允许继续处理。

```
message( "This is a message" )
```

上面的行会将“This is a message”写入控制台。使用引号是可选的，但建议使用。

**注意**:默认情况下，将为 qmake 为给定项目生成的每个 Makefile 写入消息。如果我们想确保每个项目的消息只出现一次，那么测试 build_pass 变量与一个范围，以便在构建期间过滤掉消息。例如:

```
!build_pass:message( "This is a message" )
```

### 1.23. mkpath(dirPath)

创建目录路径 dirPath。此函数是QDir::mkpath
函数的包装器。

这个函数是在 Qt 5.0 中引入的。

### 1.24. requires(condition)

评估 condition。如果条件为假，qmake 在构建时跳过这个项目(及其子目录)。

**注意**: 也可以为此使用 REQUIRES 变量。但是，我们建议使用此函数。

### 1.25. system(command)

在二级shell中执行给定的 command。如果命令以零退出状态返回，则成功;否则失败。我们可以使用范围检查此函数的返回值。

例如：

```
system("ls /bin"): HAS_BIN = TRUE
```

也可参见 system() 的替换变量。

### 1.26. touch(filename, reference_filename)

更新 filename 的时间戳以匹配 reference_filename 的时间戳。

这个函数是在 Qt 5.0 中引入的。

### 1.27. unset(variablename)

从当前上下文中删除 variablename 。

例如：

```
NARF = zort
unset(NARF)
!defined(NARF, var) {
    message("NARF is not defined.")
}
```

### 1.28. versionAtLeast(variablename, versionNumber)

测试来自 variablename 的版本号是否大于或等于 versionNumber。版本号被认为是由“.”分隔的非负的十进制数字序列;字符串的任何非数值尾部都将被忽略。从左到右分段进行比较;如果一个版本是另一个版本的前缀，则认为它更小。

这个函数是在 Qt 5.10 中引入的。

### 1.29. versionAtMost(variablename, versionNumber)

测试来自 variablename 的版本号是否小于或等于 versionNumber。工作机制类似 versionAtLeast().

这个函数是在 Qt 5.10 中引入的。

### 1.30. warning(string)

总是成功，并将 string 作为警告消息显示给用户。

### 1.31. write_file(filename, [variablename, [mode]])

将 variablename 的值写入名为 filename 的文件，每个值在单独的行上。如果没有指定 variablename，则创建一个空文件。如果 mode 是 append 并且该文件已经存在，则追加而不是替换它。

这个函数是在 Qt 5.0 中引入的。

## 2. 测试函数库

复杂的测试函数是在 .prf 文件库中实现的。

### 2.1. packagesExist(packages)

使用 PKGCONFIG 机制来确定在项目解析时给定的包是否存在。

这对于可选地启用或禁用特性非常有用。例如:

```
packagesExist(sqlite3 QtNetwork QtDeclarative) {
    DEFINES += USE_FANCY_UI
}
```

然后在代码中使用:

```
#ifdef USE_FANCY_UI
    // Use the fancy UI, as we have extra packages available
#endif
```

### 2.2. prepareRecursiveTarget(target)

通过准备一个遍历所有子目录的目标，促进类似于 install 目标的项目范围目标的创建。例如:

```
TEMPLATE = subdirs
SUBDIRS = one two three
prepareRecursiveTarget(check)
```

在 `.CONFIG` 中指定 `have_no_default` 或 `no_<target>_target` 的子目录将被排除在这个目标之外:

```
two.CONFIG += no_check_target
```

我们必须手动将准备好的目标添加到 QMAKE_EXTRA_TARGETS:

```
QMAKE_EXTRA_TARGETS += check
```

为了使目标全局化，上面的代码需要包含到每个 subdirs 子项目中。另外，为了让这些目标做任何事情，非 subdirs 子项目需要包含各自的代码。实现这一点最简单的方法是创建一个自定义特性文件。例如:

```
# <project root>/features/mycheck.prf
equals(TEMPLATE, subdirs) {
    prepareRecursiveTarget(check)
} else {
    check.commands = echo hello user
}
QMAKE_EXTRA_TARGETS += check
```

特性文件需要注入到每个子项目中，例如。qmake.conf:

```
# <project root>/.qmake.conf
CONFIG += mycheck
```

这个函数是在 Qt 5.0 中引入的。

### 2.3. qtCompileTest(test)

构建测试项目。如果测试通过，则返回 true 并将 `config_<test>` 添加到配置变量中。否则，返回false。

要使此功能可用，我们需要加载相应的功能文件:

```
# <project root>/project.pro
load(configure)
```

这还将变量 QMAKE_CONFIG_TESTS_DIR 设置为项目父目录的 `config.tests` 子目录。可以在加载特性文件后覆盖此值。

在测试目录中，每个测试必须有一个子目录，其中包含一个简单的 qmake 项目。下面的代码片段说明了项目的 .pro 文件:

```
# <project root>/config.tests/test/test.pro
SOURCES = main.cpp
LIBS += -ltheFeature
# Note that the test project is built without Qt by default.
```

下面的代码片段演示了项目的 main.cpp 文件:

```
// <project root>/config.tests/test/main.cpp
#include <TheFeature/MainHeader.h>
int main() { return featureFunction(); }
```

下面的代码片段显示了测试的调用:

```
# <project root>/project.pro
qtCompileTest(test)
```

如果测试项目成功构建，则测试通过。

测试结果被自动缓存，这也使得它们对所有子项目可用。因此，建议在顶级项目文件中运行所有配置测试。

要避免缓存结果的重用，请将 `CONFIG+=recheck`传递给 qmake。

也可参阅 load()

这个函数是在 Qt 5.0 中引入的。

### 2.4. qtHaveModule(name)

检查 name 指定的 Qt 模块是否存在。有关可能值的列表，请参见 QT 。

这个函数是在 Qt 5.0.1 中引入的。