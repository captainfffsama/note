#qt 

[toc]
# 问题
当 c++ 源码中包含中文，尤其非注释时，使用msvc+qtcreator开发，可能出现以下问题：
1. 写了个`char` 型之类的变量，编不过去，提示：常量中包含换行符
2. 可以编过去，但遇到了字符串中包含一些特定字，比如”点“，”和“，这些字会乱码，`qDebug`打不出来。
# 问题成因
Windows 识别 utf8 编码是依赖 BOM的，而 MSVC 在解析文件编译时优先使用系统编码。因此当文件是 utf8 无BOM时， MSVC解析文件就会使用系统编码比如GBK，导致涉及中文的一些硬编码错误。
解决思路就是要么文件用utf8 with bom 保存，要么强制 MSVC 用UTF8解析。考虑到跨平台，建议使用 utf8 无bom保存，强制 MSVC用UTF8解析。
# 具体解决方法
## 1 编译器宏指令
头文件上加`#pragma execution_character_set("utf-8")`

## 2 改pro文件
pro文件上加
```pro
win32-msvc* {
    QMAKE_CXXFLAGS += /source-charset:utf-8 /execution-charset:utf-8
}
```
或者
```pro
msvc {
    QMAKE_CFLAGS += /utf-8
    QMAKE_CXXFLAGS += /utf-8
}
```


## 3 文件改 utf8 with bom
使用 EmEditor 或者 vscode 把源码文件编码改成utf8 with bom。另外在 QtCreator 中工具-选项-文本编辑器-行为-文件编码处 将默认编码改为 UTF-8,UTF-8 BOM 改为 **如果编码是UTF8 则添加**。

**注意：使用1或2方法时可以不用改文本编辑器行为，UTF8 BOM可以选为 总是删除**