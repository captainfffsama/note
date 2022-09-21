#Lua
[toc]

学习主要参考[菜鸟教程](https://www.runoob.com/lua/lua-data-types.html)

# 安装
去[官方](https://www.lua.org/download.html)下载lua源码,比如`lua-5.4.4.tar.gz`,Linux执行以下命令安装:
```shell
tar zxfv lua-5.4.4.tar.gz
cd lua-5.4.4
make all test
sudo make install
```

# 环境启动
使用`lua -i`或者`lua`可以启动交互式终端,也可以编写`.lua`脚本文件.

# 数据类型
Lua 属于动态语言类型,定义的时候无需指定类型,可以直接赋值.Lua 中的8个基本类型为: nil,boolean,number,string,function,userdata,thread,table

| 数据类型 | 描述 |
| --- | --- |
| nil | 这个最简单，只有值nil属于该类，表示一个无效值（在条件表达式中相当于false）。 |
| boolean | 包含两个值：false和true。 |
| number | 表示双精度类型的实浮点数 |
| string | 字符串由一对双引号或单引号来表示 |
| function | 由 C 或 Lua 编写的函数 |
| userdata | 表示任意存储在变量中的C数据结构 |
| thread | 表示执行的独立线路，用于执行协同程序 |
| table | Lua 中的表（table）其实是一个"关联数组"（associative arrays），数组的索引可以是数字、字符串或表类型。在 Lua 里，table 的创建是通过"构造表达式"来完成，最简单构造表达式是{}，用来创建一个空表。 |

