#CPP

[toc]

假设目录结果如下:
```bash
# 最初目录结构
 .
 ├── include
 │
 ├── main.cpp
 └── src
		└── Swap.h
		└── Swap.cpp

2 directories, 3 files
```

#   静态编译
静态库本质是将一堆`.o`的文件归档打包
```bash
#  汇编生成Swap.o文件
g++ Swap.cpp -c -I../include
# 生成静态库libSwap.a
ar rs libSwap.a Swap.o

cd ..

# 链接,生成可执行文件:staticmain
g++ main.cpp -Iinclude -Lsrc -lSwap -o staticmain
```

此时会在文件下生成一个staticmain可执行文件,可以直接执行,体积比动态编译要大
 
# 动态编译
```bash
#  生成动态库libSwap.so
g++ Swap.cpp -I../include -fPIC -shared -o libSwap.so
 
## 以上命令等同于以下两条命令,现将 cpp 编译成.o文件,然后将.o文件生成动态库
# -fPIC 表示.o和路径无关
# g++ Swap.cpp -I../include -c -fPIC
# g++ -shared -o libSwap.so Swap.o

cd ..

# 链接, 生成可执行文件 sharemain
g++ main.cpp -Iinclude -Lsrc -lSwap -o sharemain
```

 此时运行需要在 `LD_LIBRARY_PATH` 中添加动态库所在的位置,比如:
 ```bash
 # 运行可执行文件
 LD_LIBRARY_PATH=src ./sharemain
 ```