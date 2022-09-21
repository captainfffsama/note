#CPP 
#vscode 

[toc]

在 `.vscode` 文件夹中通常有这三个配置文件:
- tasks.json: 通常用来设置编译构建,也可以用来做其他事情
- launch.json 设置 Debug 选项
- c_cpp_properties.json 用来设置编译路径和智能提示设置

# tasks.json
其大致内容通常如下:
```json
// tasks.json
{
    // https://code.visualstudio.com/docs/editor/tasks
    "version": "2.0.0",
    "tasks": [
        {
            "label": "Build",  // 任务的名字叫Build，注意是大小写区分的，等会在launch中调用这个名字
            "type": "shell",  // 任务执行的是shell命令，也可以是
            "command": "g++", // 命令是g++
            "args": [
                "'-Wall'",
                "'-std=c++17'",  //使用c++17标准编译
                "'${file}'", //当前文件名
                "-o", //对象名，不进行编译优化
                "'${fileBasenameNoExtension}.exe'",  //当前文件名（去掉扩展名）
            ],
          // 所以以上部分，就是在shell中执行（假设文件名为filename.cpp）
          // g++ filename.cpp -o filename.exe
            "group": { 
                "kind": "build",
                "isDefault": true   
                // 任务分组，因为是tasks而不是task，意味着可以连着执行很多任务
                // 在build组的任务们，可以通过在Command Palette(F1) 输入run build task来运行
                // 当然，如果任务分组是test，你就可以用run test task来运行 
            },
            "problemMatcher": [
                "$gcc" // 使用gcc捕获错误
            ],
        }
    ]
}
```
这里也可以使用 `${workspaceFolder}/*.cpp` 来替代 `${file}` 来编译多个 cpp 文件.

# launch.json
>下面是launch.json文件
里面主要是编译器的参数，可以使用ctrl+space来查看有哪些可用参数
也可以在configurations中存在鼠标光标的情况下，点击右下自动出现的Add Configurations按钮
为什么选gdb不选 windows？因为这个不会执行预任务，也就没法编译文件了
为什么选 launch不选attach，是因为attach用来给正在执行的文件用的，比如网页中的组件，而launch是执行新文件

大致内容如下:
```json
// launch.json

{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "(gdb) Launch", //这个应该是F1中出现的名字
            "type": "cppdbg",
            "request": "launch",
            "program": "${fileDirname}/${fileBasenameNoExtension}.exe", //需要运行的是当前打开文件的目录中，名字和当前文件相同，但扩展名为exe的程序
            "args": [],
            "stopAtEntry": false, // 默认情况下, C++ 扩展不会给源码添加任何断点,且本项为 false.若改为 true,当开始 debugging 时,调试器会停留在 main 函数上.
            "cwd": "${workspaceFolder}", // 当前工作路径：当前文件所在的工作空间
            "environment": [],
            "externalConsole": true,  // 是否使用外部控制台，选false的话，我的vscode会出现错误
            "MIMode": "gdb",
            "setupCommands": [
                {
                    "description": "Enable pretty-printing for gdb",
                    "text": "-enable-pretty-printing",
                    "ignoreFailures": true
                }
            ],
            "preLaunchTask": "Build",  //在launch之前运行的任务名，这个名字一定要跟tasks.json中的任务名字大小写一致
            "miDebuggerPath": "c:/MinGW/bin/gdb.exe",
        }
    ]
}
```

# c_cpp_properties.json
若想对 C++ 扩展进行更多控制,可以通过该文件,比如修改编译器路径,include 路径,C++ 标准(默认 17)等等.   

通过运行 `C/C++:Edit Configurations(UI)` 可以打开 C/C++ 配置的 UI.

其大致内容通常如下:
```json
{
  "configurations": [
    {
      "name": "Linux",
      "includePath": ["${workspaceFolder}/**"],
      "defines": [],
      "compilerPath": "/usr/bin/gcc",
      "cStandard": "c11",
      "cppStandard": "c++17",
      "intelliSenseMode": "clang-x64"
    }
  ],
  "version": 4
}

一些具体的设置选项可以参考: <https://code.visualstudio.com/docs/cpp/customize-default-settings-cpp>

```

# 相关参考
- <https://code.visualstudio.com/docs/editor/variables-reference>
- <https://zhuanlan.zhihu.com/p/92175757>
