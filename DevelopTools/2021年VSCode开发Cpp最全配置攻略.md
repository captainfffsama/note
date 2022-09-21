#vscode 
#待处理 


Why VSCode？
-----------

**Visual Studio Code**（简称VS Code）是一款由微软开发且跨平台（适用于 macOS、Linux 和 Windows）的免费源代码编辑器。该软件支持语法高亮、代码自动补全（又称IntelliSense）、代码重构、查看定义功能，并且内置了命令行工具和Git版本控制系统。用户可以更改主题和键盘快捷方式实现个性化设置，也可以通过内置的扩展程序商店安装扩展以拓展软件功能（轻量化）。本文将详细介绍如何在Wins10系统下配置VSCode的C/C++编译环境。

![](https://minio.cvmart.net/cvmart-community/images/202206/30/0/006C3FgEgy1gt76qcytz4j60u00e5q6502.jpg)

下载VSCode
--------

直接进入[官网](https://www.cvmart.net/community/detail/undefined "undefined")下载Windows版本，解压安装即可。

![](https://minio.cvmart.net/cvmart-community/images/202206/30/0/006C3FgEgy1gt76qfaiifj60u00fxdjl02.jpg)

![](https://minio.cvmart.net/cvmart-community/images/202206/30/0/006C3FgEgy1gt76qj1krqj60h00dt42b02.jpg)

下载编译器
-----

进入[官网](https://www.cvmart.net/community/detail/undefined "undefined")下载MinGW-w64：

![](https://minio.cvmart.net/cvmart-community/images/202206/30/0/006C3FgEgy1gt76qmeid5j60ju0aagob02.jpg)

下载完解压，并将文件夹放置到C:\\Program Files（可自定义，嫌麻烦默认系统路径），并添加到环境变量，用windows的搜索功能（快捷键是Windows徽标键+S）搜索环境变量：

![](https://minio.cvmart.net/cvmart-community/images/202206/30/0/006C3FgEgy1gt76uuuigbj60qx0gggpe02.jpg)

把路径：C:\\Program Files\\mingw64\\bin 添加进去：

![](https://minio.cvmart.net/cvmart-community/images/202206/30/0/006C3FgEgy1gt76qs0aiqj60lw0lcq8t02.jpg)

![](https://minio.cvmart.net/cvmart-community/images/202206/30/0/006C3FgEgy1gt76qy0c8uj60im0id7au02.jpg)

最后，再验证下编译器是否成功配置，打开cmd，输入gcc --version，回车：

![](https://minio.cvmart.net/cvmart-community/images/202206/30/0/006C3FgEgy1gt76r1qr1uj60r50e2wfw02.jpg)

注：GCC调试器不支持中文路径！！！

文件配置
----

这里大家可以按照这个文件目录模板先复制一遍，后期根据自己需求更改即可（注意这里.vscode文件夹 **.** 不能省略）：

![](https://minio.cvmart.net/cvmart-community/images/202206/30/0/006C3FgEgy1gt76r8c07ij60ki03pq4102.jpg)

其次，我们打开VSCode，链接到这个CODE\_C文件夹：

![](https://minio.cvmart.net/cvmart-community/images/202206/30/0/006C3FgEgy1gt76r519ypj60do0f3q4e02.jpg)

![](https://minio.cvmart.net/cvmart-community/images/202206/30/0/006C3FgEgy1gt76rbodwmj60u00jlgrv02.jpg)

按照上面的步骤，我们可以先搜索Chinese这个插件，将VSCode进行汉化。安装完之后点击右下角的Restart重启后便可以了：

![](https://minio.cvmart.net/cvmart-community/images/202206/30/0/006C3FgEgy1gt76ri2t94j60u00hgjuu02.jpg)

同样地步骤，我们再搜索C/C++插件，点击安装即可：

![](https://minio.cvmart.net/cvmart-community/images/202206/30/0/006C3FgEgy1gt76rm08fbj60u00hg7b202.jpg)

关闭VSCode后再重启，在.vscode文件下新建下列几个文件：

### tasks.json

```null
{  
    "version": "2.0.0",  
    "tasks": [  
        {//这个大括号里是‘构建（build）’任务  
            "label": "build", //任务名称，可以更改，不过不建议改  
            "type": "shell", //任务类型，process是vsc把预定义变量和转义解析后直接全部传给command；shell相当于先打开shell再输入命令，所以args还会经过shell再解析一遍  
            "command": "gcc", //编译命令，这里是gcc，编译c++的话换成g++  
            "args": [    //方括号里是传给gcc命令的一系列参数，用于实现一些功能  
                "${file}", //指定要编译的是当前文件  
                "-o", //指定输出文件的路径和名称  
                "${fileDirname}\\bin\\${fileBasenameNoExtension}.exe", //承接上一步的-o，让可执行文件输出到源码文件所在的文件夹下的bin文件夹内，并且让它的名字和源码文件相同  
                "-g", //生成和调试有关的信息  
                "-Wall", // 开启额外警告  
                "-static-libgcc",  // 静态链接libgcc  
                "-fexec-charset=GBK", // 生成的程序使用GBK编码，不加这一条会导致Win下输出中文乱码  
                "-std=c11", // 语言标准，可根据自己的需要进行修改，写c++要换成c++的语言标准，比如c++11  
            ],  
            "group": {  //group表示‘组’，我们可以有很多的task，然后把他们放在一个‘组’里  
                "kind": "build",//表示这一组任务类型是构建  
                "isDefault": true//表示这个任务是当前这组任务中的默认任务  
            },  
            "presentation": { //执行这个任务时的一些其他设定  
                "echo": true,//表示在执行任务时在终端要有输出  
                "reveal": "always", //执行任务时是否跳转到终端面板，可以为always，silent，never  
                "focus": false, //设为true后可以使执行task时焦点聚集在终端，但对编译来说，设为true没有意义，因为运行的时候才涉及到输入  
                "panel": "new" //每次执行这个task时都新建一个终端面板，也可以设置为shared，共用一个面板，不过那样会出现‘任务将被终端重用’的提示，比较烦人  
            },  
            "problemMatcher": "$gcc" //捕捉编译时编译器在终端里显示的报错信息，将其显示在vscode的‘问题’面板里  
        },  
        {//这个大括号里是‘运行(run)’任务，一些设置与上面的构建任务性质相同  
            "label": "run",   
            "type": "shell",   
            "dependsOn": "build", //任务依赖，因为要运行必须先构建，所以执行这个任务前必须先执行build任务，  
            "command": "${fileDirname}\\bin\\${fileBasenameNoExtension}.exe", //执行exe文件，只需要指定这个exe文件在哪里就好  
            "group": {  
                "kind": "test", //这一组是‘测试’组，将run任务放在test组里方便我们用快捷键执行  
                "isDefault": true  
            },  
            "presentation": {  
                "echo": true,  
                "reveal": "always",  
                "focus": true, //这个就设置为true了，运行任务后将焦点聚集到终端，方便进行输入  
                "panel": "new"  
            }  
        }  
  
    ]  
}  
  

```

### launch.json

```null
{  
    "version": "0.2.0",  
    "configurations": [  
        {//这个大括号里是我们的‘调试(Debug)’配置  
            "name": "Debug", // 配置名称  
            "type": "cppdbg", // 配置类型，cppdbg对应cpptools提供的调试功能；可以认为此处只能是cppdbg  
            "request": "launch", // 请求配置类型，可以为launch（启动）或attach（附加）  
            "program": "${fileDirname}\\bin\\${fileBasenameNoExtension}.exe", // 将要进行调试的程序的路径  
            "args": [], // 程序调试时传递给程序的命令行参数，这里设为空即可  
            "stopAtEntry": false, // 设为true时程序将暂停在程序入口处，相当于在main上打断点  
            "cwd": "${fileDirname}", // 调试程序时的工作目录，此处为源码文件所在目录  
            "environment": [], // 环境变量，这里设为空即可  
            "externalConsole": false, // 为true时使用单独的cmd窗口，跳出小黑框；设为false则是用vscode的内置终端，建议用内置终端  
            "internalConsoleOptions": "neverOpen", // 如果不设为neverOpen，调试时会跳到“调试控制台”选项卡，新手调试用不到  
            "MIMode": "gdb", // 指定连接的调试器，gdb是minGW中的调试程序  
            "miDebuggerPath": "C:\\Program Files\\mingw64\\bin\\gdb.exe", // 指定调试器所在路径，如果你的minGW装在别的地方，则要改成你自己的路径，注意间隔是\\  
            "preLaunchTask": "build" // 调试开始前执行的任务，我们在调试前要编译构建。与tasks.json的label相对应，名字要一样  
    }]  
}  
  

```

> 注：上面两个文件需要大家仔细阅读，把文中相应的路径改为你自己电脑上对应的路径，路径要统一！！！

运行调试C程序
-------

*   在C\_single文件夹下新建exercise文件夹；
*   在exercise文件下新建bin文件夹；
*   在exercise文件夹下新建hello.c文件：

```null

int main()  
{  
    char name[10];  
    printf("Input your name: ");  
    scanf("%s",name);  
    printf("Hello,%s,this is your vscode!\n",name);  
    return 0;  
}  

```

### 安装运行插件，商店搜索“Code Runner”，安装：

![](https://minio.cvmart.net/cvmart-community/images/202206/30/0/006C3FgEgy1gt76s4lo4wj60u00az0wo02.jpg)

### 同样地步骤再次搜索“C/C++ Clang Command Adapter”，安装：

![](https://minio.cvmart.net/cvmart-community/images/202206/30/0/006C3FgEgy1gt76s9jmt6j60u009nwh502.jpg)

### 点击VSCode右下角进行运行：

![](https://minio.cvmart.net/cvmart-community/images/202206/30/0/006C3FgEgy1gt76sbiiquj60u0081gnq02.jpg)

如果出现“Please install clang or check configuration clang.executable”提示，则进入这个网站 clang下载，版本下载最新版本即可。安装过程中可勾选将其添加到系统环境路径下，否则需自行手动添加：

![](https://minio.cvmart.net/cvmart-community/images/202206/30/0/006C3FgEgy1gt76sfbx7vj60pt0nun6302.jpg)

![](https://minio.cvmart.net/cvmart-community/images/202206/30/0/006C3FgEgy1gt76sjktmkj60h60ibwlr02.jpg)

重启VSCode，再次点击运行按钮即可：

![](https://minio.cvmart.net/cvmart-community/images/202206/30/0/006C3FgEgy1gt76srrm84j60u00hgq7202.jpg)

运行调试C++程序
---------

下面我们按这个工作目录新建一个工程：

![](https://minio.cvmart.net/cvmart-community/images/202206/30/0/006C3FgEgy1gt76svbpbgj60bl06tt9q02.jpg)

需要注意的是，我们要将.vscode这个文件夹放置到当前工作区目录下，而不能放到子文件夹目录下，否则会出现下面情况：即Bulid finished with error: \*\*\* 终端进程启动失败（退出代码：-1）

![](https://minio.cvmart.net/cvmart-community/images/202206/30/0/006C3FgEgy1gt76sypfhfj60j908wju302.jpg)

好了，文件目录组织好后，我们看下这三个\*.json的配置文件如何编写，注意阅读每个文件，把里面涉及到路径的地方检查一遍，一定要与你自己设置的路径保持一致：

### tasks.json

```null
{  
    "version": "2.0.0",  
    "tasks": [  
        { // 任务一  
            "label": "build", // 任务名称  
            "type": "shell", // 任务类型  
            "command": "C:\\Program Files\\mingw64\\bin\\g++.exe", // 编译命令，这里是g++，编译c的话换成gcc  
            "args": [ // 命令所需要用到的参数  
                "-g", // 生成和调试有关的信息  
                "${file}",  
                "-o", // 指定命令输出文件的路径和名称  
                "${fileDirname}\\bin\\${fileBasenameNoExtension}.exe",  
                "-Wall", // 开启额外警告  
                "-fexec-charset=GBK", // 生成的程序使用GBK编码，不加这一条会导致Win下输出中文乱码  
                "-std=c++11", // 语言标准，可根据自己需要进行修改，写C要换成C语言标准，比如C11  
            ],  
            // "options": { // 可选的编译命令  
            //  "cwd": "C:\\Program Files\\mingw64\\bin"  
            // },  
            "presentation": { // 执行这个任务的一些其他设定  
                "echo": true, // 表示在执行任务时在终端要有输出  
                "reveal": "always", // 执行任务时是否跳转到终端面板，可以为always，silent，never  
                "focus": false, // 设为true后可以使执行task时焦点聚集在终端，但对编译来说，设为true没有意义，因为运行时才涉及到输入  
                "panel": "new", // 每次执行这个task时都新建一个终端面板，也可以设置为shared，共用一个面板，不过那样会出现‘任务将被终端重用’的提示，比较烦人  
                "showReuseMessage": true,  
                "clear": false  
            },  
            "problemMatcher": [  
                "$gcc" // 捕捉编译时编译器在终端里显示的报错信息，将其显示在vscode的‘问题’面板里  
            ],  
            "group": {  
                "kind": "build", // group表示组，我们可以有很多task，然后把他们放在一个组里，“build”表示这一组任务类型是构建  
                "isDefault": true // 表示这个任务是当前这组任务中的默认任务  
            },  
            "detail": "compiler: \"C:\\Program Files\\mingw64\\bin\\g++.exe\""  
        },  
        { // 任务二  
            "label": "run",  
            "type": "shell",  
            "dependsOn": "build", // 任务依赖，因为运行必须先构建，所以执行这个任务之前必须先执行build任务  
            "command": "${fileDirname}\\bin\\${fileBasenameNoExtension}.exe",  
            "group": {  
                "kind": "test",  
                "isDefault": true  
            },  
            "presentation": {  
                "echo": true,  
                "reveal": "always",  
                "focus": true, // 这个就设置为true了，运行任务后将焦点聚集到终端，方便进行输入  
                "panel": "new",  
                "showReuseMessage": true,  
                "clear": false  
            }  
        }  
    ]  
}  

```

### launch.json

```null
{  
    // Use IntelliSense to learn about possible attributes.  
    // Hover to view descriptions of existing attributes.  
    // For more information, visit: https://go.microsoft.com/fwlink/?linkid=830387  
    "version": "0.2.0",  
    "configurations": [  
        { // 这个大括号里是我们的‘Debug’配置  
            "name": "debug", // 配置名称  
            "type": "cppdbg", // 配置类型，cppdbg对应cpptools提供的调试功能  
            "request": "launch", // 请求配置类型，可以为launch（启动）或attach（附加）  
            "program": "${fileDirname}\\bin\\${fileBasenameNoExtension}.exe",  
            "args": [],  
            "stopAtEntry": false, // 设置为true时程序将暂停在程序入口处，相当于在main上打断点，  
            "cwd": "${workspaceFolder}", // 调试程序时的工作目录  
            "environment": [], // 环境变量，这里设置为空即可  
            "externalConsole": false, // 为true时使用单独的cmd窗口，跳出小黑框；设为false则是用vscode内置终端，建议使用内置终端  
            // "internalConsoleOptions": "neverOpen", // 如果不设为neverOpen，调试时会跳到“调试控制台”选项卡，新手调试用不到  
            "MIMode": "gdb", // 指定连接的调试器，gdb是minGW中的调试程序  
            "miDebuggerPath": "C:\\Program Files\\mingw64\\bin\\gdb.exe", // 指定调试器的所在路径  
            "setupCommands": [  
                {  
                    "description": "为 gdb 启用整齐打印",  
                    "text": "-enable-pretty-printing",  
                    "ignoreFailures": true  
                }  
            ],  
            "preLaunchTask": "build", // 调试开始签执行的任务，我们在调试前要编译构建，玉tasks.json的label相对于，名字要一样  
        },  
    ]  
}  
`



`{  
    "files.associations": {  
        "ostream": "cpp"  
    }  
}  
`

好了，配置完成后，我们在hello.cpp文件下填写hello world脚本：

`
using namespace std;  
   
int main()   
{  
    cout << "Hello, World!";  
    return 0;  
}  

```

脚本写完，直接点击右上角的右三角形运行按钮即可：

![](https://minio.cvmart.net/cvmart-community/images/202206/30/0/006C3FgEgy1gt76t4b6wsj60u00hg77l02.jpg)

这时候我们发现，编译生成的.exe可执行文件出现在了与当前脚本文件下的同级目录中，这本身没什么问题。但是但我们的脚本文件过多时，会显得非常冗余，所以我们在当前目录下新建了个bin目录（也可以是build目录等，这里自己重命名即可），然后我们要做的是将所有生成的.exe文件都放置到里面去，这里提供两种方法：

#### 直接通过VSCode自带的编译

按下快捷键Ctrl+Shift+B运行生成任务，或者直接在工具栏点击即可：

![](https://minio.cvmart.net/cvmart-community/images/202206/30/0/006C3FgEgy1gt76twm6tyj60mh0aa76602.jpg)

选择.g++程序执行编译操作：

![](https://minio.cvmart.net/cvmart-community/images/202206/30/0/006C3FgEgy1gt76ts73cnj60h206qaaq02.jpg)

可以看到，.exe文件跑里面去了

![](https://minio.cvmart.net/cvmart-community/images/202206/30/0/006C3FgEgy1gt76tpz1flj60u00e0acg02.jpg)

最后，我们直接按快捷键Ctrl+F5或者点击工具栏运行当前文件即可：

![](https://minio.cvmart.net/cvmart-community/images/202206/30/0/006C3FgEgy1gt76tz4aa9j60fv0dqjtn02.jpg)

更进一步地，我们可以设置快捷键的方式去运行测试文件：点击左下角小齿轮->键盘快捷方式->搜索任务->找到运行测试任务,点击左侧加号添加键绑定，这里我们设为F4，

![](https://minio.cvmart.net/cvmart-community/images/202206/30/0/006C3FgEgy1gt76u2rnj9j60k00d50ub02.jpg)

然后回到我们的hello程序页面，按下F4即可大功告成！！！总结如下：先用“Ctrl+Shift+B”进行源文件的编译，生成.exe可执行文件；再用设置好的快捷键如"F4"运行测试文件。

#### 通过配置Code Runner来编译

在.vscode文件夹下新建"settings.json"文件，将以下内容复制进去：

```null
// {  
//     "files.associations": {  
//         "ostream": "cpp"  
//     },  
// }  
{  
  
    "files.defaultLanguage": "c++", // ctrl+N新建文件后默认的语言  
  
    "editor.formatOnType": true,  // 输入分号(C/C++的语句结束标识)后自动格式化当前这一行的代码  
  
    "editor.suggest.snippetsPreventQuickSuggestions": false, // clangd的snippets有很多的跳转点，不用这个就必须手动触发Intellisense了  
  
    "editor.acceptSuggestionOnEnter": "off", // 我个人的习惯，按回车时一定是真正的换行，只有tab才会接受Intellisense  
  
    // "editor.snippetSuggestions": "top", // （可选）snippets显示在补全列表顶端，默认是inline  
  
   
  
    "code-runner.runInTerminal": true, // 设置成false会在“输出”中输出，无法输入  
  
    "code-runner.executorMap": {  
  
        "c": "gcc '$fileName' -o '$fileNameWithoutExt.exe' -Wall -O2 -m64 -lm -static-libgcc -std=c11 -fexec-charset=GBK && &'./$fileNameWithoutExt.exe'",  
  
        "cpp": "g++ '$fileName' -o './/bin//$fileNameWithoutExt.exe' -Wall -O2 -m64 -static-libgcc -std=c++11 -fexec-charset=GBK && &'.//bin//$fileNameWithoutExt.exe'"  
  
  
    }, //   
  
    "code-runner.saveFileBeforeRun": true, // run code前保存  
  
    "code-runner.preserveFocus": true,     // 若为false，run code后光标会聚焦到终端上。如果需要频繁输入数据可设为false  
  
    "code-runner.clearPreviousOutput": false, // 每次run code前清空属于code runner的终端消息，默认false  
  
    "code-runner.ignoreSelection": true,   // 默认为false，效果是鼠标选中一块代码后可以单独执行，但C是编译型语言，不适合这样用  
  
    "code-runner.fileDirectoryAsCwd": true, // 将code runner终端的工作目录切换到文件目录再运行，对依赖cwd的程序产生影响；如果为false，executorMap要加cd $dir  
  
    "C_Cpp.clang_format_sortIncludes": true, // 格式化时调整include的顺序（按字母排序）  
  
}  

```

保存，然后再次点击右上角的运行按钮即可。现在，我们便可以愉快的玩耍拉~~~别忘记一键三连哦！

References
----------