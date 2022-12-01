#vscode  

[TOC]

[参考原文](https://code.visualstudio.com/docs/editor/variables-reference)

# 预定义变量
- **${workspaceFolder}** - VS Code 打开的那个文件夹,即当前工作目录(根目录,完整路径)
- **${workspaceFolderBasename}** - VS Code 工作目录目录名
- **${file}** - 当前打开的文件名(完整路径)
- **${fileWorkspaceBasename}** - 当前打开文件所在的工作区文件夹
- **${relativeFile}** - 当前根目录到当前打开文件的相对路径(包括文件名)
- **${relativeFileDirname}** - 当前根目录到当前打开文件的相对路径(不包括文件名)
- **${fileBasename}** - 当前打开的文件名(包括扩展名)
- **${fileBasenameNoExtension}** - 当前打开的文件名(不包括扩展名)
- **${fileDirname}** - 当前打开文件的目录
- **${fileExtname}** - 当前打开文件的扩展名
- **${cwd}** - 启动时task工作的目录
- **${lineNumber}** - 当前激活文件所选行
- **${selectedText}** - 当前激活文件中所选择的文本
- **${execPath}** - vscode执行文件所在的目录
- **${defaultBuildTask}** - 默认编译任务(build task)的名字
- **${pathSeparator}** - 操作系统的文件路径分隔符

## 预定义变量距离
假设:
1. 编辑器打开了一个文件,文件路径是: `/home/your-username/your-project/folder/file.ext`
2. 我们工作区的根目录是: `/home/your-username/your-project`

那么实际各个变量的值如下:
- **${workspaceFolder}** - `/home/your-username/your-project`
- **${workspaceFolderBasename}** - `your-project`
- **${file}** - `/home/your-username/your-project/folder/file.ext`
- **${fileWorkspaceFolder}** - `/home/your-username/your-project`
- **${relativeFile}** - `folder/file.ext`
- **${relativeFileDirname}** - `folder`
- **${fileBasename}** - `file.ext`
- **${fileBasenameNoExtension}** - `file`
- **${fileDirname}** - `/home/your-username/your-project/folder`
- **${fileExtname}** - `.ext`
- **${lineNumber}** - 光标的行号
- **${selectedText}** - 编辑器中你选定的文字
- **${execPath}** - Code.exe 所在的路径
- **${pathSeparator}** - `/` on macOS or linux, `\` on Windows

> 注意在 `tasks.json` 和 `launch.json` 可以使用智能提示来获得预设变量的完整列表
