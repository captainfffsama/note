#python 

[toc]

# -m


# -u
强制 stdout 和 stderr 流不使用缓冲。 此选项对 stdin 流无影响。

在 docker 中把python输出外挂到宿主机文件中,若不加 -u,日志不会立即刷新

# 参考
- <https://docs.python.org/zh-cn/3/using/cmdline.html?highlight=pythonunbuffered#cmdoption-u>