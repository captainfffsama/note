# 现象

使用 vscode debug 调试代码报错:

```bash
qt.qpa.plugin: From 6.5.0, xcb-cursor0 or libxcb-cursor0 is needed to load the Qt xcb platform plugin.
qt.qpa.plugin: Could not load the Qt platform plugin "xcb" in "" even though it was found.
This application failed to start because no Qt platform plugin could be initialized. Reinstalling the application may fix this problem.

Available platform plugins are: offscreen, wayland, vnc, minimalegl, vkkhrdisplay, xcb, linuxfb, eglfs, minimal, wayland-egl.
```

# 解决方法

在 `launch.json` 中配置环境变量 `QT_QPA_PLATFORM=offscreen`, 例如:

```json
{
	"name": "Python 调试程序: 当前文件",
    "type": "debugpy",
    "request": "launch",
    "program": "${file}",
    "console": "integratedTerminal",
    "env": {
            "QT_QPA_PLATFORM": "offscreen",
    }
}
```

# 参考
- <https://blog.csdn.net/baobei0112/article/details/133310070>