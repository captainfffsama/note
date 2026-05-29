# 核心点

在 `isaacsim.exp.base.kit` 文件中，在 `[settings]` 部分添加如下内容：

```ini
[settings]
persistent.isaac.asset_root.default = "/data/isaacsim_assets/isaac-sim-assets-complete-5.1.0/Assets/Isaac/5.1"

exts."isaacsim.gui.content_browser".folders = [
    "/data/Assets/Isaac/5.1/Isaac/Robots",
    "/data/Assets/Isaac/5.1/Isaac/People",
    "/data/Assets/Isaac/5.1/Isaac/IsaacLab",
    "/data/Assets/Isaac/5.1/Isaac/Props",
    "/data/Assets/Isaac/5.1/Isaac/Environments",
    "/data/Assets/Isaac/5.1/Isaac/Materials",
    "/data/Assets/Isaac/5.1/Isaac/Samples",
    "/data/Assets/Isaac/5.1/Isaac/Sensors",
]

exts."isaacsim.asset.browser".folders = [
    "/data/Assets/Isaac/5.1/Isaac/Robots",
    "/data/Assets/Isaac/5.1/Isaac/People",
    "/data/Assets/Isaac/5.1/Isaac/IsaacLab",
    "/data/Assets/Isaac/5.1/Isaac/Props",
    "/data/Assets/Isaac/5.1/Isaac/Environments",
    "/data/Assets/Isaac/5.1/Isaac/Materials",
    "/data/Assets/Isaac/5.1/Isaac/Samples",
    "/data/Assets/Isaac/5.1/Isaac/Sensors",
]

exts."omni.kit.material.library".ui_show_list = [
    "OmniPBR",
    "OmniGlass",
    "OmniSurface",
    "USD Preview Surface",
]
```

将其中的路径换成下载的离线资产路径

# Pip 安装包

Isaacsim 使用 pip 安装的，可以在启动时追加参数指定启动配置文件，比如:

```bash
isaacsim isaacsim.exp.base.kit
```

# 预编译安装

若为预编译安装，可以在安装路径的 `apps/isaacsim.exp.base.kit` 找到该文件