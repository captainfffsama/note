# 环境
Ubuntu：24.04.3

Python 3.10

使用包：[JakubAndrysek/PySpaceMouse: 🎮 Multiplatform Python library for 3Dconnexion SpaceMouse devices using raw HID.](https://github.com/JakubAndrysek/PySpaceMouse)

# 方法
```bash
sudo apt-get install libhidapi-dev
sudo echo 'KERNEL=="hidraw*", SUBSYSTEM=="hidraw", MODE="0664", GROUP="plugdev"' > /etc/udev/rules.d/99-hidraw-permissions.rules
sudo usermod -aG plugdev $USER
newgrp plugdev
pip install easyhid
pip install pyspacemouse
```

