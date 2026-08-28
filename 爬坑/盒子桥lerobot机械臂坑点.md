# 环境

Ubuntu 24.02 

Python 3.10

# libprotobuf.so.29.3.0: undefined symbol: \_ZN4absl12lts\_202501275MutexD1Ev
## 现象

执行

```python
python lerobot/scripts/control_robot.py calibrate     
--robot-path lerobot/configs/robot/so100_plus_single.yaml 
--robot-overrides '~
cameras'
```

报错：

```
Traceback (most recent call last):
  File "/home/captain/codes/python/ultron/lerobot-joycon_plus/lerobot/scripts/control_robot.py", line 106, in <module>
    from lerobot.common.robot_devices.control_utils import (
  File "/home/captain/codes/python/ultron/lerobot-joycon_plus/lerobot/common/robot_devices/control_utils.py", line 13, in <module>
    import cv2
ImportError: /home/captain/miniconda3/envs/ultron/lib/python3.10/site-packages/../.././libprotobuf.so.29.3.0: undefined symbol: _ZN4absl12lts_202501275MutexD1Ev
```

环境为：

```
opencv-python            4.10.0
opencv-python-headless   4.10.0
```

## 解决方案

重装 opencv