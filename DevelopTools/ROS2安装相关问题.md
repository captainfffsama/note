[toc]

# 全新系统级安装

直接参考 [鱼香 ROS](https://fishros.com/d2lros2/#/humble/chapt1/get_started/3.动手安装ROS2) 

```bash
wget http://fishros.com/install -O fishros && . fishros
```

# 已有系统级安装，但是想 conda 安装隔离
## 去除系统中变量

`nano ~/.bashrc` 将 `source /opt/ros/jazzy/setup.bash` 注释。

在全新终端执行

```shell
# 1. 关闭当前的 Conda 环境
conda deactivate

# 2. 强制清除 ROS 相关的环境变量 (关键步骤！)
unset AMENT_PREFIX_PATH
unset CMAKE_PREFIX_PATH
unset PYTHONPATH
unset ROS_DISTRO
unset ROS_VERSION
unset ROS_PYTHON_VERSION
unset ROS_LOCALHOST_ONLY

# 3. 重新激活 RoboStack 环境
conda activate ros_learn  # 你的环境名

# 4. 验证环境是否干净
echo $PYTHONPATH
# 输出应该是空的，或者只包含 Conda 的路径。
# 绝对不能包含 "/opt/ros/jazzy" !!!
```

清理之前 ROS2 包的编译（可选）

```bash
# 1. 删除旧的编译产物
rm -rf build/ install/ log/

# 2. 重新编译 (此时环境已纯净)
colcon build --symlink-install

# 3. Source 本地工作空间
source install/setup.bash

# 4. 运行
ros2 run test_py node_02
```

执行 `which ros2`, 若显示 `/opt/ros/jazzy/bin/ros2` 表明还是使用的系统级 ros，执行 `source /opt/ros/jazzy/setup.bash` 会重新加载系统 ROS 的库路径。

## 全新 conda 环境

直接参考 [快速入门 – RoboStack --- Getting Started - RoboStack](https://robostack.github.io/GettingStarted.html#installing-ros)

```bash
# Create a ros-jazzy desktop environment
conda create -n ros_env -c conda-forge -c robostack-jazzy ros-jazzy-desktop
# Activate the environment
conda activate ros_env
# Add the robostack channel to the environemnt
conda config --env --add channels robostack-jazzy

conda activate ros_env
conda install -c conda-forge compilers cmake pkg-config make ninja colcon-common-extensions catkin_tools rosdep
```

## 已有 conda 环境

假设已有 conda 环境 ros_learn

```bash
conda activate ros_learn

# 1.添加 conda-forge (基础依赖库) 
conda config --env --add channels conda-forge 
# 2. 添加 robostack-jazzy (ROS 包的核心源) 
conda config --env --add channels robostack-jazzy 

# 3. 设置频道优先级为 strict (严格模式) 
# 这能防止 Conda 混用 defaults 源里的包，避免 ABI 冲突 

conda config --env --set channel_priority strict

conda install ros-jazzy-desktop

conda install compilers cmake pkg-config make ninja colcon-common-extensions
```

### 验证安装
1. 清理环境变量（防止系统 ROS 干扰）

```bash
   unset AMENT_PREFIX_PATH 
   unset PYTHONPATH
```

2. 激活 conda 环境并检查 ros2 路径

```shell
conda deactivate
conda activate ros_learn

which ros2 
# 输出应该是： …/envs/ros_learn/bin/ros2 
# 绝对不能是： /opt/ros/jazzy/bin/ros2
```

# 其他报错
## Colcon build 报错
### 现象

执行 `colcon build --symlink-install` 编译报错：

```shell
Starting >>> test_py
--- stderr: test_py
usage: setup.py [global_opts] cmd1 [cmd1_opts] [cmd2 [cmd2_opts] …]
   or: setup.py --help [cmd1 cmd2 …]
   or: setup.py --help-commands
   or: setup.py cmd --help

error: option --editable not recognized
---
Failed   <<< test_py [0.32s, exited with code 1]

Summary: 0 packages finished [0.37s]
  1 package failed: test_py
  1 package had stderr output: test_py
```

### 解法

大概率 conda 环境中 setuptools 版本不对，可以尝试：

```bash
pip install wheel 

pip install setuptools==68.2.2 
rm -rf build install log 
colcon build --symlink-install
```