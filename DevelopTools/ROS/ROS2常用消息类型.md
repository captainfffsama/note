---
tags:
  - "#ROS"
---

ROS 2（Robot Operating System 2）提供了大量标准消息类型（Message），用于在不同节点之间传递数据，例如机器人关节状态、相机图像、位姿、速度指令、传感器数据等。

ROS 2 的消息通常采用以下命名方式：

```
<功能包名>/msg/<消息类型>
```

例如：

```
std_msgs/msg/Float32
geometry_msgs/msg/Twist
sensor_msgs/msg/JointState
nav_msgs/msg/Odometry
```

对于机器人开发，特别是机械臂控制、运动学、遥操作、视觉感知和机器人状态反馈，下面这些消息类型最常用。

## 一、ROS 2 常用消息类型总览

|消息包|常见消息|主要用途|
|---|---|---|
| `std_msgs` | `Bool`、`Int32`、`Float32`、`String` |基础数据通信|
| `geometry_msgs` | `Pose`、`Twist`、`Transform`、`Point` |位姿、速度、空间变换|
| `sensor_msgs` | `JointState`、`Image`、`Imu`、`PointCloud2` |关节、相机及传感器数据|
| `nav_msgs` | `Odometry`、`Path`、`OccupancyGrid` |机器人导航与里程计|
| `trajectory_msgs` | `JointTrajectory`、`JointTrajectoryPoint` |机械臂轨迹控制|
| `control_msgs` | `JointTrajectoryControllerState`、`JointJog` |机器人控制与反馈|
| `std_srvs` | `Trigger`、`SetBool`、`Empty` |常见服务接口（不是 Topic 消息）|
| `builtin_interfaces` | `Time`、`Duration` |时间戳与持续时间|
| `tf2_msgs` | `TFMessage` |TF 坐标变换|
| `visualization_msgs` | `Marker`、`MarkerArray` |RViz 可视化|
| `diagnostic_msgs` | `DiagnosticArray`、`DiagnosticStatus` |设备状态与故障诊断|

## 二、常用消息类型详解

### 1. std_msgs：基础数据类型

适合传输简单的数值、字符串和布尔量。

|消息类型|数据类型|示例用途|
|---|---|---|
| `std_msgs/msg/Bool` | `bool` |开关、使能状态|
| `std_msgs/msg/Int32` | `int32` |模式编号|
| `std_msgs/msg/Float32` | `float32` |单个传感器数值|
| `std_msgs/msg/Float64` | `float64` |关节目标位置等|
| `std_msgs/msg/String` | `string` |文本状态|
| `std_msgs/msg/Float64MultiArray` |浮点数组|简单数值数组|

例如：

```
# std_msgs/msg/Float64
data: 1.57
```

注意：`Float64MultiArray` 等多维数组消息仍可使用，但 ROS 2 更推荐针对具体业务设计语义明确的消息类型。

### 2. geometry_msgs：空间几何数据

机器人运动控制中使用频率非常高。

|消息类型|含义|
|---|---|
| `Point` |三维位置 `(x, y, z)` |
| `Vector3` |三维向量|
| `Quaternion` |四元数姿态|
| `Pose` |位置 + 姿态|
| `PoseStamped` |带时间戳和坐标系的位姿|
| `Twist` |线速度 + 角速度|
| `TwistStamped` |带时间戳和坐标系的速度|
| `Transform` |平移 + 旋转|
| `TransformStamped` |带时间戳和坐标系的坐标变换|
| `Wrench` |三维力 + 三维力矩|

例如 `geometry_msgs/msg/Pose`：

```
position:
  x: 0.4
  y: 0.1
  z: 0.3
orientation:
  x: 0.0
  y: 0.0
  z: 0.0
  w: 1.0
```

表示末端位置为 `(0.4, 0.1, 0.3)` 米，姿态是单位四元数。

`geometry_msgs/msg/Twist`：

```
linear:
  x: 0.1
  y: 0.0
  z: 0.0
angular:
  x: 0.0
  y: 0.0
  z: 0.2
```

表示沿 X 轴运动，线速度为 `0.1 m/s`，同时绕 Z 轴以 `0.2 rad/s` 旋转。

### 3. sensor_msgs：传感器数据

|消息类型|用途|
|---|---|
| `JointState` |机器人关节位置、速度和力矩/力|
| `Image` |相机原始图像|
| `CompressedImage` |JPEG/PNG 等压缩图像|
| `CameraInfo` |相机内参及标定参数|
| `Imu` |加速度、角速度和姿态|
| `LaserScan` |二维激光雷达扫描|
| `PointCloud2` |三维点云|
| `Range` |单点距离测量|
| `BatteryState` |电池信息|
| `Joy` |游戏手柄输入|

其中，机械臂开发最常用的是 `sensor_msgs/msg/JointState`：

```
name:
  - shoulder_pan_joint
  - shoulder_lift_joint
  - elbow_joint
position: [0.0, -1.2, 1.5]
velocity: [0.0, 0.1, 0.2]
effort: [0.0, 0.5, 0.3]
```

各数组通过相同索引对应关节：

- `position`：关节位置，旋转关节通常为 rad。
    
- `velocity`：关节速度，旋转关节通常为 rad/s。
    
- `effort`：关节力矩或力，通常为 N·m 或 N。
    

`JointState` 主要用于反馈关节状态，而不是表示一段完整的控制轨迹。

### 4. trajectory_msgs：机器人轨迹

这是机械臂控制中非常关键的消息包。

`trajectory_msgs/msg/JointTrajectory` 包含：

```
header:
  frame_id: base_link

joint_names:
  - joint1
  - joint2

points:
  - positions: [0.0, 0.0]
    velocities: [0.0, 0.0]
    time_from_start:
      sec: 0
      nanosec: 0

  - positions: [1.0, 0.5]
    velocities: [0.0, 0.0]
    time_from_start:
      sec: 2
      nanosec: 0
```

它表示两个关节从初始位置运动，在轨迹开始后 2 秒到达指定位置。

与 `JointState` 不同：

|对比| `JointState` | `JointTrajectory` |
|---|---|---|
|主要用途|当前关节状态反馈|目标运动轨迹|
|包含时间序列|否|是|
|包含速度|可以|可以|
|包含多个轨迹点|否|是|
|典型 Topic| `/joint_states` | `/joint_trajectory` |

实际使用 `ros2_control` 时，通常通过轨迹控制器的 `follow_joint_trajectory` Action 或轨迹 Topic 下发控制目标。

### 5. nav_msgs：导航消息

|消息类型|用途|
|---|---|
| `Odometry` |位置、姿态、速度和协方差|
| `Path` |一系列路径位姿|
| `OccupancyGrid` |二维栅格地图|
| `MapMetaData` |地图分辨率、尺寸、原点等|

主要用于移动机器人定位、建图和路径规划。

### 6. tf 2_msgs：坐标系变换

ROS 2 中常用 `/tf` 和 `/tf_static` 传递坐标系关系，消息类型为 `tf2_msgs/msg/TFMessage`，其中包含多个 `TransformStamped`。

例如：

```
world
  └── base_link
       └── shoulder_link
            └── elbow_link
                 └── end_effector
```

这些变换可以用于计算机器人末端执行器在世界坐标系下的位置和姿态。

## 三、如何查看 ROS 2 已有消息类型？

ROS 2 提供了命令行工具来查询消息定义。

1. 查看当前安装的所有消息类型

```
ros2 interface list --only-msgs
```

2. 查看指定消息结构

```
ros2 interface show sensor_msgs/msg/JointState
```

输出类似：

```
std_msgs/Header header
string[] name
float64[] position
float64[] velocity
float64[] effort
```

3. 查看一个 Topic 使用什么消息类型

```
ros2 topic type /joint_states
```

4. 查看 Topic 的实际数据

```
ros2 topic echo /joint_states
```

5. 查看当前有哪些 Topic

```
ros2 topic list -t
```

## 四、机器人开发中如何选择消息类型？

假设你在开发一个机械臂遥操作系统，可以按照下面的方式选择：

|数据内容|推荐消息类型|
|---|---|
|机器人实际关节角度| `sensor_msgs/msg/JointState` |
|目标关节角度| `trajectory_msgs/msg/JointTrajectory` |
|末端目标位置和姿态| `geometry_msgs/msg/PoseStamped` |
|末端线速度和角速度| `geometry_msgs/msg/TwistStamped` |
|末端力和力矩| `geometry_msgs/msg/WrenchStamped` |
|相机 RGB 图像| `sensor_msgs/msg/Image` |
|深度图像| `sensor_msgs/msg/Image` |
|机器人关节轨迹| `trajectory_msgs/msg/JointTrajectory` |
|设备状态或错误代码|自定义消息或 `diagnostic_msgs` |
|启停控制| `std_msgs/msg/Bool` 或 Service|

例如，你的机械臂遥操作系统可以采用以下通信架构：

示意架构：实际反馈路径和控制接口取决于所用驱动及控制器。

对于机械臂而言，建议重点掌握这 5 种消息：

1. `sensor_msgs/msg/JointState`：读取当前关节状态。
2. `geometry_msgs/msg/PoseStamped`：传递末端目标位姿。
3. `geometry_msgs/msg/TwistStamped`：传递末端目标速度。
4. `trajectory_msgs/msg/JointTrajectory`：下发关节运动轨迹。
5. `geometry_msgs/msg/TransformStamped`：表达坐标系之间的变换。
    

需要注意，消息类型只定义数据结构，并不规定控制器如何执行命令。例如高频遥操作场景中，可能使用自定义关节目标消息或支持实时 Servo 的接口，而不一定通过 `JointTrajectory` 逐点发送命令。

标准消息的完整定义可以参考 [ROS 2 common_interfaces 官方仓库](https://github.com/ros2/common_interfaces)。
