---
title: "GMR重定向配置文件(ik_config)生成步骤"
source: "https://www.bilibili.com/opus/1143562040766365699"
author:
published:
created: 2026-05-13
description:
tags:
  - "clippings"
---
目录

第一部分：准备工作

第二部分：运行初始重定向配置

一、新增配置文件

二、修改机器人关节名称

三、统一初始化所有关节的旋转

四、配置运行参数

五、运行新机器人

第三部分：可视化原始机器人状态

一、使用 mujoco 加载机器人

二、取消重力并保持站立状态

三、调整机器人姿态

四、可视化关节坐标

第四部分：修改重定向参数

一、修改旋转变换

二、修改缩放比例

三、修改位置变换

四、修改位置和旋转的权重

![](https://i2.hdslb.com/bfs/new_dyn/e761e7cf9d723ee519507c2387d13c2d2118136610.jpg@1416w_798h_1c.avif)

wang\_bryan

2025年12月07日 16:04

本篇文章主要基于已开源的 GMR 工程，增加新的机器人重定向配置文件，以期达到更好的拟合效果。

开源链接：

GMR：https://github.com/YanjieZe/GMR

参考实现：

https://zhuanlan.zhihu.com/p/1972732587438011486

[形模仿学习 Mimic Baseline培训：数据集、重定向、训练与Sim2Real\_哔哩哔哩\_bilibili](https://www.bilibili.com/video/BV1vW1pBNEF7/?spm_id_from=333.337.search-card.all.click&vd_source=c3347021967fc01a9d095550f9f43bca&spm_id_from=333.1369.0.0)

感谢 GMR 原作者以及以上作者提供的讲解与介绍！

## 第一部分：准备工作

需要有新的机器人 URDF 文件以及对应的3D模型文件；

从 Github 中 clone GMR 工程，并按照 README 配置 GMR 运行环境； 将新的机器人文件放入 GMR 工程中的 assets 目录中； 下载 Lafan1 数据集文件。

在进行第二部分配置前，可先运行 unitree\_g1 机器人，确保环境无误。

## 第二部分：运行初始重定向配置

## 一、新增配置文件

在 GMR 工程目录中的 general\_motion\_retargeting/ik\_configs 目录下新增一个配置文件（可基于已有的复制一份，并修改配置文件的名称）。

配置文件中的参数含义可参考 GMR 工程的说明 DOC.md：

![](https://i2.hdslb.com/bfs/new_dyn/3a9f9049c803173b919e1cd0d8bf98822118136610.png@1192w.avif)

## 二、修改机器人关节名称

在新建的配置文件中（以后除明确说明都是修改新的配置文件）修改机器人对应的关节名称，以及高度（高度会影响机器人脚部与地面之间距离）：

![](https://i2.hdslb.com/bfs/new_dyn/9a248d3701995c92ff4fd2348d3d65062118136610.png@762w_1786h.avif)

该修改要与 Lafan1 中的关节部位相对应。

## 三、统一初始化所有关节的旋转

将所有关节的旋转变换修改为 \[1, 0, 0, 0\]，既无任何变换（注意要初始化所有的关节包括 table1 和 table2，且在以后的修改中 table1 和 table2 要保持同步修改）：

![](https://i2.hdslb.com/bfs/new_dyn/83f29414f7d000033d51517eeb6606962118136610.png@648w_646h.avif)

## 四、配置运行参数

运行新增加的机器人，修改配置文件：general\_motion\_retargeting/params.py 。在该脚本中每一项配置对应增加新的机器人配置参数。

## 五、运行新机器人

可使用 vscode 运行 scripts/bvh\_to\_robot.py 脚本，注意 --robot 运行参数要改成新增的机器人名称；--bvh\_file 文件选择第一个即可：aiming1\_subject1.bvh。

在 while 循环中增加一个断点：

![](https://i2.hdslb.com/bfs/new_dyn/48b1130fc7f25df87e628dfd61b576962118136610.png@1192w.avif)

运行起来后要使 while 循环至少执行一次，此时可以在 mujoco 中看到显示的机器人状态和每个关节对应的坐标系：

![](https://i2.hdslb.com/bfs/new_dyn/8b72941b6f6a8d56c2750d0a31928c662118136610.png@1192w.avif)

## 第三部分：可视化原始机器人状态

## 一、使用 mujoco 加载机器人

使用命令在 mujoco 中加载机器人模型：

python -m mujoco.viewer --mjcf \[机器人 mjcf 文件路径\]

![](https://i2.hdslb.com/bfs/new_dyn/4dc499e7f54a72f79a5947e05b21be782118136610.png@1192w.avif)

## 二、取消重力并保持站立状态

运行起来后机器人是倒地的状态（unitree\_g1 机器人是可以稳定站立的，可跳过此步骤），需要保持机器人站立。

首先，取消重力影响。在左边栏找到 Physics -> Physical Parameters -> Gravity 参数，由 \[0 0 -9.81\] 改成 \[0 0 0\]。

其次，修改完之后，机器人会飘起来，因为没有重力影响，此时按键盘空格键暂停仿真（或者在左边栏的 Simulation 参数中，点击 Pause）。

最后，再重新 Reset（点击左边栏 Simulation 参数中的 Reset），此时机器人可保持站立状态。

![](https://i2.hdslb.com/bfs/new_dyn/4082934649e0a7eaf66775d1cdeb86802118136610.png@1192w.avif)

## 三、调整机器人姿态

将原始机器人的姿态调整到第二部分中运行无变换的相同姿态上。点开右边栏的 Joint 直接修改对应关节的弧度即可。

![](https://i2.hdslb.com/bfs/new_dyn/9849dbb8318f591aee891d678659c80d2118136610.png@1192w.avif)

## 四、可视化关节坐标

显示机器人关节坐标。在左边栏的 Rendering -> Frame 参数中选择 Body 即可。

![](https://i2.hdslb.com/bfs/new_dyn/1c78f43f0980fed7e0d3f46cfaf81e5a2118136610.png@1192w.avif)

修改完之后为了节省显示空间，可将左右栏关闭显示，分别按 Table 键和 Shift+Table 键关闭左边栏和右边栏。

## 第四部分：修改重定向参数

核心思路是通过对比两个坐标系的差异，将原始坐标系通过欧拉角（xyz）的方式旋转到重定向的状态上，最终再把欧拉角转换成四元素 \[w x y z\] 的形式，写入到配置文件中。

主要修改步骤包括：旋转变换 -> 缩放比例 -> 位置变换 -> 位置和旋转的权重。

![](https://i2.hdslb.com/bfs/new_dyn/5139c7e8bde24b7d7071d386a29b3b592118136610.png@1192w.avif)

## 一、修改旋转变换

以左腕关节为例，

原始坐标为： x 朝左，y 朝上，z 朝前

重定向坐标为： x 朝左，y 朝后，z 朝上

经过欧拉角旋转：x 旋转 90°

转换成四元素为 \[0.7071068, 0.7071068, 0, 0\]

四元数转换网站可参考：https://link.zhihu.com/?target=https%3A//www.andre-gaschler.com/rotationconverter/

将四元数写入到 ik\_config 中即可。

【注意】：如果再转换后发现不对，最好基于初始的 \[1 0 0 0\]旋转，避免基于已转换错误的再次旋转。

## 二、修改缩放比例

修改缩放比例主要为粗调整，将坐标系的位置接近于关节的位置，或者是在同一水平或竖直层面上即可。

![](https://i2.hdslb.com/bfs/new_dyn/c93f72a5fa4239791d2ae49837df19652118136610.png@616w.avif)

## 三、修改位置变换

经过缩放粗定位后，可能坐标系的位置和关节位置还不太接近，可通过调整位置变换，比如沿某个坐标轴方向移动一定的距离，使坐标系和关节位置更加接近。

![](https://i2.hdslb.com/bfs/new_dyn/060cdedbcf1febf8ae292a18ba2d6db52118136610.png@680w_1214h.avif)

## 四、修改位置和旋转的权重

如果确认位置和旋转参数调整较为准确，而在运行重定向的过程中，发现坐标系和关节位置相差较远，且有些位置动作可以拟合上，但是有些动作拟合不正确，可以修改位置和旋转的权重，以期达到比较好的拟合效果。

![](https://i2.hdslb.com/bfs/new_dyn/16671eae624a8a1ae7d8d81c31a6366e2118136610.png@662w_1212h.avif)

以上修改步骤可重复调整，直至达到比较好的效果。

谢谢观看！祝各位好运！

cv44051495