**TRAC-IK 更适合“给定末端位姿，求一组关节角”；Placo 更适合“同时满足多个任务和约束，让机器人持续运动”。** 选型关键在于任务复杂度，而不只是机械臂有多少自由度。

|对比项|TRAC-IK|Placo|
|---|---|---|
|核心定位|串联运动链的逆运动学求解器|面向全身运动的任务与约束求解框架|
|典型输入|末端目标位姿、初始关节角 seed、关节限位|当前状态、多个任务目标、权重及约束|
|典型输出|满足目标的一组关节角 \(q\)|当前一步的关节增量 \(\Delta q\)，可积分更新状态|
|求解方式|改进的 KDL 数值算法与 SQP 非线性优化并行运行|将任务局部线性化，构建并求解二次规划 QP|
|擅长的问题|单臂位姿求解、带关节限位的 IK|双臂协同、连续跟踪、姿态偏好、全身协调|
|工程接入|有现成 MoveIt 插件|C++ 核心、Python 接口，通常自己组织控制循环|
|动力学|主要专注 IK|另外提供逆动力学求解器|

以上定位和接口分别来自 [TRAC-IK 官方仓库](https://github.com/traclabs/trac_ik) 及 [Placo 官方文档](https://placo.readthedocs.io/en/stable/)。

**TRAC-IK 的特点：目标明确，接入直接。**

它重点解决传统 KDL IK 在关节限位附近容易求解失败的问题：一条求解路径使用带随机跳转的改进 Newton 方法，另一条使用 SQP，默认谁先找到解就返回。它也适用于七自由度等冗余机械臂。[算法说明](https://github.com/traclabs/trac_ik)

常用模式包括：

- `Speed`：尽快返回找到的可行解。
- `Distance`：在给定时间内搜索，优先返回接近 seed 的解。
- `Manip1 / Manip2`：按可操作度指标选解。

因此，如果你已经用 MoveIt 做抓取、搬运和路径规划，只需要一个更可靠的 IK 后端，**优先考虑 TRAC-IK**。[MoveIt 接入文档](https://github.com/moveit/moveit2_tutorials/blob/main/doc/how_to_guides/trac_ik/trac_ik_tutorial.rst)

它的边界是：**单次 IK 成功，只说明找到了目标关节配置。** 它本身不保证运动过程连续，也不负责完整的碰撞路径规划。连续调用时，用上一帧解作为 seed、选择 `Distance`，有助于减少跳变，但不能据此保证轨迹平滑；核心库也不根据碰撞网格排除自碰撞。[官方说明](https://github.com/traclabs/trac_ik)

**Placo 的特点：可以直接表达“同时想做到哪些事”。**

例如同一次求解中要求：

- 左右手分别跟踪目标；
- 两只手保持相对位置；
- 肘部或关节尽量靠近偏好姿态；
- 关节角度、速度不越界；
- 保持质心条件或避免自碰撞。

任务可以设为 `hard` 或 `soft`：前者在当前线性化问题中必须满足，否则求解失败；后者允许有误差，通过权重协调。**软任务权重较大，并不等于严格的任务优先级保证。** [任务与约束说明](https://placo.readthedocs.io/en/stable/kinematics/concepts.html)

Placo 的运动学求解器每次求的是局部增量，通常在循环中更新目标、求解、积分。所以，从使用方式看，它很适合连续遥操作、动作重定向和多任务跟踪；要让一个较远目标充分收敛，往往需要迭代。速度限制需要显式开启并设置时间步长。[求解循环](https://placo.readthedocs.io/en/stable/kinematics/getting_started.html)、[速度限制](https://placo.readthedocs.io/en/stable/kinematics/joint_limits.html)

代价是需要调任务权重、正则化和约束，还要处理任务冲突。它的自碰撞规避也是局部的，可能卡住，不能替代全局绕障规划。[自碰撞约束说明](https://placo.readthedocs.io/en/stable/kinematics/avoid_self_collisions_constraint.html)

按这些能力，我的选型建议是：

|你的需求|优先选择|
|---|---|
|单臂抓取：输入一个位姿，求目标关节角|**TRAC-IK**|
|已有 MoveIt，想替换 KDL IK|**TRAC-IK**|
|七自由度单臂，只要求末端到位|**TRAC-IK 通常就够用**|
|连续遥操作，还要控制肘部姿态、关节速度|**Placo 更方便表达这些要求**|
|两只手独立到位，没有耦合要求|可以分别使用 **TRAC-IK**|
|双手共同搬物，保持相对位姿，或协调腰部|**Placo**|
|人形机器人全身动作、质心和脚部约束|**Placo**|
|在复杂环境中找到完整无碰撞路径|需要再接运动规划器，二者单独都不够|

如果你关注的是双臂机器人，**“左右臂分别求解”还是“左右臂需要一起满足约束”**，通常就是这两个库之间最实用的分界。