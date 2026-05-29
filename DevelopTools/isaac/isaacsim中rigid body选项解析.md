#Isaacsim #AI生成 

以下内容来自 GPT 5.4 pro：

我对照了 NVIDIA / Omniverse 的官方文档看了一遍。先给你一个总认识：

你的这组项其实混合了三层东西：一部分是标准 USD 刚体属性 `UsdPhysics.RigidBodyAPI`，一部分是 NVIDIA PhysX 扩展属性 `PhysxSchema.PhysxRigidBodyAPI`，还有一个 `Velocities in Local Space` 严格说是 Isaac Sim/Omniverse 的**物理设置开关**，不是刚体自身的物理参数。物理量还会跟随 stage 的单位缩放；所有角度类量都用“度”而不是弧度。([NVIDIA Docs](https://docs.omniverse.nvidia.com/kit/docs/omni_usd_schema_physics/104.2/class_usd_physics_rigid_body_a_p_i.html "UsdPhysicsRigidBodyAPI Class Reference"))

还有一个很重要的细节：官方文档在“速度坐标空间”这件事上有层次差异。旧的 USD schema 文档把 `velocity/angularVelocity` 描述成“与节点 xform 同一坐标空间”；当前 Omni Physics 开发文档又写成“在刚体质心、世界坐标系下指定/读取”；而 Isaac Sim 的 Physics Settings 里明确说明 `Output Velocities in Local space` 会决定 UI 显示和用户输入按 local 还是 global 来解释。所以你在 Isaac Sim 里直接改速度数值时，**最稳妥的判断标准就是看这个开关当前状态**。另外，若开启 `Update velocities to USD`，运行仿真时速度还会被写回到 USD。([NVIDIA Docs](https://docs.omniverse.nvidia.com/kit/docs/omni_usd_schema_physics/104.2/class_usd_physics_rigid_body_a_p_i.html "UsdPhysicsRigidBodyAPI Class Reference"))

下面按你列的顺序解释。

## Rigid Body

### Rigid Body Enabled（`physics:rigidBodyEnabled`）  
是否真正启用“刚体”求解。关掉以后，这个 prim 上 `RigidBodyAPI` 的效果会被取消；它子树里的碰撞体仍可存在，但会按**静态碰撞体**处理。你可以把它理解成“保留碰撞形状，但不再作为动态/运动学刚体参与求解”。([NVIDIA Docs](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/107.0/dev_guide/rigid_bodies_articulations/rigid_bodies.html "Rigid Bodies — Omni Physics"))

### Kinematic Enabled（`physics:kinematicEnabled`）  
把刚体改成**运动学刚体**。动态刚体是“仿真写它的位置”；运动学刚体则是“仿真读你给它的位置/动画”。它仍然会和别的物体发生相互作用，而且仿真会根据你外部给的位姿变化推导出它的速度，因此它会把“运动效果”传递给碰撞到的动态物体。它不是“普通静态碰撞体做动画”的完全等价替代。另一个重要限制是：若该物体要作为 articulation 的刚体链节，官方要求它必须是**启用的、非 kinematic 的 rigid body**。([NVIDIA Docs](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/107.0/dev_guide/rigid_bodies_articulations/rigid_bodies.html "Rigid Bodies — Omni Physics"))

### Simulation Owner（`physics:simulationOwner`）  
指定这个刚体归哪个 `PhysicsScene` 来模拟。只有在一个 stage 里存在多个 `PhysicsScene` 时它才有意义；若不设置，默认使用遍历时找到的第一个 `PhysicsScene`。官方还特别说明：**不同 physics scene 里的对象彼此不会碰撞**。([NVIDIA Docs](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/latest/dev_guide/simulation_control/simulation_control.html "Simulation Control — Omni Physics"))

### Starts as Asleep（`physics:startsAsleep`）  
仿真开始时这个刚体是否一开始就处于“睡眠”状态。睡眠是 PhysX 用来节省计算量的机制：物体静止一段时间后可以暂停更新，直到再次被碰撞或被外力/用户修改唤醒。这个选项常用于一开始就应该静止的堆叠物。([NVIDIA Docs](https://docs.omniverse.nvidia.com/kit/docs/omni_usd_schema_physics/104.2/class_usd_physics_rigid_body_a_p_i.html "UsdPhysicsRigidBodyAPI Class Reference"))

### Velocities in Local Space  
这个不是单个刚体的物理本体参数，而是 Physics Settings 里的一个**显示/输入规则开关**。开着时，UI 里的速度按局部坐标显示/输入；关掉时按全局坐标显示/输入。它也直接影响你手动改“初始速度”时 Isaac Sim 如何解释你填的数值。([NVIDIA Docs](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/latest/extensions/ux/source/omni.physx.ui/docs/dev_guide/sim_management.html "Simulation Management — Omni Physics"))

### Linear Velocity（`physics:velocity`）  
线速度。单位是 stage 的 `distance / time`。在实际使用里，它通常表示刚体的初始线速度；若你开启了 `Update velocities to USD`，运行时它也可能被写回成当前仿真速度。至于它到底按 local 还是 global 看，在 Isaac Sim UI 里应以 `Velocities in Local Space` 当前设置为准。([NVIDIA Docs](https://docs.omniverse.nvidia.com/kit/docs/omni_usd_schema_physics/104.2/class_usd_physics_rigid_body_a_p_i.html "UsdPhysicsRigidBodyAPI Class Reference"))

### Angular Velocity（`physics:angularVelocity`）  
角速度，单位是**度/时间**。含义和 `Linear Velocity` 类似，只不过描述的是绕各轴的旋转速度。Isaac Sim 中同样要结合 `Velocities in Local Space` 来理解其坐标系。([NVIDIA Docs](https://docs.omniverse.nvidia.com/kit/docs/omni_usd_schema_physics/104.2/class_usd_physics_rigid_body_a_p_i.html "UsdPhysicsRigidBodyAPI Class Reference"))

### Linear Damping（`physxRigidBody:linearDamping`）  
线性阻尼系数。可以把它理解成对平移动作的“减速阻力”。值越大，物体平移时越容易慢下来。新手最常把它当成“让东西别滑太久”的参数。([NVIDIA Docs](https://docs.omniverse.nvidia.com/kit/docs/omni_usd_schema_physics/latest/class_physx_schema_physx_rigid_body_a_p_i.html "PhysxSchemaPhysxRigidBodyAPI Class Reference"))

### Angular Damping（`physxRigidBody:angularDamping`）  
角阻尼系数。和 `Linear Damping` 一样，但作用在旋转上。值越大，自转/翻滚衰减越快。([NVIDIA Docs](https://docs.omniverse.nvidia.com/kit/docs/omni_usd_schema_physics/latest/class_physx_schema_physx_rigid_body_a_p_i.html "PhysxSchemaPhysxRigidBodyAPI Class Reference"))

### Max Linear Velocity（`physxRigidBody:maxLinearVelocity`）  
刚体允许达到的最大线速度上限。PhysX 会把线速度钳制在这个范围内，常用于避免数值爆炸或极端高速导致的不稳定。([NVIDIA Docs](https://docs.omniverse.nvidia.com/kit/docs/omni_usd_schema_physics/latest/class_physx_schema_physx_rigid_body_a_p_i.html "PhysxSchemaPhysxRigidBodyAPI Class Reference"))

### Max Angular Velocity（`physxRigidBody:maxAngularVelocity`）  
刚体允许达到的最大角速度上限，单位是度/秒。常用于限制高速旋转导致的数值问题。([NVIDIA Docs](https://docs.omniverse.nvidia.com/kit/docs/omni_usd_schema_physics/latest/class_physx_schema_physx_rigid_body_a_p_i.html "PhysxSchemaPhysxRigidBodyAPI Class Reference"))

### Sleep Threshold（`physxRigidBody:sleepThreshold`）  
睡眠阈值。官方定义是“**按质量归一化后的动能阈值**”，低于它时刚体就可能进入睡眠。它不是一个简单的“速度阈值”。值越大，物体越容易被判定为可睡眠；文档还说明把这类睡眠阈值设成 0 可以禁用该对象的睡眠。([NVIDIA Docs](https://docs.omniverse.nvidia.com/kit/docs/omni_usd_schema_physics/latest/class_physx_schema_physx_rigid_body_a_p_i.html "PhysxSchemaPhysxRigidBodyAPI Class Reference"))

### Enable CCD（`physxRigidBody:enableCCD`）  
开启连续碰撞检测中的 **sweep-based / linear CCD**。它专门解决高速或很薄的物体“穿透”问题：仿真不只看离散帧头尾，而是沿运动路径做扫描。官方也说明这种 CCD 主要处理**线性运动**，会忽略角运动；而且它除了刚体自己开关外，通常还需要 scene 级别也启用 CCD。([NVIDIA Docs](https://docs.omniverse.nvidia.com/kit/docs/omni_usd_schema_physics/latest/class_physx_schema_physx_rigid_body_a_p_i.html "PhysxSchemaPhysxRigidBodyAPI Class Reference"))

### Disable Gravity（`physxRigidBody:disableGravity`）  
对这个刚体关闭重力。它仍可碰撞、受约束、受你施加的力，但不会再自动受到场景重力影响。([NVIDIA Docs](https://docs.omniverse.nvidia.com/kit/docs/omni_usd_schema_physics/latest/class_physx_schema_physx_rigid_body_a_p_i.html "PhysxSchemaPhysxRigidBodyAPI Class Reference"))

### Locked Pos Axis → X / Y / Z（`physxRigidBody:lockedPosAxis`）  
锁定某个平移轴。官方底层实现是一个按位标志位集合；在 UI 里被展开成 X/Y/Z 复选框。比如锁 X，刚体就不能沿 X 平移。常用于滑轨、2D/2.5D 约束、只允许某方向运动的机构。([NVIDIA Docs](https://docs.omniverse.nvidia.com/kit/docs/omni_usd_schema_physics/latest/class_physx_schema_physx_rigid_body_a_p_i.html "PhysxSchemaPhysxRigidBodyAPI Class Reference"))

### Locked Rot Axis → X / Y / Z（`physxRigidBody:lockedRotAxis`）  
锁定某个旋转轴。比如锁住 X/Y，只允许绕 Z 转。常用于门轴、平面小车、只允许偏航的底盘等。([NVIDIA Docs](https://docs.omniverse.nvidia.com/kit/docs/omni_usd_schema_physics/latest/class_physx_schema_physx_rigid_body_a_p_i.html "PhysxSchemaPhysxRigidBodyAPI Class Reference"))

## Advanced

### Stabilization Threshold（`physxRigidBody:stabilizationThreshold`）  
稳定化阈值。官方定义是：当刚体“按质量归一化后的动能”低于这个阈值时，它可以参与 stabilization。stabilization 的作用是给低速物体施加额外阻尼，帮助大堆叠或高交互场景更快稳定、减少抖动。**但官方明确说它会损失动量，不推荐用于机器人、工业场景或需要精确物理交互的仿真。**另外，这个参数只有在 **scene 级先启用了 stabilization** 后才有意义。([NVIDIA Docs](https://docs.omniverse.nvidia.com/kit/docs/omni_usd_schema_physics/latest/class_physx_schema_physx_rigid_body_a_p_i.html "PhysxSchemaPhysxRigidBodyAPI Class Reference"))

### Max Depenetration Velocity（`physxRigidBody:maxDepenetrationVelocity`）  
求解器为了解决物体互相穿插/重叠而引入的“分离速度”上限。值大时，重叠物体可能更猛地被弹开；值小则分离更温和，但过低时重叠可能消除得更慢。([NVIDIA Docs](https://docs.omniverse.nvidia.com/kit/docs/omni_usd_schema_physics/latest/class_physx_schema_physx_rigid_body_a_p_i.html "PhysxSchemaPhysxRigidBodyAPI Class Reference"))

### Max Contact Impulse（`physxRigidBody:maxContactImpulse`）  
接触冲量上限。官方说明，两个动态/运动学刚体接触时，实际可用的最大冲量取双方上限的较小值；静态体与动态体碰撞时，看动态体这一侧的上限。把它调低，通常会让碰撞显得“更软”、峰值作用更小。([NVIDIA Docs](https://docs.omniverse.nvidia.com/kit/docs/omni_usd_schema_physics/latest/class_physx_schema_physx_rigid_body_a_p_i.html "PhysxSchemaPhysxRigidBodyAPI Class Reference"))

### Solver Position Iteration Count（`physxRigidBody:solverPositionIterationCount`）  
这个刚体的**位置迭代**次数。求解器本质上是迭代处理接触、关节等约束；位置迭代越多，越能减少穿插/重叠，结果更准，但计算也更慢。([NVIDIA Docs](https://docs.omniverse.nvidia.com/kit/docs/omni_usd_schema_physics/latest/class_physx_schema_physx_rigid_body_a_p_i.html "PhysxSchemaPhysxRigidBodyAPI Class Reference"))

### Solver Velocity Iteration Count（`physxRigidBody:solverVelocityIterationCount`）  
这个刚体的**速度迭代**次数。官方说明它有助于避免交互时物体“凭空捡到不该有的速度”。数值越高一般越稳，但更耗时；官方还提到 TGS 求解器通常只建议很少的 velocity iterations（大约 1 次）。([NVIDIA Docs](https://docs.omniverse.nvidia.com/kit/docs/omni_usd_schema_physics/latest/class_physx_schema_physx_rigid_body_a_p_i.html "PhysxSchemaPhysxRigidBodyAPI Class Reference"))

### Enable Speculative CCD（`physxRigidBody:enableSpeculativeCCD`）  
开启 **speculative CCD**。官方描述是：根据速度动态调整 contact offset，以达到 CCD 效果。相比 sweep-based CCD，它通常更便宜，而且能照顾到角运动；但官方也提醒它在大线速度场景下可能引入“ghost collisions（幽灵碰撞）”。它**不需要** scene 级 CCD 开关；不过若你想把它和 sweep-based CCD 一起混合用，则 scene 级 CCD 仍应开启。官方还特别指出：对 **kinematic 对象只允许 speculative CCD**。([NVIDIA Docs](https://docs.omniverse.nvidia.com/kit/docs/omni_usd_schema_physics/latest/class_physx_schema_physx_rigid_body_a_p_i.html "PhysxSchemaPhysxRigidBodyAPI Class Reference"))

### Constraint-force-mixing Scale（`physxRigidBody:cfmScale`）  
CFM（constraint force mixing）缩放。官方解释是：通过“弱化约束响应”来帮助 articulation 稳定。建议使用接近默认值的小数，典型范围在 `[0, 0.1]`；而且它**只在这个刚体属于 articulation 时才会用到**。普通单刚体一般不用碰它。([NVIDIA Docs](https://docs.omniverse.nvidia.com/kit/docs/omni_usd_schema_physics/latest/class_physx_schema_physx_rigid_body_a_p_i.html "PhysxSchemaPhysxRigidBodyAPI Class Reference"))

### Contact Slop Coefficient（`physxRigidBody:contactSlopCoefficient`）  
接触“slop”系数。官方说它是“接触中角向影响的容差”，可帮助**滚动的近似碰撞形状**表现更好，例如某些近似球/轮/胶囊滚动时减少异常。这个参数很底层，没遇到滚动接触问题时通常保持默认。([NVIDIA Docs](https://docs.omniverse.nvidia.com/kit/docs/omni_usd_schema_physics/latest/class_physx_schema_physx_rigid_body_a_p_i.html "PhysxSchemaPhysxRigidBodyAPI Class Reference"))

### Enable Gyroscopic Forces（`physxRigidBody:enableGyroscopicForces`）  
是否计算刚体的陀螺力。直观理解就是让高速自旋物体的旋转响应更接近真实，尤其是高速旋转、进动明显的场景。一般普通物体保持默认即可。([NVIDIA Docs](https://docs.omniverse.nvidia.com/kit/docs/omni_usd_schema_physics/latest/class_physx_schema_physx_rigid_body_a_p_i.html "PhysxSchemaPhysxRigidBodyAPI Class Reference"))

### Retain Accelerations（`physxRigidBody:retainAccelerations`）  
是否把 forces/accelerations 在帧与帧之间保留下来，而不是每步清空。官方示例进一步说明：开启这类“保留”行为后，通过 `PhysxForceAPI` 写入的力/力矩可以持续生效，直到你显式把它们重置为 `(0,0,0)`。这对“持续推力/持续扭矩”非常有用。([NVIDIA Docs](https://docs.omniverse.nvidia.com/kit/docs/omni_usd_schema_physics/latest/class_physx_schema_physx_rigid_body_a_p_i.html "PhysxSchemaPhysxRigidBodyAPI Class Reference"))

### Solve Contact（`physxRigidBody:solveContact`）  
是否在动力学求解器里处理这个刚体的接触。官方原文很短，就是“Process the contacts of this rigid body in the dynamics solver”。实际理解上，关闭它会让这个刚体的接触不再按常规动力学方式求解；这属于比较底层、偏特化的选项，新手通常应保持开启。([NVIDIA Docs](https://docs.omniverse.nvidia.com/kit/docs/omni_usd_schema_physics/latest/class_physx_schema_physx_rigid_body_a_p_i.html "PhysxSchemaPhysxRigidBodyAPI Class Reference"))

## 给新手的实用判断

对普通 USD 资产，最常需要改的通常只有：`Rigid Body Enabled`、`Kinematic Enabled`、`Linear/Angular Velocity`、`Linear/Angular Damping`、`Disable Gravity`、`Enable CCD`、以及平移/旋转轴锁。大多数 `Advanced` 参数都属于“遇到明确问题再动”的类型。([NVIDIA Docs](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/107.0/dev_guide/rigid_bodies_articulations/rigid_bodies.html "Rigid Bodies — Omni Physics"))

若你的问题是“高速小物体穿透”，优先看 `Enable CCD`，旋转件或运动学物体再看 `Enable Speculative CCD`；若你的问题是“大堆物体抖动、难以停稳”，才考虑 `Stabilization Threshold`，但机器人和精确接触场景应谨慎使用 stabilization。([NVIDIA Docs](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/latest/dev_guide/rigid_bodies_articulations/rigid_bodies.html "Rigid Bodies — Omni Physics"))

把你的 Property 面板截图或对应的 `.usd` 片段发来，我可以再按“哪些建议改、哪些尽量别动、机器人场景该怎么设”给你标一版。
