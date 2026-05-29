#Isaacsim #AI生成 

### Collider

**Collision Enabled**  
对应 `physics:collisionEnabled`，默认值是 `1`。官方定义就是“是否启用 PhysicsCollisionAPI”。直白说，它决定这个 collider 现在是否真正参与碰撞计算；关掉以后，几何体还在，但这层碰撞行为被禁用。([NVIDIA Docs](https://docs.omniverse.nvidia.com/kit/docs/omni_usd_schema_physics/104.2/class_usd_physics_collision_a_p_i.html "UsdPhysicsCollisionAPI Class Reference"))

**Simulation Owner**  
对应 `physics:simulationOwner` 关系（relationship）。它指定“由哪个 `PhysicsScene` 来模拟这个 collider”。如果场景里有多个 `PhysicsScene`，而这个值没设，系统会用遍历时找到的第一个 `PhysicsScene`；如果父层级上的 `RigidBodyAPI` 也设置了 `simulationOwner`，那么父刚体的设置优先级更高。官方还特别说明：**不同 PhysicsScene 里的对象彼此不会碰撞**。所以大多数只有一个物理场景的项目里，这个值通常保持默认就行；只有做多场景 / 多 GPU 分摊时才需要认真指定。([NVIDIA Docs](https://docs.omniverse.nvidia.com/kit/docs/usdrt.scenegraph/7.6.2/api/classusdrt_1_1_usd_physics_collision_a_p_i.html "UsdPhysicsCollisionAPI — USDRT"))

**Approximation**  
这是 mesh collider 的“碰撞近似方式”，对应 `physics:approximation`。官方支持的主要选项有：`none`（直接用原始三角网格）、`meshSimplification`（先简化再用三角网格）、`convexHull`（单个凸包）、`convexDecomposition`（多个凸包）、`boundingSphere`（包围球）、`boundingCube`（包围盒）；另外 Omni Physics 还支持通过 `PhysxSDFMeshCollisionAPI` 使用 `sdf`（有符号距离场）。从官方性能建议看，**primitive collider 最省**，**convexHull / convexDecomposition** 是 mesh collider 里更高效的路线；**triangle mesh** 更适合大型、复杂、静态或运动学物体；而**动态刚体如果又想保留高细节的非凸碰撞**，官方建议用 **SDF**，否则就退回 convex decomposition。([NVIDIA Docs](https://docs.omniverse.nvidia.com/kit/docs/omni_usd_schema_physics/104.2/class_usd_physics_mesh_collision_a_p_i.html "UsdPhysicsMeshCollisionAPI Class Reference"))

**Hull Vertex Limit**  
这个属性出现在 `PhysxConvexHullCollisionAPI` 和 `PhysxConvexDecompositionCollisionAPI` 里，默认值都是 `64`。它限制的是**烹饪（cooking）后生成的每个凸包允许有多少顶点**，不是原始渲染网格的顶点数。实际效果上，可以把它理解成“凸包保真度上限”：值越大，通常越能贴近原网格；值越小，通常越快、越稳定，但外形会更粗。它只对 `convexHull` / `convexDecomposition` 这类近似有意义。另一个很实用的官方细节是：PhysX 说明，**convex mesh 总顶点数超过 64 时会失去 GPU collision compatibility**，这也是这个默认值很常见的原因之一。([NVIDIA Docs](https://docs.omniverse.nvidia.com/kit/docs/omni_usd_schema_physics/latest/physxschema/class_physx_schema_physx_convex_hull_collision_a_p_i.html "PhysxSchemaPhysxConvexHullCollisionAPI Class Reference"))

### Advanced

**Contact Offset**  
底层属性名是 `physxCollision:contactOffset`；Omni Physics 的说明文字有时把它叫 **Collision Offset**，但对应的就是 UI 里的 **Contact Offset**。它表示：**距离碰撞几何表面还有多远时，就开始生成接触点**。Schema 参考页给它的默认值是 `-inf`，表示让模拟器自动挑选；Collider guide 又补充说，这个自动值会综合场景重力、仿真时间步长和几何尺寸来决定。官方建议：如果你有**高速物体**或**很薄的物体**在一个时间步内“穿过去”（tunneling），可以把它调大；但它也会增加接触点数量，可能带来性能代价。官方给的替代方案是启用 **CCD**。([NVIDIA Docs](https://docs.omniverse.nvidia.com/kit/docs/omni_usd_schema_physics/latest/physxschema/class_physx_schema_physx_collision_a_p_i.html "PhysxSchemaPhysxCollisionAPI Class Reference"))

**Rest Offset**  
对应 `physxCollision:restOffset`。它表示：**真正把两物体视为“已经接触并稳定下来”的有效表面偏移**。Collider guide 明确写了：它可以是**正、零或负**；负值特别适合“渲染网格比碰撞网格略小”的情况，因为这样可以让视觉上的接触距离更自然。PhysX SDK 也说明，两物体最终静止时的距离等于双方 `restOffset` 之和：`0` 表示理想贴合，正值会留下空气隙，负值则允许一定“视觉上更合理”的穿入。需要注意的是，Omni Physics 的 schema 参考页把它写成了 `[0, contactOffset]`，这和 Collider guide / PhysX SDK 的表述有出入；这属于官方文档内部的不一致。我建议你在理解语义时，以 Collider guide 和 PhysX 的解释为主。([NVIDIA Docs](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/latest/dev_guide/rigid_bodies_articulations/collision.html "Colliders — Omni Physics"))

**Torsional Patch Radius**  
对应 `physxCollision:torsionalPatchRadius`，默认 `0`。官方定义是“用于施加**扭转摩擦（torsional friction）** 的接触 patch 半径”。通俗说，它控制的是物体在接触面上**绕接触法线拧转**擦。PhysX 进一步说明：如果这个值和 `Min Torsional Patch Radius` 都是 `0`，就**的接触 patch 半径”。通俗说，它控制的是物体在接触面上**；如果它大于 `0`，就会引入一定扭转摩擦，而且其效果与压入深度有关。PhysX 还说明，这个机制是为了近似“接触表面被压缩后带来的旋转摩擦”，并且只在 **不会施加 torsional friction** 下、且接触 patch 只有单个 anchor point 的一些情形里才会生效。([NVIDIA Docs]( https://docs.omniverse.nvidia.com/kit/docs/omni_usd_schema_physics/latest/physxschema/class_physx_schema_physx_collision_a_p_i.html "PhysxSchemaPhysxCollisionAPI Class Reference"))

**TGS solver**  
对应 `physxCollision:minTorsionalPatchRadius`，默认 `0`。它是 torsional friction patch 半径的**Min Torsional Patch Radius**。PhysX 的解释很直接：如果它为 `0`，扭转摩擦完全取决于 `torsionalPatchRadius`；如果它大于 `0`，那么**下限**。所以这个参数的作用更像是“扭转摩擦地板值”。([NVIDIA Docs](https://docs.omniverse.nvidia.com/kit/docs/omni_usd_schema_physics/latest/physxschema/class_physx_schema_physx_collision_a_p_i.html "PhysxSchemaPhysxCollisionAPI Class Reference"))

**无论压入深度或 `torsionalPatchRadius` 多小，都会保底施加一部分 torsional friction**  
这个属性属于 `PhysxConvexHullCollisionAPI` 和 `PhysxConvexDecompositionCollisionAPI`，默认值是 `0.001`，单位是距离。官方定义只有一句：**Min Thickness**。按它的名字和所在 schema 来理解，它是给凸包烹饪过程设置一个“最小厚度下限”，用来避免得到过薄、过尖、过退化的凸包特征；因此它通常只在 `convexHull` / `convexDecomposition` 路线下有意义。值调大，通常会让凸包更稳更厚实；值调小，通常更贴近原形，但更容易出现很薄的几何细节。([NVIDIA Docs](https://docs.omniverse.nvidia.com/kit/docs/omni_usd_schema_physics/latest/physxschema/class_physx_schema_physx_convex_hull_collision_a_p_i.html "PhysxSchemaPhysxConvexHullCollisionAPI Class Reference"))

### Inactive

**“Convex hull minimum thickness”**  
这个属性在官方 schema 里属于 `PhysxTriangleMeshCollisionAPI` 和 `PhysxTriangleMeshSimplificationCollisionAPI`。定义是：**Weld Tolerance**。默认值 `-inf` 表示按网格尺寸自动计算；设成 `0` 则显式关闭 welding。它本质上是三角网格 cooking 时的一个清理/合并阈值。因为它属于 triangle-mesh 路线的 schema，所以从官方 schema 归属来判断，它通常只在 `none` / `meshSimplification` 这类三角网格近似下才真正有意义；你这里它出现在 **mesh weld tolerance，控制多近的顶点会被焊接（weld）到一起**，大概率就是当前 `Approximation` 没走 triangle-mesh 这条路径。([NVIDIA Docs](https://docs.omniverse.nvidia.com/kit/docs/usdrt.scenegraph/7.5.1/api/classusdrt_1_1_physx_schema_physx_triangle_mesh_collision_a_p_i.html "PhysxSchemaPhysxTriangleMeshCollisionAPI — USDRT"))

### Contact Reporter

**Inactive**  
对应 `physxContactReport:threshold`，默认值是 `1`。官方定义是：**Contact Report Threshold**，单位是 `mass * distance / seconds^2`。也就是说，只有达到这个阈值的接触，才会触发 contact report。把它调低，会收到更多微小接触；把它调高，可以过滤掉轻微、短暂或噪声式接触。这个值同样受你 Stage 单位体系影响。([NVIDIA Docs](https://docs.omniverse.nvidia.com/kit/docs/omni_usd_schema_physics/latest/physxschema/class_physx_schema_physx_contact_report_a_p_i.html "PhysxSchemaPhysxContactReportAPI Class Reference"))

**接触报告的力阈值**  
这是 `PhysxContactReportAPI` 的 relationship。官方定义是：**Report Pairs**。这就是一个“只关注谁”的过滤器。比如你只关心夹爪和工件、轮子和地面、脚底和地板的接触，就把这些对象填进来；否则每个碰撞对象都会报。([NVIDIA Docs](https://docs.omniverse.nvidia.com/kit/docs/omni_usd_schema_physics/latest/physxschema/class_physx_schema_physx_contact_report_a_p_i.html "PhysxSchemaPhysxContactReportAPI Class Reference"))

顺便补一句：`Contact Reporter` 这一组不改变碰撞求解本身，它主要是给你输出接触事件/数据。官方文档把它定义为加在 **如果与这些对象发生接触，就发送 contact report；如果这个关系没设置，或者列表为空，就报告所有接触** 上的 API；订阅后可收到 `CONTACT_FOUND`、`CONTACT_PERSISTS`、`CONTACT_LOST` 等事件。Isaac Sim 里的 **rigid body 或 articulation prim** 其实也是构建在这个 Contact Report API 之上的更高层封装。([NVIDIA Docs](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/108.1/extensions/runtime/source/omni.physx/docs/dev_guide/contact_reports.html "Contact Reports — Omni Physics"))

给你一个面向 Isaac Sim 新手的实用记法：  
先决定 **Contact Sensor**；这个决定了后面哪些参数才有意义。`primitive` 最省，`convexHull / convexDecomposition` 是常见机器人资产选择，`triangle mesh` 更适合复杂静态物体，`SDF` 适合需要高细节非凸动态接触的情况。然后只有在出现**Approximation**时，再去动 `Contact Offset`、`Rest Offset`、torsional 参数和 `Contact Reporter`。([NVIDIA Docs](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/latest/dev_guide/rigid_bodies_articulations/collision.html "Colliders — Omni Physics"))

把你这个 prim 当前的 **穿透、接触太早/太晚、视觉接触不对、或者需要扭转摩擦 / 接触事件过滤**、它是不是 **Approximation 取值**、以及它是在 **Rigid Body** 贴出来，我可以继续帮你判断：你这张面板里哪些参数现在“真生效”，哪些只是显示出来但当前不会起作用。
