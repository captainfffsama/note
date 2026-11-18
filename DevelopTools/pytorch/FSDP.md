#pytorch #分布式训练

PyTorch **FSDP**（全称 **Fully Sharded Data Parallel**，全分片数据并行）是 PyTorch 提供的一种分布式训练技术，灵感来自 DeepSpeed ZeRO Stage 3。它可以将大规模模型的**模型参数、梯度和优化器状态**均匀分片（Shard）到多个 GPU（或 DDP 进程组）上，从而大幅降低单张 GPU 的内存占用，允许训练原本无法装入单卡的超大模型（如大型语言模型 LLM）。 [1](https://docs.pytorch.org/tutorials/intermediate/FSDP1_tutorial.html), [2](https://zhuanlan.zhihu.com/p/620333654), [3](https://github.com/pytorch/pytorch/blob/main/torch/distributed/fsdp/fully_sharded_data_parallel.py/), [4](https://www.youtube.com/watch?v=T13tYOGcclk)

核心工作流程

- **初始化分片**：在模型构造阶段，FSDP 将参数按 Rank 进行切片保存。

- **前向传播 (Forward)**：在运行某一层或模块前，通过 `all_gather` 操作从各个 Rank 收集并还原出完整的该层参数，完成计算后**立即丢弃**这些临时参数以释放显存。

- **反向传播 (Backward)**：同样通过 `all_gather` 还原完整参数进行梯度计算，随后通过 `reduce_scatter` 将计算出的梯度同步并分片到各个 Rank 上。

- **优化器更新**：每个 Rank 仅更新属于自己分片的那部分参数和优化器状态。 [1](https://zhuanlan.zhihu.com/p/620333654)

主要分片策略与配置

- **FULL_SHARD**（默认）：参数、梯度、优化器状态全部进行分片（类似 ZeRO-3）。

- **SHARD_GRAD_OP**：只对梯度和优化器状态分片，参数不分片（类似 ZeRO-2）。

- **NO_SHARD**：不分片，类似于普通的 DDP（DistributedDataParallel）。

- **HYBRID_SHARD / _HYBRID_SHARD_ZERO2**：节点内（Intra-node）分片，节点间（Inter-node）采用复制（Replicate），用于平衡通信开销与显存。

- **CPUOffload**：支持将暂时不用的参数和梯度卸载（Offload）到主机 CPU 内存中，进一步节省 GPU 显存。 [1](https://rocm.blogs.amd.com/artificial-intelligence/fsdp-training-pytorch/README.html), [2](https://github.com/pytorch/pytorch/blob/main/torch/distributed/fsdp/fully_sharded_data_parallel.py/), [3](https://zhuanlan.zhihu.com/p/620333654), [4](https://docs.pytorch.org/tutorials/intermediate/FSDP1_tutorial.html)

FSDP 与 FSDP2

- **FSDP (FSDP1)**：使用传统封装方式与 `auto_wrap_policy` 递归包装模块。

- **FSDP2**：基于 PyTorch 的 `DTensor`（分布式张量）构建，通过 `fully_shard` API 更简洁地对子模块和根模型进行分片，支持更高效的隐式/显式预取（Prefetching）与分布式检查点（Distributed Checkpoint, DCP）。