#pytorch #分布式

[toc]
# Pytorch中的分布式训练
截至1.6版本,pytorch 中分布式的方法大致可以分为三个方面:
- Distribution Data-Parallel Training(DDP): 该方法是将模型复制到每个进程上,然后在每个进程上喂不同的数据,DDP 负责将各个进程同步,收集梯度,计算,然后分布反传.这里的 DDP 包含了简单的数据并行的 DataParallel 和 常用的 DistributedDataParallel.
- RPC-Based Distributed Training(RPC): 这个是更加通用的方法
- Collective Communication(c10d): 这个库可以在组内,将 tensor 进行跨进程传送.实际上 DDP 和 RPC 都是建立在 c10d 上的.这个包的存在使得通信和计算解耦,通信可以进行更加细粒度的控制.

## DDP
单机多卡可以使用的 DDP 方法包括:
- DataParallel: 这个方法简易,需要改的代码很少,但是据说较慢,因为仅仅是模型并行
- DistributedDataParallel: 这个方法要改的代码多点,但是效率比上面好

多机多卡可以使用的:
- DistributedDataParallel 加上 launching script
- torchelastic 在训练的时候可以比较动态的加载资源.

### torch.nn.DataParallel
可以只改一行代码,但是性能不是最好的,不推荐使用.详细参加:https://pytorch.org/docs/master/generated/torch.nn.DataParallel.html
个人理解是这个是多线程的,主进程读入 batch ,然后起多个线程将 batch拆分 N 份,将自身模型参数复制 N 份到 N 个显卡,然后将多个显卡计算的梯度,通过多线程拿回主进程,然后主进程累加之后更新参数,然后重复.耗时点是多线程受 GIL 限制,且只有主进程更新权重,因此每次计算一个新的 batch 都需要将模型参数重新复制各个显卡上.

该方法属于使用的数据同步，同步更新，Parameter Server更新参数的方法。且使用单个进程管理所有GPU，汇总梯度用的是单个GPU，因此 GPU 负载不均衡，且网络通信负载大，不支持多机，不支持混合精度训练。

### [torch.nn.parallel.DistributedDataParallel](./distributed_data_parallel.md)
本方法比起 DataParallel 还要多加一步,即 `init_process_group`.并且本方法是基于多进程,不受 GIL 限制.并且每个模型副本是在一开始的时候构建的,而不是在前向传播的时候广播出去的.另外还用了一些其他的优化技术,具体可以参见 https://arxiv.org/abs/2006.15704. 

注意点:
- 由于 DDP 是借助 c10d 的 `ProcessGroup` 来进行通信,因此,程序在构建 DDP 前须先创建 `ProcessGroup`

## 参考资料:
- https://pytorch.org/tutorials/beginner/dist_overview.html
- https://pytorch.org/docs/stable/notes/ddp.html
