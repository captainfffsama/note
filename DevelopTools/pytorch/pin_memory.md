#pytorch

 在 [pytorch](pytorch.md)  创建 DataLoader 时,设置 ``pin_memory=True``,表示使用[锁页内存](pin_memory.md#%E9%94%81%E9%A1%B5%E5%86%85%E5%AD%98),此时意味着生成的 Tensor 数据最开始是属于内存中的锁页内存,这样将内存中的 Tensor 转义到 GPU 上的显存会更加快速一些.
 当计算机内存充足时,设置 ``pin_memory=True``.当系统卡住或者交换内存Swap使用过多的时候,注意设置 ``pin_memory=False``.注意 pin_memory 默认是 False.

# 锁页内存
主机中的内存分为锁页内存和不锁页内存两种,锁页内存存放的内容在任何情况下都不会和主机的虚拟内存(虚拟内存是硬盘空间)进行交换,而不锁页内存在主机内存不足时,会将数据存放在虚拟内存中.
显存全部都是锁页内存,因此显存不支持和硬盘进行交换.
