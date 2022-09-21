#分布式 

这里的 Reducer 来源于 [MapReduce](https://zhuanlan.zhihu.com/p/82399103) 编程模型，套用在深度学习分布式训练中，即各个 GPU 前向和后向计算出各个参数梯度的过程是 Map，可以称为 Mapper，而将各个 GPU 计算出的参数梯度汇总，计算和更新称为 Reducer