#并发   
期物表示中将发生的事情，而确定某件事会发生的唯一方式是执行的时间已经排定。

因此，只有排定把某件事交给`concurrent.futures.Executor`子类处理时，才会创建`concurrent.futures.Future`实例。

因此在 python 中使用`concurrent.futures`的流程一般是：先通过一个 for 循环来创建并排定期物，然后在对排定的期物序列进行查询获得期物结果。