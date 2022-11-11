在`torch.sigmoid_`和`F.relu()` 中有 `inplace` 参数,用于指示张量是否进行原地操作.**若张量是在序列结构中,没有被其他分支引用,就可以设置这个参数,从而减小显存使用**

# 参考
- <https://zhuanlan.zhihu.com/p/350316775>