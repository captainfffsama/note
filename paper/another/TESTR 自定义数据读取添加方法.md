假设目前我们需要的读入的自定义数据和过去的方式都不一样，按照 [detectron 2 文档描述](https://detectron2.readthedocs.io/en/latest/tutorials/datasets.html#register-a-dataset)，需要进行的操作包含以下两点：

1. 先自定义一个获取数据集基本信息的方法，然后使用 `DatasetCatalog.register` 进行注册。
2. 定义个映射器 `Mapper` 从 `list[dict]` 中的 `dict` 将数据转换成模型训练需要的数据格式。

# 1. 定义数据集

获取数据集基本信息的函数签名通常如下：

```python
def my_dataset(args=None):
	....
	return list[dict]
# 注册数据集，注意此时 my_dataset 并不执行
DatasetCatalog.register('my_dataset',my_dataset)
# 获取数据集，此时调用 my_dataset 函数
dataset=DatasetCatalog.get('my_dataset')
```

`DatasetCatalog.get` 的接口并不支持给 `my_dataset` 直接传入参数，若需要通过外部参数（比如数据所在文件夹）来控制 `my_dataset` 行为，一种可行的方法是借助匿名函数，具体在 **TESTR** 实现的方法是：

```python
# 1. 在adet/data/builtin.py 
# 中定义一些内置全局变量来在python导入时，固定 mydataset 的参数
_CUSTOM_LABELME_DATASETS={
    "my_dataset": ("mydataset/data")
}
def register_all_coco(root="datasets") -> None:
	....
	for key,v in _CUSTOM_LABELME_DATASETS.items():
        register_number_point_instances(key,os.path.join(root,v))

register_all_coco()
# 2. 使用 register_number_point_instances 
# 借助匿名函数传入全局变量包装一层
def register_number_point_instances(name,data_dir):
    DatasetCatalog.register(name,lambda :my_dataset(data_dir))
```

通过上面代码，就可以在 python 执行时调用 `dataset=DatasetCatalog.get('my_dataset')` 时来传入 `"datasets/bj_read_num/data"` 参数。

#  2. 定义映射器

具体实现方式可以参考 `adet/data/dataset_mapper.py` 中的 `DatasetMapperWithBasis`