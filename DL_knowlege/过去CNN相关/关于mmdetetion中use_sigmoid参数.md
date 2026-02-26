#目标检测 
#mmdetection

关于这个参数的作用可以参见:<https://github.com/open-mmlab/mmdetection/issues/3748>

`use_sigmoid=False` 代表使用 **cross entropy loss**,此时输出为目标类别+背景类.该情况下,每个目标仅有一个分类标签,分类结果之间是互斥的.
`use_sigmoid=True` 代表使用 **binary cross entropy loss**,此时输出最终数量没有背景类,针对每一类目标依次使用 `sigmoid` 进行二分类预测,预测当前目标是否属于该类.这种情况下,一个目标可能可以有多个分类标签.

注意在 mmdetetion 中, FocalLoss 系列的损失并不支持 `use_sigmoid=False`,据说原因是sigmoid的形式训练过程中会更稳定(参见RetinaNet中说法)
