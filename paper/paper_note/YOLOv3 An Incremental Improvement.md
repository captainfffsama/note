 w#目标检测 

[toc]

# YOLOv3: An Incremental Improvement.
- 原始论文: <https://readpaper.com/paper/2796347433>
- 翻译参考: [参考1](https://zhuanlan.zhihu.com/p/37201615)  [参考2](https://zhuanlan.zhihu.com/p/37201615)
- 代码: [darknet](https://github.com/pjreddie/darknet)  [ultralytics](https://github.com/ultralytics/yolov5)

# 个人的一点学习
## 关于 Yolov3 中标签分配方式

参见 [YOLOV3](../../DL_knowlege/关于目标检测中的正负标签分类策略.md#YOLOV3)

## 关于 yolov3 中的损失函数

损失函数由置信度损失,分类损失和定位损失三部分构成. 可以参考 [视频](https://www.bilibili.com/video/BV1yi4y1g7ro?p=3&t=1554.6)

### 置信度损失

使用 Binary Cross Entropy

$$
L_{conf}(o,c)=-\frac{\sum_i{(o_iln(\hat{c}_i)+(1-o_i)ln(1-\hat{c}_i))}}{N}
$$

$$
\hat{c}_i=Sigmoid(c_i)
$$

其中 $o_i \in [0,1]$ ,表示预测目标边界看于真实目标边界框的 IOU , $c$ 为预测值, $\hat{c}_i$ 为 $c$ 通过 Sigmoid 函数得到的预测置信度, N 为正样本个数.

### 分类损失

$$
L_{cla}(O,C)=-\frac{\sum_{i \in pos}\sum_{j \in cla}{(O_{ij}ln(\hat{C}_{ij})+(1-O_{ij}ln(1-\hat{C}_{ij})))}}{N_{pos}}
$$

$$
\hat{C}_{ij}=Sigmoid(C_{ij})
$$

其中 $O_{ij} \in \{0,1\}$,表示预测边界框 $i$ 中是否存在第 $j$ 类目标, $C_{ij}$ 为预测值,$\hat{C}_{ij}$ 为 $C_{ij}$ 通过 Sigmoid 函数得到的目标概率, $N_{pos}$ 为正样本个数.

注意这里使用的是二值交叉熵,输出的概率使用的激活是 Sigmoid 而不是 softmax. 因此这里输出的置信度向量并不是一个概率分布,同样一个目标可能预测出两个标签.

### 定位损失

$$
L_{loc}(t,g)=\frac{\sum_{i \in pos}{(\sigma(t^i_x)-\hat{g}_x^i)^2+(\sigma(t^i_y)-\hat{g}_y^i)^2+(\sigma(t^i_w)-\hat{g}_w^i)^2}+(\sigma(t^i_h)-\hat{g}_h^i)^2}{N_{pos}}
$$

$$
\hat{g}^i_x=g^i_x-c^i_x
$$

$$
\hat{g}^i_y=g^i_y-c^i_y
$$

$$
\hat{g}^i_w=ln(g^i_w/p^i_w)
$$

$$
\hat{g}^i_h=ln(g^i_h/p^i_h)
$$

其中: $t_x,t_y,t_w,t_h$ 为网络预测的回归参数, $g_x,g_y,g_w,g_h$ 为 GT 中心点的坐标 x,y 和宽度,高度.(映射在 Grid 网格中)