 #目标检测 

#损失函数 

#待处理 

[toc]

[原文](https://zhuanlan.zhihu.com/p/94799295)

# 一、IOU(Intersection over Union)

## **1. 特性 (优点)**

IoU 就是我们所说的**交并比**，是目标检测中最常用的指标，在 [anchor-based的方法](https://zhuanlan.zhihu.com/p/62372897) 中，他的作用不仅用来确定正样本和负样本，还可以用来评价输出框（predict box）和 ground-truth 的距离。

$$
IoU=\frac{|A \cap B|}{|A \cup B|}
$$

1.  可以说**它可以反映预测检测框与真实检测框的检测效果。** 
2.  还有一个很好的特性就是**尺度不变性**，也就是对尺度不敏感（scale invariant）， 在 regression 任务中，判断 predict box 和 gt 的距离最直接的指标就是 IoU。**(满足非负性；同一性；对称性；三角不等性)**

```python
import numpy as np
def Iou(box1, box2, wh=False):
    if wh == False:
	xmin1, ymin1, xmax1, ymax1 = box1
	xmin2, ymin2, xmax2, ymax2 = box2
    else:
	xmin1, ymin1 = int(box1[0]-box1[2]/2.0), int(box1[1]-box1[3]/2.0)
	xmax1, ymax1 = int(box1[0]+box1[2]/2.0), int(box1[1]+box1[3]/2.0)
	xmin2, ymin2 = int(box2[0]-box2[2]/2.0), int(box2[1]-box2[3]/2.0)
	xmax2, ymax2 = int(box2[0]+box2[2]/2.0), int(box2[1]+box2[3]/2.0)
    # 获取矩形框交集对应的左上角和右下角的坐标（intersection）
    xx1 = np.max([xmin1, xmin2])
    yy1 = np.max([ymin1, ymin2])
    xx2 = np.min([xmax1, xmax2])
    yy2 = np.min([ymax1, ymax2])	
    # 计算两个矩形框面积
    area1 = (xmax1-xmin1) * (ymax1-ymin1) 
    area2 = (xmax2-xmin2) * (ymax2-ymin2)
    inter_area = (np.max([0, xx2-xx1])) * (np.max([0, yy2-yy1]))　#计算交集面积
    iou = inter_area / (area1+area2-inter_area+1e-6) 　#计算交并比

    return iou
```

## **2. 作为损失函数会出现的问题 (缺点)**

1.  如果两个框没有相交，根据定义，IoU=0，不能反映两者的距离大小（重合度）。同时因为 loss=0，没有梯度回传，无法进行学习训练。
2.  IoU 无法精确的反映两者的重合度大小。如下图所示，三种情况 IoU 都相等，但看得出来他们的重合度是不一样的，左边的图回归的效果最好，右边的最差。

![](https://pic2.zhimg.com/v2-95449558cb098ff9df8c4d31474bd091_b.jpg)

# 二、GIOU(Generalized Intersection over Union)


## **1.来源**

在 CVPR2019 中，论文

的提出了 GIoU 的思想。由于 IoU 是**比值**的概念，对目标物体的 scale 是不敏感的。然而检测任务中的 BBox 的回归损失 (MSE loss, l1-smooth loss 等）优化和 IoU 优化不是完全等价的，而且 Ln 范数对物体的 scale 也比较敏感，IoU 无法直接优化没有重叠的部分。

这篇论文提出可以直接把 IoU 设为回归的 loss。

$$
GIoU=IoU-\frac{|A_c-U|}{A_c}
$$

_上面公式的意思是：先计算两个框的最小闭包区域面积_ $A_c$

 _(通俗理解：**同时包含了预测框和真实框**的最小框的面积)，再计算出 IoU，再计算闭包区域中不属于两个框的区域占闭包区域的比重，最后用 IoU 减去这个比重得到 GIoU。_

附：

## **2. 特性**

* 与 IoU 相似，GIoU 也是一种距离度量，作为损失函数的话，$L_{GIoU}=1-GIoU$,满足损失函数的基本要求

*   GIoU 对 scale 不敏感

*   GIoU 是 IoU 的下界，在两个框无限重合的情况下，IoU=GIoU=1

*   IoU 取值\[0,1\]，但 GIoU 有对称区间，取值范围\[-1,1\]。在两者重合的时候取最大值 1，在两者无交集且无限远的时候取最小值 -1，因此 GIoU 是一个非常好的距离度量指标。

* 与 IoU 只关注重叠区域不同，**GIoU 不仅关注重叠区域，还关注其他的非重合区域**，能更好的反映两者的重合度。

![](https://pic3.zhimg.com/v2-71e809433f6b26fcf30b4aeb6578413a_b.jpg)

```python
def Giou(rec1,rec2):
    #分别是第一个矩形左右上下的坐标
    x1,x2,y1,y2 = rec1 
    x3,x4,y3,y4 = rec2
    iou = Iou(rec1,rec2)
    area_C = (max(x1,x2,x3,x4)-min(x1,x2,x3,x4))*(max(y1,y2,y3,y4)-min(y1,y2,y3,y4))
    area_1 = (x2-x1)*(y1-y2)
    area_2 = (x4-x3)*(y3-y4)
    sum_area = area_1 + area_2

    w1 = x2 - x1   #第一个矩形的宽
    w2 = x4 - x3   #第二个矩形的宽
    h1 = y1 - y2
    h2 = y3 - y4
    W = min(x1,x2,x3,x4)+w1+w2-max(x1,x2,x3,x4)    #交叉部分的宽
    H = min(y1,y2,y3,y4)+h1+h2-max(y1,y2,y3,y4)    #交叉部分的高
    Area = W*H    #交叉的面积
    add_area = sum_area - Area    #两矩形并集的面积

    end_area = (area_C - add_area)/area_C    #闭包区域中不属于两个框的区域占闭包区域的比重
    giou = iou - end_area
    return giou
```

# 三、DIoU(Distance-IoU)


## **1.来源**

DIoU 要比 GIou 更加符合目标框回归的机制，**将目标与 anchor 之间的距离，重叠率以及尺度都考虑进去**，使得目标框回归变得更加稳定，不会像 IoU 和 GIoU 一样出现训练过程中发散等问题。论文中

> 基于 IoU 和 GIoU 存在的问题，作者提出了两个问题：  
> 1\. 直接最小化 anchor 框与目标框之间的归一化距离是否可行，以达到更快的收敛速度？  
> 2\. 如何使回归在与目标框有重叠甚至包含时更准确、更快？

$$
DIoU=IoU-\frac{\rho^2(b,b^{gt})}{c^2}
$$

其中， $b,b^{gt}$ 分别代表了预测框和真实框的中心点，且 $\rho$ 代表的是计算两个中心点间的欧式距离。 $c$ 代表的是能够同时包含预测框和真实框的**最小闭包区域**的对角线距离。

![](https://pic3.zhimg.com/v2-1e4b54001c4abdf392fe9d4877c83972_b.jpg)

DIoU 中对 anchor 框和目标框之间的归一化距离进行了建模

附：

<https://link.zhihu.com/?target=https>%3A<//github.com/Zzh-tju/DIoU-darknet>

## **2.优点**

* 与 GIoU loss 类似，DIoU loss（ $L_{DIoU}=1-DIoU$ ）在与目标框不重叠时，仍然可以为边界框提供移动方向。

*   DIoU loss 可以直接最小化两个目标框的距离，因此比 GIoU loss 收敛快得多。

* 对于包含两个框在水平方向和垂直方向上这种情况，DIoU 损失可以使回归非常快，而 GIoU 损失几乎退化为 IoU 损失。

*   DIoU 还可以替换普通的 IoU 评价策略，应用于 NMS 中，使得 NMS 得到的结果更加合理和有效。

## 3.实现代码 

```python
def Diou(bboxes1, bboxes2):
    rows = bboxes1.shape[0]
    cols = bboxes2.shape[0]
    dious = torch.zeros((rows, cols))
    if rows * cols == 0:#
        return dious
    exchange = False
    if bboxes1.shape[0] > bboxes2.shape[0]:
        bboxes1, bboxes2 = bboxes2, bboxes1
        dious = torch.zeros((cols, rows))
        exchange = True
    # #xmin,ymin,xmax,ymax->[:,0],[:,1],[:,2],[:,3]
    w1 = bboxes1[:, 2] - bboxes1[:, 0]
    h1 = bboxes1[:, 3] - bboxes1[:, 1] 
    w2 = bboxes2[:, 2] - bboxes2[:, 0]
    h2 = bboxes2[:, 3] - bboxes2[:, 1]
    
    area1 = w1 * h1
    area2 = w2 * h2

    center_x1 = (bboxes1[:, 2] + bboxes1[:, 0]) / 2 
    center_y1 = (bboxes1[:, 3] + bboxes1[:, 1]) / 2 
    center_x2 = (bboxes2[:, 2] + bboxes2[:, 0]) / 2
    center_y2 = (bboxes2[:, 3] + bboxes2[:, 1]) / 2

    inter_max_xy = torch.min(bboxes1[:, 2:],bboxes2[:, 2:]) 
    inter_min_xy = torch.max(bboxes1[:, :2],bboxes2[:, :2]) 
    out_max_xy = torch.max(bboxes1[:, 2:],bboxes2[:, 2:]) 
    out_min_xy = torch.min(bboxes1[:, :2],bboxes2[:, :2])

    inter = torch.clamp((inter_max_xy - inter_min_xy), min=0)
    inter_area = inter[:, 0] * inter[:, 1]
    inter_diag = (center_x2 - center_x1)**2 + (center_y2 - center_y1)**2
    outer = torch.clamp((out_max_xy - out_min_xy), min=0)
    outer_diag = (outer[:, 0] ** 2) + (outer[:, 1] ** 2)
    union = area1+area2-inter_area
    dious = inter_area / union - (inter_diag) / outer_diag
    dious = torch.clamp(dious,min=-1.0,max = 1.0)
    if exchange:
        dious = dious.T
    return dious
```

# 四、CIoU(**Complete-IoU**)

论文考虑到 bbox 回归三要素中的长宽比还没被考虑到计算中，因此，进一步在 DIoU 的基础上提出了 CIoU。其惩罚项如下面公式：

$$
R_{CIoU}=\frac{\rho^2(b,b^{gt})}{c^2}+\alpha \nu
$$

 其中 $\alpha$ 是权重函数定义为 $\alpha=\frac{\nu}{(1-IoU)+\nu}$，而 $\nu$ 用来度量长宽比的相似性，定义为 

 $$
 \nu=\frac{4}{\pi^2}(arctan \frac{w^{gt}}{h^{gt}}-arctan \frac{w}{h})^2
 $$

完整的 CIoU 损失函数定义：

$$
L_{CIoU}=1-IoU+\frac{\rho^2(b,b^{gt})}{c^2}+\alpha \nu
$$

最后，CIoU loss 的梯度类似于 DIoU loss，但还要考虑 $\nu$ 的梯度。

$$
\frac{\partial \nu}{\partial w}=\frac{8}{\pi^2}(arctan \frac{w^{gt}}{h^{gt}}-arctan \frac{w}{h}) \times \frac{h}{w^2+h^2}
$$

$$
\frac{\partial \nu}{\partial h}=\frac{8}{\pi^2}(arctan \frac{w^{gt}}{h^{gt}}-arctan \frac{w}{h}) \times \frac{w}{w^2+h^2}
$$

在长宽在 \[0,1\] 的情况下， $w^2+h^2$ 的值通常很小，会导致梯度爆炸，因此在 $\frac{1}{w^2+h^2}$ 实现时将替换成 1。

## 实现代码

```python
def bbox_overlaps_ciou(bboxes1, bboxes2):
    rows = bboxes1.shape[0]
    cols = bboxes2.shape[0]
    cious = torch.zeros((rows, cols))
    if rows * cols == 0:
        return cious
    exchange = False
    if bboxes1.shape[0] > bboxes2.shape[0]:
        bboxes1, bboxes2 = bboxes2, bboxes1
        cious = torch.zeros((cols, rows))
        exchange = True

    w1 = bboxes1[:, 2] - bboxes1[:, 0]
    h1 = bboxes1[:, 3] - bboxes1[:, 1]
    w2 = bboxes2[:, 2] - bboxes2[:, 0]
    h2 = bboxes2[:, 3] - bboxes2[:, 1]

    area1 = w1 * h1
    area2 = w2 * h2

    center_x1 = (bboxes1[:, 2] + bboxes1[:, 0]) / 2
    center_y1 = (bboxes1[:, 3] + bboxes1[:, 1]) / 2
    center_x2 = (bboxes2[:, 2] + bboxes2[:, 0]) / 2
    center_y2 = (bboxes2[:, 3] + bboxes2[:, 1]) / 2

    inter_max_xy = torch.min(bboxes1[:, 2:],bboxes2[:, 2:])
    inter_min_xy = torch.max(bboxes1[:, :2],bboxes2[:, :2])
    out_max_xy = torch.max(bboxes1[:, 2:],bboxes2[:, 2:])
    out_min_xy = torch.min(bboxes1[:, :2],bboxes2[:, :2])

    inter = torch.clamp((inter_max_xy - inter_min_xy), min=0)
    inter_area = inter[:, 0] * inter[:, 1]
    inter_diag = (center_x2 - center_x1)**2 + (center_y2 - center_y1)**2
    outer = torch.clamp((out_max_xy - out_min_xy), min=0)
    outer_diag = (outer[:, 0] ** 2) + (outer[:, 1] ** 2)
    union = area1+area2-inter_area
    u = (inter_diag) / outer_diag
    iou = inter_area / union
    with torch.no_grad():
        arctan = torch.atan(w2 / h2) - torch.atan(w1 / h1)
        v = (4 / (math.pi ** 2)) * torch.pow((torch.atan(w2 / h2) - torch.atan(w1 / h1)), 2)
        S = 1 - iou
        alpha = v / (S + v)
        w_temp = 2 * w1
    ar = (8 / (math.pi ** 2)) * arctan * ((w1 - w_temp) * h1)
    cious = iou - (u + alpha * ar)
    cious = torch.clamp(cious,min=-1.0,max = 1.0)
    if exchange:
        cious = cious.T
    return cious
```

# 五、损失函数在 YOLOv3 上的性能 (论文效果)

![](https://pic2.zhimg.com/v2-9d958cbbf6b13cddceeeb296c2f9a735_b.jpg)

# 参考

1.  特性参考 [https://zhuanlan.zhihu.com/p/57863810](https://zhuanlan.zhihu.com/p/57863810)
2.  DIoU 参考 [https://mp.weixin.qq.com/s?\_\_biz=MzUxNjcxMjQxNg==&mid=2247493985&idx=3&sn=23da3173b481d309903ec0371010d9f2&chksm=f9a19beeced612f81f94d22778481ffae16b25abf20973bf80917f9ff9b38b3f78ecd8237562&mpshare=1&scene=1&srcid=&sharer\_sharetime=1575276746557&sharer\_shareid=42a896371dfe6ebe8cc4cd474d9b747c&key=e2a6a5ccea4b8ce456e144f8db72f8becd6cfd3489f508fde8f890126594ca445adaf6bd6018077f94490c98f494d0eaf8c70165161be0cb274041ca9948ce62f6efe6e8bd9123a5b88be2b216b3da7e&ascene=1&uin=MjAyNTQwODM2NQ%3D%3D&devicetype=Windows+10&version=62070158&lang=zh\_CN&pass\_ticket=lZlnK6GAZ9ytbMcunsgTln9TaxVld4X1XGi8tTmIAmsi3d5CrasWo8RlWqYnGtqv](https://mp.weixin.qq.com/s?__biz=MzUxNjcxMjQxNg==&mid=2247493985&idx=3&sn=23da3173b481d309903ec0371010d9f2&chksm=f9a19beeced612f81f94d22778481ffae16b25abf20973bf80917f9ff9b38b3f78ecd8237562&mpshare=1&scene=1&srcid=&sharer_sharetime=1575276746557&sharer_shareid=42a896371dfe6ebe8cc4cd474d9b747c&key=e2a6a5ccea4b8ce456e144f8db72f8becd6cfd3489f508fde8f890126594ca445adaf6bd6018077f94490c98f494d0eaf8c70165161be0cb274041ca9948ce62f6efe6e8bd9123a5b88be2b216b3da7e&ascene=1&uin=MjAyNTQwODM2NQ%3D%3D&devicetype=Windows+10&version=62070158&lang=zh_CN&pass_ticket=lZlnK6GAZ9ytbMcunsgTln9TaxVld4X1XGi8tTmIAmsi3d5CrasWo8RlWqYnGtqv)
3.  DIOU 代码实现 [https://blog.csdn.net/TJMtaotao/article/details/103317267](https://blog.csdn.net/TJMtaotao/article/details/103317267)
4.  AAAI 2020 | DIoU 和 CIoU：IoU 在目标检测中的正确打开方式 [https://bbs.cvmart.net/articles/1396](https://bbs.cvmart.net/articles/1396)
5. <https://blog.csdn.net/TJMtaotao/article/details/103317267>