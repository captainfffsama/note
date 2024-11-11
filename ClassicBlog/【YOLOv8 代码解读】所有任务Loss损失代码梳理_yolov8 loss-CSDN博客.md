YOLOv8官方将各类任务（目标检测，关键点检测，实例分割，旋转目标框检测，图像分类）的损失函数封装了在ultralytics\\utils\\loss.py中，本文主要梳理一下各类任务Loss的大致组成，不涉及到具体的原理。
-----------------------------------------------------------------------------------------------------------------

一、[目标检测](https://so.csdn.net/so/search?q=%E7%9B%AE%E6%A0%87%E6%A3%80%E6%B5%8B&spm=1001.2101.3001.7020)任务
--------------------------------------------------------------------------------------------------------

1.1 class v8DetectionLoss
-------------------------

YOLOv8目标检测任务主要由`分类损失`，和`矩形框回归损失`（iou loss，DFL loss）组成。  
![](https://i-blog.csdnimg.cn/blog_migrate/af6dc909a9a28f675d39eb92d270217c.png)
  
![](https://i-blog.csdnimg.cn/blog_migrate/804ae42ddad3b78afae42e4360f692ec.png)

1.2 [分类损失](https://so.csdn.net/so/search?q=%E5%88%86%E7%B1%BB%E6%8D%9F%E5%A4%B1&spm=1001.2101.3001.7020)
--------------------------------------------------------------------------------------------------------

`YOLOv8用的多分类损失是N个目标的二元交叉熵损失，而不是一般我们认为的多目标的softmax交叉熵损失。  
这里的BECWithLogitsLoss=BCELoss（二元交叉熵）+Sigmoid（激活函数）`  
![](https://i-blog.csdnimg.cn/blog_migrate/f400000125794b792ec81e2630fdb93b.png)
  
这里的分类损失是把N个目标的二元交叉熵损失求和，再取平均  
![](https://i-blog.csdnimg.cn/blog_migrate/be97d8f22f071f35f55036b846af0b8e.png)
  
二分类，多分类损失可以参考`http://t.csdnimg.cn/y89vH`

1.3 矩形框回归损失
-----------

YOLOv8用的矩形框损失主要由iou loss和DFL loss组成。  
![](https://i-blog.csdnimg.cn/blog_migrate/4ca76fdc64256edf1ceab68d963ae298.png)
  
![](https://i-blog.csdnimg.cn/blog_migrate/8b5edcbe62cf10d0f5ec1ad5c3571d16.png)

### 1.3.1 iou loss

iou loss有`CIoU，DIoU，GIoU` 三种loss可选择  
![](https://i-blog.csdnimg.cn/blog_migrate/ed50fcd46b2f5b920e85c84788725d44.png)

### 1.3.2 DFL loss

论文：`https://ieeexplore.ieee.org/document/9792391`  
`Distribution Focal Loss (DFL)` 是在 `Generalized Focal Loss（GLF）`中提出，用来让网络快速聚焦到标签附近的数值，使标签处的概率密度尽量大。思想是使用交叉熵函数，来优化标签y附近左右两个位置的概率，使网络分布聚焦到标签值附近。  
![](https://i-blog.csdnimg.cn/blog_migrate/dc3ab6381e1a45acd2a581805f6874e4.png)
  
![](https://i-blog.csdnimg.cn/blog_migrate/d0aa6da2aee4d68daf746cc216b149d8.png)

1.4 loss加权
----------

![](https://i-blog.csdnimg.cn/blog_migrate/db69afc613b1b64b6eb831dfe107c98e.png)

二、关键点检测任务（姿态估计）
---------------

2.1 class v8PoseLoss(v8DetectionLoss)
-------------------------------------

YOLOv8关键点检测任务主要由关键点相关的`回归损失`和`obj损失`和通过继承了目标检测任务的`分类损失`，和`矩形框回归损失`（iou loss，DFL loss）组成。

2.2 关键点损失
---------

关键点的损失主要由各个关键点的`obj loss`和`pose loss`组成。

### 2.2.1 obj loss

obj loss也是使用的N个关键点的二元交叉熵之和，表示当前关键点是否存在（可见）。  
![](https://i-blog.csdnimg.cn/blog_migrate/31690f1754e293fccb1c497187adb83f.png)
  
![](https://i-blog.csdnimg.cn/blog_migrate/79ced44eed81d09ef2e06c9ca121fdca.png)

### 2.2.2 pose loss

pose loss由每个关键点的预测坐标与gt坐标的L2损失，再除以gt矩形框面积，再除以比例系数计算而得。  
![](https://i-blog.csdnimg.cn/blog_migrate/33bc4af7b67397298f6e3edfd1b47905.png)
  
![](https://i-blog.csdnimg.cn/blog_migrate/f879c54f21bec3f2bdd845bb4140aa5c.png)

2.3 分类损失
--------

参考目标检测任务的分类损失1.2小节，继承的目标检测任务中的分类损失。  
![](https://i-blog.csdnimg.cn/blog_migrate/b43179c22aacc9ad489799be7006e8ad.png)

2.4 矩形框回归损失
-----------

参考目标检测任务的矩形框回归损失1.3小节，继承的目标检测任务中的矩形框回归损失。  
![](https://i-blog.csdnimg.cn/blog_migrate/374de84c9bf8476ce95abcc908e0e62d.png)

2.4 loss加权
----------

![](https://i-blog.csdnimg.cn/blog_migrate/f6a34f1ea549a093a4420f0bb0287a39.png)

三、实例分割任务
--------

3.1 class v8SegmentationLoss(v8DetectionLoss)
---------------------------------------------

YOLOv8关键点检测任务主要由分割相关的`损失`和通过继承了目标检测任务的`分类损失`，和`矩形框回归损失`（iou loss，DFL loss）组成。

3.2 分割损失
--------

分割损失主要是预测分割区域与gt分割区域进行逐像素的计算二元交叉熵损失。  
`F.binary_cross_entropy_with_logits()`对应的类是torch.nn.BCEWithLogitsLoss，在使用时会自动添加sigmoid，然后计算loss。（其实就是nn.sigmoid和nn.BCELoss的合体）  
![](https://i-blog.csdnimg.cn/blog_migrate/6265b6611ce3e0db6419429f1c020dfd.png)

3.3 分类损失
--------

参考目标检测任务的分类损失1.2小节，继承的目标检测任务中的分类损失。  
![](https://i-blog.csdnimg.cn/blog_migrate/265d9751cda60b97f77fdbf4e45582c2.png)

3.4 矩形框回归损失
-----------

参考目标检测任务的矩形框回归损失1.3小节，继承的目标检测任务中的矩形框回归损失。  
![](https://i-blog.csdnimg.cn/blog_migrate/1efb8bb77b4dccb4cc9710da4a3bb9a3.png)

3.5 loss加权
----------

![](https://i-blog.csdnimg.cn/blog_migrate/ea7ad75ff16e9c61df788c53ac39ba3d.png)

四、旋转目标框检测任务
-----------

4.1 class v8OBBLoss(v8DetectionLoss)
------------------------------------

YOLOv8旋转目标框检测任务主要由关键点相关的`回归损失`和`obj损失`和通过继承了目标检测任务的`分类损失`，和`矩形框回归损失`（iou loss，DFL loss）组成。

4.2 分类损失
--------

参考目标检测任务的分类损失1.2小节，继承的目标检测任务中的分类损失。  
![](https://i-blog.csdnimg.cn/blog_migrate/9025dead380e4aa01092aebb2926f743.png)
  
![](https://i-blog.csdnimg.cn/blog_migrate/0d620fb7105472f95c27603e9ed5974f.png)

4.3 `旋转矩形框`回归损失
---------------

YOLOv8-OBB用的旋转矩形框损失主要由`probiou loss`和DFL loss组成。

### 4.3.1 ProbIoU loss

yolov8-obb通过预测旋转框的左上角点，右下角点，角度，结合anchor点进行选择框解码。

```
`def dist2rbox(pred_dist, pred_angle, anchor_points, dim=-1):
    """
    Decode predicted object bounding box coordinates from anchor points and distribution.

    Args:
        pred_dist (torch.Tensor): Predicted rotated distance, (bs, h*w, 4).
        pred_angle (torch.Tensor): Predicted angle, (bs, h*w, 1).
        anchor_points (torch.Tensor): Anchor points, (h*w, 2).
    Returns:
        (torch.Tensor): Predicted rotated bounding boxes, (bs, h*w, 4).
    """
    # top_left, bottom_right  xaing dui ju li
    lt, rb = pred_dist.split(2, dim=dim)
    # angle
    cos, sin = torch.cos(pred_angle), torch.sin(pred_angle)
    # (bs, h*w, 1)
    xf, yf = ((rb - lt) / 2).split(1, dim=dim)
    x, y = xf * cos - yf * sin, xf * sin + yf * cos
    xy = torch.cat([x, y], dim=dim) + anchor_points
    return torch.cat([xy, lt + rb], dim=dim)` 

*   1
*   2
*   3
*   4
*   5
*   6
*   7
*   8
*   9
*   10
*   11
*   12
*   13
*   14
*   15
*   16
*   17
*   18
*   19
*   20


```

再通过prouiou函数，根据海林格距离（用来计算两个高斯概率分布之间的相似性）来计算两个旋转矩形框之间的相似度。  
ProbIoU原论文 ：https://arxiv.org/pdf/2106.06072v1.pdf

```
`def probiou(obb1, obb2, CIoU=False, eps=1e-7):
    """
    Calculate the prob iou between oriented bounding boxes, https:

    Args:
        obb1 (torch.Tensor): A tensor of shape (N, 5) representing ground truth obbs, with xywhr format.
        obb2 (torch.Tensor): A tensor of shape (N, 5) representing predicted obbs, with xywhr format.
        eps (float, optional): A small value to avoid division by zero. Defaults to 1e-7.

    Returns:
        (torch.Tensor): A tensor of shape (N, ) representing obb similarities.
    """
    x1, y1 = obb1[..., :2].split(1, dim=-1)
    x2, y2 = obb2[..., :2].split(1, dim=-1)
    a1, b1, c1 = _get_covariance_matrix(obb1)
    a2, b2, c2 = _get_covariance_matrix(obb2)

    t1 = (
        ((a1 + a2) * (torch.pow(y1 - y2, 2)) + (b1 + b2) * (torch.pow(x1 - x2, 2)))
        / ((a1 + a2) * (b1 + b2) - (torch.pow(c1 + c2, 2)) + eps)
    ) * 0.25
    t2 = (((c1 + c2) * (x2 - x1) * (y1 - y2)) / ((a1 + a2) * (b1 + b2) - (torch.pow(c1 + c2, 2)) + eps)) * 0.5
    t3 = (
        torch.log(
            ((a1 + a2) * (b1 + b2) - (torch.pow(c1 + c2, 2)))
            / (4 * torch.sqrt((a1 * b1 - torch.pow(c1, 2)).clamp_(0) * (a2 * b2 - torch.pow(c2, 2)).clamp_(0)) + eps)
            + eps
        )
        * 0.5
    )
    bd = t1 + t2 + t3
    bd = torch.clamp(bd, eps, 100.0)
    hd = torch.sqrt(1.0 - torch.exp(-bd) + eps)
    iou = 1 - hd
    if CIoU:  # only include the wh aspect ratio part
        w1, h1 = obb1[..., 2:4].split(1, dim=-1)
        w2, h2 = obb2[..., 2:4].split(1, dim=-1)
        v = (4 / math.pi**2) * (torch.atan(w2 / h2) - torch.atan(w1 / h1)).pow(2)
        with torch.no_grad():
            alpha = v / (v - iou + (1 + eps))
        return iou - v * alpha  # CIoU
    return iou` 

*   1
*   2
*   3
*   4
*   5
*   6
*   7
*   8
*   9
*   10
*   11
*   12
*   13
*   14
*   15
*   16
*   17
*   18
*   19
*   20
*   21
*   22
*   23
*   24
*   25
*   26
*   27
*   28
*   29
*   30
*   31
*   32
*   33
*   34
*   35
*   36
*   37
*   38
*   39
*   40
*   41
*   42


```

### 4.3.2 DFL loss

参考目标检测任务1.3.2章节。