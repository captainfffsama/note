[toc]
[原文](https://blog.csdn.net/m0_51004308/article/details/122841046)

# IOU-loss

![](https://img-blog.csdnimg.cn/e327ddc1f6e24b7fab203810dfad2045.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAWlpZX2Rs,size_12,color_FFFFFF,t_70,g_se,x_16)

$$
IoU=\frac{|A \cap B|}{|A \cup B|}
$$

## 算法作用：
Iou的就是交并比，预测框和真实框相交区域面积和合并区域面积的比值，计算公式如下，Iou作为损失函数的时候只要将其对数值输出就好了。

## 算法代码：

```python
def Iou_loss(preds, bbox, eps=1e-6, reduction='mean'):
    '''
    preds:[[x1,y1,x2,y2], [x1,y1,x2,y2],,,]
    bbox:[[x1,y1,x2,y2], [x1,y1,x2,y2],,,]
    reduction:"mean"or"sum"
    return: loss
    '''
    x1 = torch.max(preds[:, 0], bbox[:, 0])
    y1 = torch.max(preds[:, 1], bbox[:, 1])
    x2 = torch.min(preds[:, 2], bbox[:, 2])
    y2 = torch.min(preds[:, 3], bbox[:, 3])

    w = (x2 - x1 + 1.0).clamp(0.)
    h = (y2 - y1 + 1.0).clamp(0.)
    inters = w * h
    print("inters:\n",inters)

    uni = (preds[:, 2] - preds[:, 0] + 1.0) * (preds[:, 3] - preds[:, 1] + 1.0) + (bbox[:, 2] - bbox[:, 0] + 1.0) * (
            bbox[:, 3] - bbox[:, 1] + 1.0) - inters
    print("uni:\n",uni)
    ious = (inters / uni).clamp(min=eps)
    loss = -ious.log()

    if reduction == 'mean':
        loss = torch.mean(loss)
    elif reduction == 'sum':
        loss = torch.sum(loss)
    else:
        raise NotImplementedError
    print("last_loss:\n",loss)
    return loss
if __name__ == "__main__":
    pred_box = torch.tensor([[2,4,6,8],[5,9,13,12]])
    gt_box = torch.tensor([[3,4,7,9]])
    loss = Iou_loss(preds=pred_box,bbox=gt_box)


"""
inters:
 tensor([20.,  3.])
uni:
 tensor([35., 63.])
last_loss:
 tensor(1.8021)
"""
```

# GIOU-loss


## 问题分析
当预测框和真实框不相交时Iou值为0，导致很大范围内损失函数没有梯度。针对这一问题，提出了Giou作为损失函数。

## 算法公式及其解释：
其实想法也很简单（但这一步很难）：假如现在有两个box A，B，我们找到一个最小的封闭形状C，让C可以把A，B包含在内，然后再计算C中没有覆盖A和B的面积占C总面积的比值，然后用A与B的IoU减去这个比值：

$$
GIoU=IoU-\frac{|C-A \cup B|}{|C|}
$$
GIoU其实一种泛化版的IoU，是IoU的下界，与IoU只关注重叠部分不同，GIoU不仅关注重叠区域，还关注其他非重合区域，能更好地反映二者的重合度。

## 算法代码：

```python
def Giou_loss(preds, bbox, eps=1e-7, reduction='mean'):
    '''
    https://github.com/sfzhang15/ATSS/blob/master/atss_core/modeling/rpn/atss/loss.py#L36
    :param preds:[[x1,y1,x2,y2], [x1,y1,x2,y2],,,]
    :param bbox:[[x1,y1,x2,y2], [x1,y1,x2,y2],,,]
    :return: loss
    '''
    ix1 = torch.max(preds[:, 0], bbox[:, 0])
    iy1 = torch.max(preds[:, 1], bbox[:, 1])
    ix2 = torch.min(preds[:, 2], bbox[:, 2])
    iy2 = torch.min(preds[:, 3], bbox[:, 3])

    iw = (ix2 - ix1 + 1.0).clamp(0.)
    ih = (iy2 - iy1 + 1.0).clamp(0.)


    inters = iw * ih
    print("inters:\n",inters)

    uni = (preds[:, 2] - preds[:, 0] + 1.0) * (preds[:, 3] - preds[:, 1] + 1.0) + (bbox[:, 2] - bbox[:, 0] + 1.0) * (
            bbox[:, 3] - bbox[:, 1] + 1.0) - inters + eps
    print("uni:\n",uni)

    ious = inters / uni
    print("Iou:\n",ious)
    ex1 = torch.min(preds[:, 0], bbox[:, 0])
    ey1 = torch.min(preds[:, 1], bbox[:, 1])
    ex2 = torch.max(preds[:, 2], bbox[:, 2])
    ey2 = torch.max(preds[:, 3], bbox[:, 3])
    ew = (ex2 - ex1 + 1.0).clamp(min=0.)
    eh = (ey2 - ey1 + 1.0).clamp(min=0.)


    enclose = ew * eh + eps
    print("enclose:\n",enclose)

    giou = ious - (enclose - uni) / enclose
    loss = 1 - giou

    if reduction == 'mean':
        loss = torch.mean(loss)
    elif reduction == 'sum':
        loss = torch.sum(loss)
    else:
        raise NotImplementedError
    print("last_loss:\n",loss)
    return loss
if __name__ == "__main__":
    pred_box = torch.tensor([[2,4,6,8],[5,9,13,12]])
    gt_box = torch.tensor([[3,4,7,9]])
    loss = Giou_loss(preds=pred_box,bbox=gt_box)


"""
inters:
 tensor([20.,  3.])
uni:
 tensor([35., 63.])
Iou:
 tensor([0.5714, 0.0476])
enclose:
 tensor([36., 99.])
last_loss:
 tensor(0.8862)
"""
```

# DIOU-loss


![](https://img-blog.csdnimg.cn/ebb243b25bba44aca1e301bd6f471cb3.png)

## 问题分析
虽然GIoU可以缓解不重叠情况下梯度消失问题，但是它依然存在一些局限性。
![](https://img-blog.csdnimg.cn/a10923b97951474f96c273055b64d7ea.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAWlpZX2Rs,size_20,color_FFFFFF,t_70,g_se,x_16)

基于IoU和GIoU存在的问题，作者提出了**两个问题**：

*   直接最小化anchor框与目标框之间的归一化距离是否可行，以达到更快的收敛速度？
*   如何使回归在与目标框有重叠甚至包含时更准确、更快？

其实在边界框回归的过程中需要考虑的几点：

*   重叠面积
*   中心点距离
*   长宽比

实际上，无论是IoU还是GIoU都只考虑了第一点重叠面积。因此提出的DIoU中考虑了第二点中心点距离。

算法公式及其解释：
$$
DIoU=IoU-\frac{\rho(b^p,b^g)}{c^2}
$$

其中， $b,b^{gt}$ 分别代表了预测框和真实框的中心点，且 $\rho$ 代表的是计算两个中心点间的欧式距离。 $c$ 代表的是能够同时包含预测框和真实框的**最小闭包区域**的对角线距离。DIoU loss为$L_{DIoU}=1-DIoU$

DIoU在IoU的基础上加入了一个惩罚项，用于度量目标框和预测框之间中心点的距离。
## 算法代码：

```python
def Diou_loss(preds, bbox, eps=1e-7, reduction='mean'):
    '''
    preds:[[x1,y1,x2,y2], [x1,y1,x2,y2],,,]
    bbox:[[x1,y1,x2,y2], [x1,y1,x2,y2],,,]
    eps: eps to avoid divide 0
    reduction: mean or sum
    return: diou-loss
    '''
    ix1 = torch.max(preds[:, 0], bbox[:, 0])
    iy1 = torch.max(preds[:, 1], bbox[:, 1])
    ix2 = torch.min(preds[:, 2], bbox[:, 2])
    iy2 = torch.min(preds[:, 3], bbox[:, 3])

    iw = (ix2 - ix1 + 1.0).clamp(min=0.)
    ih = (iy2 - iy1 + 1.0).clamp(min=0.)


    inters = iw * ih


    uni = (preds[:, 2] - preds[:, 0] + 1.0) * (preds[:, 3] - preds[:, 1] + 1.0) + (bbox[:, 2] - bbox[:, 0] + 1.0) * (
            bbox[:, 3] - bbox[:, 1] + 1.0) - inters


    iou = inters / (uni + eps)
    print("iou:\n",iou)


    cxpreds = (preds[:, 2] + preds[:, 0]) / 2
    cypreds = (preds[:, 3] + preds[:, 1]) / 2

    cxbbox = (bbox[:, 2] + bbox[:, 0]) / 2
    cybbox = (bbox[:, 3] + bbox[:, 1]) / 2

    inter_diag = (cxbbox - cxpreds) ** 2 + (cybbox - cypreds) ** 2
    print("inter_diag:\n",inter_diag)


    ox1 = torch.min(preds[:, 0], bbox[:, 0])
    oy1 = torch.min(preds[:, 1], bbox[:, 1])
    ox2 = torch.max(preds[:, 2], bbox[:, 2])
    oy2 = torch.max(preds[:, 3], bbox[:, 3])

    outer_diag = (ox1 - ox2) ** 2 + (oy1 - oy2) ** 2
    print("outer_diag:\n",outer_diag)

    diou = iou - inter_diag / outer_diag
    diou = torch.clamp(diou, min=-1.0, max=1.0)

    diou_loss = 1 - diou
    print("last_loss:\n",diou_loss)

    if reduction == 'mean':
        loss = torch.mean(diou_loss)
    elif reduction == 'sum':
        loss = torch.sum(diou_loss)
    else:
        raise NotImplementedError
    return loss
if __name__ == "__main__":
    pred_box = torch.tensor([[2,4,6,8],[5,9,13,12]])
    gt_box = torch.tensor([[3,4,7,9]])
    loss = Diou_loss(preds=pred_box,bbox=gt_box)


"""
iou:
 tensor([0.5714, 0.0476])
inter_diag:
 tensor([ 1, 32])
outer_diag:
 tensor([ 50, 164])
last_loss:
 tensor([0.4286, 0.9524])
"""
```

# CIOU-loss


## 问题分析
Ciou在Diou的基础上进行改进，认为Diou只考虑了中心点距离和重叠面积，但是没有考虑到长宽比。

算法公式及其解释：
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
## 算法代码：

```python
import math
def Ciou_loss(preds, bbox, eps=1e-7, reduction='mean'):
    '''
    https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/loss/multibox_loss.py
    :param preds:[[x1,y1,x2,y2], [x1,y1,x2,y2],,,]
    :param bbox:[[x1,y1,x2,y2], [x1,y1,x2,y2],,,]
    :param eps: eps to avoid divide 0
    :param reduction: mean or sum
    :return: diou-loss
    '''
    ix1 = torch.max(preds[:, 0], bbox[:, 0])
    iy1 = torch.max(preds[:, 1], bbox[:, 1])
    ix2 = torch.min(preds[:, 2], bbox[:, 2])
    iy2 = torch.min(preds[:, 3], bbox[:, 3])

    iw = (ix2 - ix1 + 1.0).clamp(min=0.)
    ih = (iy2 - iy1 + 1.0).clamp(min=0.)


    inters = iw * ih


    uni = (preds[:, 2] - preds[:, 0] + 1.0) * (preds[:, 3] - preds[:, 1] + 1.0) + (bbox[:, 2] - bbox[:, 0] + 1.0) * (
            bbox[:, 3] - bbox[:, 1] + 1.0) - inters


    iou = inters / (uni + eps)
    print("iou:\n",iou)


    cxpreds = (preds[:, 2] + preds[:, 0]) / 2
    cypreds = (preds[:, 3] + preds[:, 1]) / 2

    cxbbox = (bbox[:, 2] + bbox[:, 0]) / 2
    cybbox = (bbox[:, 3] + bbox[:, 1]) / 2

    inter_diag = (cxbbox - cxpreds) ** 2 + (cybbox - cypreds) ** 2


    ox1 = torch.min(preds[:, 0], bbox[:, 0])
    oy1 = torch.min(preds[:, 1], bbox[:, 1])
    ox2 = torch.max(preds[:, 2], bbox[:, 2])
    oy2 = torch.max(preds[:, 3], bbox[:, 3])

    outer_diag = (ox1 - ox2) ** 2 + (oy1 - oy2) ** 2

    diou = iou - inter_diag / outer_diag
    print("diou:\n",diou)


    wbbox = bbox[:, 2] - bbox[:, 0] + 1.0
    hbbox = bbox[:, 3] - bbox[:, 1] + 1.0
    wpreds = preds[:, 2] - preds[:, 0] + 1.0
    hpreds = preds[:, 3] - preds[:, 1] + 1.0
    v = torch.pow((torch.atan(wbbox / hbbox) - torch.atan(wpreds / hpreds)), 2) * (4 / (math.pi ** 2))
    alpha = v / (1 - iou + v)
    ciou = diou - alpha * v
    ciou = torch.clamp(ciou, min=-1.0, max=1.0)

    ciou_loss = 1 - ciou
    if reduction == 'mean':
        loss = torch.mean(ciou_loss)
    elif reduction == 'sum':
        loss = torch.sum(ciou_loss)
    else:
        raise NotImplementedError
    print("last_loss:\n",loss)
    return loss
if __name__ == "__main__":
    pred_box = torch.tensor([[2,4,6,8],[5,9,13,12]])
    gt_box = torch.tensor([[3,4,7,9]])
    loss = Ciou_loss(preds=pred_box,bbox=gt_box)


"""
iou:
 tensor([0.5714, 0.0476])
diou:
 tensor([0.5714, 0.0476])
last_loss:
 tensor(0.6940)
"""
```

# EIOU-loss和Focal EIOU-loss

文章链接：[点击](https://link.zhihu.com/?target=https://arxiv.org/pdf/2101.08158.pdf)

解决问题
CIOU Loss虽然考虑了**边界框回归的重叠面积、中心点距离、纵横比**。但是通过其公式中的v反映的纵横比的差异，而不是宽高分别与其置信度的真实差异，所以有时会**阻碍模型有效的优化相似性**。针对这一问题，有学者在CIOU的基础上将纵横比拆开，提出了EIOU Loss，并且加入Focal聚焦优质的锚框，该方法出自于2021年的一篇文章《**[Focal and Efficient IOU Loss for Accurate Bounding Box Regression](https://arxiv.org/abs/2101.08158)**》

算法公式及其解释：
$$
L_{EIoU}=L_{IoU}+L_{dis}+L_{asp} \\
=1-IoU+\frac{\rho^2(b,b^{gt})}{c^2}+\frac{\rho^2(w,w^{gt})}{C^2_w}+\frac{\rho^2(h,h^{gt})}{C^2_h}
$$


 **该损失函数包含三个部分：重叠损失，中心距离损失，宽高损失**，前两部分延续CIOU中的方法，但是宽高损失直接使目标盒与锚盒的宽度和高度之差最小，使得收敛速度更快。其中 Cw 和 Ch 是覆盖两个Box的最小外接框的宽度和高度。考虑到 **BBox的回归中也存在训练样本不平衡的问题**，即在一张图像中回归误差小的高质量锚框的数量远少于误差大的低质量样本，质量较差的样本会产生过大的梯度影响训练过程。作者在 **EIOU的基础上结合Focal Loss提出一种Focal EIOU Loss**，梯度的角度出发，把高质量的锚框和低质量的锚框分开，公式如下

$$
L_{Focal-EIoU}=IoU^{\gamma}L_{EIoU}
$$


其中$IoU = |A \cap B|/|A \cup B|$， $\gamma$为控制异常值抑制程度的参数。该损失中的Focal与传统的Focal Loss有一定的区别，传统的Focal Loss针对越困难的样本损失越大，起到的是困难样本挖掘的作用；而根据上述公式：IOU越高的损失越大，相当于加权作用，给越好的回归目标一个越大的损失，有助于提高回归精度。

存在的问题
本文针对边界框回归任务，在之前基于CIOU损失的基础上提出了两个优化方法：

*   将纵横比的损失项拆分成预测的宽高分别与最小外接框宽高的差值，加速了收敛提高了回归精度；
*   引入了Focal Loss优化了边界框回归任务中的样本不平衡问题，即减少与目标框重叠较少的大量锚框对BBox 回归的优化贡献，使回归过程专注于高质量锚框。

不足之处或许在于Focal的表达形式是否有待改进。

算法代码

```python
def bbox_iou(box1, box2, x1y1x2y2=True, GIoU=False, DIoU=False, CIoU=False,  EIoU=False, eps=1e-7):

    box2 = box2.T


    if x1y1x2y2:
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
    else:
        b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
        b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
        b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
        b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2


    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)


    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
    union = w1 * h1 + w2 * h2 - inter + eps

    iou = inter / union
    if GIoU or DIoU or CIoU or EIoU:
        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)
        if CIoU or DIoU or EIoU:
            c2 = cw ** 2 + ch ** 2 + eps
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 +
                    (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4
            if DIoU:
                return iou - rho2 / c2
            elif CIoU:
                v = (4 / math.pi ** 2) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
                with torch.no_grad():
                    alpha = v / (v - iou + (1 + eps))
                return iou - (rho2 / c2 + v * alpha)
            elif EIoU:
                rho_w2 = ((b2_x2 - b2_x1) - (b1_x2 - b1_x1)) ** 2
                rho_h2 = ((b2_y2 - b2_y1) - (b1_y2 - b1_y1)) ** 2
                cw2 = cw ** 2 + eps
                ch2 = ch ** 2 + eps
                return iou - (rho2 / c2 + rho_w2 / cw2 + rho_h2 / ch2)
        else:
            c_area = cw * ch + eps
            return iou - (c_area - union) / c_area
    else:
        return iou
```

# alpha IOU

## 算法代码：

```python
def bbox_alpha_iou(box1, box2, x1y1x2y2=False, GIoU=False, DIoU=False, CIoU=False, alpha=3, eps=1e-7):

    box2 = box2.T


    if x1y1x2y2:
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
    else:
        b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
        b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
        b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
        b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2


    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)


    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
    union = w1 * h1 + w2 * h2 - inter + eps



    iou = torch.pow(inter/union + eps, alpha)

    if GIoU or DIoU or CIoU:
        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)
        if CIoU or DIoU:
            c2 = (cw ** 2 + ch ** 2) ** alpha + eps
            rho_x = torch.abs(b2_x1 + b2_x2 - b1_x1 - b1_x2)
            rho_y = torch.abs(b2_y1 + b2_y2 - b1_y1 - b1_y2)
            rho2 = ((rho_x ** 2 + rho_y ** 2) / 4) ** alpha
            if DIoU:
                return iou - rho2 / c2
            elif CIoU:
                v = (4 / math.pi ** 2) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
                with torch.no_grad():
                    alpha_ciou = v / ((1 + eps) - inter / union + v)

                return iou - (rho2 / c2 + torch.pow(v * alpha_ciou + eps, alpha))
        else:


            c_area = torch.max(cw * ch + eps, union)
            return iou - torch.pow((c_area - union) / c_area + eps, alpha)
    else:
        return iou
```

# SIOU-loss


文章链接：[点击](https://arxiv.org/ftp/arxiv/papers/2205/2205.12740.pdf)

解决问题
已有方法匹配真实框和预测框之间的IoU、中心点距离、宽高比等，它们均未考虑真实框和预测框之间不匹配的方向。这种不足导致收敛速度较慢且效率较低，因为预测框可能在训练过程中“徘徊”，最终生成更差的模型。

算法优势

SCYLLA-IoU（SIoU），考虑到期望回归之间向量的角度，重新定义角度惩罚度量，它可以使预测框快速漂移到最近的轴，随后则只需要回归一个坐标（X或Y），这有效地减少了自由度的总数。

算法公式及其解释：
![](https://img-blog.csdnimg.cn/514da45305284e8eb7570c8c173c8fae.png)

其中， Intersection  表示检测框和真实框的交集面积， Union 表示检测框和真实框的并集面积， Scylla 是指检测框与真实框的长宽比例差异，可以定义为：
![](https://img-blog.csdnimg.cn/f80a44b9d4254a8092bb1982c6d68928.png)

其中， widthd \\text{width}{\\text{d}} widthd 和 heightd \\text{height}{\\text{d}} heightd 分别表示检测框的宽度和高度， widthg \\text{width}{\\text{g}} widthg 和 heightg \\text{height}{\\text{g}} heightg 分别表示真实框的宽度和高度。

Scylla-IoU的取值范围是 \[ 0 , 1 \] \[0,1\] \[0,1\]，其值越大表示检测框与真实框的匹配程度越高，当其值等于1时，表示检测框与真实框完全重合。

存在的问题
Scylla-IoU是一种用于目标检测模型评估的指标，它在计算IoU时采用了多个不同阈值，从而能够更全面地评估模型的性能。然而，Scylla-IoU也存在一些问题，主要包括以下几点：

*   模型评估结果的可重复性问题：Scylla-IoU采用了多个不同的IoU阈值，这意味着对于同一组测试数据，在不同的IoU阈值下，同一个模型的评估结果可能会有所不同。这会给模型的评估和比较带来困难。

*   IoU阈值的选择问题：Scylla-IoU需要指定多个IoU阈值，而这些阈值的选择通常需要根据具体的数据集和任务进行调整。不同的IoU阈值选择可能会导致评估结果的不同，从而使得模型的比较变得困难。

*   可解释性问题：Scylla-IoU的计算过程较为复杂，需要对多个IoU阈值进行计算并进行加权平均。这可能会使得结果难以解释，给结果的可信度带来一定影响。


因此，在使用Scylla-IoU进行模型评估时，需要注意这些问题，并综合考虑使用其他指标进行辅助评估。

算法代码

```python
import torch

def calculate_iou(boxes_a, boxes_b):
    """
    Calculate Intersection over Union (IoU) of two bounding box tensors.
    :param boxes_a: Tensor of shape (N, 4) representing bounding boxes (x1, y1, x2, y2)
    :param boxes_b: Tensor of shape (M, 4) representing bounding boxes (x1, y1, x2, y2)
    :return: Tensor of shape (N, M) representing IoU between all pairs of boxes_a and boxes_b
    """

    x1 = torch.max(boxes_a[:, 0].unsqueeze(1), boxes_b[:, 0].unsqueeze(0))
    y1 = torch.max(boxes_a[:, 1].unsqueeze(1), boxes_b[:, 1].unsqueeze(0))
    x2 = torch.min(boxes_a[:, 2].unsqueeze(1), boxes_b[:, 2].unsqueeze(0))
    y2 = torch.min(boxes_a[:, 3].unsqueeze(1), boxes_b[:, 3].unsqueeze(0))


    intersection_area = torch.clamp(x2 - x1 + 1, min=0) * torch.clamp(y2 - y1 + 1, min=0)


    boxes_a_area = (boxes_a[:, 2] - boxes_a[:, 0] + 1) * (boxes_a[:, 3] - boxes_a[:, 1] + 1)
    boxes_b_area = (boxes_b[:, 2] - boxes_b[:, 0] + 1) * (boxes_b[:, 3] - boxes_b[:, 1] + 1)


    union_area = boxes_a_area.unsqueeze(1) + boxes_b_area.unsqueeze(0) - intersection_area


    iou = intersection_area / union_area

    return iou

def scylla_iou_loss(pred_boxes, target_boxes):
    """
    Compute the SCYLLA-IoU loss between predicted and target bounding boxes.
    :param pred_boxes: Tensor of shape (N, 4) representing predicted bounding boxes (x1, y1, x2, y2)
    :param target_boxes: Tensor of shape (N, 4) representing target bounding boxes (x1, y1, x2, y2)
    :return: SCYLLA-IoU loss
    """
    iou = calculate_iou(pred_boxes, target_boxes)


    si = torch.min(pred_boxes[:, 2], target_boxes[:, 2]) - torch.max(pred_boxes[:, 0], target_boxes[:, 0]) + 1
    sj = torch.min(pred_boxes[:, 3], target_boxes[:, 3]) - torch.max(pred_boxes[:, 1], target_boxes[:, 1]) + 1
    s_union = (pred_boxes[:, 2] - pred_boxes[:, 0] + 1) * (pred_boxes[:, 3] - pred_boxes[:, 1] + 1) + \
              (target_boxes[:, 2] - target_boxes[:, 0] + 1) * (target_boxes[:, 3] - target_boxes[:, 1] + 1)
    s_intersection = si * sj


    siou = iou - (s_intersection / s_union)
```

WIOU-loss
---------

文章链接：[点击](https://arxiv.org/pdf/2301.10051v1.pdf)

解决问题
传统的Intersection over Union（IoU）只考虑了预测框和真实框的重叠部分，没有考虑两者之间的区域，导致在评估结果时可能存在偏差。

算法优势
WIOU（Weighted Intersection over Union）通过考虑预测框和真实框之间的区域来对IoU进行加权，解决了传统IoU在评估结果时可能存在的偏差问题。

算法公式：
具体而言，WIOU计算方法如下：

1.  计算预测框和真实框的IoU得分。
2.  计算两个框之间的区域：用预测框和真实框的边框中心点计算它们之间的距离，并将这个距离作为两个框之间的最大距离，进而计算两个框之间的区域。
3.  根据两个框之间的区域，计算权重系数，该系数衡量了两个框之间的关系，可以用于加权IoU得分。
4.  通过引入框之间的区域和权重系数，WIOU可以更准确地评估目标检测结果，避免了传统IoU的偏差问题。

W I O U = ∑ i = 1 n w i I O U ( b i , g i ) ∑ i = 1 n w i WIOU = \\frac{\\sum_{i=1}^{n}w\_i IOU(b\_i, g\_i)}{\\sum\_{i=1}^{n}w_i} WIOU=∑i=1n​wi​∑i=1n​wi​IOU(bi​,gi​)​

其中， n n n表示物体框的数量， b i b_i bi​表示第 i i i个物体框的坐标， g i g_i gi​表示第 i i i个物体的真实标注框的坐标， I O U ( b i , g i ) IOU(b\_i, g\_i) IOU(bi​,gi​)表示第 i i i个物体框与真实标注框之间的IoU值， w i w_i wi​表示权重值。

在WIOU中，每个物体框的权重值取决于其与真实标注框的重叠程度。重叠程度越大的物体框权重越高，重叠程度越小的物体框权重越低。通过这种方式，WIOU能够更好地评估检测结果，并且在存在大小物体不平衡的情况下也能给出更准确的评价。

具体代码：

```python
import torch

def wiou(prediction, target, weight):
	"""其中，prediction和target是大小为[batch_size, height, width]的张量，
	weight是大小为[batch_size, height, width]的张量，表示每个像素的权重。"""
    intersection = torch.min(prediction, target) * weight
    union = torch.max(prediction, target) * weight
    numerator = intersection.sum(dim=(1, 2))
    denominator = union.sum(dim=(1, 2)) + intersection.sum(dim=(1, 2))
    iou = numerator / denominator
    wiou = (1 - iou ** 2) * iou
    return wiou.mean().item()

import torch

def w_iou_loss(pred_boxes, target_boxes, weight=None):
    """
    Compute the Weighted IoU loss between predicted and target bounding boxes.

	其中，输入pred_boxes和target_boxes分别是形状为(N, 4)的预测边界框和目标边界框张量。
	如果需要使用权重，则输入形状为(N,)的权重张量weight，否则默认为None。函数返回一个标量，表示计算出的加权IoU损失。
    Args:
        pred_boxes (torch.Tensor): Predicted bounding boxes, with shape (N, 4).
        target_boxes (torch.Tensor): Target bounding boxes, with shape (N, 4).
        weight (torch.Tensor, optional): Weight tensor with shape (N,). Defaults to None.

    Returns:
        torch.Tensor: Weighted IoU loss scalar.
    """

    x1 = torch.max(pred_boxes[:, 0], target_boxes[:, 0])
    y1 = torch.max(pred_boxes[:, 1], target_boxes[:, 1])
    x2 = torch.min(pred_boxes[:, 2], target_boxes[:, 2])
    y2 = torch.min(pred_boxes[:, 3], target_boxes[:, 3])
    intersection_area = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
    pred_boxes_area = (pred_boxes[:, 2] - pred_boxes[:, 0]) * (pred_boxes[:, 3] - pred_boxes[:, 1])
    target_boxes_area = (target_boxes[:, 2] - target_boxes[:, 0]) * (target_boxes[:, 3] - target_boxes[:, 1])
    union_area = pred_boxes_area + target_boxes_area - intersection_area
    iou = intersection_area / union_area


    if weight is None:
        w_iou = 1 - iou.mean()
    else:
        w_iou = (1 - iou) * weight
        w_iou = w_iou.sum() / weight.sum()

    return w_iou
```

总对比
---

边界框回归的三大几何因素：重叠面积、中心点距离、纵横比

- IOU Loss：考虑了重叠面积，归一化坐标尺度；
- GIOU Loss：考虑了重叠面积，基于IOU解决边界框不相交时loss等于0的问题；
- DIOU Loss：考虑了重叠面积和中心点距离，基于IOU解决GIOU收敛慢的问题；
- CIOU Loss：考虑了重叠面积、中心点距离、纵横比，基于DIOU提升回归精确度；
- EIOU Loss：考虑了重叠面积，中心点距离、长宽边长真实差，基于CIOU解决了纵横比的模糊定义，并添加Focal Loss解决BBox回归中的样本不平衡问题。
- SIOU Loss：在EIOU的基础上加入了类别信息的权重因子，以提高检测模型的分类准确率。