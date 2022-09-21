#线性插值

[toc]

在 pytorch 中和图像采样相关的函数,比如`affine_grid`,[`grid_sample`](grid_sample.md)  等,常有个 `align_corners` 参数,用来指导像素边界如何进行对齐.
这个参数的效果如下图所示:  
![align_corners](../../Attachments/align_corners.png)    

如图,展示了4\*4大小缩放到8\*8,在这些情况中,我们将像素考虑成是一个1\*1大小的小格子,像素座标实际取的是像素格的左上角点的座标.图中点就是一个像素的四个角点.

# align_corners = True
本情况下,我们将像素视为网格,像素值是在网格角上. 在网格尺上插入新的点,便是将原始的图片进行放大.新插入点的值根据距离它最近两端的原始网格的值,按照线性插值计算.
此时,是没有外插值的,内插值也要比 False 的情况下要少. 此时计算方式如下:
```python
# align_corners = True
# h_ori is the height in original image
# h_up is the height in the upsampled image
stride = (h_ori - 1) / (h_up - 1)
x_ori_list = []
# append the first coordinate
x_ori_list.append(0)
for i in range(1, h_up - 1):
    x_ori_list.append(0 + i * stride)
# append the last coordinate
x_ori_list.append(h_ori - 1)
```

此时显然边界值是直接放进去的,没有进行插值.这种方式通常在语义分割等像素级任务中常用. MXNet 据说默认是这种情况

# align_corners = False
本情况下,我们将像素视为1x1大小格子,像素值是格子中心的值.  
这种情况下外插和内插值比较多,但是座标换算方便.   计算时超出原座标的值进行截断

```python
# align_corners = False
# x_ori is the coordinate in original image
# x_up is the coordinate in the upsampled image
factor=h_up/h_ori
x_ori = (x_up + 0.5) / factor - 0.5
```

关于这里0.5是因为,建设像素座标为(x,y),像素中心的座标是(x+0.5,y+0.5). 具体可以参考:[如何理解双线性插值中图像坐标映射带0.5偏移](https://zhuanlan.zhihu.com/p/161457977)

# 实验
```python
import torch
import torch.nn as nn

input = torch.Tensor([1,2,5]).view(1, 1, 3)
input
```
> tensor(\[\[\[1., 2., 5.\]\]\])

```python
m = nn.Upsample(size=5, mode='linear',align_corners=False)
m(input)
```
> tensor(\[\[\[1.0000, 1.4000, 2.0000, 3.8000, 5.0000\]\]\])

```python
m = nn.Upsample(size=5, mode='linear',align_corners=True)
m(input)
```
> tensor(\[\[\[1.0000, 1.5000, 2.0000, 3.5000, 5.0000\]\]\])
# 参考资料
- https://zhuanlan.zhihu.com/p/161457977
- https://discuss.pytorch.org/t/what-we-should-use-align-corners-false/22663/9
- https://zhuanlan.zhihu.com/p/87572724