#DBNet #文字检测

一般网络结构是通过 backbone 获得 feature map 之后, 取一个固定阈值来对 feature map 做二值化, 然后得到文字的外接矩形.

而 DBNet 会输出两个 feature map, feature map 1 是传统的文字分割人力图, feature map 2 是用来指导 feature map 1 进行二值化. 这里 FPN 使用来动态卷积.

![](../../Attachments/Pasted%20image%2020240709162911.png)

另外为了方便优化, DBNet 使用了一个可微的近似操作来替代二值化, 具体可以参见 [B 站](https://www.bilibili.com/video/BV1xf4y1p7Gf?t=994.8)

![](../../Attachments/Pasted%20image%2020240709163055.png)

以上 probability map 和 threshold map 分别需要一个监督信号, 其监督信号的可视化大约如下:

![](../../Attachments/Pasted%20image%2020240709163802.png)

实际情况下文字块区域每一点值大小都不一样. 

图中 $P_{gt}$ 的获取方式如下:

对原始标注区域进行向内收缩 (shrink) 操作, 向内的距离计算方式为:

$$
D=\frac{A(1-r^2)}{L},r=0.4
$$

$A,L,D$ 分别为标注区域的面积, 周长以及向内收缩距离.

$T_{gt}$ 中区域的获取方式类似 $P_{gt}$ ,只不过是将原始标注区域向外扩张, 扩张计算方式相同.

![](../../Attachments/Pasted%20image%2020240709165001.png)

$T_{gt}$ 中各点的值的计算方式为, 现计算各点到原始标注轮廓上最近点的距离. 然后归一化到 [0,1] 之后, 然后用 1 取减即可.

## 损失函数计算

具体参见 [B 站](https://www.bilibili.com/video/BV1xf4y1p7Gf?t=2630.4)

![](../../Attachments/Pasted%20image%2020240709165558.png)

## 推理

推理阶段会去掉 threshold map 和 approximate binary map. 仅仅使用 probability map. 再得到 $P_{map}$ 之后, 取固定阈值过滤, 固定阈值为 0.2, 然后按照类似以上的公式做膨胀:

$$
D'=\frac{A' \times r'}{L'},r=1.5
$$