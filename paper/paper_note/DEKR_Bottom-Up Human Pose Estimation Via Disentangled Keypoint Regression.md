[toc]

# DEKR: Bottom-Up Human Pose Estimation Via Disentangled Keypoint Regression

- 会议: CVPR2021
- 文章: https://arxiv.org/abs/2104.02300
- 代码: https://github.com/HRNet/DEKR

随着深度学习的发展，运用计算机视觉中的人体姿态估计技术已经能够高精度地从人体的图片中检测出人体关键点，并恢复人体位姿。在应用端，此技术也已经在人机交互、影视制作、运动分析、游戏娱乐等各领域大放异彩。

相比单人姿态检测，由于不知道图像中每个人的位置和总人数，多人姿态检测技术在预测图片中每个人的不同关键点所在的位置时更加困难。其困难在于：不仅要定位不同种类的关键点，还要确定哪些关键点属于同一个人。

针对这一困难，学术界有两种解决方案，一种是自顶向下的方法，先检测出人体目标框，再对框内的人体完成单人姿态检测，这种方法的优点是更准确，但开销花费也更大；另一种则是自底向上的方法，常常先用热度图检测关键点，然后再进行组合，该方法的优点是其运行效率比较高，但需要繁琐的后处理过程。

最近，也有学者采用了基于密集关键点坐标回归的框架（CenterNet）对图片中的多人姿态进行检测。此方法要求对于图中的每个像素点都要直接回归 K 个关键点的位置，虽然简洁，但在位置的准确度方面却一直都显著低于先检测再组合的方法。

而微软亚洲研究院的研究员们认为，回归关键点坐标的特征必须集中注意到关键点周围的区域，才能够精确回归出关键点坐标。基于此，微软亚洲研究院提出了一种基于密集关键点坐标回归的方法：解构式关键点回归（Disentangled Keypoint Regression, DEKR）。这种直接回归坐标的方法超过了以前的关键点热度图检测并组合的方法，并且在 COCO 和 CrowdPose 两个数据集上达到了目前自底向上姿态检测的最好结果。相关工作“DEKR: Bottom-Up Human Pose Estimation Via Disentangled Keypoint Regression”已经被 CVPR 2021 收录。

![](https://www.msra.cn/wp-content/uploads/2021/06/cvpr-2021-dekr-1.png)

论文地址：https://arxiv.org/pdf/2104.02300.pdf

代码地址：https://github.com/HRNet/DEKR

DEKR 方法有两个关键：

1．用自适应的卷积激活关键点区域周围的像素，并利用这些激活的像素去学习新的特征；

2．利用了多分支的结构，每个分支都会针对某种关键点利用自适应卷积学习专注于关键点周围的像素特征，并且利用该特征回归这种关键点的位置。

通过以上两个技术关键，DEKR 学到的特征更加专注于目标关键点周围，因此，回归出的关键点具有很高的精度。

## 方法：解构式关键点回归（DEKR）

密集关键点回归框架对于每个像素点都会通过回归一个 2K 维度的偏移值向量来估计一个姿态。这个偏移值向量图是通过一个关键点回归模块处理骨干网络得到的特征而获得的。

解构式关键点回归（DEKR）的框架如下图所示，研究员们将骨干网络生成的特征分为 K 份，每份送入一个单独的分支。每个分支用各自的自适应卷积去学习一种关键点的特征，最后用一个卷积输出一个这种关键点的二维偏移值向量。在图中，为了表示方便，假设了 K=3，事实上，在 COCO 数据集的实验中，K 为 17。

![](https://www.msra.cn/wp-content/uploads/2021/06/cvpr-2021-dekr-2.png)

_图 1：解构式关键点回归（DEKR）的框架_

## 关键点 1：自适应卷积

一般来说，人体的关键点与中心点的距离比较远，一个在中心点处的普通卷积只能看到中心点周围像素的信息，而一系列在中心点处的普通卷积则可以看到更远的在目标关键点周围的像素信息，但是它不能集中地去激活这些关键点周围的像素。

因此，研究员们采用了自适应的卷积学习来激活关键点周围信息的特征。换句话说，自适应卷积就是普通卷积的加强版。

![](https://www.msra.cn/wp-content/uploads/2021/06/cvpr-2021-dekr-3.png)

以上公式中，W 代表卷积核的权重。q 是二维的位置，g_si^q 代表了偏移量，而两者相加代表了被激活的像素点。这个偏移量可以像可形变卷积（Deformable Convolution）一样用额外的普通卷积进行估计，也可以将空间形变卷积（Spatial transformer network）从全局模式扩展到逐像素模式。

在上述选择中，研究员们采用了后者：为每个像素预测仿射变换矩阵和平移矩阵。从而将这两个矩阵作用于普通卷积的卷积核，得到自适应卷积的卷积核。

![](https://www.msra.cn/wp-content/uploads/2021/06/cvpr-2021-dekr-4.png)

![](https://www.msra.cn/wp-content/uploads/2021/06/cvpr-2021-dekr-5.png)

## 关键点 2：多分支回归

研究员们还进一步用了多分支的结构。每个分支分别用不同的自适应卷积激活对应的关键点周围的像素，然后回归出相应的关键点。研究员们将骨干网络输出的特征分成了 K 个特征，然后利用这些特征估计对应关键点的偏移值向量图。

![](https://www.msra.cn/wp-content/uploads/2021/06/cvpr-2021-dekr-6.png)

上述每个公式都代表了其中一个分支为了回归一种关键点所进行的操作。可以看出，这些分支结构相同，但是被相互独立地训练。

图 2 展示了鼻子、左肩、左膝、左脚踝这些关键点对应分支学到的人的中心点处自适应卷积激活的像素位置。可以观察到通过多分支的结构，每个分支都可以用其自己的自适应卷积集中激活相应关键点位置附近的像素。

通过多分支结构，研究员们显式地将特征解构分别预测了不同的关键点，这让优化变得更加容易，进而让模型可以准确地预测出关键点的位置。

![](https://www.msra.cn/wp-content/uploads/2021/06/cvpr-2021-dekr-7.png)

_图 2：鼻子、左肩、左膝、左脚踝这几个关键点对应分支学到的人的中心点处自适应卷积激活的像素位置。_

在损失函数方面，关键点偏移值向量图的损失函数是用人的大小来正则化的光滑 L1 损失函数。公式如下：

![](https://www.msra.cn/wp-content/uploads/2021/06/cvpr-2021-dekr-8.png)

此外，在关键点和中心点热度图的损失函数设计中，研究员们还用一个独立的热度图估计分支估计了 K 张关键点热度图和一张中心点热度图，中心点热度图显示了像素是人的中心点的置信度。热度图用来给回归出的人体姿态进行打分并且排序，热度图的损失函数为加权的 L2 损失函数。

![](https://www.msra.cn/wp-content/uploads/2021/06/cvpr-2021-dekr-9.png)

结合 L1 和 L2 损失函数就可生成总体的损失函数。具体而言就是由关键点回归的损失函数与关键点和中心点热度图的损失函数加权而成。

![](https://www.msra.cn/wp-content/uploads/2021/06/cvpr-2021-dekr-10.png)

在上式中，λ 是一个权重系数，研究员们设置为 0.03。

## 实验结果：COCO 和 CrowdPose 的双重验证

在 COCO 数据集中，研究员们首先将模型在 COCO train2017 上进行了训练，训练时使用了随机旋转、随机缩放、随机平移、随机水平翻转的增广方式，然后对于 HRNet-w32 骨干网络，将图片裁剪到 512×512；对于 HRNet-w48，将图片裁剪到 640×640，并用 Adam 优化器训练网络，一共训练了 140 回，初始学习率为 1e-3，分别在第 90 回和第 120 回降为 1e-4 和 1e-5。

随后，研究员们又在 COCO val2017 和 COCO test-dev2017 上进行了测试。在测试时，保持了图片长宽比不同，把图片的短边缩放到 512/640。此外，还采用了翻转测试，将原始的热度图、关键点位置图和翻转后的热度图、关键点位置图分别做了平均。同时研究员们也尝试了多尺度测试，平均了各个尺度的热度图，并且收集了三个尺度的回归结果。

在 COCO val2017 测试集上的结果如表 1 所示，表中的“AE”指的是 Associative Embedding。可以看到 DEKR 在 COCO val2017 达到了很好的结果。

![](https://www.msra.cn/wp-content/uploads/2021/06/cvpr-2021-dekr-11.png)

_表 1：在 COCO val2017 中的测试结果_

在 COCO test-dev2017 上的结果如表 2 所示，DEKR 方法是已知第一个在 COCO test-dev2017 上仅用单尺度测试就达到了 70AP 的方法。

![](https://www.msra.cn/wp-content/uploads/2021/06/cvpr-2021-dekr-12.png)

_表 2：在 COCO test-dev2017 中的测试结果_

在 CrowdPose 数据集中，研究员们在 CrowdPose 训练集和验证集上训练了网络，训练的方法除了回合数，其余的都与 COCO 数据集完全一致。模型在 CrowdPose 数据集一共训练了 300 回合，初始学习率为 1e-3，分别在第 200 回和第 260 回降到 1e-4 和 1e-5。

在 CrowdPose 测试集上的结果如表 3 所示。

![](https://www.msra.cn/wp-content/uploads/2021/06/cvpr-2021-dekr-13.png)

_表 3：在 CrowdPose 测试集中的测试结果_

在消融实验中，自适应卷积在基准方法的基础上将 AP 提高了 3.5，然后多分支结构进一步将 AP 提高了 2.6，两者结合将基准的 AP 提高了 6.1。

![](https://www.msra.cn/wp-content/uploads/2021/06/cvpr-2021-dekr-14.png)

_表 4：消融实验结果_

此外，通过错误分析工具可以看到 Jitter 和 Miss 两种错误显著减少，分别减少了 4.6 和 1.5。这也证明了 DEKR 方法确实提高了关键点回归的位置准确度。

图 3 展示出了模型在回归关键点时注意到的区域，左栏为基准方法，右栏为 DEKR。为了展示得更加清楚，研究员们只用了鼻子和两个脚踝这三个关键点作为例子。从图中可以看到 DEKR 的自适应卷积和多分支结构确实让特征更加集中注意到关键点周围的区域。

![](https://www.msra.cn/wp-content/uploads/2021/06/cvpr-2021-dekr-15.png)

_图 3：模型在回归关键点时注意到的区域，左栏为基准方法，右栏为 DEKR_

研究员们将回归出的关键点匹配到了距其最近的从热度图检测出的关键点，这种匹配的做法对单尺度测试（ss）结果影响不大，但是提高了多尺度测试（ms）的结果。其在三个测试数据集不同的骨干网络下的结果如表 5 所示。

![](https://www.msra.cn/wp-content/uploads/2021/06/cvpr-2021-dekr-16.png)

_表 5：不同的骨干网络下的结果，“D-32”表示使用了 HRNet-W32 骨干网络，“D-48”表示使用了 HRNet-W48 骨干网络。_

综上，DEKR 显著地提高了关键点回归的质量，并且达到了现阶段自底向上姿态检测的最好结果。DEKR 将用于回归的特征解构，这样每个特征都可以集中注意到相应关键点区域，进而更准确地回归对应关键点。

# 问题

1. 文章中说分了 17 个分支，而代码中的 NUM_CHANNELS_PERKPT =15 是什么参数？  
   这只是随便设置的一个中间的超参
2. HRNet 中所有卷积都没有 bias,为何?
   很多 cbr 结构都不要 bias,bn 前使用 偏置没有意义,参见[杂坑](../../爬坑/杂坑.md#^6d6d93)

3. 为何 heatmap 的通道数是 kp 点数+1?
   参加 `CrowdPoseKeypoints` 类 heatmap 通道是`self.num_joints_with_center = self.num_joints+1`,即添加了一个中心点,中心点是 各个标注了的点的座标的平均值
4.

# 自适应卷积实现:

```python
class AdaptBlock(nn.Module):
    expansion = 1

    # 参数: ADAPTIVE,15,15,2,1
    def __init__(self, inplanes, outplanes, stride=1,
            downsample=None, dilation=1, deformable_groups=1):
        super(AdaptBlock, self).__init__()
        # 这个 regular_matrix 代表了每个位置在计算对应卷积时涉及到的其他位置,其他位置相对于当前位置的偏移
        regular_matrix = torch.tensor([[-1, -1, -1, 0, 0, 0, 1, 1, 1],\
                                       [-1, 0, 1, -1 ,0 ,1 ,-1, 0, 1]])
        # 这里使用 register_buffer 是表示这个也是模型的参数,但是不参与更新.
        self.register_buffer('regular_matrix', regular_matrix.float())
        self.downsample = downsample
        self.transform_matrix_conv = nn.Conv2d(inplanes, 4, 3, 1, 1, bias=True)
        self.translation_conv = nn.Conv2d(inplanes, 2, 3, 1, 1, bias=True)
        self.adapt_conv = ops.DeformConv2d(inplanes, outplanes, kernel_size=3, stride=stride, \
            padding=dilation, dilation=dilation, bias=False, groups=deformable_groups)
        self.bn = nn.BatchNorm2d(outplanes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x

        N, _, H, W = x.shape
        transform_matrix = self.transform_matrix_conv(x)
        transform_matrix = transform_matrix.permute(0,2,3,1).reshape((N*H*W,2,2))
        offset = torch.matmul(transform_matrix, self.regular_matrix)
        # QUE: 为何这里要减? 而非加? 这里其实加减都是同一区域
        offset = offset-self.regular_matrix
        offset = offset.transpose(1,2).reshape((N,H,W,18)).permute(0,3,1,2)

        translation = self.translation_conv(x)
        # 这里这样取可以保持形状,直接索引会导致维度-1
        offset[:,0::2,:,:] += translation[:,0:1,:,:]
        offset[:,1::2,:,:] += translation[:,1:2,:,:]

        out = self.adapt_conv(x, offset)
        out = self.bn(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

```

# 相关参考资料

- https://www.bilibili.com/read/cv12646511
