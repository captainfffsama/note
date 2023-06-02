#异常检测 

[toc]

# Sub-Image Anomaly Detection with Deep Pyramid Correspondences

- 代码: <https://github.com/byungjae89/SPADE-pytorch>
- 文章: <https://arxiv.org/abs/2005.02357>

## 摘要  

对深度预训练特征应用 KNN 方法来进行异常检测是个不错的思路.但是常规 KNN 无法得到异常在图像中的位置分隔.基于此,我们提出了一个新颖的异常分割方法,这个方法是基于将异常图片和数个正常图片对齐来实现的,即语义金字塔异常分割 (Semantic Pyramid Anomaly Detection,SPADE).实验证明,SPADE SOTA 且无需训练.

关键词: 异常检测 (anomaly detection),最邻近 (nearest-neighbors),特征金字塔 (feature pyramid)

## 1. 引言

Humans observe many images throughout their lifetimes, most of which are of little interest. Occasionally, an image indicating an opportunity or danger appears. A key human ability is to detect the novel images that deviate from previous patterns triggering particular vigilance on the part of the human agent.Due to the importance of this function, allowing computers to detect anomalies is a key task for artificial intelligence.

反正就是异常检测是一个很重要的课题.  
以装配线异常检测为例.大部分商品肯定都是正常的,但是不排除少部分商品有故障等. balabala….  
本文提出的方法无需额外的训练图片,它的步骤如下:

- i).使用预训练的网络来提取特征
- ii).搜索距离目标最近的 K 个正常图片
- iii).将目标图片和正常图片之间建立像素级对应,目标图片中找不到最近对应区域的记为异常.

## 2. 前人工作

接下来本文将对图片级和子图片级的异常检测工作进行一个综述:   
*图片级方法:*   图像级方法一般不分隔异常.很多方法甚至不是专门针对图像的.主要可以分为三大类: 基于重建的,基于分布的和基于分类的.

略.

## 3. 基于对照的子图像异常检测

我们方法主要包含以下三步:

### 3.1 特征提取

直接在正常图像上使用自监督来学习特征似乎是个不错的选择,但是在小数据集的情况下,学习到的特征是否足够鲁棒是值得怀疑的. Bergmanet 证明了使用自监督训练的模型在进行异常检测时性能不如通用的 ImageNet 训练的模型.因此我们选择了 ImageNet 训练的 Resnet 作为特征提取器. 图像级的特征,我们使用的是全局池化之后卷积层出来的特征向量作为图像特征 $f_i$.  
在初始化时,我们将所有训练图片的特征都存起来.在推理的时候,仅仅提取目标图片的特征.   

### 3.2 K 最邻近法搜索正常图像

首先,我们使用 [DN2](https://arxiv.org/abs/2005.02359) 来确定哪张图片包含了异常.对于测试图片 $y$,我们选取最近的 K 个正常图片 $N_k(f_y)$.直接使用了欧式距离来作为图像级特征之间的判定.

通过验证 kNN 距离是否大于阈值来确定类型.预期是多数图像是正常的,少数图像是异常的.

### 3.3 使用图像对齐来进行子图像的异常检测

当图像被标记为异常之后,接下来目标就是定位和分割出一个或多个异常像素.当然,若是图像被错误的归类为异常图片,那么我们会认为没有像素是异常的.

那么自然而然的,考虑将测试图像和正常图像对齐,然后发现两者之间的差异进而定位出异常的像素.然是这个朴素的方法有几个问题:

- 对于由几个正常部位组合起来的目标,在针对特定正常图片对位时可能失败. assume that there are multiple normal parts the object may possibly consist of, alignment to particular normal images may fail.
- 对于变化复杂的小数据集,我们可能压根找不到一个和测试图像相似的正常训练图像,从而导致误检.
- 对于损失函数来说,计算图像间的不同可能会变得很敏感.  

为了克服以上问题,我们提出了多图像对照方法. 我们在所有像素位置都构建了一个 K 临近 $G=\{F(x_1,p)|p \in P\} \cup \{F(x_2,p)|p \in P\} .. \cup\{F(x_K,p)|p \in P\}$.像素 p 的异常分数将由目标图像的特征和最近的 K 个特征的平均距离决定.然后设定一个阈值,若距离大于阈值,就认为像素是异常的.

### 3.4 特征金字塔匹配

通过密集对照来对齐图像是一个有效判定图像异常和正常的方法.为了有效对齐,确定匹配的特征是必要的.在之前的步骤中,我们使用的都是预训练的 ResNet 特征,和图像金字塔类似,较早的层会有较高的分辨率,编码包含的周围区域越少.我们将最后 M 个块的特征图 cat 起来.这样就不必进行显示的对齐了.   

### 3.5 实验细节

用的 Wide-ResNet 50 x2 网络. MVTec 图片 resize 到 256X256 然后 crop 出中间区域. STC 图片 resize 到 256.对于 STC 数据集.我们对训练集下采样了将近 5000 张图片.下采样因子是 5,缩放方法用 `cv2.INTERAREA`. 以下试验中若没有特别指明,使用的是 ResNet 最后三个 block 特征 56\*56,28\*28,14\*14,大小.对于 MVT 使用的 K=50, 对于 STC 使用 K=1. 在实验中,我们都是使用的 k=1.  
最终图片的分数图,我们使用了高斯平滑,$\sigma=4$
