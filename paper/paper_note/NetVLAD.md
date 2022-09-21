[原文](http://www.liuxiao.org/2019/02/%E8%AE%BA%E6%96%87%E7%AC%94%E8%AE%B0%EF%BC%9Anetvlad-cnn-architecture-for-weakly-supervised-place-recognition/)

NetVLAD1是一个较早的使用 CNN 来进行图像检索或者视频检索的工作，后续在此工作的基础上陆续出了很多例如 NetRVLAD、NetFV、NetDBoW 等等的论文，思想都是大同小异。

VLAD 和 BoW、Fisher Vector 等都是图像检索领域的经典方法，这里仅简介下图像检索和 VLAD 的基本思想。

图像检索（实例搜索）是这样的一个经典问题：

1、我们有一个图像数据库 Ii I\_i 通过函数可以得到每一个图像的特征 f(Ii) f(I\_i)；

2、我们有一个待查询图像 q q 通过函数得到它的特征 f(q) f(q)；

3、则我们获得的欧氏距离 d(q,I)\=∥f(q)−f(I)∥ d(q, I) = \\parallel  f(q) - f(I)\\parallel 应该满足越相近的图像 d(q,I) d(q, I) 越小。

而 VLAD （Vector of Locally Aggregated Descriptors）则是图像检索中一个经典方法，也就可以理解为怎么获得上述的 f f 的方法，记为 fVLAD f\_{VLAD}。通常在传统方法中我们会获得一系列的局部特征（SIFT、SURF、ORB）之类，假设为 N 个 D 维的局部特征（通常 N 可能比较大，而且每幅图特征多少不一，N 可能数量也不一定），我们希望通过这 N\*D 维特征获得一个可以表示图像全局 K\*D 维特征的方法（通常K是我们指定的数目，例如128维）。VLAD 的主要流程如下：

1、对全部 N\*D 维局部特征进行 K-Means 聚类获得 K 个聚类中心，记为 Ck C\_k

2、通过以下公式将 N\*D 维局部特征编写为一个全局特征 V，特征向量维数为 K\*D，其中 k∈K k\\in K j∈D j\\in D，公式如下：

V(j,k)\=∑i\=1Nak(xi)(xi(j)−ck(j)) V(j, k) = \\sum\_{i=1}^N a\_k(x\_i) (x\_i(j) - c\_k(j))

其中 xi x\_i 为第i个局部图像特征，ck c\_k 为第k个聚类中心，xi x\_i 和 ck c\_k 都是 D 维向量。ak(xi) a\_k(x\_i) 是一个符号函数，当且仅当 xi x\_i 属于聚类中心 ck c\_k 时，ak(xi)\=1 a\_k(x\_i)=1 ，否则 ak(xi)\=0 a\_k(x\_i)=0。

经过上面对于经典 VLAD 方法的解释，我们可以看出来 fVLAD f\_{VLAD} 是一个将若干局部特征压缩为一个特定大小全局特征的方法，通过聚类，实现了将特征降维，同时用特征与聚类中心的差值作为新的特征值，在实践中 VLAD 方法具有较好的检索效果。

1、经典 VLAD 公式的可微化
----------------

经典 VLAD 方法显然是一个不可导的函数，其中主要不可导的地方在于 ak(xi) a\_k(x\_i) 这样一个符号函数，因此为了将 VLAD 变可训练的函数，必须将其变成可微计算。作者将 ak(xi) a\_k(x\_i) 平滑化：

aˉk(xi)\=e−α∥xi−ck∥2∑k′e−α∥xi−ck′∥2∈(0,1) \\bar{a}\_k(x\_i)=\\frac{e^{-\\alpha\\left \\| x\_i - c\_k \\right \\|^2}}{\\sum\_{{k}'}e^{-\\alpha\\left \\| x\_i - c\_{{k}'} \\right \\|^2}}\\in (0,1)

这里面 α \\alpha 是一个正值参数，显然当 α→∞ \\alpha\\rightarrow \\infty 时，aˉk(xi) \\bar{a}\_k(x\_i) 更趋近于 0 和 1 的两级。

再将 −α∥xi−ck∥2 -\\alpha \\left \\| x\_i - c\_k \\right \\|^2 展开，显然 e−α∥xi∥2 e^{-\\alpha\\left \\| x\_i \\right \\|^2} 可以被约掉，因此得到：

αˉk(xi)\=ewkTxi+bk∑k′ewk′T+bk′ \\bar{\\alpha}\_k(x\_i)=\\frac{e^{w^T\_kx\_i+b\_k}}{\\sum\_{{k}'}e^{w^T\_{{k}'}+b\_{{k}'}}}

其中 wk\=2αck w\_k=2\\alpha c\_k ，bk\=−α∥ck∥2 b\_k=-\\alpha\\left \\| c\_k \\right \\|^2，其实仔细观察这个公式，就是一个 softmax。

这样 VLAD 公式就被改写为：

V(j,k)\=∑i\=1NewkTxi+bk∑k′ewk′T+bk′(xi(j)−ck(j)) V(j, k) = \\sum\_{i=1}^N\\frac{e^{w^T\_kx\_i+b\_k}}{\\sum\_{{k}'}e^{w^T\_{{k}'}+b\_{{k}'}}}(x\_i(j) - c\_k(j))

显然这里面 wk w\_k 、bk b\_k 和 ck c\_k 都是 NetVLAD 需要学习的参数。

2、NetVLAD 通过监督学习获得聚类中心的好处
-------------------------

我们这样将 ck c\_k 作为学习参数有什么好处呢，论文中用了一幅图进行了直观解释：

![](http://cdn.liuxiao.org/wp-content/uploads/2019/02/Screenshot-from-2019-02-19-16-38-47.png?x-oss-process=image/resize,m_fill,w_990,h_656#)

传统 VLAD 的中心是聚类出来的，没有监督的标签数据 ckVLAD c\_k^{VLAD} ，在聚类时我们使用了很多图像这些图像的描述符之间没有关系，那么也就很可能把本来不是一个物体的描述符聚为一类，使得我们原本期望的类内描述符都是一个物体的feature不太容易达到。而在使用监督数据进行训练时，我们可以已知图像数据属于同一物体，那么训练时就可以只把属于同一物体的特征聚在一起而把不是的划分到其他类别，这样就可能学习出一个更好的 ckNetVLAD c\_k^{NetVLAD} 聚类中心，使得最终的特征更有区分度。

3、网络实现
------

![](http://cdn.liuxiao.org/wp-content/uploads/2019/02/Screenshot-from-2019-02-19-17-07-48.png?x-oss-process=image/resize,m_fill,w_2620,h_576#)

首先，由于是 NN 的方法，我们这里使用 CNN Feature 代替了传统 VLAD 中的 N 个局部描述子，CNN 是一个全局的特征，它的 Feature Map 是 W\*H\*D 大小，那么类比于我们之前的传统方法 N\*D，我们这里 NetVLAD 目标就是将 W\*H\*D （N=W\*H）的特征转换为 K\*D 的特征；

其次，我们将整个 NetVLAD 看做一个 pooling layer，它的作用是实现降最终和 VLAD 一样获得我们想要的 K\*D 维描述子。

具体实现上，我们就分步骤来做，这部分作者的 PPT 里面有张图非常清晰：

![](http://cdn.liuxiao.org/wp-content/uploads/2019/02/Screen-Shot-2019-02-20-at-11.32.06.png?x-oss-process=image/resize,m_fill,w_2240,h_1432#)

1、实现 zk\=wkTxi+bk z\_k=w\_k^Tx\_i+b\_k，也就是公式中的蓝色部分，论文里直接通过一个1x1的卷积来做。这也是1x1卷积的一个应用；

2、实现 σk(z)\=ezk∑k′ezk′ \\sigma\_k (z)=\\frac{e^{z\_k}}{\\sum\_{{k}'} e^{z\_{{k}'}} }，也就是公式中的黄色部分，如之前所述这实际上就是一个 softmax 公式，论文里直接通过一个 softmax 来做；

3、实现 xi(j)−ck(j) x\_i (j)- c\_k(j)，也就是公式中的绿色部分，这部分就是一个减法，直接用一个 VLAD core来做；

4、1~3已经实现了 V(j,k) V(j,k) 的计算了，后面按照 All about VLAD 论文还要对 V(j,k) V(j,k) 做两步简单的归一化（这篇论文证明了归一化可以提升检索性能），包括：

1）intra-normalization

这个主要意思是将每一个 VLAD block（也就是每个聚类中心的所有残差）分别作 l2 normalization。

2） l2 normalization

这个是将所有 VLAD 特征一起做 l2 normalization。

1、弱监督数据的获取
----------

在图像检索中，训练通常需要明确的图像到底是哪个物体的标注。而论文针对的领域是地点识别，因此可以利用图片的位置信息，将 Google Street View 的数据作为训练图片，将同一地点、不同视角、不同时间的数据作为同一物体标签进行训练。相对并没有明确的标注图上面都是那个物体，只是说这几幅图中可能看到同样的一个物体，数据比较容易获取。这在机器人领域同样也是可行的。

2、弱监督 triplet ranking loss
--------------------------

我们将整个网络看做一个特征提取函数 fθ f\_{\\theta} ，那我们的目标自然就是：对于一个需要检索的图像 q q ，我们有一个数据库 I I，我们希望位置最近的图片Ii∗ I\_{i\*} 的特征欧氏距离，比其他所有图片Ii I\_{i} 的距离都要小：

dθ(q,Ii∗)<dθ(q,Ii) d\_{\\theta}(q, I\_{i\*}) < d\_{\\theta}(q, I\_i)

然而理想情况是这样，如刚才小节所述我们实际上不能获取最近的图片 Ii∗ I\_{i\*}，但我们可以获取相近的图片集合（作为正样本） pi∗q p\_{i\*}^q，这些正样本集合一定包含了我们想要的 Ii∗ I\_{i\*}，但是我们不知道是哪一个；同样，我们也可以获取绝对不可能是同一物体的负样本集合 njq n\_j^q ，比如远离所选地点的图片。那么我们至少可以期望获得这样的结果：所有正样本集合的特征距离，应该要比负样本集合的特征距离要小：

dθ(q,piq)<dθ(q,njq) d\_{\\theta}(q, p\_{i}^q) < d\_{\\theta}(q, n\_j^q)

并且正样本中距离最近的应该就是我们想要的最近图片：

pi∗q\=argminpiqdθ(q,piq) p\_{i\*}^q=\\underset{p\_i^q}{argmin}d\_{\\theta}(q, p\_i^q)

据此，我们构造如下 loss：

Lθ\=∑jl(min⁡idθ2(q,piq)+m−dθ2(q,njq)) L\_{\\theta}=\\underset{j}{\\sum}l\\left ( \\underset{i}{\\min} d\_{\\theta}^2 (q,p\_i^q)+m-d\_{\\theta}^2(q,n\_j^q)\\right )

其中 l(x)\=max(x,0) l(x)=max(x,0) 也就是所说的 hinge loss，m是一个常量表示我们给负样本设置的 margin。这个 loss 与 triplet loss 是很像的。

3、训练&实现
-------

作者在附录中给出了详细的实现细节，网络提取部分作者使用 VGG-16 但做了一些修改，对于 VLAD 的聚类中心使用 K=64，m=0.1。

训练 lr = 0.001 或 0.0001，优化器 SGD，momentun 0.9，weight decay 0.001，batchsize=4 turples。这里不一一例举。具体还是看作者的代码比较好。

1、PDF：[NetVLAD: CNN architecture for weakly supervised place recognition](http://www.liuxiao.org/wp-content/uploads/2019/02/NetVLAD-CNN-architecture-for-weakly-supervised-place-recognition.pdf)

2、PPT：[cvpr16\_NetVLAD\_presentation.pptx](http://file.liuxiao.org/blog/cvpr16_NetVLAD_presentation.pptx)

3、Code：

[https://github.com/Relja/netvlad](https://github.com/Relja/netvlad) （作者的，使用 MatConv 框架）

[https://github.com/sitzikbs/netVLAD/blob/master/netVLAD.py](https://github.com/sitzikbs/netVLAD/blob/master/netVLAD.py) （第三方实现，不过只有 VLAD layer 部分）

[https://github.com/shamangary/LOUPE\_Keras](https://github.com/shamangary/LOUPE_Keras) （第三方实现，不过只有 VLAD layer 部分）

1.  1.
    
    Arandjelovic R, Gronát P, Torii A, Tomás Pajdla, Sivic J. NetVLAD: CNN architecture for weakly supervised place recognition. _CoRR_. 2015;abs/1511.07247.