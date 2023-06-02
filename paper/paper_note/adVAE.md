#异常检测 

[TOC]

# adVAE: a Self-adversarial Variational Autoencoder with Gaussian Anomaly Prior Knowledge for Anomaly Detection

Xu hong Wang,∗, Ying Du, Shijie Lin, Ping Cui, Yuntian Shen, Yupu Yanga

Shanghai Jiao Tong University, Shanghai,

China Wu han University, Wuhan,

China University of California, Davis , U.S.A.

----

## Abstract

Recently, deep generative models have become increasingly popular in unsupervised anomaly detection. However,deep generative models aim at recovering the data distribution  rather  than detecting anomalies. Moreover, deep generative models have the risk of overfitting training samples, which has disastrous effects on anomaly detection performance. To solve the above two problems, we propose a self-adversarial variational autoencoder (adVAE) with a Gaussian anomaly prior assumption. We assume that both the anomalous and the normal prior distribution are Gaussian and have overlaps in the latent space. Therefore, a Gaussian transformer net T is trained to synthesize anomalousbut near-normal latent variables. Keeping the original training objective of a variational autoencoder, a generator G tries to distinguish between the normal latent variables encoded by E and the anomalous latent variables synthesized by T, and the encoder E is trained to discriminate whether the output of G is real. These new objectives we added not only give both G and E the ability to discriminate, but also become an additional regularization mechanism to prevent overfitting. Compared with other competitive methods, the proposed model achieves significant improvements in extensive experiments. The employed datasets and our model are available in a Github repository.

最近，深度生成模型在无间督异常检测领域变得越来越流行。然而，深度生成模型旨在恢复数据分布而非检测异常.此外,深度生成模型还存在过拟合样本的风险,这会极大影响其异常检测性能.为了解决以上两个问题,本文提出了带有高斯异常先验假设的自对抗的变分自编码器.我们假设异常和正常样本的先验分布都是服从高斯分布且在隐空间有重叠区域.基于此,本文训练一个高斯变换网络T来合成接近正常隐向量的异常.然后保留变分自编码的原始训练目的,使用一个生成器G来分辨由编码器E产生的正常隐向量和由T合成的异常隐向量,同时编码器E还负责分辨G的输出是否真实.本文添加的这些新目的不仅使得G和E由了分辨能力,同时还成为了一种额外的正则策略来抑制了过拟合.和其他方法相比,本文模型在大量的实验中取得了显著的进步.实验使用的数据集和模型可以在Github取得.

*Keywords*:anomaly detection, outlier detection, novelty detection, deep generative model, variational autoencoder

关键词: 异常检测,离群点检测,新颖性检测,深度生成模型,变分自编码器

----

## 1.Introduction

Anomaly detection (or outlier detection) can be regarded as the task of identifying rare data items that differ from the majority of the data. Anomaly detection is applicable in a variety of domains, such as intrusion detection, fraud detection, fault detection,health monitoring, and security checking [1–5].Owing to the lack of labeled anomaly samples, there is alarge skew between normal and anomaly class distributions. Some attempts [6–8] use several imbalanced-learning methods to improve the performance of supervised anomaly detection models. Moreover, unsupervised models are more popular than supervised models in the anomaly detection field. Reference [9] reviewed machine-learning-based anomaly detection algorithms comprehensively.

异常检测(离群点检测)可以被视为一个区分罕见数据和常见数据区别的任务.异常检测在入侵检测,欺诈检测,故障检测,健康监测和安全检查等诸多领域都有应用[2-5].由于缺少标记的异常样本,异常样本和正常样本的分布有很大差异.一些工作[6-8]尝试使用不平衡的学习方法来提升异常检测监督模型的性能.然而,无监督模型比监督模型在异常检测领域要更受欢迎.文献[9]非常全面的总结了各类基于机器学习的异常检测算法.

Recently, deep generative models have become increasingly popular in anomaly detection [10]. A generative model can learn a probability distribution model by being trained on an anomaly-free dataset. Afterwards, outliers can be detected by their deviation from the probability model. The most famous deep generative models are variational autoencoders (VAEs) [11]and generative adversarial networks (GANs) [12].

最近,深度生成模型在异常检测领域变得越来越流行[10].深度生成模型可以在正常样本上学习到数据的概率分布.之后,通过衡量概率模型的偏离值就可以检测离群点.变分自编码器(VAE)和GAN是最出名的深度生成模型.

VAEs have been used in many anomaly detection studies. The first work using VAEs for anomaly detection [13] declared that VAEs generalize more easily than autoencoders (AEs), because VAEs work on probabilities. [14, 15] used different types of RNN-VAE architectures to recognize the outliers of time series data. [1]and [16] implemented VAEs for intrusion detection and internet server monitoring, respectively. Furthermore,GANs and adversarial autoencoders [17] have also been introduced into image [4, 18] and video [19] anomaly detection.

在很多异常检测研究里都用到了VAE.文献[13]是第一个在异常检测上使用VAE的工作,它揭示了由于VAE是在概率分布上工作的,因此VAE的生成能力比AE要好.[14,15]使用了不同结构的RNN-VAE来识别时间序列上的离群点.[1]和[16]则将VAE应用于入侵检测和互联网服务器监测.此外,GAN和对抗AE也被用于图像和视频的异常检测.

However, there are two serious problems in deep-generative-model-based anomaly detection methods.

然而,基于深度生成模型的异常检测方法依然存在两个严重问题.

(1) Deep generative models only aim at recovering the data distribution of the training set, which has an indirect contribution to detecting anomalies. Those earlier studies paid little attention to customize their models for anomaly detection tasks. Consequently, there is an enormous problem that those models only learnfrom available normal samples, without attempting todiscriminate the anomalous data. Owing to the lack of discrimination, it is hard for such models to learn useful deep representations for anomaly detection tasks.

(1)深度生成模型主要是尝试恢复训练数据的数据分布,以此来间接实现异常检测.之前的工作都很少关注针对异常检测任务进行模型的改进.这导致这些模型注重从可用的正常样本中学习分布而没有试图判别异常数据.由于缺少判别性,这些模型难以学习到对异常检测任务有用的深度表征.

(2) Plain VAEs use the regularization of the Kullback–Leibler divergence (KLD) to limit the capacity of the encoder, but there is no regularization implemented in the generator. Because neural networksare universal function approximators, generators can, intheory, cover any probability distribution, even withoutdependence on the latent variables [20]. However, previous attempts [21, 22] have found it hard to benefit from using an expressive and powerful generator. [20]uses Bits-Back Coding theory [23] to explain this phenomenon: if the generator can model the data distribution $p_{data}(x)$ without using information from the latent code $z$ , it is more inclined not to use $z$ . In this case, to reduce the KLD cost, the encoder loses much information about $x$ and maps $p_{data}(x)$ to the simple prior $p(z)$  (e.g., $N(0,I)$) rather than the true posterior $p(z|x)$ . Once this undesirable training phenomenon occurs, VAEs tend to overfit the distribution of existing normal data, which leads to a bad result (e.g., a high false positive rate),especially when the data are sparse. Therefore, the capacity of VAEs should be limited by an additional regularization in anomaly detection tasks.

(2)常规VAE使用KL散度来限制编码器,但是对于生成器却没有正则策略.由于神经网络是通用的函数拟合器,那么理论上生成器可以不依赖隐变量而覆盖任何概率分布[20].然而[21,22]发现即使使用能力强大的生成器也收效甚微.文献[20]使用Bits-Back编码理论[23]解释了这种现象:若生成器可以在不使用隐变量编码 $z$ 的情况下也可以对数据 $P_{data}(x)$ 进行建模,那么生成器将更加倾向不使用 $z$ 来进行建模.在这种情况下,减小KL散度的损失,编码器将丢失更多关于 $x$ 的信息,且将分布 $p_{data}(x)$ 映射到简单的先验分布 $p(z)$ 而非后验分布 $p(z|x)$ .一旦这种不良的训练情况发生,VAE将倾向于过拟合现存的正常数据分布,这将导致模型出现很高的假阳率,特别是在数据稀疏时.因此需要额外的正则策略来限制VAE的能力.

There are only a handful of studies that attempt to solve the above two problems. MO-GAAL method [24]made the generator of a GAN stop updating before convergence in the training process and used the nonconverged generator to synthesize outliers. Hence, the discriminator can be trained to recognize outliers in a supervised manner. To try to solve the mode collapse issue of GANs, [24] expands the network structure from a single generator to multiple generators with different objectives. [25] proposed an assumption that the anomaly prior distribution is the complementary set of the normal prior distribution in latent space. However, we believe that this assumption may not hold true. If the anomalous and the normal data have complementary distributions, which means that they are separated in the latent space,then we can use a simple method (such as KNN) to detect anomalies and achieve satisfactory results, but this is not the case. Both normal data and outliers are generated by some natural pattern. Natural data ought to conform to a common data distribution, and it is hard to imagine a natural pattern that produces such a strange distribution.

只有少数研究尝试解决以上两个问题.MO-GAAL方法[24]在训练收敛前停止来生成器GAN,并使用未收敛的生成器来合成离群点.这样判别器可以以监督学习的方式来训练识别离群点.另外为了解决GAN模式崩溃的问题,[24]还将网络结构从单一的生成器扩展成多个不同目标的生成器.[25]则假设异常先验分布是正常先验分布在隐空间中的补集.但是,我们认为这个假设可能不成立.若异常和正常数据是互补分布,那么这意味着他们在隐空间中是分离的,那么我们是使用KNN等方法就可以较好的检测出异常样本,但显然这是不可能的.正常数据和离群点都是由某个自然分布产生的,自然数据应该符合一个通用的数据分布,并且很难想象是怎样的一个自然模式会产生这样一个奇怪的分布.

To enhance deep generative models to distinguish between normal and anomalous samples and to prevent them from overfitting the given normal data, we propose a self-adversarial Variational Autoencoder (ad-VAE) with a Gaussian anomaly prior assumption and a self-adversarial regularization mechanism. The basic idea of this self-adversarial mechanism is adding discrimination training objectives to the encoder and the generator through adversarial training. These additional objectives will solve the above two problems at the same time; the details are as follows.

为了加强深度生成模型区分正常和异常样本的能力,同时防止模型在正常数据上过拟合,我们提出了一个具有高斯异常先验假设和自对抗正则机制的自对抗VAE.这个自对抗机制的大致思想是通过对抗训练,给编码器和生成器添加判别训练数据的目标.这些额外的目标将同时解决上述两个问题;详见后文.

The encoder can be trained to discriminate the original sample and its reconstruction, but we do not have any anomaly latent code to train the generator. To synthesize the anomaly latent code, we propose a Gaussian anomaly hypothesis to describe the relationship between normal and anomaly latent space. Our assumption is described in Figure 1(a); both the anomalous and the normal prior distributions are Gaussian and have overlaps in the latent space. It is an extraordinarily weak and reasonable hypothesis, because the Gaussian distri-bution is the most widespread in nature. The basic struc-ture of our self-adversarial mechanism is shown in Fig-ure 1(b). The encoder is trained to discriminate the original sample $x$ and its reconstruction $x_r$, and the generator tries to distinguish between the normal latent variables $z$ encoded by the encoder and the anomalous ones $z_T$ synthesized by $T$. These new objectives we added not only give G and E the ability to discern, but also introduce anadditional regularization to prevent model overfitting.

编码器可以被训练成可以判别原始数据和重建数据,但我们没有任何的异常隐编码来训练生成器.为了合成异常隐编码,我们提出来高斯异常假说来描述正常和异常隐空间的关系.Fig.1(a)描述我们的假设;异常和正常先验分布都是高斯分布且在隐空间上由重叠.这是一个脆弱但是合理的假设,因为高斯分布在自然界是普遍存在的.Fig.1(b)展示来本文自对抗机制的基本结构.编码器被训练来判别原始样本 $x$ 和重建样本 $x_r$ ,生成器则尝试区别由编码器编码的正常隐向量 $z$ 和由 $T$ 合成的异常样本 $z_T$ .这些被添加的新目标不仅使G和E有了判别能力,还引入来额外的正则化机制来防止模型过拟合.


>![Fig 1](https://raw.githubusercontent.com/hqabcxyxz/MarkDownPics/master/image/20200901183009.png)
Figure 1: (a) Our assumption is that the normal data prior distribution is a Gaussian distribution close to $N(0,I)$ , and the anomalous prior is another Gaussian distribution, whose mean and variance are unknown and different from the normal prior, with overlaps in the latent space. (b) This figure illustrates the basic structure of our self-adversarial mechanism. We propose additional discrimination objectives for both the encoder and the generator by adding two competitive relationships .$x$ and $z$ are normal items because of the anomaly-free training dataset. $x_r$ and $z_T$ can be regarded as anomalous item.G is trained to distinguish $z$ and $z_T$ , and E tries to discern $x$ and $x_r$ .  
图1:(a) 我们假设正常样本先验分布是一个接近 $N(0,1)$ 的高斯分布,异常先验是另外一个高斯分布,其期望和方差未知且都和正常先验分布不一样,并且两者在隐空间中有重叠区域. (b) 本图展示来本文自对抗机制的大致结构.通过添加两个竞争关系,为编码器和生成器设定来额外的判别目标.$x$ 和 $z$ 是正常样本. $x_r$ 和 $z_T$ 被视为异常样本.G被训练来区分 $z$ 和 $z_T$ ,E尝试区分 $x$ 和 $x_r$

Our training process can be divided into two steps:
(1)T tries to mislead the generator G; meanwhile G works as a discriminator.
(2)G generates realistic-like samples, and the encoder E acts as a discriminator of G.
To make the training phase more robust, inspired by [26], we train alternatively between the above two steps in a mini-batch iteration.

本文训练过程可以被分为两步:
(1)T尝试误导生成器G;此时G作为一个判别器.
(2)G产生类似真实的样本,编码器E作为G的判别器.
为了使训练过程更加鲁棒,受[26]启发,在一个batch迭代中,我们交替执行以上两个步骤.

Our main contributions are summarized as follows:
1. We propose a novel and important concept that deep generative models should be customized to learn to discriminate outliers rather than being used in anomaly detection directly without any suitable customization.
2. We propose a novel self-adversarial mechanism,which is the prospective customization of a plain VAE, enabling both the encoder and the generator to discriminate outliers.
3. The proposed self-adversarial mechanism also provides a plain VAE with a novel regularization,which can significantly help VAEs to avoid over-fitting normal data.
4. We propose a Gaussian anomaly prior knowledge assumption that describes the data distribution of anomalous latent variables. Moreover, we propose a Gaussian transformer net T to integrate this prior knowledge into deep generative models.
5. The decision thresholds are automatically learned from normal data by a kernel density estimation(KDE) technique, whereas earlier anomaly detection works (e.g., [13, 16]) often omit the importance of automatic learning thresholds.

综上,本文主要贡献如下:
1. 本文提出一个新颖而重要的概念:深度生成模型应该被制定学区别离群点而不是直接被用于异常检测.
2. 本文提出一个新颖的自对抗机制:通过设计一个朴素VAE,使得编码器和生成器可以判别离群点.
3. 提出的自对抗机制为这个朴素VAE提供了一个新颖的正则化,使得VAE可以避免在正常数据上过拟合.
4. 本文提出来一个高斯异常先验知识假说来描述异常隐向量的数据分布.此外,我们提出来一个高斯变换网络T来将这些先验知识融合到深度生成模型中.
5. 决策的阈值是通过核密度估计(KDE)的方法来自动计算出来的,之前的工作常常忽略来自动学习阈值的重要性.

----
## 2.Preliminary
### 2.1 Conventional Anomaly Detection  

Anomaly detection methods can be broadly categorized into probabilistic, distance-based, boundary-basedand reconstruction-based.

异常检测方法可以粗分为基于概率,基于距离,基于边界和基于重构的方法.

(1) Probabilistic approach, such as GMM [27] and KDE [28], uses statistical methods to estimate the probability density function of the normal class. A data pointis defined as an anomaly if it has low probability density.

(1)概率类方法,例如GMM[27]和KDE[28],使用统计方法来估计正常样本的概率密度函数.低概率密度的数据点被定义为异常点.

(2) Distance-based approach has the assumption that normal data are tightly clustered, while anomaly data occur far from their nearest neighbours. These methods depend on the well-defined similarity measure between two data points. The basic distance-based methods are LOF [29] and its modification [30].

(2) 距离类的方法假设正常样本是紧密聚集的,异常数据通常离它们较远.这类方法效果依赖于两类数据之间定义的相似性度量.基础的距离类方法有LOF[29]及其变体[30].

(3) Boundary-based approach, mainly involving OCSVM [31] and SVDD [32], typically try to define a boundary around the normal class data. Whether the unknown data is an anomaly instance is determined by their location with respect to the boundary. 

(3)边界类方法,典型的有OCSVM[31]和SVDD[32],尝试寻找出正常样本的边界.待测数据是否是异常数据取决于相对边界的位置.

(4) Reconstruction-based approach assumes that anomalies are incompressible and thus cannot be effectively reconstructed from low-dimensional projections. In this category, PCA [33] and its variations [34, 35] are widely used, effective techniques to detect anomalies. Besides, AE and VAE basedmethods also belong to this category, which will be explained detailedly in the next two subsections.

(4)基于重构的方法假设异常是不可压缩的,因此无法从低维投影中有效的重建出来.PCA[33]和其变体[34,35]是被广泛应用且有效的技术.除此之外,AE和VAE的方法也属于这类方法,详见下两小节.

### 2.2 Autoencoder-based Anomaly Detection

An AE, which is composed of an encoder and a decoder, is a neural network used to learn reconstructions as close as possible to its original inputs. Given a datapoint  $x∈R^d$ ( $d$ is the dimension of $x$ ), the loss function can be viewed as minimizing the reconstruction error between the training data and the outputs of the AE, and $\theta$ and $\phi$ denote the hidden parameters of the encoder E and the decoder G:

AE由编码器和解码器组成,是一个用于学习重建尽量接近原始输入的神经网络.给定一个数据点 $x\in{R^d}$ ( $d$ 是 $x$ 的维数),损失函数可以被视为最小化训练数据和AE输出的重建误差, $\theta$ 和 $\phi$ 被定义为编码器E和解码器G的隐藏参数:

$$
L_{AE}(x,\phi,\theta)=\parallel x-G_\theta(E_\phi(x)) \parallel^2     \tag{1.1}
$$

After training, the reconstruction error of each test data will be regarded as the anomaly score. The data with a high anomaly score will be defined as anomalies,because only the normal data are used to train the AE.The AE will reconstruct normal data very well, while failing to do so with anomalous data that the AE has not encountered.

经过训练后,将每个测试数据的重构误差作为异常分数.由于只有正常数据被用于AE的训练,所以有较高重构误差的数据可以被视为异常点.AE可以很好的重构正常数据,而异常数据则无法很好的重构.

### 2.3 VAE-based Anomaly Detection

The net architecture of VAEs is similar to that of AEs, with the difference that the encoder of VAEs forces the representation code $z$ to obey some type of prior probability distribution $p(z)$ (e.g., $N(0,I)$ ). Then, the decoder generates new realistic data with code $z$ sampled from $p(z)$ . In VAEs, both the encoder and decoder conditional distributions are denoted as $q_\phi(z|x)$ and $p_\theta(z|x)$ . The data distribution $p_\theta(x)$ is intractable by analytic methods, and thus variational inference methods are introduced to solve the maximum likelihood log $p_\theta(x)$ :

VAE的网络结构和AE类似,不同之处是VAE的编码器强制表征编码 $z$ 服从某个先验概率分布 $p(z)$ .然后,解码器将从 $p(z)$ 中采样得到 $z$ ,然后生成新的类似真实数据.在VAE中, $q_\phi(z|x)$ 和 $p_\theta(z|x)$ 分布代表编码器和生成器的条件分布.数据分布 $p_\phi(x)$ 难以被解析法求解,因此引入来变分推理的方法来求解极大似然对数log $p_\phi(x)$ :

$$
L(x) =\log{p_\phi(x)}-KL[q_\theta(z|x)||p_\phi(z|x)]  \\
     =E_{q_\phi(z|x)}[\log{p_\theta(x)}+\log{p_\theta(z|x)}-\log{q_\phi(z|x)}]  \\ 
     =-KL(q_\phi(z|x)||p_\theta(z))+E_{q_\phi(z|x)}[\log{p_\theta(x|z)}] \tag{2}
$$

KLD is a similarity measure between two distributions. To estimate this maximum likelihood, VAE needs to maximize the evidence variational lower bound(ELBO) $L(x)$. To optimize the KLD betweenq $q_\phi(z|x)$ and $p_θ(z)$ , the encoder estimates the parameter vectorsof the Gaussian distribution $q_\phi(z|x)$ : mean $μ$ and standard deviation $σ$ . There is an analytical expression for their KLD, because bothqφ(z|x) andpθ(z) are Gaussian.To optimize the second term of equation (2), VAEs min-imize the reconstruction errors between the inputs andthe outputs. Given a datapointx∈Rd, the objectivefunction can be rewritten as

KLD(KL散度)是衡量两个分布相似性的一种方法.为了估计这个极大似然,VAE需要最大化`evidence variational lower bound(ELBO)` $L(x)$ .为了优化$q_\phi(z|x)$ 和 $p_\theta(z)$ 的KLD,编码器将估计高斯分布 $q_\phi(z|x)$ 的期望 $u$ 和标准差 $\sigma$ .以下是两个分布的KLD的解析式子,其中 $q_\phi(z|x)$ 和 $p_\theta(z)$ 都服从高斯分布.为了优化公式(2),VAE将最小化输入和输出之间的重建误差.给定一个数据点 $x\in{R^d}$ ,目标函数可以写成:

$$
L_{VAE}=L_{MSE}(x,G_\theta(z))+\lambda L_{KLD}(E_\phi(x))
$$
$$
=L_{MSE}(x,x_r)+\lambda L_{KLD}(u,\sigma)  \tag{3}  
$$

$$
L_{MSE}(x,x_r)=||x-x_r||^2 \tag{4}
$$

$$
L_{KLD}(u,\sigma)=KL(q_\phi(z|x)||p_\theta(z))
$$
$$
=KL(N(z;u,\sigma^2)||N(z;0,I))
$$
$$
=\int{N(z;u,\sigma^2)\log{\frac{N(z;u,\sigma^2)}{N(z;0,I)}}\mathrm{d}z}    \tag{5}
$$
$$
=\frac{1}{2}(1+\log{\sigma^2}-u^2-\sigma^2)
$$


The first term $L_{MSE}(x,x_r)$ is the mean squared error (MSE) between the inputs and their reconstructions.The second term $L_{KLD}(μ,σ)$ regularizes the encoder by encouraging the approximate posterior $q_\phi(z|x)$ to match the prior $p(z)$ . To hold the trade off between these two targets, each KLD target term is multiplied by a scaling hyper-parameter $\lambda$ .

$L_{MSE}(x,x_r)$ 是输入和重建的平方差, $L_{KLD}(u,\sigma)$ 是通过使 $q_\phi(z|x)$ 逼近 $p(z)$ 来对编码器进行正则的.通过对每个KLD乘上一个权重超参$\lambda$来平衡这两个目标.

AEs define the reconstruction error as the anomaly score in the test phase, whereas VAEs use the reconstruction probability [13] to detect outliers. To estimate the probabilistic anomaly score, VAEs sample $z$ according to the prior $p_θ(z)$ for $L$ times and calculate the average reconstruction error as the reconstruction probability. This is why VAEs work more robustly than traditional AEs in the anomaly detection domain.

AE将重构误差定义为测试阶段的异常分数,而VAE则使用重构概率[13]来检测离群点.为了估计异常分数概率,VAE从先验 $p_\theta(z)$ 中采样 $L$ 次来得到样本 $z$ ,并计算平均重构误差来作为重构概率.这就是为何VAE比传统AE在异常检测领域更加鲁棒的原因.

### 2.4 GAN-based Anomaly Detection
Since GANs [12] were first proposed in 2014, GANs have become increasingly popular and have been applied for diverse tasks. A GAN model comprises two components, which contest with each other in a cat-and-mouse game, called the generator and discriminator. The generator creates samples that resemble the real data, while the discriminator tries to recognize the fake samples from the real ones. The generator of a GAN synthesizes informative potential outliers to assist the discriminator in describing a boundary that can separate outliers from normal data effectively [24]. When a sample is input into a trained discriminator, the output of the discriminator is defined as the anomaly score.However, suffering from the mode collapsing problem,GANs usually could not learn the boundaries of normal data well, which reduces the effectiveness of GANs in anomaly detection applications. To solve this problem, [24] propose MOGAAL and suggest stopping optimizing the generator before convergence and expanding the network structure from a single generator to multiple generators with different objectives. In addition, WGAN-GP [36], one of the most advanced GAN frameworks, proposes a Wasserstein distance and gradient penalty trick to avoid mode collapsing. Is our experiments, we also compared the anomaly detection performance between a plain GAN and WGAN-GP.

自2014年GAN[12]出世,GAN已变得愈加流行,并在各种任务中被应用.GAN包含了两个像进行猫鼠游戏一样的模块,生成器和判别器.生成器生成和真实数据相似的样本,判别器则尽力取区分真假样本.生成器合成来潜在离群点的信息,这可以帮助判别器更加有效的找到正常数据和离群数据的边界[24].样本在判别器的输出被定义为异常分数.然而由于存在模式崩溃的问题,GAN经常无法很好的学习到正常样本的变价,导致其效果不尽人意.为了解决这个问题,文献[24]提出来MOGAAL,建议在生成器拟合前停止优化并将网络结构从单个生成器扩展到多个带有不同优化目标的生成器.此外,[[../../ClassicBlog/From GAN to WGAN|WGAN-GP]][36]提出[[../../ClassicBlog/From GAN to WGAN|Wasserstein距离]]和梯度惩罚策略来规避模式崩溃.在本文实验中,我们比较了普通GAN和WGAN-GP在异常检测中的性能.

## 3.Self-adversarisal Variational Autoencoder

In this section,a self-adversarial Variational Autoencoder (adVAE) for anomaly detection is pro-posed. To customize plain VAE to fit anomaly detection tasks, we propose the assumption of a Gaussian anomaly prior and introduce the self-adversarial mechanism into traditional VAE. The proposed method consists of three modules: an encoder net E, a generative net G, and a Gaussian transformer net T.

在本章中,提出了adVAE来进行异常检测.通过引入高斯异常先验假设和自对抗机制到传统VAE,来得到一个定制化的适合异常检测的VAE.该方法包括三个部分:编码器E,生成器G和高斯变换网络T.

There are two competitive relationships in the training phase of our method: (1) To generate a potential anomalous prior distribution and enhance the generator’s ability to discriminate between normal and anomalous priors, we train the Gaussian transformer T and the generator G with adversarial objectives simultaneously.(2) To produce more realistic samples in a competitive manner and make the encoder learn to discern, we train the generator and the encoder analogously to the generator and discriminator in GANs.

本文方法的训练步骤有两个竞争关系:(1)为了生成潜在异常先验分布和增强生成器区分正常和异常的能力,我们使用对抗训练来训练高斯变换网络T和生成器G.(2)另外为了在竞争状态下产生更加真实的样本,使得编码器和学习到不同,我们以GAN中生成器和判别器那样训练生成器和编码器.

According to equation (3), there are two components in the objective function of VAEs: $L_{MSE}$ and $L_{KLD}$ . The cost function of adVAE is a modified combination objective of these two terms. In the following, we describe the training phase in subsections 3.1 to 3.3 and subsections 3.4 to 3.6 address the testing phase.

根据公式3,VAE的目标函数包括两个部分 $L_{MSE}$ 和 $L_{KLD}$ .adVAE的损失函数结合来这两个部分.3.1~3.3节阐释来训练步骤,3.4~3.6展示来测试细节.

### 3.1 Training Step 1: Competition between T and G
The generator of plain VAE is often so powerful that it maps all the Gaussian latent code to the high-dimensional data space, even if the latent code is encoded from anomalous samples. Through the competition between T and G, we introduce an effective regularization into the generator.

普通VAE的生成器有能力将所有高斯隐编码映射到高维数据空间,甚至这个隐编码是异常样本的编码.通过T和G的竞争,我们引入了一个有效的正则方法到生成器中.

Our anomalous prior assumption suggests that it is difficult for the generator of plain VAE to distinguish the normal and the anomalous latent code, because they have overlaps in the latent space. To solve this problem,we synthesize anomalous latent variables and make the generator discriminate the anomalous from the normal latent code. As shown in Figure 2 (a), we freeze the weights of E and update G and T in this training step.The Gaussian transformer T receives the normal Gaussian latent variables $z$ encoded from the normal training samples as the inputs and transforms $z$ to the anomalous Gaussian latent variables $z_T$ with different mean $μ_T$ and standard deviation $σ_T$ .T aims at reducing the KLD between $\{z;μ,σ\}$ and $\{z_T;μ_T,σ_T\}$ , and G tries to generate as different as possible samples from such two similar latent codes.

我们有一个异常先验假设:普通VAE的生成器难以区分正常和异常隐编码,因为它们在隐空间中有重叠.为了解决这个问题,我们合成来异常隐变量并使生成器判别异常和正常隐编码.如Fig.2(a)所示,训练时,我们冻结E的权重,更新G和T.高斯变换网络T将正常训练样本的高斯隐变量 $z$ 作为输入,并将 $z$ 变换到异常高斯隐变量 $z_T$ ,其期望使 $u_T$ ,标准差是 $\sigma_T$ .T将最小化 $\{z;u,\sigma\}$ 和 $\{z_T;u_T,\sigma_T\}$ 的KL散度,G尝试从这两个相似的隐编码中生成尽可能不同的样本.

>![Fig2](https://raw.githubusercontent.com/hqabcxyxz/MarkDownPics/master/image/20200903134711.png)  
>Figure 2: Architecture and training flow of adVAE. As adVAE is a variation of plain VAE, $\lambda$ is the hyperparameter derived from VAE. We add discrimination objectives to both the encoder and the generator by adversarial learning. The larger the $\gamma$ or $m_z$ , the larger the proportion of the encoder discrimination objective in the total loss. The larger the $m_x$ , the larger the proportion of the generator discrimination objective. (a) Updating the decoder G and the Gaussian transformer T. (b) Updating the encoder E. (c) Anomaly score calculation. Hatch lines indicate that the weights of the corresponding networks are frozen.   
>Fig2:adVAE的结构和训练流程.超参 $\lambda$ 使从VAE中派生出来的.通过对抗学习,我们为编码器和生成器都提添加来区分目标的能力 .$\gamma$ 和 $m_z$ 越大,编码器区分目标的损失在整个损失中比重越大. $m_x$ 越大,生成器区分目标在整个损失中比重越大.(a)更新解码器G和高斯变换网络T.(b)更新编码器E.(c)计算异常分数.阴影表示对应的网络权重被冻结.

Given a datapoint $x\in{R^d}$ , the objective function in this competition process can be defined as:

给定数据 $x\in{R^d}$ ,目标函数可以定义为:

$$
L_G=L_{G_z}+L_{G_{z_T}} \tag{6}
$$

$$
L_{G_z}=L_{MSE}(x,G(z))+\gamma L_{KLD}(E(G(z))) 
$$
$$
=L_{MSE}(x,x_{\gamma})+\gamma L_{KLD}(u_r,\sigma_r)  \tag{7}
$$

$$
L_{G_{z_T}}=[m_x-L_{MSE}(G(z),G(z_T))]^++\gamma [m_z-L_{KLD}(E(G(z_T)))]^+
$$
$$
=[m_x-L_{MSE}(x_r,x_{T_r})]^++\gamma [m_z-L_{KLD}(u_{T_r},\sigma_{T_r})]^+    \tag{8}
$$

$$
L_T=KL(N(z;u,\sigma^2)||N(z;u_T,\sigma^2_T))
$$
$$
=\int{N(z;u,\sigma^2)\log{\frac{N(z;u,\sigma^2)}{N(z;u_T,\sigma^2_T)}}\mathrm{d}z}  \tag{9}
$$
$$
=\log{\frac{\sigma_T}{\sigma}}+\frac{\sigma^2+(u-u_T)^2}{2\sigma^2_T}-\frac{1}{2}
$$

$[·]^+=max(0,·)$ , $m_x$ is a positive margin of the MSE target, and $m_z$ is a positive margin of the KLD target.The aim is to hold the corresponding target term below the margin value for most of the time. $L_{G_z}$ is the objective for the data flow of $z$ , and $L_{G_{z_T}}$ is the objective for the pipeline of $z_T$ .

== $[·]^+=max(0,·)$ , $m_x$ 是MSE的正边界, $m_z$ 是KLD的正边界.这样做的目的是让对应的目标项在多数时候低于边界值. $L_{G_z}$ 是数据流 $z$ 的优化目标, $L_{G_{z_T}}$ 是数据流 $z_T$ 的优化目标. ==

$L_T$ and $L_G$ are two adversarial objectives, and the total objective function in this training step is the sum of the two: $L_T+λL_G$ . Objective $L_T$ encourages T to mislead G by synthesizing $z_T$ similar to $z$ , such that G cannot distinguish them. Objective $L_G$ forces the generator to distinguish between $z$ and $z_T$ .T hopes that $z_T$ is close to $z$ , whereas G hopes that $z_T$ is farther away from $z$ . After iterative learning,T and G will reach a balance.T will generate anomalous latent variables close to the normal latent variables, and the generator will distinguish them by different reconstruction errors. Although the anomalous latent variables synthesized by T are not necessarily real, it is helpful for the models as long as they try to identify the outliers.

$L_T$ 和 $L_G$ 是竞争关系,他俩的和 $L_T+\lambda L_G$ 是训练阶段的总的损失函数.$L_T$ 通过生成和 $z$ 相似的 $z_T$ 使T来尽力蒙骗G. $L_G$ 则强迫G区分 $z$ 和$z_T$ .T希望 $z_T$ 尽量接近 $z$ 而G希望 $z_T$ 和 $z$ 差异尽量大.在经过迭代学习之后,T和G应达到一个平衡.T将可以生成和正常隐变量相似的异常隐变量,生成器则可以通过重建误差来分辨它们.尽管T合成的异常隐变量不是真的数据,但是对于模型识别离群点还是很有帮助的.

Because the updating of E will affect the balance of T and G, we freeze the weights of E when training T and G. If we do not do this, it will be an objective of three networks’ equilibrium, which is extremely difficult to optimize

由于更新E将影响T和G的平衡,因此我们将在训练T和G时冻结E的权重.若不这么做,很难同时优化这三个网络.

### 3.2 Training Step 2: Training E like a Discriminator
In the first training step demonstrated in the previous subsection, we freeze the weights of E. Instead, as shown in Figure 2 (b), we now freeze the weights of T and G and update the encoder E. The encoder not only attempts to project the data samples $x$ to Gaussian latent variables $z$ like the original VAE, but also works like a discriminator in GANs. The objective of the encoder is as follows:

之前章节展示来训练的第一步,通过冻结E的权重.如Fig2(b)显示,现在我们将冻结T和G并更新E.编码器不仅尝试像普通VAE那种从高斯隐变量 $z$ 中产生样本 $x$ ,同时还作为GAN的判别器.其目标函数如下:
$$
L_E=L_{MSE}(x,G(z))+\lambda L_{KLD}(E(x)) \\
+\gamma [m_z-L_{KLD}(E(G(z)))]^+ \\
+\gamma [m_z-L_{KLD}(E(G(z_T)))]^+   \\
$$

$$
=L_{MSE}(x,x_r)+\lambda L_{KLD}(u,\sigma) \\
+\gamma [m_z-L_{KLD}(u_r,\sigma_r)]^+ \\
+\gamma [m_z-L_{KLD}(u_{Tr},\sigma_{Tr})]^+  \tag{10}
$$

The first two terms of Equation 10 are the objective function of plain VAE. The encoder is trained to encode the inputs as close to the prior distribution when the inputs are from the training dataset. The last two terms are the discriminating loss we proposed. The encoder is prevented from mapping the reconstructions of training data to the latent code of the prior distribution

公式10的前两项是普通VAE的目标函数.编码器使输入编码尽量接近先验分布.后两项判别损失是本文提出的.防止编码器将训练数据的重构映射到先验分布的隐编码.

The objective $L_E$ provides the encoder with the ability to discriminate whether the input is normal because the encoder is encouraged to discover differences between the training samples (normal) and their reconstructions (anomalous). It is worth mentioning that the encoder with discriminating ability also helps the generator distinguish between the normal and the anomalous latent code.

目标 $L_E$ 使得编码器能够分辨输入是否是正常的,这是由于编码器被促使发现训练数据(正常)和它们重建(异常)的.另外,具有分辨能力的编码器还有助于区分正常和异常的隐编码.
![](https://raw.githubusercontent.com/hqabcxyxz/MarkDownPics/master/image/20200903175845.png)

### 3.3 Alternating between the Above Two Steps
As described in Algorithm 1, we train alternatively between the above two steps in a mini-batch iteration.  These two steps are repeated until convergence. $detach(·)$ indicates that the back propagation of the gradients is stopped at this point.

如算法1所示,我们在一次小批量迭代中交替执行以上两步,直到拟合. $detach(·)$ 表示在该点的时候停止梯度的反传.

In  the  first  training  step,  the  Gaussian  transformer converts normal latent variables into anomalous latent variables.  At the same time, the generator is trained to generate realistic-like samples when the latent variables are normal and to synthesize a low-quality reconstruction when they are not normal.  It offers the generator the  ability  to  distinguish  between  the  normal  and  the anomalous latent variables.  In the second training step,the encoder not only maps the samples to the prior latent distribution, but also attempts to distinguish between the real data $x$ and generated samples $x_r$.

在第一步训练中,高斯变换网络将正常隐变量转换为异常隐变量.与此同时,生成器当隐变量是正常的时生成生成真实的图片,当隐变量是异常时合成低质量的重建.这使得生成器可以分辨隐变量是正常还是异常.在第二步中,编码器不仅将样本映射到先验隐分布,还尝试区分真实样本 $x$ 和生成样本 $x_r$ .

Importantly, we introduce the competition of E and G into our adVAE model by training alternatively between these two steps.  Analogously to GANs, the generator is trained to fool the encoder in training step 1, and the encoder is encouraged to discriminate the samples generated by the generator in step 2.  In addition to benefitting from adversarial alternative learning as in GANs,the encoder and generator models will also learn jointly for the given training data to maintain the advantages of VAEs.

重点是,通过交替以上两部,我们将E和G的竞争引入到来adVAE.类似GAN,在训练步骤1中,生成器尝试蒙骗编码器,在训练步骤2中,编码器尝试分辨由生成器生成的样本.这样不仅能像GAN那样从对抗学习中获益,还可以让编码器和生成器联合学习训练数据,保持VAE的优势.

### 3.4 Anomaly Score
As demonstrated in Figure 2 (c), only the generator and the encoder are used in the testing phase, as in a traditional VAE. Given a test data point $x∈R^d$ as the input, the encoder estimates the parameters of the latent Gaussian variables $μ$ and $σ$ as the output.Then, the reparameterization trick is used to sample $z=\{z^{(1)},z^{(2)},...,z^{(L)}\}$ according to the latent distribution $N(μ,σ^2)$ , i.e., $z(l)=μ+σ\bigodotε^{(l)}$ , where $ε∼N(0,I)$ and $l=1,2,...L$ . L is set to 1000 in this work and used to improve the robustness of adVAE’s performance. The generator receives $z^{(l)}$ as the input and outputs the reconstruction $x^{(l)}_r∈R^d$ .

如Fig2(c), 如同传统VAE,测试阶段仅使用生成器和编码器.给定输入数据点 $x\in{R^d}$ ,编码器估计输出的隐高斯变量的 $u$ 和 $\sigma$ .一个重参数化的技巧是,根据隐分布 $N(u,\sigma^2)$ 采样 $z=\{z^{(1)},z^{(2)},...,z^{(L)}\}$ ,这里 $z(l)=μ+σ\bigodotε^{(l)}$,$ε∼N(0,I)$ , $l=1,2,...L$ .L被设置为1000来提高adVAE的鲁棒性.生成器以 $z^{(l)}$ 作为输入,将重构 $x^{(l)}_r∈R^d$ 作为输出.

The error between the inputs $x$ and their average reconstruction $∑^L_{l=1}x^{(l)}_r$ reflects the deviation between the testing data and the normal data distribution learned by adVAE, such that the anomaly score of a mini-batch data $x∈R^(n×d)$ ( $n$ is the batch size) is defined as follows:

输入 $x$ 个其平均重建 $∑^L_{l=1}x^{(l)}_r$ 的误差反映来测试数据和adVAE学到的正常数据分布的差异,一个batch的异常分数 $x∈R^(n×d)$ ( $n$ 是batchsize)定义如下:

$$
S=ee^T \\
$$

$$
=(x-\frac{1}{L}\sum^L_{l=1}{x^{(l)}_r})(x-\frac{1}{L}\sum^L_{l=1}x^{(l)}_r)^T,   \tag{11}
$$

$$
s=\{S_{11},S_{22},...,S_{nn}\}   \tag{12}
$$

with error matrix $e∈R^{n×d}$ and squared error matrix $S∈R^{n×n}$ . $s∈R^n$ is the anomaly scores vector of a mini-batch dataset $x$ . The dimension of $s$ is always equal to the batch size of $x$ .  Data points with a high anomaly score are classified as anomalies. To determine whether an anomaly score is high enough, we need a decision threshold. In the next subsection, we will illustrate how to decide the threshold automatically by KDE [37].

$e∈R^{n×d}$ 是误差矩阵, $S∈R^{n×n}$ 是平方误差矩阵. $s\in{R^n}$ 是小批量数据 $x$ 的异常分数向量. $s$ 的维数和 $x$ 的batch-size相等.有着高异常分数的数据点被分为异常类.为了确定异常分数是否足够高,我们设定了一个阈值.在下一节中,我们将介绍如何使用KDE(核密度估计)[37]来计算阈值.

### 3.5 Decision Threshold
Earlier VAE-based anomaly detection studies [13, 16]often overlook the importance of threshold selection.Because we have no idea about what the values of reconstruction error actually represent, determining the threshold of reconstruction error is cumbersome. Some studies [13–15] adjust the threshold by cross-validation.However, building a big enough validation set is a luxury in some cases, as anomalous samples are challenging to collect. Other attempts [16, 38] simply report the best performance in the test dataset to evaluate models, which makes it difficult to reproduce the results in practical applications. Thus, it is exceedingly important to let the anomaly detection model automatically determine the threshold.

早先基于VAE的异常检测研究[13,16]常常忽视来阈值选择的重要性.因为我们不知道重建误差的值究竟代表着什么,确定重建误差的值是很复杂的.一些研究[13-15]采用交叉验证的方式来调整阈值,但建立一个大而全的验证集代价很大,且一些异常样本难以收集.一些工作[16,38]则干脆只汇报在测试集上最好的一次结果作为对模型性能的衡量,导致来这类方法在实际中难以复现.因此,让异常检测模型自动确定阈值是很必要且重要的.

The KDE technique [37] is used to determine the decision threshold in the case where only normal samples are provided. Given the anomaly scores vectors of the training dataset, KDE estimates the probability density function (PDF) $p(s)$ in a nonparametric way:

在只有正常样本提供的情况下,KDE[37]常被用来确定决策阈值.给定训练集的异常分数向量,通过非参数的方式,KDE可以估计出其概率密度函数(PDF) $p(s)$ :

$$
p(s)≈\frac{1}{mh}∑^m_{i=1}K(\frac{s−s_i}{h}) \tag{13}
$$

where $m$ is the size of the training dataset, ${si},i=1,2...,m$ , is the training dataset’s anomaly scores vector, $K(·)$ is the kernel function, and $h$ is the bandwidth.

训练集大小为 $m$ , ${si},i=1,2...,m$ 是训练集的异常分数向量, $K(·)$ 是核函数, $h$ 是其带宽. 

Among all kinds of kernel functions, radial basis functions (RBFs) are the most commonly used in density estimation. Therefore, a RBF kernel is used to estimate the PDF of the normal training dataset:

这里我们使用径向基函数作为核函数:

$$
K_{RBF}(s;h)∝exp(−\frac{s^2}{2h^2}).  \tag{14}
$$

In practice, the choice of the bandwidth parameter $h$ has a significant influence on the effect of the KDE. To make a good choice of bandwidth parameters, Silverman [39] derived a selection criterion:

在实际中,带宽参数 $h$ 的选择对于KDE的性能影响很大.为了更好的选出这个参数,Silverman[39]推导出了以下选择准则:

$$
h=(\frac{m(d+2)}{4})^{−\frac{1}{(d+4)}}   \tag{15}
$$

where $m$ is the number of training data points and $d$ the number of dimensions. After obtaining the PDF $p(s)$ of the training dataset by KDE, the cumulative distribution function (CDF) $F(s)$ can be obtained by equation (16):

这里 $m$ 为训练集数据数量, $d$ 为其维度,使用KDE得到训练集的概率密度函数 $p(s)$ 之后,累计分布函数(CDF) $F(s)$ 可以按照公式16计算得到:

$$
F(s)=∫^s_{−∞}p(s)d s   \tag{16}
$$

Given a significance level $α∈[0,1]$ and a CDF, we can find a decision threshold $s_α$ that satisfies $F(s_α)=1−α$ .In this case, there is at least $(1−α)100\%$ probability that a sample with the anomaly score $s≥s_α$ is an outlier. Because KDE decides the threshold by estimating the interval of normal data’s anomaly scores, it is clear that using KDE to decide the threshold is more objective and reasonable than simply relying on human experience. A higher significance level $α$ leads to a lower missing alarm rate, which means that models have fewer chances to mislabel outliers as normal data. On the contrary, a lower $α$ means a lower false alarm rate. Therefore, the choice of the significance level $α$ is a tradeoff.The significance level $α$ is recommended to be set to 0.1 for anomaly detection tasks.

给定重要权重 $α∈[0,1]$ 和累加分布函数,我们可以找到一个阈值 $s_α$ 满足 $F(s_α)=1−α$ .这意味着一个样本的异常分数 $s≥s_α$ ,它有 $(1−α)100\%$ 的机率是一个离群点.因为KDE是通过估计正常样本异常分数的间隔来确定阈值的,比起使用人工经验,这样做显然更有目的性和合理性. $α$ 越高,漏报率越低,反之 $α$ 越低,误报率越低.如何选择需要权衡.在异常检测任务中,本文推荐 $α$ 为0.1.

>![](https://raw.githubusercontent.com/hqabcxyxz/MarkDownPics/master/image/20200903200758.png)  
>Figure 3: Work flow of using a trained adVAE model to choose threshold automatically and detect outliers.  
>Fig3:使用训练好的adVAE模型来自动选择阈值和检测离群点的工作流程.

### 3.6 Detecting Outliers
In this subsection, we summarize how to use a trained adVAE model and the KDE technique to learn the threshold from a training dataset and detect outliers in a testing dataset. As illustrated in Figure 3, this process is divided into two parts.

本节将总结如何使用训练好的adVAE模型和KED从训练集中学到的阈值来检测测试集中的离群点.如Fig3所示,这分为两个部分.

Part I focuses on the training process. Given a training dataset matrix  $X∈R^{m×d}$ consisting of normal samples, we can calculate their anomaly scores vector $s∈R^m$ (described in subsection 3.4), where $m$ is the size of training dataset and $d$ is the dimension of the data. As described in subsection 3.5, the PDF $p(s)$ of the anomaly scores vector is obtained by KDE and the CDFF(s) is obtained by $∫^s_{−∞}p(s)d s$ . Then, the decision threshold $s_α$ can be determined from a given significance level $α$ and CDFF(s).

第一部分主要是在训练过程.给定一个全是正常样本的数据集矩阵 $X∈R^{m×d}$ ,我们可以计算出它们的异常分数向量 $s∈R^m$ (如3.4节所述),这里 $m$ 是训练集大小 ,$d$ 是数据维度.如3.5小节,通过KDE可以得到概率密度函数 $p(s)$ ,按照 $∫^s_{−∞}p(s)d s$ 可以得到累计概率密度函数.再给定重要等级 $α$ ,就可以确定阈值 $s_α$ .

Part II is simple and easy to understand. The anomaly score $s_{new}$ of a new sample $x_{new}$ is calculated by adVAE.If $s_{new}≥s_α$ , then $x_{new}$ is defined as an outlier. If not,then $x_{new}$ is regarded as a normal sample.

部分二就很简单来.使用adVAE计算新样本 $x_{new}$ 的异常分数 $s_{new}$ ,然后根据阈值确定是否异常.

## 4.Experiments
### 4.1 Datasets
Most previous works used image datasets to test their anomaly detection models. To eliminate the impact of different convolutional structures and other image tricks on the test performance, we chose five publicly available and broadly used tabular anomaly detection datasets to evaluate our adVAE model. All the dataset characteristics are summarized in Table 1. For each dataset, 80% of the normal data were used for the training phase, and then the remaining 20% and all the outliers were used for testing. More details about the datasets can be found in their references or our [Github repository](https://github.com/WangXuhongCN/adVAE).

过去很多工作都是在图像数据集上测试异常检测算法。为了消除不同卷积结构和其他图像的技巧对测试性能的影响，本文选了5个公开的表格异常检测数据集来评价我们的adVAE模型。表1总结了所有数据集的特征。对于每个数据集，使用80%的正常样本来进行训练，剩余20%的正常样本和所有的离群样本都被用于测试。更多的关于数据集的细节参见[Github](https://github.com/WangXuhongCN/adVAE)。

### 4.2 Evaluation Metric
The anomaly detection community defines anomalous samples as positive and defines normal samples as negative, hence the anomaly detection tasks can also be regarded as a two-class classification problem with a large skew in class distribution. For the evaluation of a two-class classifier, the metrics are divided into two categories; one category is defined at a single threshold,and the other category is defined at all possible thresholds.

在异常检测中通常定义异常样本为正正常样本为负，因此异常检测任务也可被视为是一个类分布差距极大的二分类任务。针对评价二分类分类器，评价方法可以被分为两类；一类是仅在单个阈值上定义，另一类是在所有可能的阈值上都定义。

**Metrics at a single threshold.** Accuracy, precision,recall, and the F1 score are the common metrics to evaluate models performance at a single threshold. Because the class distribution is skew, accuracy is not a suitable metric for anomaly detection model evaluation.

**在单一阈值上衡量。** 精确度，准确度，召回率，F1分数是在单一阈值上衡量一个模型性能的常用手段。因为类分布是极端不平衡的，因此精确度不适合用来衡量。

High precision means the fewer chances of misjudging normal data, and a high recall means the fewer chances of models missing alarming outliers. Even if models predict a normal sample as an outlier, people can still correct the judgment of the model through expert knowledge, because the anomalous samples are of small quantity. However, if models miss alarming outliers, we cannot find anomalous data in such a huge dataset. Thus, precision is not as crucial as recall. The F1 score is the harmonic average of precision and recall. Therefore, we adopt recall and the F1 score as the metrics for comparing at a single threshold.

高准确度表示对于正常样本的误判更少，高召回率表示对于漏检离群点的几率更小。尽管一个模型将正常样本预测为离群点，但由于异常样本数量极少，人们依然可以通过先验知识来纠正模型的结果。但是，一旦模型漏检了一个离群点，我们将很难在从海量的数据中找到异常样本了。因此准确度不及召回率重要。而F1参数是用来调节二者平衡的。这里我们使用召回率和F1分数来作为模型在单一阈值下的衡量标准。

**Metrics at all possible thresholds.** The anomaly detection community often uses receiver operator characteristic (ROC) and precision–recall (PR) curves, which aggregate over all possible decision thresholds, to evaluate the predictive performance of each method. When the class distribution is close to being uniform, ROC curves have many desirable properties. However, because anomaly detection tasks always have a large skew in the class distribution, PR curves give a more accurate picture of an algorithm’s performance [45].

**在所有阈值上衡量.** ROC曲线和RP曲线是异常检测中常用的评价手段,它们可以汇集所有的阈值,然后评价模型的性能.当类别分布均衡时,ROC曲线的效果不错.然而,对于异常检测这个类别分布极端不均衡的任务,PR曲线用于衡量算法性能反而更加精准.

Rather than comparing curves, it is useful and clear to analyze the model performance quantitatively using a single number. Average precision (AP) and area under the ROC curve (AUC) are the common metrics to measure performance, with the former being preferred under class imbalance.

比起比较曲线,使用单一指标评价模型性能会更清晰有效.在类别不平衡的情况下,使用平均精度(AP)比ROC曲线面积(AUC)更好.

AP summarizes a PR curve by a sum of precisions at each threshold, multiplied by the increase in recall,which is a close approximation of the area under the PR curve: $AP=∑_nP_n∆R_n$ , where $P_n$ is the precision at the $n^{th}$ threshold and $∆R_n$ is the increase in recall from the $n−1^{th}$ to the $n^{th}$ threshold. Because the PR curve is more useful than the ROC curve in anomaly detection, we recommend using AP as an evaluation metric for anomaly detection models rather than AUC. In our experiments, recall, F1 score, AUC, and AP were used to evaluate the models performance.

AP就是PR曲线下面积的近视: $AP=∑_nP_n∆R_n$ , $P_n$ 是 $n^{th}$ 阈值下的精度, $∆R_n$ 是阈值 $n−1^{th}$ 到 $n^{th}$ 召回的变化量.由于PR曲线在异常检测任务中比ROC曲线更好,因此我们推荐使用AP作为评价指标而不是AUC.本文实验指标采用召回,F1 分数,AUC和AP.

### 4.3 Model Configurations
The network structures of adVAE used are summarized in Figure 4, where the size of each layer was proportionate to the input dimension dim(x). The Adam optimizer was used in all datasets with a learning rate of 0.001. We trained each model for a maximum 20000 mini-batch iterations with a batch size of 128. Kaiming uniform weight initialization was used for all three subnetworks.

Fig4展示来adVAE的网络结构.其中每层大小和输入的维度 dim(x)成比例.使用学习率为0.001的adam.以128大小的Batch size训练20000步.每个网络使用 Kaiming 初始化.

>![Fig4](https://raw.githubusercontent.com/hqabcxyxz/MarkDownPics/master/image/20200904100701.png)  
>Figure 4: Network structure of adVAE. For the encoder, generator, and transformer networks, we adopted fully connected neural networks consisting of three hidden layers. For the individual dataset, we adjust the number of neurons according to the dimension of the training data dim(x)   
>Fig4:adVAE的网络结构,对于编码器,生成器,变换网络,我们都使用一个三层的全链接网络.对于每个单独的数据集,我们将根据数据维度调增神经元数量.

As shown in Table 2, there are four hyperparameters in adVAE model. $λ$is derived from plain VAE, and the other three new parameters are used to maintain the balance between the original training objective and the additional discrimination objective. The larger the $γ$ or $m_z$,the larger the proportion of the encoder discrimination objective in the total loss. The larger the $m_x$ , the larger the proportion of the generator discrimination objective.

如表2所示,adVAE主要有4个超参. $λ$ 是普通VAE所有的,其他三个是用来平衡原始训练目标和额外的判别目标的, $γ$ 或 $m_z$ 越大,编码器判别所占损失的比重越大. $m_x$ 越大,生成器判别所占损失越大.

![Tab2](https://raw.githubusercontent.com/hqabcxyxz/MarkDownPics/master/image/20200904101101.png)

The KLD margin $m_z$ can be selected to be slightly larger than the training KLD value of plain VAEs and the MSE margin $m_x$ was set equal to 2 for each dataset $γ$ and $m_z$ have similar effects, $γ$ is always suggested to be fixed as 0.001. Adjusting the parameters becomes easier, because we can simply consider tuning $m_x$ and $m_z$ . Note that the structure and hyperparameters used in this study proved to be sufficient for our applications, although they can still be improved.

可以将KLD边界 $m_z$ 设置的比普通VAE的大,MSE的边界 $m_x$ 可以设置为2.由于 $γ$  和 $m_z$ 有同等效力, $γ$ 设置成0.001.为了简化调参,我们仅仅考虑调整 $m_x$ 和  $m_z$ .虽然本文实验显示这组参数很有效,但它们仍有提升的空间.

### 4.4 Compared Methods
We compare adVAE with the eleven most advanced outlier detection algorithms, which include both traditional and deep-learning-based methods. To obtain a convincing conclusion, the hyperparameters of competing methods are searched in a range of optional values. They can be divided into seven categories: (i) A boundary-based approach, OCSVM [31]. It is a widely used semi supervised outlier detection algorithm, and we use a RBF kernel in all tasks. Because OCSVM needs an important parameter $ν$, its value will be searched in the range {0.01, 0.05, 0.1, 0.15, 0.2}. (ii) An ensemble-learning-based approach, IForest [46]. The hyperparameter to be tuned is the number of decision trees $n_t$,which is chosen from the set {50, 100, 200, 300}. (iii)A probabilistic approach, ABOD [47]. (iv) Distance-based approaches, SOD [48] and HBOS [49]. Because the performance of ABOD and SOD will be dramatically affected by the size of the neighborhood, we tune it in the range of{5, 8, 16, 24, 40}. For HBOS, the number of bins is chosen from{5, 10, 20, 30, 50}. (v) Three GAN-based models, GAN [12], MOGAAL [24], and WGAN-GP [36]. GAN shares the same network structure as adVAE, except that the output layer of the discriminator is one-dimensional, and the configurations of MOGAAL refer to its official open source code.Based on the network structure of GAN, WGAN-GP removes the output activation function of the generator. (vi) Three reconstruction-based methods, AE [13],DAGMM [38], and VAE [13]. The AE, VAE, and DAGMM share the same network structure as adVAE,except that the hyperparameter dimension of DAGMM is required to be 1. VAE shares the same hyperparameter $λ$ as adVAE. (vii) Two ablation models of the proposed adVAE, named E-adVAE and G-adVAE, are also com-pared to demonstrate how the two discrimination objectives affect the performance. Figure 5 and 6 show the architecture of our two ablation models, respectively. E-adVAE, G-adVAE, and adVAE share the same parameters and network structure. 

适用来11种先进的离群点检测算法来作为对比,这里面既有传统的也有基于深度学习的. 为了得到更加有说服力的结论,方法的超参将在一系列值中取最好.这些方法可以被分为7大类:
(i) 基于边界的方法OCSVM[31],这是一种被广泛适用的半监督离群点检测算法,我们适用RBF核函数.它的重要参数$ν$将在{0.01, 0.05, 0.1, 0.15, 0.2}中搜索.
(ii)嵌入学习类的方法,IForest[46].需要调整的超参是决策树数量$n_t$,从{50, 100, 200, 300}中选.
(iii)基于统计的方法,ABOD[47].
(iv)基于距离的方法.SOD[48]和HBOS[49].ABOD和SOD的性能和邻域大小有关,我们从{5, 8, 16, 24, 40}中选.对于HBOS,bin 的数量从{5, 10, 20, 30, 50}选.
(v)三个基于GAN的方法,GAN[12],MOGAAL[24],WGAN-GP[36].GAN 和 adVAE 网络结构相同,只不过将判别器的输出换成1维,MOGAAL的配置参考官方代码.WGAN-GP则去掉了生成器的输出激活函数.
(vi)三个重建类的方法,AE[13],DAGMM[38] 和VAE[13].这三和 adVAE 网络结构相同,只不过DAGMM的隐维度是1.VAE和 adVAE 共享超参$λ$.
(vii) 两个消融模型.叫E-adVAE 和 G-adVAE,用来衡量两个判别条件对于最终测试性能的影响.图5和6展示来这两的网络结构,这两和 adVAE 共享参数和网络结构.

All deep-learning methods are implemented in pytorch and share the same optimizer, learning rate, batch size, and iteration times as adVAE, except that the parameters of DAGMM and MOGAAL refer to their author’s recommendation. All traditional methods are implemented on a common outlier detection framework PyOD [50].

所有深度学习类方法用 pytorch 实现,用和 adVAE 同样的优化器,学习率,批量大小,迭达次数.DAGMM和MOGAAL则使用来原作者推荐参数.所用传统方法实现均来自离群检测框架 PyOD[50].

>![fig5](https://raw.githubusercontent.com/hqabcxyxz/MarkDownPics/master/image/20200904110330.png)
>Figure 5:  Structure of E-adVAE. As adVAE is formed by adding a discrimination objective to both the generator and the encoder of plain VAE, we also propose an ablation model E-adVAE, in which only the encoder has the discrimination objective
>Fig5:E-adVAE结构.即仅在编码器上添加判别目标.

>![Fig6](https://raw.githubusercontent.com/hqabcxyxz/MarkDownPics/master/image/20200904110548.png)   
>Figure 6: Structure of G-adVAE. This is the other ablation model G-adVAE: only the generator has the discrimination objective.   
>Fig6:G-adVAE结构.仅在生成器上添加判别目标.

### 4.5 Results and Discussion
Table 3 shows the experimental results for the AP and AUC metrics, and Table 4 indicates the results for the recall and F1 score. According to the experimental results, the adVAE model achieved 11 best and 5 second-best results from 24 comparisons. Therefore, adVAE is significantly better than other compared methods.

表3展示了AP和AUC,表4展示来F1分数和召回.在24项比较中,adVAE有11最好,5项次好.由此可以看出 adVAE 比其他方法好.

Among all the experimental results, the performance of VAE is generally slightly better than AE, because VAE has a KLD regularization of the encoder, which proves the importance of regularization in AE-based anomaly detection algorithms.

在所有实验中显示,VAE普遍比AE结果略好,这是由于VAE对于编码器有KLD正则,这显示来基于AE的异常检测算法中,正则是很重要的

We customize plain VAE by the proposed self-adversarial mechanism, reaching the highest AP improvement of 0.255 (40.8%). Compared to its base technique VAE, on average, adVAE improved by 0.113(14.9%) in AP, 0.036 (3.9%) in AUC, 0.046 (5.3%) in recall, and 0.049 (7.3%) in F1 score. The outstanding results of adVAE prove the superiority of the proposed self-adversarial mechanism, which helps deep generative models to play a greater role in anomaly detection research and applications.

对普通VAE采用自对抗策略,最高AP可以提升0.255(40.8%).相较于基础VAE,adVAE AP提升来0.113(14.9%),AUC提升0.036(3.9%),recall提升0.046(5.3%),F1分数提升0.049(7.3%).这个结果正米来自我对抗机制的优越性,这将有助于深度生成模型在异常检测研究和应用中发挥更大的作用.

OCSVM performed well in the cardio, satellite, and pen datasets, but achieved horrible results in the letter and optical datasets. The results indicate that the conventional method OCSVM is not stable enough, leading the AP gap from adVAE to reach an astonishing 0.8 in the optical dataset. Actually, all traditional anomaly detection methods (e.g., IForest, HBOS, and SOD) have the same problem as OCSVM: They perform well on some datasets, but they perform extremely poorly on other datasets. ABOD’s performance is more stable than others, but it still suffers from the above problem.

OCSVM(one-class SVM)在心脏,卫星,笔数据集上表现不错,但是在字母和光学数据上表现很糟糕.显然传统的OCSVM方法不够鲁棒,在光学上AP和adVAE甚至差了0.8.实际上,所有的传统异常检测方法(孤立森林,HBOS和SOD)都有着和OCSVM一样的问题:不是对所有数据集都适用.ABOD比这几个略好点,但也好不到哪去.

The reason for this is that conventional methods often have a strict data assumption. If training data are high-dimensional or not consistent with the data assumption, their performance may be significantly impaired. Because neural-network-based methods do not need a strict data assumption and are suitable for big and high-dimensional data, they have more stable performance on diverse datasets with different data distributions.

原因就是传统方法对于数据有严格的假设.若训练数据是高维或和数据假设不符合,它们的性能就不行来.由于基于神经网络的方法不需要严格的数据假设,并且适用于高维的海量数据,因此这类方法在各种不同的数据分布上都有更加稳定的性能.

In our experiments, both VAE and AE outperformed the GAN-based methods in most datasets. The results are analyzed as follows: Because the mode collapse issue of GANs cannot be characterized in a training loss curve, people usually monitor the generated images during the training phase. However, for tabular datasets, it is difficult for us to ensure that the generated samples are diverse enough. A GAN with mode collapse would mislabel normal data as outliers and cause a high false positive rate, because GAN learns an incomplete normal data distribution in this case. To better recover the distribution of normal data, MOGAAL suggests stopping optimizing the generator before convergence and expanding the network structure from a single generator to multiple generators with different objectives. However, a nonconverged model brings more uncertainty to experimental results, because it is impossible to know what the model has learned precisely. According to the results, it does have a performance improvement over plain GAN, but still has a performance gap with AE-based methods. WGAN-GP tries to avoid mode collapsing, which leads to better results of all four metrics than plain GAN. However, it still results in worse performance than encoder-decoder-based methods. GAN is better at generating high-quality samples rather than learning the complete probability distribution of the training dataset. Thus GAN-based methods do not perform well for outlier detection.

在本文实验中,VAE和AE都比基于GAN的方法要好.原因如下:训练的损失曲线无法描述GAN的模式崩溃,在训练阶段人们会监控生成图像.但是对于表格数据集,这很难保证生成的样本是多样的.有模式崩溃的GAN会将正常样本误识别离群点从而得到一个高的假阳率，因为GAN学习到的是不完整的正常数据分布。为了更好的恢复正常数据的分布，MOGAAL建议在生成器收敛前停止优化并将网络结构从单一生成器拓展到多个带有不同目标的生成器。然而没有收敛的模型会给实验结果带来更多的不确定性，因为不知道模型究竟学到什么。根据结果，比起普通的GAN这种方法确实有提升，但是和AE类的方法相比仍然有差距。WGAN-GP尝试避免模式崩溃，它的4个衡量指标比普通GAN更好，但是还没有AE类的方法好。GAN在生成高质量的样本上更有优势，而不是学习整个数据集的分布。因此基于GAN的方法在离群点检测任务中不够适合。

For three reconstruction-based neural network models, there is no significant performance gap between AE and VAE, whereas DAGMM is worse than them in our tests. This is explained by the fact that the latent dimension of DAGMM is required to be set as 1,which severely limits the ability of the model to summarize datasets. The results also indicate this explanation: DAGMM only achieves good results in the cardio dataset, which has the lowest data size and dimensions.The larger the size and dimensions of the dataset, the larger performance gap between DAGMM and AE.

针对三个基于重建的神经网络方法，和AE，VAE相比没有明显的性能差异，其中DAGMM性能最差。这是由于DAGMM要求隐维度是1，这个限制来模型学习整个数据集的能力。由此可以得出一个结论：DAGMM 在 cardio 数据集上取得好的效果是因为数据维度低且数据量小。数据维度和数量越大，AE和DAGMM的表现差距越大。

In conclusion, the AE and VAE methods are better than most of the other anomaly detection methods according to our tests. Moreover, adVAE learns to detect outliers through the self-adversarial mechanism, which further enhances the anomaly detection capability and achieves state-of-the-art performance. We believe that the proposed self-adversarial mechanism has immense potentiality, and it deserves to be extended to other pattern recognition tasks.

综上，AE和VAE的方法比其他异常检测方法要好。其中 adVAE 通过自对抗机制学习离群点，这大大加强来它检测异常的能力，取得来 sort 性能。我们相信自对抗机制有着巨大的潜力，值得推广到其他模式识别任务上。

> ![Tab3](https://raw.githubusercontent.com/hqabcxyxz/MarkDownPics/master/image/20200904111218.png)   
> Table 3:  AP and AUC comparisons between the baselines and adVAE. AP is a better metric than AUC in anomaly detection.  Note that the best result is typeset in bold and the second best in italic typeface. The dataset names are represented as their first three letters and Avg. represents the average.   
> Tab3:adVAE 和一些基线方法的AP和AUC。AP比AUC在异常检测上更加有优势。粗体是最好结果，斜体是次好的。每个数据集名称用前三个字母替代，Avg.代表来平均值。

>![Tab4](https://raw.githubusercontent.com/hqabcxyxz/MarkDownPics/master/image/20200904111325.png)   
>Table 4: Recall and F1 score comparisons between the baselines and adVAE. For all reconstruction-based methods (AE, VAE, DAGMM, G-adVAE,E-adVAE, and adVAE), the significance level $α$ was set to 0.1. Note that the best result is typeset in bold and the second best in italic typeface. The dataset names are represented as their first three letters and Avg. represents the average.   
>Tab4:基线方法和adVAE的召回和F1分数.

### 4.6. Ablation Study
To  illustrate  the  impact  of  different  designs  on  the model results, we design three different types of ablation experiments:

为了展示不同设计对模型结果的影响,我们设计来三个不同类型的消融实验:

(i) Because adVAE is formed by adding two different training objectives to plain VAE, it is important to demonstrate how the two discrimination objectives affect the performance.  Hence we propose E-adVAE and G-adVAE models (Figure 5 and 6) to demonstrate the effect of removing one of the discrimination objectives from  adVAE.  According  to  Table  3  and  4,  the  results show that both G-adVAE and E-adVAE achieve performance improvements compared to plain VAE, which indicates the validity of the two discriminative objectives of  adVAE  In  addition,  E-adVAE  is  more  competitive than G-adVAE in most datasets. This result is analyzed as follows.  Because the latent variables $z$ have a lower dimension than the original data $x$ , $x$ has more information than $z$ . Therefore, the benefit of adding the discriminating target to the encoder E is more significant than adding it to the generator G.

(i) adVAE 是通过给普通VAE添加两个不同训练目标得来的,因此显示这两个判别目标对性能影响的重要性是必要的.我们提出来E-adVAE 和G-adVAE,分别是去掉来一个判别目标.根据表3和4,结果显示G-adVAE 和 E-adVAE 都比普通VAE好,其中在多数数据集上,E-adVAE 要更好一些.这是由于隐变量 $z$ 比原始数据 $x$ 的维度少,信息不如原始数据多.因此编码器E比生成器G获益更多.

(ii) Because the encoder and generator work together to detect outliers, we performed a latent space visualization to prove the improvement in the encoder’s anomaly detection capability. Figure 8 illustrates that the encoder of  adVAE  has  a  better  discrimination  ability,  because the latent variables are separated in the latent space and more easily distinguished by the later generator.

(ii)因为编码器和生成器一同检测离群点,因此我们将隐空间可视化出来证明编码器的异常检测能力.Fig8显示来adVAE的编码器有更好的异常检测能力,因为隐变量在隐空间中是分离的的,比生成器的更好分辨.

(iii)  To  independently  verify  the  anomaly  detection capabilities of the generator, we synthesize latent outlier variables $z_o$ by adding five types of noise to the normal latent variables $z_n$ .  These latent variables are the outputs  of  the  five  corresponding  encoders  and  the  input data of the encoders are 200 normal samples $x_n$ from the pen [44] datasets The normal scores vector $s_n∈R^{200}$ can be calculated from $x_n$ and $G(z_n)$ , and the anomalous scores vector $s_o∈R^{200}$ can be calculated from $x_n$ and $G(z_o)$ (as in section 3.4).  Afterwards, the Wasserstein distance between $s_n$ and $s_o$ is used to measure the detection capabilities of each generator. Figure 7 indicates that the generator G of adVAE also has a better ability to discriminate latent variables.

(iii)为了分开验证生成器的异常检测能力,我们通过对正常隐变量 $z_n$ 添加五种噪声来合成来隐离群变量 $z_o$ .这些
隐变量是5个对应编码器的数据,输入数据是来自pen数据集的200张正常样本$x_n$.正常分数向量 $s_n∈R^{200}$ 可以根据 $x_n$ 和 $G(z_n)$ 计算出来,异常分数向量 $s_o∈R^{200}$ 可以根据 $x_n$ 和 $G(z_o)$ 计算(如3.4节).$s_n$ 和 $s_o$ 之间的[[../../ClassicBlog/From GAN to WGAN|Wasserstein]]距离被用来衡量每个生成器的检测能力.图7展示来adVAE的生成器G有更好的能力分辨隐变量.

In conclusion,  benefiting from better discrimination abilities,  adVAE  has  better  anomaly  detection  performance.  This proves that the proposed self-adversarial mechanism is a prospective way of customizing generative models to fit outlier detection tasks.

综上,收益于更好的分辨能力,adVAE 有更好的异常检测性能.这也证明了本文提出的自对抗即使能够通过定制声称模型来适合离群点检测任务.

>![Fig7](https://raw.githubusercontent.com/hqabcxyxz/MarkDownPics/master/image/20200904114855.png)   
>Figure  7:   To  independently  verify  the  sensitivity  of  the  generator to anomalies, we add different noises to the normal latent variables.Noise 1:  adding a uniform distribution $U(0,1)$ ;  Noise 2:  adding a Gaussian distribution $N(0,1)$ ; Noise 3:  adding a constant value 0.5;Noise 4:  multiplying the last half dimensions of the latent vector by 0.5; Noise 5: setting the first half dimensions of the latent vector to 0.The larger the Wasserstein distance, the better the anomaly detection capability.  The generator of adVAE achieves the best discrimination performance, and G-adVAE’s generator is the second best. Benefiting from the adversarial training between E and G, the detection performance of the E-adVAE’s generator is also improved in comparison to plain VAE.   
>图7:为独立验证生成器对异常点的敏感性,我们添加来不同的噪声到正常隐变量.噪声1:平均分布噪声.噪声2:高斯分布.噪声3:哼定制0.5.噪声4:对一半维度的隐变量乘以0.5.噪声5:将隐变量的前一半置零.Wasserstein 距离越大,异常检测能力越强.adVAE 的生成器有最好的性能,G-adVAE 的生成器次之.受益于E和G的对抗训练,E-adVAE 的生成器比一般的VAE也有更好的检测性能.

>![Fig8](https://raw.githubusercontent.com/hqabcxyxz/MarkDownPics/master/image/20200904115005.png)   
>Figure 8: T-SNE visualization of the latent space in the letter dataset. The orange points represent the latent variables encoded from normal data, and the blue points are encoded from anomalous data. The KDE curves at the top indicate that adVAE’s latent space overlaps are smaller than VAE’s,which means that the encoder of adVAE has a better ability to distinguish the normal data from the outliers.  The visualization plot of G-adVAE does not show significant advantages compared to plain VAE, and E-adVAE’s plot proves that the encoder does benefit from the discrimination objective.   
>图8:T-SEN将 letter 数据集的隐空间可视化.橙色点代表了正常数据的隐编码,蓝色是异常.头上KDE曲线显示来 adVAE 的隐空间重叠比VAE的更小,这意味着 adVAE 的编码器有更强的离群点检测能力.G-adVAE 的可视化显示和普通 VAE差距不大,而E-adVAE 图则证明来编码器确实有从判别目标中获益.

>![Fig9](https://raw.githubusercontent.com/hqabcxyxz/MarkDownPics/master/image/20200904115104.png)   
>Figure 9: Robustness experiment results. (a) $λ$ is the hyperparameter derived from VAE, which has a significant impact on performance, and hence it should e set carefully. (b) $γ$ and $m_z$ have a similar role: enhancing the encoder’s discrimination objective. Because adVAE with $γ=0.01$ achieves the best results in all datasets, we suggest fixing $γ$ as 0.01 and tuning $m_z$ instead. (c) The model performance is not sensitive to $m_z$, and it should be considered to simply set it between 20 and 40. (d) $m_x$ is used to enhance the generator’s discrimination objective, and the choice of this parameter is not very critical.  For convenience, we fixed it as 2 for all tests.  (e) As long as the scale of the neural network is large enough, the number of network layers or the number of neurons will not have a significant impact on the results.  (f) Contaminated training data negatively affect the detection accuracy and adVAE is more sensitive, especially when the contamination ratio is larger than 3%, because the additional discrimination objectives of adVAE are more susceptible to contamination data.   
>图9:鲁棒性实验结果.(a) $λ$ 是从VAE继承来的超参,对于性能的影响很大,要小心设置.(b) $γ$ 和 $m_z$ 作用类似,用于加强编码器的判别目标,因为 adVAE 在 $γ=0.01$ 时取得来最好的结果,因此我们建议将 $γ$ 固定0.01,调整 $m_z$ 就行.(c) $m_z$ 对模型影响不大,建议设置在20到40.(d) $m_x$ 是用于将加强生成器的判别目标的,且该参数影响不大,为了简化实验,我们固定为2.(e)只要神经网络规模足够大,网络层数量或神经元数量就不会对结果产生显著影响.(f)脏的训练数据对检测精度有负作用,并且 adVAE 对此更加敏感,尤其是当污染率大于3%.这是因为额外的判别目标更易受脏数据影响.

### 4.7 Robustness Experiments
Because adVAE involves four primary hyperparameters, it is essential to implement in-depth hyperparameter experiments comprehensively.  As shown in Figure9(a)–9(d), we tested the results for four hyperparameters. Actually, only $m_z$ and $m_x$ need to be tuned, because $λ$ is derived from VAE and $γ$ is fixed as 0.01.   Based on that, finding good hyperparameters of adVAE is not more complicated than plain VAE, because it is not sensitive to $m_z$ and $m_x$ according to Figure 9(c) and 9(d).

因为 adVAE 引入了4个主要的超参,因此全面而深入的进行超参实验是必要的.如图9(a)~9(d),我们测试了4个超参的结果.实际上只有 $m_z$ 和$m_x$ 需要调整. $λ$ 是从VAE引入的且 $γ$ 固定为0.01.根据图9(c)和9(d),找到 adVAE 良好的超参并不必普通 VAE复杂,因为$m_z$ 和 $m_x$ 影响不大.

As for the learning ability of neural networks, we tune the number of layers from 2 to 6 and adjust the number of hidden neurons from 1/3 dim(x) to dim(x).  The results are shown in Figure 9(e),  where V1–V5 represent the parameter variation.  It can be seen that V1 achieved as lightly worse detection capability, because the insufficient networks cannot learn enough information of the normal training data. As long as the scale of neural network is within a reasonable range, adVAE is robust to network structure.

至于神经网络学习能力,我们将层数从2调到6,将隐藏神经元数目从 1/3 dim(x) 调到 dim(x).如图9(e),V1-V5代表来各种参数组合.V1性能最差,因为网络容量不足以学习正常训练数据的足够信息,只要神经网络容量合理,adVAE 会对网络结构保持相对稳健.

Because adVAE needs an anomaly-free dataset, it is important to investigate how adVAE responds to contaminated  training  data.   Meanwhile,  we  also  choose three  semi supervised  anomaly  detection  methods  for comparison.  ADASYN [51] is used to synthesize new anomalous  samples.   The  results  of  Figure  9(f)  show that OCSVM is more robust with a contaminated dataset than the other three AE-based methods. This is because OCSVM  can  ignore  certain  noises  when  learning  the boundary of the training data, whereas AE-based models always try to reduce the reconstruction error for all training data. Moreover, a high contamination ratio will more  easily  disrupt  the  proposed  discrimination  loss,which  suggests  training  an  AE-based  anomaly  detection model with high-quality data (i.e., a clean or low-contamination-ratio dataset). In practice, normal data is easy to collect, and thus the contamination ratio usually remains at a low level.

由于 adVAE 需要一个纯正常的数据集,因此研究 脏数据集对 adVAE 的影响是必要的.同时,我们还选择来三个半监督异常检测方法作为比较,ADASYN[51] 被用来合成新的异常样本.图9(f)显示OCSVM比3个AE类的方法对脏数据更加鲁棒.这是由于OCSVM在学习训练数据边界时会忽略某些噪声,而AE类方法总是尝试最小化所有训练数据的重构误差.此外,脏数据更容易破坏判别目标损失,因此建议对于AE类的异常检测模型,应适用一个高质量的数据集.在实际中,正常数据容易收集,因此一般数据集污染率比较低.

## 5. Conclusion
In  this  paper,  we  have  proposed  a  self-adversarial Variational Auto-encoder  (adVAE)  with  a  Gaussian anomaly prior assumption and a self-adversarial mechanism. The proposed adVAE is encouraged to distinguish the normal latent code and the anomalous latent variables generated by the Gaussian transformer T, which can  also  be  regarded  as  an  outstanding  regularization introduced  into  VAE-based  outlier  detection  method.The results demonstrate that the proposed adVAE out-performs than other state-of-the-art anomaly detection methods.  In the future, we will try to redesign the discrimination objective of the generator to further enhance the generator’s ability to recognize anomalies.

本文提出来带有高斯先验假设和自对抗机制的自对抗变分自编码器(adVAE).adVAE 可以区分正常隐变量和由高斯变换网络T产生的异常隐变量,这也可以视为是对VAE类离群点检测方法引入的一个优秀的正则策略.结果显示了本文方法很牛逼.在后期工作中,我们将尝试重新设计生成器的判别目标来进一步增强生成器识别异常的能力.

## References
[1]  G.  Osada,  K.  Omote,  T.  Nishide,  Network  intrusion  detection based on semi-supervised variational auto-encoder, in: European  Symposium  on  Research  in  Computer  Security  (ES-ORICS), Springer, 2017, pp. 344–361 (2017).
[2]  A. Abdallah, M. A. Maarof, A. Zainal, Fraud detection system:A  survey,  Journal  of  Network  and  Computer  Applications  68(2016) 90–113 (2016).
[3]  P.  Cui,  C.  Zhan,  Y.  Yang,  Improved  nonlinear  process  monitoring  based  on  ensemble  kpca  with  local  structure  analysis,Chemical  Engineering  Research  and  Design  142  (2019)  355–368 (2019).
[4]  T.  Schlegl,  P.  Seeb ̈ock,  S.  M.  Waldstein,  U.  Schmidt-Erfurth,G.  Langs,  Unsupervised  anomaly  detection  with  generative adversarial  networks  to  guide  marker  discovery,  in:   International Conference on Information Processing in Medical Imaging (IPMI), Springer, 2017, pp. 146–157 (2017).
[5]  S. Akcay, A. A. Abarghouei, T. P. Breckon, Ganomaly:  Semi-supervised anomaly detection via adversarial training, in: Asian Conference on Computer Vision (ACCV), Springer, 2018, pp.622–637 (2018).
[6]  C. Zhang, J. Bi, S. Xu, E. Ramentol, G. Fan, B. Qiao, H. Fu-jita, Multi-imbalance:  An open-source software for multi-class imbalance  learning,  Knowl.-Based  Syst.  174  (2019)  137–143(2019).
[7]  F. Zhou,  S. Yang,  H. Fujita,  D. Chen,  C. Wen,  Deep learning fault diagnosis method based on global optimization gan for unbalanced data, Knowl.-Based Syst. (2019).
[8]  G. Lemaˆıtre,  F. Nogueira,  C. K. Aridas,  Imbalanced-learn:  A python toolbox to tackle the curse of imbalanced datasets in ma-chine learning, Journal of Machine Learning Research 18 (17)(2017) 1–5 (2017).
[9]  M. A. F. Pimentel, D. A. Clifton, L. A. Clifton, L. Tarassenko, A review of novelty detection, Signal Processing 99 (2014) 215–249 (2014).
[10]  R. Chalapathy, S. Chawla, Deep Learning for Anomaly Detection: A Survey, arXiv e-prints (2019) arXiv:1901.03407 (2019).
[11]  D. P. Kingma,  M. Welling,  Auto-Encoding Variational Bayes,in:International   Conference   on   Learning   Representations(ICLR), 2014 (2014).
[12]  I. J. Goodfellow, J. Pouget-Abadie, M. Mirza, B. Xu, D. Warde-Farley, S. Ozair, A. C. Courville, Y. Bengio, Generative adversarial nets, in:  Annual Conference on Neural Information Processing Systems (NeurIPS), MIT Press,  2014,  pp. 2672–2680(2014).    
[13]  J. An, S. Cho, Variational autoencoder based anomaly detection using reconstruction probability, Tech. rep., SNU Data Mining Center (2015).
[14]  D. Park, Y. Hoshi, C. C. Kemp, A multimodal anomaly detectorfor  robot-assisted  feeding  using  an  lstm-based  variational  au-toencoder, IEEE Robotics and Automation Letters 3 (3) (2018)1544–1551 (2018).
[15]  S. Suh, D. H. Chae, H. Kang, S. Choi, Echo-state conditionalvariational autoencoder for anomaly detection, in: InternationalJoint Conference on Neural Networks (IJCNN), IEEE, 2016, pp.1015–1022 (2016).
[16]  H. Xu, W. Chen, N. Zhao, Z. Li, J. Bu, Z. Li, Y. Liu, Y. Zhao,D.  Pei,  Y.  Feng,  J.  Chen,  Z.  Wang,  H.  Qiao,  Unsupervisedanomaly detection via variational auto-encoder for seasonal kpisin web applications, in: International World Wide Web Confer-ence (WWW), ACM, 2018, pp. 187–196 (2018).
[17]  A.  Makhzani,  J.  Shlens,  N.  Jaitly,  I.  Goodfellow,  Adversarialautoencoders, in:  International Conference on Learning Repre-sentations (ICLR), 2016 (2016).
[18]  S. Pidhorskyi, R. Almohsen, G. Doretto, Generative probabilis-tic  novelty  detection  with  adversarial  autoencoders,  in:   An-nual  Conference  on  Neural  Information  Processing  Systems(NeurIPS), MIT Press, 2018, pp. 6823–6834 (2018).
[19]  M. Ravanbakhsh, M. Nabi, E. Sangineto, L. Marcenaro, C. S.Regazzoni, N. Sebe, Abnormal event detection in videos usinggenerative adversarial nets, in: International Conference on Im-age Processing (ICIP), IEEE, 2017, pp. 1577–1581 (2017).
[20]  X.  Chen,  D.  P.  Kingma,  T.  Salimans,  Y.  Duan,  P.  Dhariwal,J. Schulman, I. Sutskever, P. Abbeel, Variational lossy autoen-coder, in: International Conference on Learning Representations(ICLR), 2017 (2017).
[21]  M. Fraccaro, S. K. Sønderby, U. Paquet, O. Winther, Sequentialneural models with stochastic layers, in: Annual Conference onNeural  Information  Processing  Systems  (NeurIPS),  2016,  pp.2199–2207 (2016).
[22]  I. V. Serban, A. Sordoni, R. Lowe, L. Charlin, J. Pineau, A. C.Courville,  Y.  Bengio,  A  hierarchical  latent  variable  encoder-decoder model for generating dialogues, in:  AAAI Conferenceon Artificial Intelligence (AAAI), 2017, pp. 3295–3301 (2017).
[23]  A. Honkela, H. Valpola, Variational learning and bits-back cod-ing:  an information-theoretic view to bayesian learning, IEEETransactions on Neural Networks 15 (4) (2004) 800–810 (2004).
[24]  Y. Liu, Z. Li, C. Zhou, Y. Jiang, J. Sun, M. Wang, X. He, Gener-ative adversarial active learning for unsupervised outlier detec-tion,  IEEE Transactions on Knowledge and Data Engineering(2019).
[25]  Y. Kawachi, Y. Koizumi, N. Harada, Complementary set varia-tional autoencoder for supervised anomaly detection, in:  Inter-national Conference on Acoustics, Speech and Signal Process-ing (ICASSP), IEEE, 2018, pp. 2366–2370 (2018).
[26]  H. Huang, Z. Li, R. He, Z. Sun, T. Tan, Introvae:  Introspectivevariational autoencoders for photographic image synthesis, in:Annual Conference on Neural Information Processing Systems(NeurIPS), MIT Press, 2018, pp. 52–63 (2018).
[27]  J. Ilonen, P. Paalanen, J. Kamarainen, H. K ̈alvi ̈ainen, Gaussianmixture  pdf  in  one-class  classification:  computing  and  utiliz-ing confidence values, in:  International Conference on PatternRecognition (ICPR), IEEE, 2006, pp. 577–580 (2006).
[28]  D.  Yeung,  C.  Chow,  Parzen-window  network  intrusion  de-tectors,  in:   International  Conference  on  Pattern  Recognition(ICPR), IEEE, 2002, pp. 385–388 (2002).[29]  M. M. Breunig, H. Kriegel, R. T. Ng, J. Sander, LOF: identifyingdensity-based local outliers,  in:  ACM SIGMOD InternationalConference on Management of Data (SIGMOD), ACM, 2000,pp. 93–104 (2000).
[30]  B. Tang, H. He, A local density-based approach for outlier de-tection, Neurocomputing 241 (2017) 171–180 (2017).
[31]  B. Sch ̈olkopf, J. C. Platt, J. Shawe-Taylor, A. J. Smola, R. C.Williamson, Estimating the support of a high-dimensional dis-tribution, Neural Computation 13 (7) (2001) 1443–1471 (2001).
[32]  L. Yin, H. Wang, W. Fan, Active learning based support vectordata description method for robust novelty detection,  Knowl.-Based Syst. 153 (2018) 40–52 (2018).
[33]  D. J. Olive, Principal Component Analysis, Springer, 2017, pp.189–217 (2017).
[34]  F. Harrou, F. Kadri, S. Chaabane, C. Tahon, Y. Sun, Improvedprincipal component analysis for anomaly detection:  Applica-tion to an emergency department, Computers & Industrial Engi-neering 88 (2015) 63–77 (2015).
[35]  R.  Baklouti,  M.  Mansouri,  M.  Nounou,  H.  Nounou,  A.  B.Hamida, Iterated robust kernel fuzzy principal component anal-ysis and application to fault detection, Journal of ComputationalScience 15 (2016) 34–49 (2016).
[36]  I.  Gulrajani,  F.  Ahmed,  M.  Arjovsky,  V.  Dumoulin,  A.  C.Courville,  Improved  training  of  wasserstein  gans,  in:    An-nual  Conference  on  Neural  Information  Processing  Systems(NeurIPS), 2017, pp. 5767–5777 (2017).
[37]  A. Gramacki, J. Gramacki, Fft-based fast bandwidth selector formultivariate kernel density estimation, Computational Statistics& Data Analysis 106 (2017) 27–45 (2017).
[38]  B. Zong, Q. Song, M. R. Min, W. Cheng, C. Lumezanu, D. Cho,H.  Chen,  Deep  autoencoding  gaussian  mixture  model  for  un-supervised anomaly detection, in:  International Conference onLearning Representations (ICLR), 2018 (2018).
[39]  B. W. Silverman, Density estimation for statistics and data anal-ysis, Routledge, 2018, Ch. 3 (2018).
[40]  S. Rayana, L. Akoglu, Less is more: Building selective anomalyensembles,  ACM  Trans.  Knowl.  Discov.  Data  10  (4)  (2016)42:1–42:33 (2016).
[41]  S. Sathe, C. C. Aggarwal, LODES: local density meets spectraloutlier detection,  in:  SIAM International Conference on DataMining (SDM), SIAM, 2016, pp. 171–179 (2016).
[42]  F. T. Liu, K. M. Ting, Z. Zhou, Isolation forest, in: InternationalConference on Data Mining (ICDM), IEEE, 2008, pp. 413–422(2008).
[43]  C.  C.  Aggarwal,  S.  Sathe,  Theoretical  foundations  and  algo-rithms  for  outlier  ensembles,  SIGKDD  Explor.  Newsl.  17  (1)(2015) 24–47 (2015).
[44]  F. Keller, E. M ̈uller, K. B ̈ohm, Hics: High contrast subspaces fordensity-based outlier ranking, in:  International Conference onData Engineering (ICDE), IEEE, 2012, pp. 1037–1048 (2012).
[45]  J.  Davis,  M.  Goadrich,  The  relationship  between  precision-recall and ROC curves, in: International Conference on MachineLearning (ICML), ACM, 2006, pp. 233–240 (2006).
[46]  F. T. Liu, K. M. Ting, Z. Zhou, Isolation forest, in: InternationalConference on Data Mining (ICDM), IEEE Computer Society,2008, pp. 413–422 (2008).
[47]  H. Kriegel, M. Schubert, A. Zimek, Angle-based outlier detec-tion in high-dimensional data, in:  ACM Knowledge Discoveryand Data Mining (KDD), ACM, 2008, pp. 444–452 (2008).
[48]  H. Kriegel, P. Kr ̈oger, E. Schubert, A. Zimek, Outlier detectionin axis-parallel subspaces of high dimensional data, in: Pacific-Asia  Conference  on  Knowledge  Discovery  and  Data  Mining(PAKDD),  Vol.  5476  of  Lecture  Notes  in  Computer  Science,Springer, 2009, pp. 831–838 (2009).
[49]  M. Goldstein, A. Dengel, Histogram-based outlier score (hbos):A fast unsupervised anomaly detection algorithm, German Con-ference on Artificial Intelligence (KI-2012):  Poster and DemoTrack (2012) 59–63 (2012).
[50]  Y. Zhao, Z. Nasrullah, Z. Li, Pyod:  A python toolbox for scalable  outlier  detection,  Journal  of  Machine  Learning  Research20 (96) (2019) 1–7 (2019).
[51]  H. He, Y. Bai, E. A. Garcia, S. Li, ADASYN: adaptive synthetic sampling  approach  for  imbalanced  learning,  in:  International Joint Conference on Neural Networks (IJCNN), IEEE, 2008, pp.1322–1328 (2008)