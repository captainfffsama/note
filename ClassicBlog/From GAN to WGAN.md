#GAN
#生成模型 
#待处理 

[原文链接](https://lilianweng.github.io/lil-log/2017/08/20/from-GAN-to-WGAN.html)

> This post explains the maths behind a generative adversarial network (GAN) model and why it is hard to be trained. Wasserstein GAN is intended to improve GANs’ training by adopting a smooth metric for measuring the distance between two probability distributions.

> 这篇文章解释了生成对抗网络(GAN)模型背后的数学原理以及为何难以训练。[[../DL_knowlege/Wasserstein]] GAN旨在通过采用平滑度量来测量两个概率分布之间的距离来改善GAN的训练。

[Generative adversarial network](https://arxiv.org/pdf/1406.2661.pdf) (GAN) has shown great results in many generative tasks to replicate the real-world rich content such as images, human language, and music. It is inspired by game theory: two models, a generator and a critic, are competing with each other while making each other stronger at the same time. However, it is rather challenging to train a GAN model, as people are facing issues like training instability or failure to converge.

生成对抗网络（GAN）

Here I would like to explain the maths behind the generative adversarial network framework, why it is hard to be trained, and finally introduce a modified version of GAN intended to solve the training difficulties.

Kullback–Leibler and Jensen–Shannon Divergence
----------------------------------------------

Before we start examining GANs closely, let us first review two metrics for quantifying the similarity between two probability distributions.

(1) [KL (Kullback–Leibler) divergence](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence) measures how one probability distribution p diverges from a second expected probability distribution q.

DKL(p‖q)=∫xp(x)log⁡p(x)q(x)dx

DKL achieves the minimum zero when p(x) == q(x) everywhere.

It is noticeable according to the formula that KL divergence is asymmetric. In cases where p(x) is close to zero, but q(x) is significantly non-zero, the q’s effect is disregarded. It could cause buggy results when we just want to measure the similarity between two equally important distributions.

(2) [Jensen–Shannon Divergence](https://en.wikipedia.org/wiki/Jensen%E2%80%93Shannon_divergence) is another measure of similarity between two probability distributions, bounded by \[0,1\]. JS divergence is symmetric (yay!) and more smooth. Check this [Quora post](https://www.quora.com/Why-isnt-the-Jensen-Shannon-divergence-used-more-often-than-the-Kullback-Leibler-since-JS-is-symmetric-thus-possibly-a-better-indicator-of-distance) if you are interested in reading more about the comparison between KL divergence and JS divergence.

DJS(p‖q)=12DKL(p‖p+q2)+12DKL(q‖p+q2)

![](https://lilianweng.github.io/lil-log/assets/images/KL_JS_divergence.png)

_Fig. 1. Given two Gaussian distribution, p with mean=0 and std=1 and q with mean=1 and std=1. The average of two distributions is labelled as m=(p+q)/2. KL divergence DKL is asymmetric but JS divergence DJS is symmetric._

Some believe ([Huszar, 2015](https://arxiv.org/pdf/1511.05101.pdf)) that one reason behind GANs’ big success is switching the loss function from asymmetric KL divergence in traditional maximum-likelihood approach to symmetric JS divergence. We will discuss more on this point in the next section.

Generative Adversarial Network (GAN)
------------------------------------

GAN consists of two models:

*   A discriminator D estimates the probability of a given sample coming from the real dataset. It works as a critic and is optimized to tell the fake samples from the real ones.
*   A generator G outputs synthetic samples given a noise variable input z (z brings in potential output diversity). It is trained to capture the real data distribution so that its generative samples can be as real as possible, or in other words, can trick the discriminator to offer a high probability.

![](https://lilianweng.github.io/lil-log/assets/images/GAN.png)

_Fig. 2. Architecture of a generative adversarial network. (Image source: [www.kdnuggets.com/2017/01/generative-…-learning.html](http://www.kdnuggets.com/2017/01/generative-adversarial-networks-hot-topic-machine-learning.html))_

These two models compete against each other during the training process: the generator G is trying hard to trick the discriminator, while the critic model D is trying hard not to be cheated. This interesting zero-sum game between two models motivates both to improve their functionalities.

Given,

| Symbol | Meaning | Notes |
| --- | --- | --- |
| pz | Data distribution over noise input z | Usually, just uniform. |
| pg | The generator’s distribution over data x |   |
| pr | Data distribution over real sample x |   |

On one hand, we want to make sure the discriminator D’s decisions over real data are accurate by maximizing Ex∼pr(x)\[log⁡D(x)\]. Meanwhile, given a fake sample G(z),z∼pz(z), the discriminator is expected to output a probability, D(G(z)), close to zero by maximizing Ez∼pz(z)\[log⁡(1−D(G(z)))\].

On the other hand, the generator is trained to increase the chances of D producing a high probability for a fake example, thus to minimize Ez∼pz(z)\[log⁡(1−D(G(z)))\].

When combining both aspects together, D and G are playing a **minimax game** in which we should optimize the following loss function:

minGmaxDL(D,G)=Ex∼pr(x)\[log⁡D(x)\]+Ez∼pz(z)\[log⁡(1−D(G(z)))\]=Ex∼pr(x)\[log⁡D(x)\]+Ex∼pg(x)\[log⁡(1−D(x)\]

(Ex∼pr(x)\[log⁡D(x)\] has no impact on G during gradient descent updates.)

### What is the optimal value for D?

Now we have a well-defined loss function. Let’s first examine what is the best value for D.

L(G,D)=∫x(pr(x)log⁡(D(x))+pg(x)log⁡(1−D(x)))dx

Since we are interested in what is the best value of D(x) to maximize L(G,D), let us label

x~=D(x),A=pr(x),B=pg(x)

And then what is inside the integral (we can safely ignore the integral because x is sampled over all the possible values) is:

f(x~)=Alogx~+Blog(1−x~)df(x~)dx~=A1ln101x~−B1ln1011−x~=1ln10(Ax~−B1−x~)=1ln10A−(A+B)x~x~(1−x~)

Thus, set df(x~)dx~=0, we get the best value of the discriminator: D∗(x)=x~∗=AA+B=pr(x)pr(x)+pg(x)∈\[0,1\].

Once the generator is trained to its optimal, pg gets very close to pr. When pg=pr, D∗(x) becomes 1/2.

### What is the global optimal?

When both G and D are at their optimal values, we have pg=pr and D∗(x)=1/2 and the loss function becomes:

L(G,D∗)=∫x(pr(x)log⁡(D∗(x))+pg(x)log⁡(1−D∗(x)))dx=log⁡12∫xpr(x)dx+log⁡12∫xpg(x)dx=−2log⁡2

### What does the loss function represent?

According to the formula listed in the [previous section](https://lilianweng.github.io/lil-log/2017/08/20/from-GAN-to-WGAN.html#kullbackleibler-and-jensenshannon-divergence), JS divergence between pr and pg can be computed as:

DJS(pr‖pg)=12DKL(pr||pr+pg2)+12DKL(pg||pr+pg2)=12(log⁡2+∫xpr(x)log⁡pr(x)pr+pg(x)dx)+12(log⁡2+∫xpg(x)log⁡pg(x)pr+pg(x)dx)=12(log⁡4+L(G,D∗))

Thus,

L(G,D∗)=2DJS(pr‖pg)−2log⁡2

Essentially the loss function of GAN quantifies the similarity between the generative data distribution pg and the real sample distribution pr by JS divergence when the discriminator is optimal. The best G∗ that replicates the real data distribution leads to the minimum L(G∗,D∗)=−2log⁡2 which is aligned with equations above.

> **Other Variations of GAN**: There are many variations of GANs in different contexts or designed for different tasks. For example, for semi-supervised learning, one idea is to update the discriminator to output real class labels, 1,…,K−1, as well as one fake class label K. The generator model aims to trick the discriminator to output a classification label smaller than K.

**Tensorflow Implementation**: [carpedm20/DCGAN-tensorflow](https://github.com/carpedm20/DCGAN-tensorflow)

Problems in GANs
----------------

Although GAN has shown great success in the realistic image generation, the training is not easy; The process is known to be slow and unstable.

### Hard to achieve Nash equilibrium

[Salimans et al. (2016)](http://papers.nips.cc/paper/6125-improved-techniques-for-training-gans.pdf) discussed the problem with GAN’s gradient-descent-based training procedure. Two models are trained simultaneously to find a [Nash equilibrium](https://en.wikipedia.org/wiki/Nash_equilibrium) to a two-player non-cooperative game. However, each model updates its cost independently with no respect to another player in the game. Updating the gradient of both models concurrently cannot guarantee a convergence.

Let’s check out a simple example to better understand why it is difficult to find a Nash equilibrium in an non-cooperative game. Suppose one player takes control of x to minimize f1(x)=xy, while at the same time the other player constantly updates y to minimize f2(y)=−xy.

Because ∂f1∂x=y and ∂f2∂y=−x, we update x with x−η⋅y and y with y+η⋅x simulitanously in one iteration, where η is the learning rate. Once x and y have different signs, every following gradient update causes huge oscillation and the instability gets worse in time, as shown in Fig. 3.

![](https://lilianweng.github.io/lil-log/assets/images/nash_equilibrium.png)

_Fig. 3. A simulation of our example for updating x to minimize xy and updating y to minimize −xy. The learning rate η=0.1. With more iterations, the oscillation grows more and more unstable._

### Low dimensional supports

| Term | Explanation |
| --- | --- |
| [Manifold](https://en.wikipedia.org/wiki/Manifold) | A topological space that locally resembles Euclidean space near each point. Precisely, when this Euclidean space is of **dimension n**, the manifold is referred as **n-manifold**. |
| [Support](https://en.wikipedia.org/wiki/Support_(mathematics)) | A real-valued function f is the subset of the domain containing those elements which are not mapped to **zero**. |

[Arjovsky and Bottou (2017)](https://arxiv.org/pdf/1701.04862.pdf) discussed the problem of the [supports](https://en.wikipedia.org/wiki/Support_(mathematics)) of pr and pg lying on low dimensional [manifolds](https://en.wikipedia.org/wiki/Manifold) and how it contributes to the instability of GAN training thoroughly in a very theoretical paper [“Towards principled methods for training generative adversarial networks”](https://arxiv.org/pdf/1701.04862.pdf).

The dimensions of many real-world datasets, as represented by pr, only appear to be **artificially high**. They have been found to concentrate in a lower dimensional manifold. This is actually the fundamental assumption for [Manifold Learning](http://scikit-learn.org/stable/modules/manifold.html). Thinking of the real world images, once the theme or the contained object is fixed, the images have a lot of restrictions to follow, i.e., a dog should have two ears and a tail, and a skyscraper should have a straight and tall body, etc. These restrictions keep images aways from the possibility of having a high-dimensional free form.

pg lies in a low dimensional manifolds, too. Whenever the generator is asked to a much larger image like 64x64 given a small dimension, such as 100, noise variable input z, the distribution of colors over these 4096 pixels has been defined by the small 100-dimension random number vector and can hardly fill up the whole high dimensional space.

Because both pg and pr rest in low dimensional manifolds, they are almost certainly gonna be disjoint (See Fig. 4). When they have disjoint supports, we are always capable of finding a perfect discriminator that separates real and fake samples 100% correctly. Check the [paper](https://arxiv.org/pdf/1701.04862.pdf) if you are curious about the proof.

![](https://lilianweng.github.io/lil-log/assets/images/low_dim_manifold.png)

_Fig. 4. Low dimensional manifolds in high dimension space can hardly have overlaps. (Left) Two lines in a three-dimension space. (Right) Two surfaces in a three-dimension space._

### Vanishing gradient

When the discriminator is perfect, we are guaranteed with D(x)=1,∀x∈pr and D(x)=0,∀x∈pg. Therefore the loss function L falls to zero and we end up with no gradient to update the loss during learning iterations. Fig. 5 demonstrates an experiment when the discriminator gets better, the gradient vanishes fast.

![](https://lilianweng.github.io/lil-log/assets/images/GAN_vanishing_gradient.png)

_Fig. 5. First, a DCGAN is trained for 1, 10 and 25 epochs. Then, with the **generator fixed**, a discriminator is trained from scratch and measure the gradients with the original cost function. We see the gradient norms **decay quickly** (in log scale), in the best case 5 orders of magnitude after 4000 discriminator iterations. (Image source: [Arjovsky and Bottou, 2017](https://arxiv.org/pdf/1701.04862.pdf))_

As a result, training a GAN faces a **dilemma**:

*   If the discriminator behaves badly, the generator does not have accurate feedback and the loss function cannot represent the reality.
*   If the discriminator does a great job, the gradient of the loss function drops down to close to zero and the learning becomes super slow or even jammed.

This dilemma clearly is capable to make the GAN training very tough.

### Mode collapse

During the training, the generator may collapse to a setting where it always produces same outputs. This is a common failure case for GANs, commonly referred to as **Mode Collapse**. Even though the generator might be able to trick the corresponding discriminator, it fails to learn to represent the complex real-world data distribution and gets stuck in a small space with extremely low variety.

![](https://lilianweng.github.io/lil-log/assets/images/mode_collapse.png)

_Fig. 6. A DCGAN model is trained with an MLP network with 4 layers, 512 units and ReLU activation function, configured to lack a strong inductive bias for image generation. The results shows a significant degree of mode collapse. (Image source: [Arjovsky, Chintala, & Bottou, 2017.](https://arxiv.org/pdf/1701.07875.pdf))_

### Lack of a proper evaluation metric

Generative adversarial networks are not born with a good objection function that can inform us the training progress. Without a good evaluation metric, it is like working in the dark. No good sign to tell when to stop; No good indicator to compare the performance of multiple models.

Improved GAN Training
---------------------

The following suggestions are proposed to help stabilize and improve the training of GANs.

First five methods are practical techniques to achieve faster convergence of GAN training, proposed in [“Improve Techniques for Training GANs”](http://papers.nips.cc/paper/6125-improved-techniques-for-training-gans.pdf). The last two are proposed in [“Towards principled methods for training generative adversarial networks”](https://arxiv.org/pdf/1701.04862.pdf) to solve the problem of disjoint distributions.

(1) **Feature Matching**

Feature matching suggests to optimize the discriminator to inspect whether the generator’s output matches expected statistics of the real samples. In such a scenario, the new loss function is defined as ‖Ex∼prf(x)−Ez∼pz(z)f(G(z))‖22, where f(x) can be any computation of statistics of features, such as mean or median.

(2) **Minibatch Discrimination**

With minibatch discrimination, the discriminator is able to digest the relationship between training data points in one batch, instead of processing each point independently.

In one minibatch, we approximate the closeness between every pair of samples, c(xi,xj), and get the overall summary of one data point by summing up how close it is to other samples in the same batch, o(xi)=∑jc(xi,xj). Then o(xi) is explicitly added to the input of the model.

(3) **Historical Averaging**

For both models, add ‖Θ−1t∑i=1tΘi‖2 into the loss function, where Θ is the model parameter and Θi is how the parameter is configured at the past training time i. This addition piece penalizes the training speed when Θ is changing too dramatically in time.

(4) **One-sided Label Smoothing**

When feeding the discriminator, instead of providing 1 and 0 labels, use soften values such as 0.9 and 0.1. It is shown to reduce the networks’ vulnerability.

(5) **Virtual Batch Normalization** (VBN)

Each data sample is normalized based on a fixed batch (_“reference batch”_) of data rather than within its minibatch. The reference batch is chosen once at the beginning and stays the same through the training.

**Theano Implementation**: [openai/improved-gan](https://github.com/openai/improved-gan)

(6) **Adding Noises**.

Based on the discussion in the [previous section](https://lilianweng.github.io/lil-log/2017/08/20/from-GAN-to-WGAN.html#low-dimensional-supports), we now know pr and pg are disjoint in a high dimensional space and it causes the problem of vanishing gradient. To artificially “spread out” the distribution and to create higher chances for two probability distributions to have overlaps, one solution is to add continuous noises onto the inputs of the discriminator D.

(7) **Use Better Metric of Distribution Similarity**

The loss function of the vanilla GAN measures the JS divergence between the distributions of pr and pg. This metric fails to provide a meaningful value when two distributions are disjoint.

[Wasserstein metric](https://en.wikipedia.org/wiki/Wasserstein_metric) is proposed to replace JS divergence because it has a much smoother value space. See more in the next section.

Wasserstein GAN (WGAN)
----------------------

### What is Wasserstein distance?

[Wasserstein Distance](https://en.wikipedia.org/wiki/Wasserstein_metric) is a measure of the distance between two probability distributions. It is also called **Earth Mover’s distance**, short for EM distance, because informally it can be interpreted as the minimum energy cost of moving and transforming a pile of dirt in the shape of one probability distribution to the shape of the other distribution. The cost is quantified by: the amount of dirt moved x the moving distance.

Let us first look at a simple case where the probability domain is _discrete_. For example, suppose we have two distributions P and Q, each has four piles of dirt and both have ten shovelfuls of dirt in total. The numbers of shovelfuls in each dirt pile are assigned as follows:

P1=3,P2=2,P3=1,P4=4Q1=1,Q2=2,Q3=4,Q4=3

In order to change P to look like Q, as illustrated in Fig. 7, we:

*   First move 2 shovelfuls from P1 to P2 =\> (P1,Q1) match up.
*   Then move 2 shovelfuls from P2 to P3 =\> (P2,Q2) match up.
*   Finally move 1 shovelfuls from Q3 to Q4 =\> (P3,Q3) and (P4,Q4) match up.

If we label the cost to pay to make Pi and Qi match as δi, we would have δi+1=δi+Pi−Qi and in the example:

δ0=0δ1=0+3−1=2δ2=2+2−2=2δ3=2+1−4=−1δ4=−1+4−3=0

Finally the Earth Mover’s distance is W=∑|δi|=5.

![](https://lilianweng.github.io/lil-log/assets/images/EM_distance_discrete.png)

_Fig. 7. Step-by-step plan of moving dirt between piles in P and Q to make them match._

When dealing with the continuous probability domain, the distance formula becomes:

W(pr,pg)=infγ∼Π(pr,pg)E(x,y)∼γ\[‖x−y‖\]

In the formula above, Π(pr,pg) is the set of all possible joint probability distributions between pr and pg. One joint distribution γ∈Π(pr,pg) describes one dirt transport plan, same as the discrete example above, but in the continuous probability space. Precisely γ(x,y) states the percentage of dirt should be transported from point x to y so as to make x follows the same probability distribution of y. That’s why the marginal distribution over x adds up to pg, ∑xγ(x,y)=pg(y) (Once we finish moving the planned amount of dirt from every possible x to the target y, we end up with exactly what y has according to pg.) and vice versa ∑yγ(x,y)=pr(x).

When treating x as the starting point and y as the destination, the total amount of dirt moved is γ(x,y) and the travelling distance is ‖x−y‖ and thus the cost is γ(x,y)⋅‖x−y‖. The expected cost averaged across all the (x,y) pairs can be easily computed as:

∑x,yγ(x,y)‖x−y‖=Ex,y∼γ‖x−y‖

Finally, we take the minimum one among the costs of all dirt moving solutions as the EM distance. In the definition of Wasserstein distance, the inf ([infimum](https://en.wikipedia.org/wiki/Infimum_and_supremum), also known as _greatest lower bound_) indicates that we are only interested in the smallest cost.

### Why Wasserstein is better than JS or KL divergence?

Even when two distributions are located in lower dimensional manifolds without overlaps, Wasserstein distance can still provide a meaningful and smooth representation of the distance in-between.

The WGAN paper exemplified the idea with a simple example.

Suppose we have two probability distributions, P and Q:

∀(x,y)∈P,x=0 and y∼U(0,1)∀(x,y)∈Q,x=θ,0≤θ≤1 and y∼U(0,1)

![](https://lilianweng.github.io/lil-log/assets/images/wasserstein_simple_example.png)

_Fig. 8. There is no overlap between P and Q when θ≠0._

When θ≠0:

DKL(P‖Q)=∑x=0,y∼U(0,1)1⋅log⁡10=+∞DKL(Q‖P)=∑x=θ,y∼U(0,1)1⋅log⁡10=+∞DJS(P,Q)=12(∑x=0,y∼U(0,1)1⋅log⁡11/2+∑x=0,y∼U(0,1)1⋅log⁡11/2)=log⁡2W(P,Q)=|θ|

But when θ=0, two distributions are fully overlapped:

DKL(P‖Q)=DKL(Q‖P)=DJS(P,Q)=0W(P,Q)=0=|θ|

DKL gives us inifity when two distributions are disjoint. The value of DJS has sudden jump, not differentiable at θ=0. Only Wasserstein metric provides a smooth measure, which is super helpful for a stable learning process using gradient descents.

### Use Wasserstein distance as GAN loss function

It is intractable to exhaust all the possible joint distributions in Π(pr,pg) to compute infγ∼Π(pr,pg). Thus the authors proposed a smart transformation of the formula based on the Kantorovich-Rubinstein duality to:

W(pr,pg)=1Ksup‖f‖L≤KEx∼pr\[f(x)\]−Ex∼pg\[f(x)\]

where sup ([supremum](https://en.wikipedia.org/wiki/Infimum_and_supremum)) is the opposite of inf (infimum); we want to measure the least upper bound or, in even simpler words, the maximum value.

**Lipschitz continuity?**

The function f in the new form of Wasserstein metric is demanded to satisfy ‖f‖L≤K, meaning it should be [K-Lipschitz continuous](https://en.wikipedia.org/wiki/Lipschitz_continuity).

A real-valued function f:R→R is called K-Lipschitz continuous if there exists a real constant K≥0 such that, for all x1,x2∈R,

|f(x1)−f(x2)|≤K|x1−x2|

Here K is known as a Lipschitz constant for function f(.). Functions that are everywhere continuously differentiable is Lipschitz continuous, because the derivative, estimated as |f(x1)−f(x2)||x1−x2|, has bounds. However, a Lipschitz continuous function may not be everywhere differentiable, such as f(x)=|x|.

Explaining how the transformation happens on the Wasserstein distance formula is worthy of a long post by itself, so I skip the details here. If you are interested in how to compute Wasserstein metric using linear programming, or how to transfer Wasserstein metric into its dual form according to the Kantorovich-Rubinstein Duality, read this [awesome post](https://vincentherrmann.github.io/blog/wasserstein/).

Suppose this function f comes from a family of K-Lipschitz continuous functions, {fw}w∈W, parameterized by w. In the modified Wasserstein-GAN, the “discriminator” model is used to learn w to find a good fw and the loss function is configured as measuring the Wasserstein distance between pr and pg.

L(pr,pg)=W(pr,pg)=maxw∈WEx∼pr\[fw(x)\]−Ez∼pr(z)\[fw(gθ(z))\]

Thus the “discriminator” is not a direct critic of telling the fake samples apart from the real ones anymore. Instead, it is trained to learn a K-Lipschitz continuous function to help compute Wasserstein distance. As the loss function decreases in the training, the Wasserstein distance gets smaller and the generator model’s output grows closer to the real data distribution.

One big problem is to maintain the K-Lipschitz continuity of fw during the training in order to make everything work out. The paper presents a simple but very practical trick: After every gradient update, clamp the weights w to a small window, such as \[−0.01,0.01\], resulting in a compact parameter space W and thus fw obtains its lower and upper bounds to preserve the Lipschitz continuity.

![](https://lilianweng.github.io/lil-log/assets/images/WGAN_algorithm.png)

_Fig. 9. Algorithm of Wasserstein generative adversarial network. (Image source: [Arjovsky, Chintala, & Bottou, 2017.](https://arxiv.org/pdf/1701.07875.pdf))_

Compared to the original GAN algorithm, the WGAN undertakes the following changes:

*   After every gradient update on the critic function, clamp the weights to a small fixed range, \[−c,c\].
*   Use a new loss function derived from the Wasserstein distance, no logarithm anymore. The “discriminator” model does not play as a direct critic but a helper for estimating the Wasserstein metric between real and generated data distribution.
*   Empirically the authors recommended [RMSProp](http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf) optimizer on the critic, rather than a momentum based optimizer such as [Adam](https://arxiv.org/abs/1412.6980v8) which could cause instability in the model training. I haven’t seen clear theoretical explanation on this point through.

* * *

Sadly, Wasserstein GAN is not perfect. Even the authors of the original WGAN paper mentioned that _“Weight clipping is a clearly terrible way to enforce a Lipschitz constraint”_ (Oops!). WGAN still suffers from unstable training, slow convergence after weight clipping (when clipping window is too large), and vanishing gradients (when clipping window is too small).

Some improvement, precisely replacing weight clipping with **gradient penalty**, has been discussed in [Gulrajani et al. 2017](https://arxiv.org/pdf/1704.00028.pdf). I will leave this to a future post.

Example: Create New Pokemons!
-----------------------------

Just for fun, I tried out [carpedm20/DCGAN-tensorflow](https://github.com/carpedm20/DCGAN-tensorflow) on a tiny dataset, [Pokemon sprites](https://github.com/PokeAPI/sprites/). The dataset only has 900-ish pokemon images, including different levels of same pokemon species.

Let’s check out what types of new pokemons the model is able to create. Unfortunately due to the tiny training data, the new pokemons only have rough shapes without details. The shapes and colors do look better with more training epoches! Hooray!

![](https://lilianweng.github.io/lil-log/assets/images/pokemon-GAN.png)

_Fig. 10. Train [carpedm20/DCGAN-tensorflow](https://github.com/carpedm20/DCGAN-tensorflow) on a set of Pokemon sprite images. The sample outputs are listed after training epoches = 7, 21, 49._

If you are interested in a commented version of [carpedm20/DCGAN-tensorflow](https://github.com/carpedm20/DCGAN-tensorflow) and how to modify it to train WGAN and WGAN with gradient penalty, check [lilianweng/unified-gan-tensorflow](https://github.com/lilianweng/unified-gan-tensorflow).

* * *

Cited as:

```
@article{weng2017gan,
  title   = "From GAN to WGAN",
  author  = "Weng, Lilian",
  journal = "lilianweng.github.io/lil-log",
  year    = "2017",
  url     = "http://lilianweng.github.io/lil-log/2017/08/20/from-GAN-to-WGAN.html"
} 
```

OR

```
@misc{weng2019gan,
    title={From GAN to WGAN},
    author={Lilian Weng},
    year={2019},
    eprint={1904.08994},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
} 
```

_If you notice mistakes and errors in this post, don’t hesitate to contact me at \[lilian dot wengweng at gmail dot com\] and I would be super happy to correct them right away!_

See you in the next post :D

References
----------

\[1\] Goodfellow, Ian, et al. [“Generative adversarial nets.”](https://arxiv.org/pdf/1406.2661.pdf) NIPS, 2014.

\[2\] Tim Salimans, et al. [“Improved techniques for training gans.”](http://papers.nips.cc/paper/6125-improved-techniques-for-training-gans.pdf) NIPS 2016.

\[3\] Martin Arjovsky and Léon Bottou. [“Towards principled methods for training generative adversarial networks.”](https://arxiv.org/pdf/1701.04862.pdf) arXiv preprint arXiv:1701.04862 (2017).

\[4\] Martin Arjovsky, Soumith Chintala, and Léon Bottou. [“Wasserstein GAN.”](https://arxiv.org/pdf/1701.07875.pdf) arXiv preprint arXiv:1701.07875 (2017).

\[5\] Ishaan Gulrajani, Faruk Ahmed, Martin Arjovsky, Vincent Dumoulin, Aaron Courville. [Improved training of wasserstein gans.](https://arxiv.org/pdf/1704.00028.pdf) arXiv preprint arXiv:1704.00028 (2017).

\[6\] [Computing the Earth Mover’s Distance under Transformations](http://robotics.stanford.edu/~scohen/research/emdg/emdg.html)

\[7\] [Wasserstein GAN and the Kantorovich-Rubinstein Duality](https://vincentherrmann.github.io/blog/wasserstein/)

\[8\] [zhuanlan.zhihu.com/p/25071913](https://zhuanlan.zhihu.com/p/25071913)

\[9\] Ferenc Huszár. [“How (not) to Train your Generative Model: Scheduled Sampling, Likelihood, Adversary?.”](https://arxiv.org/pdf/1511.05101.pdf) arXiv preprint arXiv:1511.05101 (2015).

