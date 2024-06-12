#薄板样条采样

[原文](https://blog.csdn.net/g11d111/article/details/128641313)

[toc]

# 0. 前言

本文的目标是详细分析一个经典的基于 [landmark](https://so.csdn.net/so/search?q=landmark&spm=1001.2101.3001.7020)(文章后面有时也称之为控制点 control point) 的图像 warping(扭曲/变形) 算法: Thin Plate Spine (TPS).

[TPS](https://so.csdn.net/so/search?q=TPS&spm=1001.2101.3001.7020) 被广泛的应用于各类的任务中, 尤其是生物形态中应用的更多: 人脸, 动物脸等等, TPS 是 cubic spline 的 2D 泛化形态. 值得注意的是, 图像处理中常用的**仿射变换 (Affine Transformation)**, 可以理解成 TPS 的一个特殊的变种.

* 什么是图像扭曲/变形问题?`[3]`  
    给定两张图片中一些相互对应的控制点 (landmark, 如图绿色连接线所示)，将 **图片 A(参考图像)** 进行特定的形变，使得其控制点可以与 **图片 B(目标模板)** 的 landmark 重合.

TPS 是其中较为经典的方法, 其基础假设是:  
`如果用一个薄钢板的形变来模拟这种2D形变, 在确保landmarks能够尽可能匹配的情况下，怎么样才能使得钢板的弯曲量(deflection)最小。`

* 用法示例  
    TPS 算法在我的实践中, 用法是: **根据图像的 landmark**(下图左黑色三角), 将 2D 图像**按照映射关系**(绿色连接线) 到的逻辑变形 (warping) 到**目标模板**(下图右).  
    ![](https://img-blog.csdnimg.cn/7206f5f7b1954861ab349e7f67bbcd00.png)
    

# 1. 理论

**Thin-Plate-Spline**, 本文剩余部分均用其简称**TPS**来替代. TPS 其实是一个数学概念 `[1]`:

TPS 是 1D cubic spline 的二维模拟, 它是 **双调和方程** (Biharmonic Equation) `[2]` 的基本解, 其形式如下:

$$
U(r)=r^2ln(r)
$$

## 1.1 U ( r ) 形式的由来

那么为什么形式是这样的呢? Bookstein`[10]` 在 1989 年发表的论文 **“Principle Warps: Thin-Plate Splines and the Decomposition of Deformation”** 中以双调和函数 (**Biharmonic Equation**) 的基础解进行展开:

首先, $r$ 表示的是 $\sqrt{x^2+y^2}$  ​(笛卡尔坐标系), 在论文中, Bookstein 用的是 $U(r) = -r^2 \ln(r)$, 其目的只是为了可视化方便: **In this pose, it appears to be a slightly dented but otherwise convex surface viewed from above**(即为了看起来中心 X 点附近的区域是一种 _凹陷 (dented)_ 的感觉).  
![](https://img-blog.csdnimg.cn/1f27d895f3ee48b799e60f28a89d3d1c.png)

  

这个函数天然的满足如下方程:

$$
\Delta^2U = (\frac{\partial ^{2}}{\partial x^{2}} + \frac{\partial ^{2}}{\partial y^{2}})^2 U \propto \delta_{(0,0)}
$$​

公式的左侧和(0,0)的泛函 $\delta_{(0,0)}$ 等价(泛函介绍如下), $\delta_{(0,0)}$ ​是在除了(0,0)处不等于0外, 任何其它位置都为0的泛函, 其积分为1(我猜, 狄拉克δ函数应该可以理解成这个泛函的一个形态).

所以, 由于双调和函数(**Biharmonic Equation**)的形式就是 $\Delta^2U=0$ , 那么显然, $U(r) = (\pm) r^2 \ln(r)$ 都满足这个条件, 所以它被称为双调和函数的**基础解(fundamental solution)**.

> 泛函简单来说, 就是**定义域为函数集，而值域为实数或者复数的映射**, 从**知乎**`[11]`处借鉴来一个泛函的例子：2D平面的两点之间直线距离最短.  
> ![](https://img-blog.csdnimg.cn/2e3101886f2c47cf8b135bb66b0adcb3.png)
>   
> 如图所示二维平面空间，从坐标原点(0,0)到点(a,b)的连接曲线是 $y = y(x)$ , 而连接曲线的微元 $\Delta$ 或者 $ds = \sqrt{1+(\frac{dy}{dx})^2dx}$ ​, 对总的长度, 即为 $ds$ 在  $[0, a]$ 上的积分:  
> $$
s = \int_{0}^{a}(1+y^{'2})^{1/2}dx
 $$ 这里, $s$ 是**标量(scalar)**, $y^{'}(x)$ 就是**泛函(functional)**, 通常也记作 $s(y^{'})$ . 那么上面的问题就转变成: 找出一条曲线 $y(x)$ ，使得泛函 $s(y^{'})$最小.

好的, $U$ 的来源和定义清楚了, 那么我们的目标是:

给定一组样本点，以每个样本点为中心的薄板样条(TPS)的加权组合给出了精确地通过这些点的插值函数，同时使所谓的**弯曲能量(bending energy)** 最小化.

那么, 什么是所谓的**弯曲能量**呢?

## 1.2 弯曲能量: Bending Energy

根据 `[1]`, 弯曲能量在这里定义为二阶导数的平方对实数域 $R^2$ (在我看来, 这里的 $R^2$可以**直接理解成2D image的Height and Width, 即高度和宽度**)的积分:

$$

I[f(x, y)] = \iint (f_{xx}^2 + 2f_{xy}^2+ f_{yy}^2)dxdy

$$
优化的目标是要让 $I[f(x, y)]$ 最小化.

好了, 弯曲能量的数学定义到此结束, 很自然的，我们会如下的疑问:

*   $f(x, y)$ 是如何定义的?
*   对图像这样的2D平面, 其样条的加权组合后的**弯曲的方向**应该是什么样的, 才能使得**弯曲能量**最小?

首先我们先分析下**弯曲的方向**的问题, 并在**1.4**中进行 $f(x, y)$ 定义的介绍.

## 1.3 弯曲的方向

首先, 回顾一下TPS的命名, TPS起源于一个**物理的类比**: _the bending of a thin sheet of metal_ (薄金属片的弯曲).

在物理学上来讲, 弯曲的方向(deflection)是 $z$ 轴, 即垂直于2D图像平面的轴.  
为了将这个idea应用于坐标转换的实际问题当中, 我们将TPS理解成是**将平板进行拉升 or 降低, 再将拉升/降低后的平面投影到2D图像平面**, 即得到根据参考图像和目标模板的landmark对应关系进行warping(形变)后的图像结果.

如下所示, 将平面上设置4个控制点, 其中**最后一个不是边缘角点**, 在做拉扯的时候, 平面就自然产生了一种局部被拉高或者降低的效果.  
![](https://img-blog.csdnimg.cn/2ffb17570f604702b35fc41072205120.gif)

显然, 这种warping在一定程度上也是一种**坐标转换(coordinate transformation)**, 如下图所示, 给定参考landmark红色 X 和目标点蓝色 ⚪ . TPS warping将会将这些 X X X完美的移动到 ⚪上.

![](https://img-blog.csdnimg.cn/051098c055d0497a89f45f31b38ba961.png)

问题来了, 那么这个 X→⚪移动的方案是如何实现的呢?

## 1.4 如何实现2D plane的coordinate transformation (a.k.a warping)?

如下图 `[7]`, 2D plane上的坐标变换其实就是2个方向的变化: $\mathbf{X}$  和 $\mathbf{Y}$方向. 来实现这2个方向的变化, TPS的做法是:

**用2个样条函数分别考虑 $\mathbf{X}$ 和 $\mathbf{Y}$ 方向上的位移(displacement)**.

```
TPS actually use two splines, 
one for the displacement in the X direction 
and one for the displacement in the Y direction` 
```

![](https://img-blog.csdnimg.cn/b935737df2b5438dad4209d48347a2f8.png)
  
这2个样条函数的定义如下 `[7]` ( $N$ 指的是对应的landmark数量, 如上图所示, $N=5$ ):  
$$

f_{(x')}(x,y)=a_1+a_xx+a_yy+\sum_{i=1}^N{w_iU(||(x_i,y_i)-(x,y)||)}

$$
$$

f_{(y')}(x,y)=a_1+a_xx+a_yy+\sum_{i=1}^N{w_iU(||(x_i,y_i)-(x,y)||)}

$$

注意, 每个方向 $(\mathbf{X}, \mathbf{Y})$ 的位移( $\mathbf{\Delta X}, \mathbf{\Delta Y})$ 可以被视为 $N$ 个点**高度图(height map)**, 因此样条的就像在3D空间拟合 **散点(scatter point)** 一样, 如下图所示 `[7]`.  
![](https://img-blog.csdnimg.cn/fbe3145f46ff401fad670493d0bd864b.png)
  
在样条函数的定义公式中,

*   前3个系数 $a_1, a_x, a_y$ 表示线性空间的部分(line part), 用于在线性空间拟合 $X ( x_i, y_i​)$ 和 $⚪ (x_i^{'}, y_i^{'}​)$ .
*   紧接着的系数 $w_i, i \in [1, N]$ 表示每个控制点 $i$ 的**kernel weight**, 它用于乘以控制点 $X(x_i, y_i​)$ 和其最终的 $x, y$之间的**位移**(displacement).
*   最后的一项是 $U(|| (x_i, y_i) - (x, y) ||)$ , 即控制点 $X(x_i, y_i)$ 和其最终的 $x, y$ 之间的**位移**. 需要注意的是, $U(|| (x_i, y_i) - (x, y) ||)$ 用的是L2范数 `[8]`. 这里 $U$ 定义如下: $U(r) = r^2 \ln(r)$ .
	* 这里我们需要revisit一下TPS的**RBF函数(radial basis function)** : $U(r) = r^2 \ln(r)$ , 根据 `[9]` 所述, 像RBF这种Gaussian Kernel, 是一种用于**衡量相似性的方法(Similarity measurement)**.

## 1.5 具体计算方案

对于每个方向 $(\mathbf{X}, \mathbf{Y})$ 的样条函数的系数 $a_1, a_x, a_y, w_i$ 可以通过求解如下linear system来获得:  
![](https://img-blog.csdnimg.cn/769d80fa626b48b9877f73b15acc01a4.png)
  
其中, $K_{ij} = U(|| (x_i, y_i) - (x_j, y_j) ||)$ , $P$ 的第 $i$ 行是齐次表示 $(1, x_i, y_i)$ , $O$ 是3x3的全0矩阵, $o$ 是3x1的全0列向量, $w$ 和 $v$ 是 $w_i$ ​和 $v_i$ ​组成的列向量. $a$ 是由 $[a_1, a_x, a_y]$组成的列向量.

具体地, 左侧的大矩阵形式如下`[9-10]`:  
![](https://img-blog.csdnimg.cn/2abaedf8376648f8970892ced63e81ea.png)
  
以 $N=3$ (控制点数量为3)为例, $\mathbf{X}$ 方向的样条函数的线性矩阵表达为:  
$$

\begin{bmatrix} 

U_{11} & U_{21} & U_{31} & 1 & x_1 & y_1 \\ 

U_{12} & U_{22} & U_{32} & 1 & x_2 & y_2 \\ 

U_{13} & U_{23} & U_{33} & 1 & x_3 & y_3 \\ 

1 & 1 & 1 & 0 & 0 & 0 \\ 

x_1 & x_2 & x_3 & 0 & 0 & 0 \\ 

y_1 & y_2 & y_3 & 0 & 0 & 0 \end{bmatrix} 

\times

\begin{bmatrix} 

w_1 \\ w_2 \\ w_3 \\ a_1 \\ a_x \\ a_y \end{bmatrix} = 

\begin{bmatrix} 

x'_1 \\ x'_2 \\ x'_3 \\ 0 \\ 0 \\ 0 \end{bmatrix}

$$


同样地, $\mathbf{Y}$ 的样条函数的线性矩阵表达为:
$$

\begin{bmatrix} 

U_{11} & U_{21} & U_{31} & 1 & x_1 & y_1 \\ 

U_{12} & U_{22} & U_{32} & 1 & x_2 & y_2 \\ 

U_{13} & U_{23} & U_{33} & 1 & x_3 & y_3 \\ 

1 & 1 & 1 & 0 & 0 & 0 \\ 

x_1 & x_2 & x_3 & 0 & 0 & 0 \\ 

y_1 & y_2 & y_3 & 0 & 0 & 0 \end{bmatrix} 

\times

\begin{bmatrix} 

w_1 \\ w_2 \\ w_3 \\ a_1 \\ a_x \\ a_y \end{bmatrix} = 

\begin{bmatrix} 

y'_1 \\ y'_2 \\ y'_3 \\ 0 \\ 0 \\ 0 \end{bmatrix}

$$


显然可见, N+3个函数来求解N+3个未知量, 能得到相应的 $\begin{bmatrix} w \\ a \end{bmatrix}$


# 2. 代码实现

我使用的TPS是cheind/py-thin-plate-spline项目`[6]`, 这里会对代码进行详细拆解, 以达到理解公式和实现的对应关系.

## 2.1 核心计算逻辑

核心逻辑在函数`warp_image_cv`中: `tps.tps_theta_from_points`, `tps.tps_grid`和`tps.tps_grid_to_remap`,  
最基本的示例代码如下:

```python
def show_warped(img, warped, c_src, c_dst):
    fig, axs = plt.subplots(1, 2, figsize=(16,8))
    axs[0].axis('off')
    axs[1].axis('off')
    axs[0].imshow(img[...,::-1], origin='upper')
    axs[0].scatter(c_src[:, 0]*img.shape[1], c_src[:, 1]*img.shape[0], marker='^', color='black')
    axs[1].imshow(warped[...,::-1], origin='upper')
    axs[1].scatter(c_dst[:, 0]*warped.shape[1], c_dst[:, 1]*warped.shape[0], marker='^', color='black')
    plt.show()

def warp_image_cv(img, c_src, c_dst, dshape=None):
    dshape = dshape or img.shape
    theta = tps.tps_theta_from_points(c_src, c_dst, reduced=True)
    grid = tps.tps_grid(theta, c_dst, dshape)
    mapx, mapy = tps.tps_grid_to_remap(grid, img.shape)
    return cv2.remap(img, mapx, mapy, cv2.INTER_CUBIC)

img = cv2.imread('test.jpg')

c_src = np.array([
    [0.44, 0.18],
    [0.55, 0.18],
    [0.33, 0.23],
    [0.66, 0.23],
    [0.32, 0.79],
    [0.67, 0.80],
])

c_dst = np.array([
    [0.693, 0.466],
    [0.808, 0.466],
    [0.572, 0.524],
    [0.923, 0.524],
    [0.545, 0.965],
    [0.954, 0.966],
])

warped_front = warp_image_cv(img, c_src, c_dst, dshape=(512, 512))
show_warped(img, warped1, c_src_front, c_dst_front)
```

![](https://img-blog.csdnimg.cn/6ec065bf67324d03a689e0156a7b023f.png)
  
此开源代码有2个版本: numpy和torch. 这里我的分析以numpy版本进行, 以便没有GPU用的朋友进行hands-on的测试.

> 核心类TPS
 
```python
class TPS:
   @staticmethod
   def fit(c, lambd=0., reduced=False):
        n = c.shape[0]
        
        U = TPS.u(TPS.d(c, c))
        K = U + np.eye(n, dtype=np.float32)*lambd
        
        P = np.ones((n, 3), dtype=np.float32)
        P[:, 1:] = c[:, :2]
        v = np.zeros(n+3, dtype=np.float32)
        v[:n] = c[:, -1]

        A = np.zeros((n+3, n+3), dtype=np.float32)
        A[:n, :n] = K
        A[:n, -3:] = P
        A[-3:, :n] = P.T
        theta = np.linalg.solve(A, v) # p has structure w,a
        return theta[1:] if reduced else thete
        ...
        
   @staticmethod
   def z(x, c, theta):
        x = np.atleast_2d(x)
        U = TPS.u(TPS.d(x, c))
        w, a = theta[:-3], theta[-3:]
        reduced = theta.shape[0] == c.shape[0] + 2
        if reduced:
            w = np.concatenate((-np.sum(w, keepdims=True), w))
        b = np.dot(U, w)
        return a[0] + a[1]*x[:, 0] + a[2]*x[:, 1] + b
```

#### 2.2 `tps.tps_theta_from_points`

此函数的作用是为了求解样条函数的 $\begin{bmatrix} w \\ a \end{bmatrix}$
![](https://img-blog.csdnimg.cn/769d80fa626b48b9877f73b15acc01a4.png)

```python
def tps_theta_from_points(c_src, c_dst, reduced=False):
    delta = c_src - c_dst
    
    cx = np.column_stack((c_dst, delta[:, 0]))
    cy = np.column_stack((c_dst, delta[:, 1]))
        
    theta_dx = TPS.fit(cx, reduced=reduced)
    theta_dy = TPS.fit(cy, reduced=reduced)

    return np.stack((theta_dx, theta_dy), -1)
```

1.  **delta** 是在参考图的控制点和目标模板的控制点之间的插值 $\Delta x_i, \Delta y_i$ 
    
2.  **cx**和**cy**是在 `c_dst` 的基础上, 分别加了 $\Delta x_i$ ​和 $\Delta y_i$ ​的列向量
    
3.  **theta_dx**和**theta_dy**的reduce参数默认为False/True时. 其结果是1D向量, **长度为9/8** . 其计算过程需要看TPS核心类的`fit`函数.
    

① `TPS.d(cx, cx, reduced=True)` or `TPS.d(cy, cy, reduced=True)` **计算L2**
```python
@staticmethod
def d(a, b):
    # a[:, None, :2] 是把a变成[N, 1, 2]的tensor/ndarray
    # a[None, :, :2] 是把a变成[1, N, 2]的tensor/ndarray
    return np.sqrt(np.square(a[:, None, :2] - b[None, :, :2]).sum(-1))
```

其作用是计算样条中的 $|| (x_i, y_i) - (x, y) ||$ (**L2**), 得出的结果是shape为 $N, N$的中间结果.  
![](https://img-blog.csdnimg.cn/6e4d8d26fe3b4d5fba4e841fff76420b.png)

② `TPS.u(...)` **计算 $U(...)$ **

和公式完全一样: $U(r) = r^2 \ln(r)$ , 为了防止 $r$ 太小, 加了个epsilon系数 $1e^{-6}$ . 这一步得到 $K$ , shape仍 $N, N$和①一样.  
![](https://img-blog.csdnimg.cn/7e0ccc174fca4a39843a6f3ae2585f4e.png)

```python
def u(r):
    return r**2 * np.log(r + 1e-6)
```

③ 根据`cx`和`cy`, 简单拼接即可生成`P`.  
![](https://img-blog.csdnimg.cn/ff339888c0b940029843961a99e668eb.png)

```python
P = np.ones((n, 3), dtype=np.float32)
P[:, 1:] = c[:, :2] # c就是cx or cy.
```

④ 根据 $\Delta x_i$ ​ (`cx` 得最后一列向量, `cy` 同理), 得到 $v$  
![](https://img-blog.csdnimg.cn/846225e2a2a34ca1a7c748f12a411d1c.png)

```python
# c = cx or cy
v = np.zeros(n+3, dtype=np.float32)
v[:n] = c[:, -1]
```

⑤ 组装矩阵 `A`, 即 `[10]` 论文中的 $L$ 矩阵.  
![](https://img-blog.csdnimg.cn/5867a6118a2e4db585caac44292af0dd.png)

```python
A = np.zeros((n+3, n+3), dtype=np.float32)
A[:n, :n] = K
A[:n, -3:] = P
A[-3:, :n] = P.T
```

⑥ 现在 $L$ 和 $Y$ 已知, $Y=\begin{bmatrix} v \\ o \end{bmatrix}$ , 那么 $W$ 和 $a_1, a_x, a_y$​的向量可以直接线性求解  
![](https://img-blog.csdnimg.cn/a811496f6c8346868665cdfee51d7c56.png)
$\begin{bmatrix} w \\ a \end{bmatrix} = L^{-1}Y$

```python
class TPS:       
    @staticmethod
    def fit(c, lambd=0., reduced=False):
        # 1. TPS.d
        U = TPS.u(TPS.d(c, c))
        K = U + np.eye(n, dtype=np.float32)*lambd

        P = np.ones((n, 3), dtype=np.float32)
        P[:, 1:] = c[:, :2]

        v = np.zeros(n+3, dtype=np.float32)
        v[:n] = c[:, -1]

        A = np.zeros((n+3, n+3), dtype=np.float32)
        A[:n, :n] = K
        A[:n, -3:] = P
        A[-3:, :n] = P.T

        theta = np.linalg.solve(A, v) # p has structure w,a
        return theta[1:] if reduced else theta
        
    @staticmethod
    def d(a, b):
        return np.sqrt(np.square(a[:, None, :2] - b[None, :, :2]).sum(-1))

    @staticmethod
    def u(r):
        return r**2 * np.log(r + 1e-6)
```

![](https://img-blog.csdnimg.cn/67870e3bfaa14f22bb364f2dcba92d4e.png)

即函数返回的 `theta` 就是$\begin{bmatrix} w \\ a \end{bmatrix}$. 由于我们是2个方向(X, Y)都要这个 `theta`, 因此

```python
theta = tps.tps_theta_from_points(c_src, c_dst)
```

返回的theta是 $(N+3, 2)$的形式.

#### 2.3 `tps.tps_grid`

此函数是为了求解image plane在x和y方向上的偏移量(offset).

```python
def warp_image_cv(img, c_src, c_dst, dshape=None):
    dshape = dshape or img.shape
    # 2.2
    theta = tps.tps_theta_from_points(c_src, c_dst, reduced=True)
    # 2.3
    grid = tps.tps_grid(theta, c_dst, dshape)
    # 2.4
    mapx, mapy = tps.tps_grid_to_remap(grid, img.shape)
    return cv2.remap(img, mapx, mapy, cv2.INTER_CUBIC)
```

由核心代码部分可以看出, 当求出 `theta`, 也就是$\begin{bmatrix} w \\ a \end{bmatrix}$. 我们下面用`tps_grid`函数进行网格的warping操作.

函数如下:

```python
def tps_grid(theta, c_dst, dshape):
    # 1) uniform_grid(…)    
    ugrid = uniform_grid(dshape)

    reduced = c_dst.shape[0] + 2 == theta.shape[0]
    # 2) 求dx和dy.
    dx = TPS.z(ugrid.reshape((-1, 2)), c_dst, theta[:, 0]).reshape(dshape[:2])
    dy = TPS.z(ugrid.reshape((-1, 2)), c_dst, theta[:, 1]).reshape(dshape[:2])
    dgrid = np.stack((dx, dy), -1)

    grid = dgrid + ugrid
    
    return grid # H'xW'x2 grid[i,j] in range [0..1]
```

其输入是3个参数:

*   **theta** `reduced=True` (N+2, 2) or `reduced=False` (N+3, 2)
*   **c_dst** (N, 2), 是**目标模板**上的control points or landmarks.

```python
c_dst = np.array([
    [0.693, 0.466],
    [0.808, 0.466],
    [0.572, 0.524],
    [0.923, 0.524],
    [0.545, 0.965],
    [0.954, 0.966],
])
```

*   **dshape** (H, W, 3), 是给定参考图像的分辨率.

输出是1个:

*   **grid** (H, W, 2).  
    其可视化效果见`2.3.1`.

##### 2.3.1 `uniform_grid`

`tps.tps_grid` 函数的第一步是**ugrid = uniform_grid(dshape)**, 此函数的定义如下, 作用是创建1个 $(H, W, 2)$ 的grid, 里面的值都是0到1的线性插值 `np.linspace(0, 1, W(H))`.

```python
def uniform_grid(shape):
    '''Uniform grid coordinates.
    '''

    H,W = shape[:2]    
    c = np.empty((H, W, 2))
    c[…, 0] = np.linspace(0, 1, W, dtype=np.float32)
    c[…, 1] = np.expand_dims(np.linspace(0, 1, H, dtype=np.float32), -1)

    return c
```

返回的 `ugrid` 就是一个 $(H, W, 2)$的grid, 其X, Y方向值的大小按方向线性展开, 如下图所示.

> X方向  
> ![](https://img-blog.csdnimg.cn/a9e526bd340d4881bd52071306189504.png)
>   
> Y方向  
> ![](https://img-blog.csdnimg.cn/6d90f0c105114ede8e78fbe8cd4b7e6e.png)

##### 2.3.2 `TPS.z`求解得到`dx`和`dy`

```python
# 2) 求dx和dy.
dx = TPS.z(ugrid.reshape((-1, 2)), c_dst, theta[:, 0]).reshape(dshape[:2]) # [H, W]
dy = TPS.z(ugrid.reshape((-1, 2)), c_dst, theta[:, 1]).reshape(dshape[:2]) # [H, W]
dgrid = np.stack((dx, dy), -1)  # [H, W, 2]

grid = dgrid + ugrid
```

由下面的`TPS.z`定义容易看出, 这个函数就是求解X和Y方向的**样条函数**:

$$

f_{(x/y)^{'}}(x, y) = a_1 + a_x x + a_y y + \sum_{i=1}^{N} w_i U(|| (x_i, y_i) - (x, y) ||)

$$

可能让人有困惑的点是说, **为什么在`2.2`的时候, `TPS.d()`的传参是一样的(`cx(cy)`), 而这里的x是shape为`(H*W), 2`, 而`c`仍旧是`c_dst (N,2)`**, 我的理解是说, 由于`2.3`这一步的目标是为了真正的让image plane按照控制点的位置进行移动(**最小化弯曲能量**), 所以通过`ugrid`均匀对平面采样的点进行offset计算(`dx`和`dy`), 使其得到满足推导条件下的offset解析解`dgrid`.

```python
class TPS:
    …
    @staticmethod
    def z(x, c, theta):
        x = np.atleast_2d(x)
        U = TPS.u(TPS.d(x, c)) # [H*W, N] 本例中H=W=800, N=6
        w, a = theta[:-3], theta[-3:]
        reduced = theta.shape[0] == c.shape[0] + 2
        if reduced:
            w = np.concatenate((-np.sum(w, keepdims=True), w))
        b = np.dot(U, w)
        return a[0] + a[1]*x[:, 0] + a[2]*x[:, 1] + b
```

所以对 `ugrid` + `dgrid`, 即得到整个图像平面按照样条函数计算出来的 $dx,dy(offset)$ 加到均匀的 `ugrid` 的结果: 显然可以看出, 这个结果相比 `2.3.1` 的 `ugrid`, 在 $\mathbf{X}, \mathbf{Y}$ 方向有了相应的变化.

> X方向  
> ![](https://img-blog.csdnimg.cn/0e091cacc757417db77ace74c0336314.png)
>   
> ![](https://img-blog.csdnimg.cn/d099b95b075f44d993dc2ba2da011512.png)
>   
> Y方向  
> ![](https://img-blog.csdnimg.cn/270376e80a764bb194f1bf6fb6ce8276.png)
> ![](https://img-blog.csdnimg.cn/daaf1e315b5d4cff82ae5840e9cbd716.png)

到这里, `2.3` 这步返回的其实就是一个在 $\mathbf{X}, \mathbf{Y}$ 方向相应扭曲的**grid(格子)** $(H,W,2)$ , 其可视化结果如上, 值的范围都在 **-1到1** 之间.

#### 2.4 `tps.tps_grid_to_remap`

这一步很简单了, 就是把 `2.3` 计算得到的\*\*grid(格子)\*\*按 $\mathbf{X}, \mathbf{Y}$ 方向分别乘以对应的 $W$  和 $H$ . 然后送去 `cv2.remap` 函数进行图像的扭曲操作.

```python
def warp_image_cv(img, c_src, c_dst, dshape=None):
    dshape = dshape or img.shape
    # 2.2
    theta = tps.tps_theta_from_points(c_src, c_dst, reduced=True)
    # 2.3
    grid = tps.tps_grid(theta, c_dst, dshape)
    # 2.4
    mapx, mapy = tps.tps_grid_to_remap(grid, img.shape)
    return cv2.remap(img, mapx, mapy, cv2.INTER_CUBIC)
```

##### 2.4.1 `tps_grid_to_remap` 简单的把grid乘以宽和高

```python
def tps_grid_to_remap(grid, sshape):
    '''Convert a dense grid to OpenCV's remap compatible maps.
    Returns
    -------
    mapx : HxW array
    mapy : HxW array
    '''
    mx = (grid[:, :, 0] * sshape[1]).astype(np.float32)
    my = (grid[:, :, 1] * sshape[0]).astype(np.float32)

    return mx, my
```

![](https://img-blog.csdnimg.cn/15b7990e18854f029684ddff9685f6c8.png)

##### 2.4.2 `cv2.remap(img, mapx, mapy, cv2.INTER_CUBIC)` 得到warp后的结果.

`cv2.remap`是允许用户自己定义**映射关系**的函数, 不同于通过变换矩阵进行的**仿射变换**和**透视变换**, 更加的灵活, `TPS`就是使用的这种映射. 具体示例参考`[12]`.

![](https://img-blog.csdnimg.cn/9473ab13803d486f8cf510afff333fd0.png)

需要注意的是, 这个结果之所以和前言中的不一样, 是因为在前言里, 我们用了mask来做遮罩.

# 总结

到这里, `TPS`的分析就告一段落了, 这种算法是瘦脸, 纹理映射等任务中最常见的, 也是很灵活的warping算法, 目前还仍然在广泛使用, 如果文章哪里写的有谬误或者问题, 欢迎大家在下面指出,  
感谢 ^ . ^

# 参考文献

1.  [Thin Plate Spline: MathWorld](https://mathworld.wolfram.com/ThinPlateSpline.html)
2.  [Biharmonic Equation: MathWorld](https://mathworld.wolfram.com/BiharmonicEquation.html)
3.  [c0ldHEart: Thin Plate Spline TPS薄板样条变换基础理解](https://blog.csdn.net/c0ldHEart/article/details/121336266)
4.  [MIT: WarpMorph](http://groups.csail.mit.edu/graphics/classes/CompPhoto06/html/lecturenotes/14_WarpMorph.pdf)
5.  [Approximation Methods for Thin Plate Spline Mappings and Principal Warps](https://escholarship.org/uc/item/00n325f2)
6.  [cheind/py-thin-plate-spline](https://github.com/cheind/py-thin-plate-spline)
7.  [Thin-Plate-Splines-Warping](https://khanhha.github.io/posts/Thin-Plate-Splines-Warping/)
8.  [Wikipedia: Thin plate spline](https://en.wikipedia.org/wiki/Thin_plate_spline)
9.  [Deep Shallownet: Radial Basis Function Kernel - Gaussian Kernel](https://www.youtube.com/watch?v=I8r0cJIpeA0)
10.  [Bookstein: Principle Warps: Thin Plate Splines and the Decomposition of Deformations](http://user.engineering.uiowa.edu/~aip/papers/bookstein-89.pdf)
11.  [知乎:「泛函」究竟是什么意思？](https://www.zhihu.com/question/21938224)
12.  [【opencv】5.5 几何变换-- 重映射 cv2.remap()](https://blog.csdn.net/weixin_37804469/article/details/112316884)