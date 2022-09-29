#优化理论

[原文](https://zhuanlan.zhihu.com/p/42383070)

> _很多问题最终归结为一个最小二乘问题，如SLAM算法中的Bundle Adjustment，位姿图优化等等。求解最小二乘的方法有很多，高斯-牛顿法就是其中之一。_

# 推导

对于一个非线性最小二乘问题：

$$
x =argmin_x \frac{1}{2}||f(x)||^2   \tag{1}
$$

高斯牛顿的思想是把 f(x) 利用泰勒展开，取一阶线性项近似。
$$
f(x+ \Delta x)=f(x)+f'(x)\Delta x=f(x)+J(x)\Delta x   \tag{2}
$$
带入到(1)式：
$$
\frac{1}{2}||f(x+\Delta x)||^2=\frac{1}{2}\{f(x)^Tf(x)+2f(x)^TJ(x)\Delta x+ \Delta x^TJ(x)^TJ(x) \Delta x\}  \tag{3}
$$


对上式求导，令导数为0。

$$
J(x)^TJ(x) \Delta x =-J(x)^Tf(x)  \tag{4}
$$


令 $H=J^TJ$ ，$B=-J^Tf$, 式（4）即为
$$
H \Delta x =B  \tag{5}
$$


求解式（5），便可以获得调整增量 $\Delta x$ 。这要求 $H$可逆（正定），但实际情况并不一定满足这个条件，因此可能发散，另外步长$\Delta x$可能太大，也会导致发散。

综上，高斯牛顿法的步骤为

> _STEP1. 给定初值 $x_0$_  
> _STEP2. 对于第k次迭代，计算 雅克比 J ， 矩阵H ， B ；根据（5）式计算增量 $\Delta x_k$;_ 
> _STEP3. 如果 $\Delta x_k$ 足够小，就停止迭代，否则，更新 $x_{k+1}=x_k+ \Delta x_k$ . 
> _STEP4. 循环执行STEP2. SPTE3，直到达到最大循环次数，或者满足STEP3的终止条件。_

