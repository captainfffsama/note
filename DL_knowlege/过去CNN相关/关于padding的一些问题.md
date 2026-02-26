#目标跟踪 

主要是看到 ICLR 2021 [Mind the pad](https://readpaper.com/paper/3123715274) 这篇文章

想起了过去看 SiamRPN++ 时提到的一些关于  padding的问题.    
SiamRPN++ [提到 padding 会引入位置偏见](https://zhuanlan.zhihu.com/p/56254712).而过去 SiamFC 的训练方式是:  
图A 进行 padding,使得目标在图片中间,图 B 进行以目标为中心进行剪裁,这样最终的监督label都是以中心为高斯分布的label.
参见:<https://blog.csdn.net/BearLeer/article/details/115050445>

这种方式显然是位置强相关的
