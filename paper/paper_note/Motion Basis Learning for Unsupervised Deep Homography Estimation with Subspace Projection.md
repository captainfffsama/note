#图像配准 

[toc]

# Motion Basis Learning for Unsupervised Deep Homography Estimation with Subspace Projection
- 文章: <https://readpaper.com/pdf-annotate/note?pdfId=4665270569806086145&noteId=735298692965552128>
- 代码: <https://github.com/megvii-research/BasesHomo>
- 会议: [ICCV 2021 Oral](../../Tag/ICCV.md)


## 摘要
本文引入了一个无监督深度单应性估计的新框架.我们的贡献包括3点.第一,不同于以往使用4个偏移来表示单应性,我们提出来单应性流表示法,即通过一个8个预定义的单应性基流的加权和来估计单应性流.第二,单应性仅包含8个自由度(DOFs),这个比网络特征的层级要少很多,我们提出来一个低等表示块(Low Rank Representation,LRR)来减少特征的层级,使得和主要运动对应的特征将被保留,而其他特征将被消除.最后,我们提出了特征鉴别损失(Feature Identity Loss,FIL)来强迫学习到的特征是扭曲不变的,这意味着就算交换扭曲操作和特征提取的顺序,其结果也是不变的.有了这个约束，可以更有效地实现无监督优化，并学习到更稳定的特征。进行了广泛的实验以证明所有新提出的组件的有效性，结果表明，我们的方法在定性和定量上都优于单应性基准数据集的最新技术.代码见: https://github.com/megvii-research/BasesHomo .   

