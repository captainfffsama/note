出处参见 Attention is all you need.

按照[参考1](https://www.zhihu.com/question/347678607/answer/864217252)说法,由于 Transformer 结构中没有什么结构上机制来利用位置之间的关系.为此需要对位置进行编码,通过嵌入编码的方式来使用这种信息.

那么要求上:
1. 编码的数值范围是有界的
2. 编码可以体现一定范围内的字句先后顺序,且编码差异和文本长度无关.即若是简单的按照文本长度对编码数值进行归一化,那么在数值上,间隔一个字的短文本和间隔数个字的长文本编码数值上相等,这无法在一定文本范围内保证编码的效果相似.

按照上诉要求,一个可行的方式是使用有界的周期函数来做编码.

原文公式为:
$$
PE_{(pos,2i)}=sin(\frac{pos}{10000^{2i/d_{model}}})
$$
$$
PE_{(pos,2i+1)}=cos(\frac{pos}{10000^{2i/d_{model}}})
$$

即奇数位置使用 cos 函数,偶数位置使用 sin 函数产生编码.$d_{model}$ 为维度长度.

但是使用 sin 和 cos 并非是必须的,这里使用三角函数也许仅仅出于经验.

另外按照[参考3](https://zhuanlan.zhihu.com/p/46990010)说法,Position Embedding本身是一个绝对位置的信息，但在语言中，相对位置也很重要，Google选择前述的位置向量公式的一个重要原因是：由于我们有
$$
sin(\alpha+\beta)=sin \alpha cos \beta+cos \alpha sin \beta
$$
$$
cos(\alpha+\beta)=cos \alpha cos \beta - sin \alpha sin \beta
$$
这表明位置p+k的向量可以表示成位置p的向量的线性变换，这提供了表达相对位置信息的可能性。 

# 参考
- https://www.zhihu.com/question/347678607/answer/864217252
- https://zhuanlan.zhihu.com/p/98641990
- https://zhuanlan.zhihu.com/p/46990010