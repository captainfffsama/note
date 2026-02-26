来自论文**SimCSE: Simple Contrastive Learning of Sentence Embeddings**

SimCSE（Simple Contrastive Learning of Sentence Embeddings）是一种基于对比学习的句子嵌入模型算法，旨在提升句向量表征的质量。其核心思想是通过对比学习框架，利用 BERT 模型的 dropout 机制作为数据增强手段：将同一个句子两次输入模型，由于随机 dropout 导致的不同输出作为正例对，同一批次中的其他句子作为负例，通过优化对比损失函数来增大正例相似度、减小负例相似度。

具体来说：

1. **无监督训练**：利用预训练语言模型（如 BERT）的 dropout 特性，同一句子两次前向传播得到两个不同向量作为正例，同一 batch 内其他句子作为负例，损失函数为：

   $$ \mathcal{L} = -\log \frac{e^{\text{sim}(h_i, h_i^+)/\tau}}{\sum_{j=1}^N e^{\text{sim}(h_i, h_j^+)/\tau}} $$

   其中 $\tau$ 为温度参数， $\text{sim}$ 为余弦相似度。

2. **有监督训练**：在自然语言推理（NLI）等标注数据中，将蕴含关系的句子对作为正例，矛盾关系的句子对作为硬负例，进一步提升模型性能。

SimCSE 的优势包括：

- 改善句向量的**各向异性**问题，提升嵌入空间的均匀性（uniformity）。
- 在 STS-B 等语义相似度任务中超越传统方法（如 Sentence-BERT），且计算效率高。
- 代码开源（[GitHub链接](https://github.com/princeton-nlp/SimCSE)），复现和使用便捷。

注：用户问题中的“SimSCE”可能存在拼写误差，正确名称为“SimCSE”（文献 [2][3][4] 均使用此名称）。