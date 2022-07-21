# Sentence Bert

SBERT在BERT/RoBERTa的输出结果上增加了一个Pooling操作，从而生成一个固定维度的句子Embedding。实验中采取了三种Pooling策略做对比：

- CLS：直接用CLS位置的输出向量作为整个句子向量

- MEAN：计算所有Token输出向量的平均值作为整个句子向量

- MAX：取出所有Token输出向量各个维度的最大值作为整个句子向量

![simi](../images/simi8.png)