## 基于bert的抽取式摘要

### 基本原理
将文档分割为句子，利用bert将每个句子转化为向量化表达，对这些句向量应用kmeans算法，初始化k个聚类中心，最终在每个cluster中找出与该cluster的centroid距离最近的句向量，文档的摘要则由这k个句向量所对应的句子所组成。

### Reference
- https://github.com/dmmiller612/bert-extractive-summarizer
