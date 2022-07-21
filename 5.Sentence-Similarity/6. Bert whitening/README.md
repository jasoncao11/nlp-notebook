# Bert whitening

Bert-whitening是直接对Bert生成的句向量做转换，将当前坐标系变换到标准正交基下，进而实现句向量空间的各向同性。

## PCA：
降维问题的优化目标：将一组N维向量降为K维（K大于0，小于N），其目标是选择K个单位（模为1）正交基，使得原始数据变换到这组基上后，各字段两两间协方差为0，而字段的方差则尽可能大（在正交的约束下，取最大的K个方差）。

![simi3](../../images/simi3.png)
![simi4](../../images/simi4.png)

## 参考：

- https://kexue.fm/archives/8069
- https://blog.csdn.net/u010376788/article/details/46957957