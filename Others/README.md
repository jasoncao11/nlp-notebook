## 其他

### (1). 分类问题名称 & 输出层使用激活函数 & 对应的损失函数 & pytorch loss function
- 多标签 & sigmoid & binary cross entropy & BCELoss/BCEWithLogitsLoss
- 多分类 & softmax & categorical cross entropy & NLLLoss/CrossEntropyLoss
- 二分类可看作输出层有一个logit，对应sigmoid和binary cross entropy，或有两个logits，对应softmax和categorical cross entropy

### (2). 对抗训练 
- [NLP中的对抗训练 + PyTorch实现](https://zhuanlan.zhihu.com/p/91269728)

### (3). 数据增强
- [华为开源的tinyBert数据增强方法](https://github.com/huawei-noah/Pretrained-Language-Model/blob/master/TinyBERT/data_augmentation.py#L146)
