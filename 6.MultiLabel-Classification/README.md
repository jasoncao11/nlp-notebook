### 多标签文本分类

#### 数据集

数据来源：[2020语言与智能技术竞赛：事件抽取任务](https://aistudio.baidu.com/aistudio/competition/detail/32?isFromCcf=true)

#### 模型

TextRCNN

#### 损失函数

- 1. 将多标签分类变成多个二分类问题
- 2. 借助于logsumexp的性质，将“softmax + 交叉熵”推广到多标签分类任务中

#### 参考
- 1. https://kexue.fm/archives/7359/comment-page-2#comments
- 2. https://kexue.fm/archives/3290