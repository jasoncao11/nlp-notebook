# Bert fine-tuning with TextRCNN model

## Concept of Bert

- [The Illustrated BERT](https://jalammar.github.io/illustrated-bert/)
- [A Visual Guide to Using BERT for the First Time](https://jalammar.github.io/a-visual-guide-to-using-bert-for-the-first-time/)

```
1. 取最后一层的隐向量，验证集上的准确率得分：0.9712857142857143
2. 取倒数第二层的隐向量，验证集上的准确率得分：0.9717142857142858

选择倒数第二层而不是最后一层，是因为倒数第二层不那么接近任务，但是又能学习到句子的较高层的语义。
```
