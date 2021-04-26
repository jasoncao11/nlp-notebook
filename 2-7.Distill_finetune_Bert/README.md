# Distill fine-tune Bert

## 1. Concept of Bert and distillation

- [The Illustrated BERT](https://jalammar.github.io/illustrated-bert/)
- [A Visual Guide to Using BERT for the First Time](https://jalammar.github.io/a-visual-guide-to-using-bert-for-the-first-time/)
- [Hands-on coding in PyTorch — Compressing BERT](https://medium.com/huggingface/distilbert-8cf3380435b5)

## 2. 数据预处理

load_data.py 里，将数据处理成 [CLS] + tokens + [SEP] + padding 的形式，并生成相应的mask

## 3. 首先fine tune Bert
```
1. 结构为 Bert+TextRCNN
2. 模型中选择Bert倒数第二层的隐向量传入Bilstm，而不是最后一层，是因为倒数第二层不那么接近任务，但是又能学习到句子的较高层的语义。

python train_eval.py
```
## 4. 蒸馏上一步生成的大模型至小模型
```
1. 大模型也被称为教师模型，小模型被称为学生模型。学生模型学习的是教师模型的softmax层的输出概率分布，以便直接学习到其泛化能力。
2. 温度的高低改变的是学生模型训练过程中对负标签的关注程度。
- 从有部分信息量的负标签中学习 --> 温度高一些
- 防止受负标签中的噪声的影响 --> 温度低一些
- 温度的选择和学生模型的大小有关，代码里选择T=5作为demo。
3. 我们用K-L散度(https://www.countbayesie.com/blog/2017/5/9/kullback-leibler-divergence-explained)，来量化学生模型和教师模型的softmax层的差异。

python distill.py
```
