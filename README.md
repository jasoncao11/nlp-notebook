## 项目描述
基于pytorch实现的词向量、中文文本分类、实体识别等任务。 

## 依赖
```
python  3.7
pytorch 1.8.0
torchtext 0.5.0
optuna 2.6.0
```

## 目录

#### 1. Basic Embedding Model

- 1-1. [Word2Vec(Skip-gram)](1-1.Word2Vec)

#### 2. Text Classification (每个模型内部使用[optuna](https://optuna.org/)进行调参) 

- 2-1. [TextCNN](2-1.TextCNN)
- 2-2. [FastText](2-2.FastText)
- 2-3. [TextRCNN](2-3.TextRCNN)
- 2-4. [TextRNN_Att](2-4.TextRNN_Att)
- 2-5. [DPCNN](2-5.DPCNN)
- 2-6. [XGboost](2-6.XGboost)
  
数据集(data文件夹)： 二分类舆情数据集

数据集划分：

数据集|数据量
--|--
训练集|56700
验证集|7000
测试集|6300

#### 3-1. [BILSTM_CRF_NER](3-1.NER)

数据集在 NER/data 文件夹内

#### 4-1. [Transformer pytorch implementation step by step](4-1.Transformer)

- Reference
- https://wmathor.com/index.php/archives/1455/
- http://nlp.seas.harvard.edu/2018/04/03/attention.html

## 对应论文

[1] Convolutional Neural Networks for Sentence Classification

[2] Recurrent Neural Network for Text Classification with Multi-Task Learning

[3] Attention-Based Bidirectional Long Short-Term Memory Networks for Relation Classification

[4] Recurrent Convolutional Neural Networks for Text Classification

[5] Bag of Tricks for Efficient Text Classification

[6] Deep Pyramid Convolutional Neural Networks for Text Categorization

[7] Attention Is All You Need
