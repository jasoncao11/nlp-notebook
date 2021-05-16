# BILSTM_CRF_NER

A PyTorch implementation of batch support bidirectional LSTM-CRF.

## Usage

Training/Test data should be formatted as shown in data directory.

### To train:

```
python train.py

output:
Epoch[0] - Loss:60.69867333498868
Epoch[1] - Loss:21.227553984131475
Epoch[2] - Loss:17.732702394928594
Epoch[3] - Loss:15.213178186705619
Epoch[4] - Loss:13.035220449621027
Epoch[5] - Loss:11.137125511362095
.
.
.
Epoch[98] - Loss:0.09522969230557933
Epoch[99] - Loss:0.09701536212003592
Training costs:1282.8690948486328 seconds
```

### To evaluate:

```
python eval.py

output:

......

Sent: 这是第一位访华的希腊大船主，两周时间，他饱览中国的山水名胜，接触了各界人士，参观港口、码头和造船厂，探索与中国合作的途径。
NER: [['华', 'LOC'], ['希腊', 'LOC'], ['中国', 'LOC'], ['中国', 'LOC']]
Predicted NER: [['华', 'LOC'], ['希腊', 'LOC'], ['中国', 'LOC'], ['中国', 'LOC']]
---------------

Sent: 在希腊，人们常称他为“小奥纳西斯”。
NER: [['希腊', 'LOC'], ['奥纳西斯', 'PER']]
Predicted NER: [['希腊', 'LOC'], ['奥纳西斯', 'PER']]
---------------

Sent: 其实，他与那位娶了美国总统肯尼迪遗孀杰奎琳·肯尼迪的希腊船王奥纳西斯毫无血缘关系。
NER: [['美国', 'LOC'], ['肯尼迪', 'PER'], ['杰奎琳·肯尼迪', 'PER'], ['希腊', 'LOC'], ['奥纳西斯', 'PER']]
Predicted NER: [['美国', 'LOC'], ['肯尼迪', 'PER'], ['杰奎琳·肯尼迪', 'PER'], ['希腊', 'LOC'], ['奥纳西斯', 'PER']]
---------------

Sent: 这么称呼他，是因为他们有相似的经历：胆识过人，白手起家，历经一二十年创建了庞大的船队，成为政界、商界的风云人物。
NER: []
Predicted NER: []

......

gold_num = 6180
predict_num = 5835
correct_num = 4710
precision = 0.8071979434447301
recall = 0.7621359223300971
f1-score = 0.7840199750312109
```
### TODO:
```
- 将IOB格式转换成IOBES格式
- 使用预训练的bert字向量，并拼接如下向量(参考https://github.com/AlexYangLi/ccks2019_el)：

1.字符所在词向量。将实体词load到jieba词典，分词得到训练语料的词序列后，使用word2vec的方法进行训练，得到一定维度的词向量。然后为每个字都拼接上其所在的词的词向量，这样来自同一个实体的字都具有相同的词向量，有利于实体识别。

2. 字符所在词的位置特征向量。使用 BMES 标记字符在词中的位置。如句子 “比特币吸粉无数” 被 jieba 切成的词序列为：['比特币', '吸粉', '无数']，则字符的位置信息将会被标注为 [B, M, E, B, E, B, E]。为这四个标记分别随机初始化一个50维向量，然后在模型训练时再进行优化。

- 对<PAD>标记进行MASK，不参与Loss的计算, 简单的思路是在计算正确路径(_score_sentence_parallel)和所有路径(_forward_alg_parallel)的得分的时候，将：

1. transition matrix 中任何 tag 转移至 <PAD> 的概率置为0
2. emission matrix 中每个 sequence 里的 <PAD> 对应的得分置为0， 如经过_get_lstm_features_parallel后，我们得到scores，维度[batch_size, seq_len, tag_nums]：
scores=[[[0.0659, 0.2049, 0.5691, 0.9560],
         [0.6628, 0.8559, 0.2735, 0.8353],
         [0.6732, 0.9289, 0.1579, 0.0020]],

        [[0.2270, 0.4444, 0.8364, 0.6438],
         [0.1887, 0.7808, 0.4555, 0.4346],
         [0.5038, 0.0862, 0.7574, 0.5452]]]
  
假设mask为[[1,1,0],[1,1,1]], 即第一个序列最后一个字为<PAD>，我们可以用scores*mask.reshape(-1,seq_len,1)，得到新的scores。
```

### Others

1. settings.py is where the parameters are defined.

2. load_data.py implements a data generator for producing the tensor of batch size.

3. pytorch_tutorial_vec.py is the vectorized version of the [pytorch BiLSTM-CRF tutorial](https://pytorch.org/tutorials/beginner/nlp/advanced_tutorial.html), modified based on [https://zhuanlan.zhihu.com/p/97676647](https://zhuanlan.zhihu.com/p/97676647).
