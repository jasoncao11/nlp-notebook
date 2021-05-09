# GPT2摘要生成项目

## 1. Concept of GPT-2

- [The Illustrated GPT-2](https://jalammar.github.io/illustrated-gpt2/)

## 2. 数据预处理
```
load_data.py 里，将数据处理成 [CLS] + content tokens + [SEP] + summary tokens + [SEP] + [PAD](0个或多个) 的形式，并生成相应的 token_type_ids。
```
## 3. 训练
```
1. config/config.json: 模型的配置信息，包含n_ctx、n_embd、n_head、n_layer等。
2. vocab/vocab.txt: 字典文件，该字典为大小为13317，删除了将原始字典中的“##中文”，并且增加了“[Content]”、“[Summary]”等标记。
3. train.py: 通过新闻生成摘要的GPT2模型的训练文件，训练过程中只计算summary部分的loss
4. generate_summary.py: 根据训练好的模型，进行摘要生成
```

## 4. Reference
```
1. https://github.com/liucongg/GPT2-NewsTitle
2. https://github.com/qingkongzhiqian/GPT2-Summary
```