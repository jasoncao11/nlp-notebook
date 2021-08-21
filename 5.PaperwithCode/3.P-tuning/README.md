# GPT Understands, Too

## Structure
![GPT Understands, Too](../../images/pt.png)

## Reference
- https://github.com/THUDM/P-tuning/tree/main/LAMA
- https://arxiv.org/pdf/2103.10385.pdf

## Usage
- train.jsonl: 训练样本数据，其中，obj_label和sub_label为临近的国家或城市，需构建带有obj_label的模板预测sub_label
- load_data.py: 构造训练数据
- construct_query_label_demo.py：假设，template的size为(2,2,2)， 则：
query的格式为[cls_token_id, pseudo_token_id, pseudo_token_id, mask_token_id, pseudo_token_id, pseudo_token_id, obj_label_token_id, pseudo_token_id, pseudo_token_id, sep_token_id]
pseudo_token_id的embedding需要通过额外训练的prompt encoder获得，cls_token_id，mask_token_id，obj_label_token_id，sep_token_id则由BertForMaskedLM预训练模型的embedding layer获得。
label的格式为[-100, -100, -100,  xxx, -100, -100, -100, -100, -100, -100]，xxx为sub_label对应的token id
- prompt_encoder.py: LSTM+MLP， 通过训练用于获取pseudo_token_id的embedding
- model.py: 模型文件
- train.py: 训练脚本
