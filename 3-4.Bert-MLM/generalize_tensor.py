# -*- coding: utf-8 -*-
"""
Created on Sun Mar 26 11:36:38 2023

@author: tieku
"""

import torch
import torch.utils.data as tud
from transformers import BertTokenizer, BertModel
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
torch.cuda.empty_cache()

TRAIN_DATA_PATH = './data/train_data'
BERT_PATH = './bert-base-chinese'
BATCH_SIZE = 8
MAX_LEN = 512  # 输入模型的最大长度，不能超过config中n_ctx的值


def collate_fn(batch_data):
    """
    DataLoader所需的collate_fun函数，将数据处理成tensor形式
    Args:
        batch_data: batch数据
    Returns:
    """
    input_ids_list, label_ids_list, attention_mask_list = [], [], []
    for instance in batch_data:
        # 按照batch中的最大数据长度,对数据进行padding填充
        input_ids_temp = instance["input_ids"]
        label_ids_temp = instance["label_ids"]
        attention_mask_temp = instance["attention_mask"]
        # 添加到对应的list中
        input_ids_list.append(torch.tensor(input_ids_temp, dtype=torch.long))
        label_ids_list.append(torch.tensor(label_ids_temp, dtype=torch.long))
        attention_mask_list.append(torch.tensor(attention_mask_temp, dtype=torch.long))
    # 使用pad_sequence函数，会将list中所有的tensor进行长度补全，补全到一个batch数据中的最大长度，补全元素为padding_value
    return {"input_ids": pad_sequence(input_ids_list, batch_first=True, padding_value=0),
            "label_ids": pad_sequence(label_ids_list, batch_first=True, padding_value=-100),
            "attention_mask": pad_sequence(attention_mask_list, batch_first=True, padding_value=0)}


class NERDataset(tud.Dataset):
    def __init__(self, data_path, tokenizer_path, max_len):
        super(NERDataset, self).__init__()
        label2id = {"O": 0, "B-ORG": 1, "I-ORG": 2, "B-PER": 3, "I-PER": 4, "B-LOC": 5, "I-LOC": 6}
        tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
        chars = []
        label_ids = []
        self.data_set = []
        with open(data_path, encoding='utf8') as rf:
            for line in rf:
                if line != '\n':
                    char, label = line.strip().split()
                    chars.append(char)
                    label_ids.append(label2id[label])
                else:
                    if len(chars) + 2 > max_len:
                        chars = chars[:max_len - 2]
                        label_ids = label_ids[:max_len - 2]
                    input_ids = [tokenizer.convert_tokens_to_ids(c) for c in chars]
                    input_ids = [tokenizer.cls_token_id] + input_ids + [tokenizer.sep_token_id]
                    label_ids = [0] + label_ids + [0]
                    assert len(input_ids) == len(label_ids)
                    self.data_set.append({"input_ids": input_ids,
                                          "label_ids": label_ids,
                                          "attention_mask": [1] * len(input_ids)})
                    chars = []
                    label_ids = []

    def __len__(self):
        return len(self.data_set)

    def __getitem__(self, idx):
        return self.data_set[idx]


traindataset = NERDataset(TRAIN_DATA_PATH, BERT_PATH, MAX_LEN)
traindataloader = tud.DataLoader(traindataset, BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

device = "cuda" if torch.cuda.is_available() else 'cpu'
model = BertModel.from_pretrained(BERT_PATH)
model.to(device)

B_ORG = torch.zeros(768, )
I_ORG = torch.zeros(768, )
B_PER = torch.zeros(768, )
I_PER = torch.zeros(768, )
B_LOC = torch.zeros(768, )
I_LOC = torch.zeros(768, )

pbar = tqdm(traindataloader)
for batch_idx, batch_data in enumerate(pbar):
    input_ids = batch_data["input_ids"].to(device)
    attention_mask = batch_data["attention_mask"].to(device)
    label_ids = batch_data["label_ids"]
    output = model(input_ids, attention_mask)
    sequence_output, pooled_output = output.last_hidden_state, output.pooler_output
    # print(input_ids, input_ids.size())
    # print(attention_mask, attention_mask.size())
    sequence_output = sequence_output.to('cpu')
    # print(sequence_output, sequence_output.size())
    # print(label_ids, label_ids.size())
    # print('---------------------------------')
    for ind, t in enumerate([B_ORG, I_ORG, B_PER, I_PER, B_LOC, I_LOC], 1):
        x = torch.where(label_ids == ind, 1, 0).nonzero().numpy()
        if x.size > 0:
            # print(ind)
            # print(x)
            num = 0
            for k in x:
                # print(sequence_output[k[0]][k[1]][0])
                num += 1
                t += sequence_output[k[0]][k[1]]
            # print('--------------', t[0])
            # print(num)
            t /= num
            # print('--------------', t[0])
            # print('++++++++++++')

#print(B_ORG)