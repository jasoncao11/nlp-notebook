# -*- coding: utf-8 -*-
import torch
import torch.utils.data as tud
from transformers import BertTokenizer
from torch.nn.utils.rnn import pad_sequence

TRAIN_DATA_PATH = './data/train_data'
TEST_DATA_PATH = './data/test_data'
TOKENIZER_PATH = './bert-base-chinese'
BATCH_SIZE = 64
MAX_LEN = 512 #输入模型的最大长度，不能超过config中n_ctx的值

START_TAG, STOP_TAG = "<START>", "<STOP>"
label2idx = {START_TAG: 0, "O": 1, "B-PER": 2, "I-PER": 3, "B-LOC": 4, "I-LOC": 5, "B-ORG": 6, "I-ORG": 7, STOP_TAG: 8}
idx2label = {idx: label for label, idx in label2idx.items()}

def collate_fn(batch_data):
    """
    DataLoader所需的collate_fun函数，将数据处理成tensor形式
    Args:
        batch_data: batch数据
    Returns:
    """
    input_ids_list, label_ids_list, attention_mask_list, real_lens_list = [], [], [], []
    for instance in batch_data:
        # 按照batch中的最大数据长度,对数据进行padding填充
        input_ids_temp = instance["input_ids"]
        label_ids_temp = instance["label_ids"]
        attention_mask_temp = instance["attention_mask"]
        real_len = instance["real_len"]
        # 将input_ids_temp和label_ids_temp,attention_mask_temp添加到对应的list中
        input_ids_list.append(torch.tensor(input_ids_temp, dtype=torch.long))
        label_ids_list.append(torch.tensor(label_ids_temp, dtype=torch.long))
        attention_mask_list.append(torch.tensor(attention_mask_temp, dtype=torch.long))
        real_lens_list.append(real_len)
    # 使用pad_sequence函数，会将list中所有的tensor进行长度补全，补全到一个batch数据中的最大长度，补全元素为padding_value
    return {"input_ids": pad_sequence(input_ids_list, batch_first=True, padding_value=0),
            "label_ids": pad_sequence(label_ids_list, batch_first=True, padding_value=1),#"O"对应的ID为1
            "attention_mask": pad_sequence(attention_mask_list, batch_first=True, padding_value=0),
            "real_lens": real_lens_list}
    
class NERDataset(tud.Dataset):
    def __init__(self, data_path, tokenizer_path, max_len, label2idx):
        super(NERDataset, self).__init__()
        tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
        chars = []
        label_ids = []
        self.data_set = []
        with open (data_path, encoding='utf8') as rf:
            for line in rf:
                if line != '\n':
                    char, label = line.strip().split()
                    chars.append(char)
                    label_ids.append(label2idx[label])
                else:
                    input_ids = [tokenizer.convert_tokens_to_ids(c) for c in chars]
                    input_ids = [tokenizer.cls_token_id] + input_ids + [tokenizer.sep_token_id]
                    label_ids = [label2idx['O']] + label_ids + [label2idx['O']] #拼接上[CLS],[SEP]对应的label id
                    if len(input_ids) > max_len:
                        input_ids = input_ids[0] + input_ids[:max_len-2] + input_ids[-1]
                        label_ids = label_ids[0] + label_ids[:max_len-2] + label_ids[-1]
                    assert len(input_ids) == len(label_ids)
                    real_len = len(input_ids)
                    self.data_set.append({"input_ids": input_ids, "label_ids": label_ids, "attention_mask":[1]*len(input_ids), "real_len": real_len})
                    chars = []
                    label_ids = []                  
               
    def __len__(self):
        return len(self.data_set)
    
    def __getitem__(self, idx):    
        return self.data_set[idx]
    
traindataset = NERDataset(TRAIN_DATA_PATH, TOKENIZER_PATH, MAX_LEN, label2idx)
traindataloader = tud.DataLoader(traindataset, BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

valdataset = NERDataset(TEST_DATA_PATH, TOKENIZER_PATH, MAX_LEN, label2idx)
valdataloader = tud.DataLoader(valdataset, 1, shuffle=False, collate_fn=collate_fn)