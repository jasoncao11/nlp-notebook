# -*- coding: utf-8 -*-
import torch
import torch.utils.data as tud
from transformers import BertTokenizer, BertModel
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

TRAIN_DATA_PATH = './data/train_data'
TEST_DATA_PATH = './data/test_data'
BERT_PATH = './bert-base-chinese'
BATCH_SIZE = 4
MAX_LEN = 512 #输入模型的最大长度，不能超过config中n_ctx的值


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
        label2id = {"B-ORG": 1, "I-ORG": 2, "B-PER": 3, "I-PER": 4, "B-LOC": 5, "I-LOC": 6}
        tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
        chars = []
        labels = []
        self.data_set = []
        num = 0
        with open(data_path, encoding='utf8') as rf:
            for line in rf:
                if line != '\n':
                    char, label = line.strip().split()
                    chars.append(char)
                    labels.append(label)
                else:
                    num += 1
                    if len(chars) + 2 > max_len:
                        chars = chars[:max_len - 2]
                        labels = labels[:max_len - 2]
                    input_ids = [tokenizer.convert_tokens_to_ids(c) for c in chars]
                    label_ids = []
                    for input_id, label in zip(input_ids, labels):
                        if label in label2id:
                            label_ids.append(label2id[label])
                        else:
                            label_ids.append(input_id)
                    input_ids = [tokenizer.cls_token_id] + input_ids + [tokenizer.sep_token_id]
                    #print(input_ids)
                    label_ids = [-100] + label_ids + [-100]
                    #print(label_ids)
                    assert len(input_ids) == len(label_ids)
                    self.data_set.append({"input_ids": input_ids,
                                          "label_ids": label_ids,
                                          "attention_mask": [1]*len(input_ids)})
                    chars = []
                    labels = []
                    if num > 3000:
                        break
               
    def __len__(self):
        return len(self.data_set)
    
    def __getitem__(self, idx):    
        return self.data_set[idx]


traindataset = NERDataset(TRAIN_DATA_PATH, BERT_PATH, MAX_LEN)
traindataloader = tud.DataLoader(traindataset, BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

valdataset = NERDataset(TEST_DATA_PATH, BERT_PATH, MAX_LEN)
valdataloader = tud.DataLoader(valdataset, 1, shuffle=False, collate_fn=collate_fn)

# pbar = tqdm(traindataloader)
# for batch_idx, batch_data in enumerate(pbar):
#     input_ids = batch_data["input_ids"]
#     attention_mask = batch_data["attention_mask"]
#     label_ids = batch_data["label_ids"]
#     print(input_ids)
#     print(attention_mask)
#     print(label_ids)
#     print('-----------------------------------')