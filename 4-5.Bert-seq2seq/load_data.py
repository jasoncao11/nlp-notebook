# -*- coding: utf-8 -*-
import csv
import torch
import torch.utils.data as tud
from torch.nn.utils.rnn import pad_sequence
from tokenizer import Tokenizer, load_chinese_base_vocab

TRAIN_DATA_PATH = './data/train.tsv'
DEV_DATA_PATH = './data/dev.tsv'
MAX_LEN = 256
BATCH_SIZE = 32
VOCAB_PATH = '../bert-base-chinese/vocab.txt'
word2idx, keep_tokens = load_chinese_base_vocab(VOCAB_PATH, simplfied=True)

def collate_fn(batch_data):
    """
    DataLoader所需的collate_fun函数，将数据处理成tensor形式
    Args:
        batch_data: batch数据
    Returns:
    """
    input_ids_list, token_type_ids_list, token_type_ids_for_mask_list, target_ids_list = [], [], [], []
    for instance in batch_data:
        # 按照batch中的最大数据长度,对数据进行padding填充
        input_ids_temp = instance["input_ids"]
        token_type_ids_temp = instance["token_type_ids"]
        token_type_ids_for_mask_temp = instance["token_type_ids_for_mask"]
        target_ids_temp = instance["target_ids"]

        input_ids_list.append(torch.tensor(input_ids_temp, dtype=torch.long))
        token_type_ids_list.append(torch.tensor(token_type_ids_temp, dtype=torch.long))
        token_type_ids_for_mask_list.append(torch.tensor(token_type_ids_for_mask_temp, dtype=torch.long))
        target_ids_list.append(torch.tensor(target_ids_temp, dtype=torch.long))
    # 使用pad_sequence函数，会将list中所有的tensor进行长度补全，补全到一个batch数据中的最大长度，补全元素为padding_value
    return {"input_ids": pad_sequence(input_ids_list, batch_first=True, padding_value=0),
            "token_type_ids": pad_sequence(token_type_ids_list, batch_first=True, padding_value=0),
            "token_type_ids_for_mask": pad_sequence(token_type_ids_for_mask_list, batch_first=True, padding_value=-1),
            "target_ids": pad_sequence(target_ids_list, batch_first=True, padding_value=0)}

class BertDataset(tud.Dataset):
    def __init__(self, data_path):
        super(BertDataset, self).__init__()
        self.tokenizer = Tokenizer(word2idx)
        
        self.data_set = []
        with open (data_path, 'r', encoding='utf8') as rf:
            r = csv.reader(rf, delimiter='\t')
            next(r)
            for row in r:
                summary = row[0]
                content = row[1]
                
                input_ids, token_type_ids, token_type_ids_for_mask = self.tokenizer.encode(content, summary, max_length=MAX_LEN)
                target_ids = input_ids[1:]
                       
                self.data_set.append({"input_ids": input_ids, 
                                      "token_type_ids": token_type_ids, 
                                      "token_type_ids_for_mask": token_type_ids_for_mask,
                                      "target_ids": target_ids})
               
    def __len__(self):
        return len(self.data_set)
    
    def __getitem__(self, idx):
        return self.data_set[idx]
        
traindataset = BertDataset(TRAIN_DATA_PATH)
traindataloader = tud.DataLoader(traindataset, BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

valdataset = BertDataset(DEV_DATA_PATH)
valdataloader = tud.DataLoader(valdataset, BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

# for batch in valdataloader:
#     print(batch["input_ids"])
#     print(batch["input_ids"].shape)
#     print('------------------')
    
#     print(batch["token_type_ids"])
#     print(batch["token_type_ids"].shape)
#     print('------------------')
    
#     print(batch["token_type_ids_for_mask"])
#     print(batch["token_type_ids_for_mask"].shape)
#     print('------------------')
    
#     print(batch["target_ids"])
#     print(batch["target_ids"].shape)
#     print('------------------')