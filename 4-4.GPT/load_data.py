# -*- coding: utf-8 -*-
import csv
import torch
import torch.utils.data as tud
from transformers import BertTokenizer
from torch.nn.utils.rnn import pad_sequence

TRAIN_DATA_PATH = './data/train.tsv'
DEV_DATA_PATH = './data/dev.tsv'
TOKENIZER_PATH = './vocab'
BATCH_SIZE = 2
MAX_LEN = 512 #输入模型的最大长度，不能超过config中n_ctx的值

def collate_fn(batch_data):
    """
    DataLoader所需的collate_fun函数，将数据处理成tensor形式
    Args:
        batch_data: batch数据
    Returns:
    """
    input_ids_list, token_type_ids_list, attention_mask_list = [], [], []
    for instance in batch_data:
        # 按照batch中的最大数据长度,对数据进行padding填充
        input_ids_temp = instance["input_ids"]
        token_type_ids_temp = instance["token_type_ids"]
        attention_mask_temp = instance["attention_mask"]
        # 将input_ids_temp和token_type_ids_temp添加到对应的list中
        input_ids_list.append(torch.tensor(input_ids_temp, dtype=torch.long))
        token_type_ids_list.append(torch.tensor(token_type_ids_temp, dtype=torch.long))
        attention_mask_list.append(torch.tensor(attention_mask_temp, dtype=torch.long))
    # 使用pad_sequence函数，会将list中所有的tensor进行长度补全，补全到一个batch数据中的最大长度，补全元素为padding_value
    return {"input_ids": pad_sequence(input_ids_list, batch_first=True, padding_value=0),
            "token_type_ids": pad_sequence(token_type_ids_list, batch_first=True, padding_value=0),
            "attention_mask": pad_sequence(attention_mask_list, batch_first=True, padding_value=0)}
    
class SummaryDataset(tud.Dataset):
    def __init__(self, data_path, tokenizer_path, max_len):
        super(SummaryDataset, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
        #self.tokenizer.add_tokens("[Content]", special_tokens=True)
        #self.tokenizer.add_tokens("[Summary]", special_tokens=True)
        #内容正文和摘要分别用content_id，summary_id区分表示
        self.content_id = self.tokenizer.convert_tokens_to_ids("[Content]")
        self.summary_id = self.tokenizer.convert_tokens_to_ids("[Summary]")        
        self.sep_id = self.tokenizer.sep_token_id
        self.cls_id = self.tokenizer.cls_token_id
        
        self.data_set = []
        with open (data_path, 'r', encoding='utf8') as rf:
            r = csv.reader(rf, delimiter='\t')
            next(r)
            for row in r:
                
                input_ids = []
                token_type_ids = []      
                
                summary = row[0]
                summary_tokens = self.tokenizer.tokenize(summary)
                
                content = row[1]
                content_tokens = self.tokenizer.tokenize(content)
                # 如果正文过长，进行截断
                if len(content_tokens) > max_len - len(summary_tokens) - 3:
                    content_tokens = content_tokens[:max_len - len(summary_tokens) - 3]                
                
                input_ids.append(self.cls_id)
                token_type_ids.append(self.content_id)
                input_ids.extend(self.tokenizer.convert_tokens_to_ids(content_tokens))
                token_type_ids.extend([self.content_id] * len(content_tokens))
                input_ids.append(self.sep_id)
                token_type_ids.append(self.content_id)
                input_ids.extend(self.tokenizer.convert_tokens_to_ids(summary_tokens))
                token_type_ids.extend([self.summary_id] * len(summary_tokens))
                input_ids.append(self.sep_id)
                token_type_ids.append(self.summary_id)

                assert len(input_ids) == len(token_type_ids)
                assert len(input_ids) <= max_len
                
                self.data_set.append({"input_ids": input_ids, "token_type_ids": token_type_ids, "attention_mask":[1]*len(input_ids)})
               
    def __len__(self):
        return len(self.data_set)
    
    def __getitem__(self, idx):    
        return self.data_set[idx]
    
traindataset = SummaryDataset(TRAIN_DATA_PATH, TOKENIZER_PATH, MAX_LEN)
traindataloader = tud.DataLoader(traindataset, BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

valdataset = SummaryDataset(DEV_DATA_PATH, TOKENIZER_PATH, MAX_LEN)
valdataloader = tud.DataLoader(valdataset, BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

# tokenizer = BertTokenizer.from_pretrained(TOKENIZER_PATH)
# print(len(tokenizer))
# for batch_idx, batch_data in enumerate(traindataloader):
#     print(batch_idx)
#     input_ids = batch_data["input_ids"]
#     print(input_ids.shape)
#     token_type_ids = batch_data["token_type_ids"]
#     print(token_type_ids.shape)
#     attention_mask = batch_data["attention_mask"]
#     print(attention_mask.shape)
#     for input_, type_, mask_ in zip(input_ids, token_type_ids, attention_mask):
#         print(input_)
#         print(tokenizer.decode(input_))
#         print(type_)
#         print(tokenizer.decode(type_))
#         print(mask_)
#     print('-----------')
#     break