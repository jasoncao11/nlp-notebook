# -*- coding: utf-8 -*-
import csv
import jieba
import torch
import torch.utils.data as tud
from torch.nn.utils.rnn import pad_sequence

TRAIN_DATA_PATH = '../data/train.tsv'
DEV_DATA_PATH = '../data/dev.tsv'
BATCH_SIZE = 128
MIN_FREQ = 5

#Make word dict
word2id = {'<pad>':0, '<unk>':1}
word2freq = {}
with open(TRAIN_DATA_PATH, 'r', encoding='utf8') as rf:
    r = csv.reader(rf, delimiter='\t')
    next(r)
    for row in r:
        text = row[2]
        tokens = [tok for tok in jieba.cut(text)]
        for token in tokens:
            word2freq[token] = word2freq.get(token, 0) + 1
filtered_words = [word for word, freq in word2freq.items() if freq >= MIN_FREQ]
for ind, word in enumerate(filtered_words, 2):
    word2id[word] = ind

def collate_fn(batch_data):
    """
    DataLoader所需的collate_fun函数，将数据处理成tensor形式
    Args:
        batch_data: batch数据
    Returns:
    """
    input_ids_list, labels_list = [], []
    for instance in batch_data:
        # 按照batch中的最大数据长度,对数据进行padding填充
        input_ids_temp = instance["input_ids"]
        input_ids_list.append(torch.tensor(input_ids_temp, dtype=torch.long))
        labels_list.append(instance["label"])
    # 使用pad_sequence函数，会将list中所有的tensor进行长度补全，补全到一个batch数据中的最大长度，补全元素为padding_value
    return {"input_ids": pad_sequence(input_ids_list, batch_first=True, padding_value=0),
            "labels": torch.tensor(labels_list, dtype=torch.long)}
    
class SentiDataset(tud.Dataset):
    def __init__(self, data_path, mode='train'):
        super(SentiDataset, self).__init__()  
        self.data_set = []
        with open (data_path, 'r', encoding='utf8') as rf:
            r = csv.reader(rf, delimiter='\t')
            next(r)
            for row in r:
                text = row[2]
                tokens = [tok for tok in jieba.cut(text)]
                input_ids = [word2id.get(tok, 1) for tok in tokens]
                label = int(row[1])           
                self.data_set.append({"input_ids": input_ids, "label": label})
                if mode == 'train':
                    self.data_set.append({"input_ids": input_ids, "label": label})
               
    def __len__(self):
        return len(self.data_set)
    
    def __getitem__(self, idx):    
        return self.data_set[idx]
    
traindataset = SentiDataset(TRAIN_DATA_PATH)
traindataloader = tud.DataLoader(traindataset, BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

valdataset = SentiDataset(DEV_DATA_PATH, mode='dev')
valdataloader = tud.DataLoader(valdataset, BATCH_SIZE, shuffle=False, collate_fn=collate_fn)