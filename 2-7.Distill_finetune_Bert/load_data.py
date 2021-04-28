# -*- coding: utf-8 -*-
import csv
import torch
import torch.utils.data as tud
from transformers import BertTokenizer

TRAIN_DATA_PATH = '../data/train.tsv'
DEV_DATA_PATH = '../data/dev.tsv'
TOKENIZER_PATH = './bert-base-chinese'
PAD_SIZE = 35
BATCH_SIZE = 128

class BinarySentiDataset(tud.Dataset):
    def __init__(self, data_path, tokenizer_path, pad_size):
        super(BinarySentiDataset, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
        self.pad_size = pad_size
        self.text = []
        self.labels = []
        with open (data_path, 'r', encoding='utf8') as rf:
            r = csv.reader(rf, delimiter='\t')
            next(r)
            for row in r:
                self.labels.append(int(row[1]))
                self.text.append(row[2])
               
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        raw_data = self.text[idx]
        label = self.labels[idx]
        tokens_ids = self.tokenizer.encode(raw_data)
        
        seq_len = len(tokens_ids)
        if seq_len < self.pad_size:
            mask = [1] * seq_len + [0] * (self.pad_size - seq_len)
            tokens_ids = tokens_ids + [0] * (self.pad_size - seq_len)
        else:
            mask = [1] * self.pad_size
            tokens_ids = tokens_ids[:self.pad_size]
            
        tokens_ids = torch.tensor(tokens_ids)
        mask = torch.tensor(mask)
        label = torch.tensor(label)
        
        return tokens_ids, mask, label
        
traindataset = BinarySentiDataset(TRAIN_DATA_PATH, TOKENIZER_PATH, PAD_SIZE)
traindataloader = tud.DataLoader(traindataset, BATCH_SIZE, shuffle=False)

valdataset = BinarySentiDataset(DEV_DATA_PATH, TOKENIZER_PATH, PAD_SIZE)
valdataloader = tud.DataLoader(valdataset, BATCH_SIZE, shuffle=False)