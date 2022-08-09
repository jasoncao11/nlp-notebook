# -*- coding: utf-8 -*-
from torch.utils.data import Dataset
import json
from torch.utils.data import DataLoader

from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

class LAMADataset(Dataset):
    def __init__(self, filename, tokenizer):
        super().__init__()

        records = []
        with open(filename, "r") as f:
            for line in f.readlines():
                try:
                    records.append(json.loads(line))
                except:
                    continue

        self.x_hs, self.x_ts = [], []
        vocab = tokenizer.get_vocab()
        print(len(vocab))
        for d in records:
            if d['obj_label'].lower() not in vocab or d['sub_label'].lower() not in vocab:
                continue
            self.x_ts.append(d['obj_label'].lower())
            self.x_hs.append(d['sub_label'].lower())

    def __len__(self):
        return len(self.x_hs)

    def __getitem__(self, i):
        return self.x_hs[i], self.x_ts[i]

train_set = LAMADataset('../train.jsonl' ,tokenizer)
train_loader = DataLoader(train_set, batch_size=8, shuffle=True, drop_last=True)