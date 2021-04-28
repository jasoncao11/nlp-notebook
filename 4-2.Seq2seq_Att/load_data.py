# -*- coding: utf-8 -*-
import csv
import torch
import jieba
import torch.utils.data as tud

min_freq = 2
vocab_count = {}
with open('./data/train.tsv', 'r', encoding='utf8') as rf:
    r = csv.reader(rf, delimiter='\t')
    next(r)
    for row in r:
        text = f'{row[0]} {row[1]}'
        for tok in jieba.cut(text):
            if tok not in vocab_count:
                vocab_count[tok] = 1
            else:
                vocab_count[tok] += 1

print(len(vocab_count))                
#vocab_count = sorted(vocab_count.items(), key=lambda x:x[1], reverse=True)[:max_len]
vocab = [word for word, count in vocab_count.items() if count >= min_freq]
vocab.extend(['<UNK>', '<PAD>', '<SOS>', '<EOS>'])
print(len(vocab))
vocab_size = len(vocab) 
vocab2idx = {w:i for i, w in enumerate(vocab)}
#print(vocab2idx['披露'])       
idx2vocab = {i:w for w, i in vocab2idx.items()}
#print(idx2vocab[16336])

UNK_IDX = vocab2idx['<UNK>']
PAD_IDX = vocab2idx['<PAD>']
SOS_IDX = vocab2idx['<SOS>']
EOS_IDX = vocab2idx['<EOS>']
PAD_SIZE_C = 350
PAD_SIZE_S = 35
BATCH_SIZE = 512
TRAIN_DATA_PATH = './data/train.tsv'
DEV_DATA_PATH = './data/dev.tsv'

class SummaryDataset(tud.Dataset):
    def __init__(self, data_path):
        super(SummaryDataset, self).__init__()
        self.content = []
        self.summary = []
        with open (data_path, 'r', encoding='utf8') as rf:
            r = csv.reader(rf, delimiter='\t')
            next(r)
            for row in r:
                self.summary.append(row[0])
                self.content.append(row[1])
               
    def __len__(self):
        return len(self.summary)
    
    def __getitem__(self, idx):
        content_tokens = [tok for tok in jieba.cut(self.content[idx])]
        summary_tokens = [tok for tok in jieba.cut(self.summary[idx])]
        
        content_idx = [SOS_IDX] + [vocab2idx.get(word, UNK_IDX) for word in content_tokens] + [EOS_IDX]
        summary_idx = [SOS_IDX] + [vocab2idx.get(word, UNK_IDX) for word in summary_tokens] + [EOS_IDX]
        
        content_len = len(content_idx)
        if content_len < PAD_SIZE_C:
            content_idx = content_idx + [PAD_IDX] * (PAD_SIZE_C - content_len)
        else:
            content_idx = content_idx[:PAD_SIZE_C]
        
        summary_len = len(summary_idx)
        if summary_len < PAD_SIZE_S:
            summary_idx = summary_idx + [PAD_IDX] * (PAD_SIZE_S - summary_len)
        else:
            summary_idx = summary_idx[:PAD_SIZE_S]
        
        return torch.tensor(summary_idx), torch.tensor(content_idx)
        
traindataset = SummaryDataset(TRAIN_DATA_PATH)
traindataloader = tud.DataLoader(traindataset, BATCH_SIZE, shuffle=True)

valdataset = SummaryDataset(DEV_DATA_PATH)
valdataloader = tud.DataLoader(valdataset, BATCH_SIZE, shuffle=True)

#for summary, content in traindataloader:
#    print(f'summary idx = {summary[0]}')
#    print(f"summary = {' '.join([idx2vocab[idx.item()] for idx in summary[0]])}")
#    print(f'content idx = {content[0]}')
#    print(f"content = {' '.join([idx2vocab[idx.item()] for idx in content[0]])}")
#    print(f'-----------\n')