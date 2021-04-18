# -*- coding: utf-8 -*-
import csv
import random as rnd
import torch
from settings import START_TAG, STOP_TAG, BATCH_SIZE, UNK_TAG, PAD_TAG, CHAR_VOCAB_PATH

idx2vocab = {}
idx2vocab[0] = PAD_TAG
with open(CHAR_VOCAB_PATH, "r", encoding="utf8") as rf:
    r = csv.reader(rf)
    for ind, line in enumerate(r,1):
        idx2vocab[ind] = line[0].strip()
    idx2vocab[ind+1] = UNK_TAG
vocab2idx = {char: idx for idx, char in idx2vocab.items()}

label2idx = {PAD_TAG: 0, "O": 1, "B-PER": 2, "I-PER": 3, "B-LOC": 4, "I-LOC": 5, "B-ORG": 6, "I-ORG": 7, START_TAG: 8, STOP_TAG: 9}
idx2label = {idx: label for label, idx in label2idx.items()}

def data_generator(corpus_path, vocab2idx, label2idx, shuffle=False):
    datas, labels = [], []
    with open(corpus_path, encoding='utf-8') as fr:
        lines = fr.readlines()
    sent_, tag_ = [], []
    for line in lines:
        if line != '\n':
            [char, label] = line.strip().split()
            sent_.append(char)
            tag_.append(label)
        else:
            datas.append(sent_)
            labels.append(tag_)
            sent_, tag_ = [], []
    
    num_lines = len(datas)
    lines_index = [*range(num_lines)]
    if shuffle:
        rnd.shuffle(lines_index)
 
    index = 0
    flag = False
    while True:        
        buffer_datas = []
        buffer_labels = []        
        
        max_len = 0
        for i in range(BATCH_SIZE):
            if index >= num_lines:
                flag = True
                break
                        
            buffer_datas.append(datas[lines_index[index]])
            buffer_labels.append(labels[lines_index[index]])            
 
            lenx = len(datas[lines_index[index]])
            if lenx > max_len:
                max_len = lenx
            
            index += 1
            
        pad_datas = [x+[PAD_TAG]*(max_len-len(x)) for x in buffer_datas]
        pad_labels = [x+[PAD_TAG]*(max_len-len(x)) for x in buffer_labels]
        
        pad_datas = torch.tensor([[vocab2idx[w] if w in vocab2idx else vocab2idx[UNK_TAG] for w in seq] for seq in pad_datas], dtype=torch.long)
        pad_labels = torch.tensor([[label2idx[t] for t in tag] for tag in pad_labels], dtype=torch.long)
           
        yield pad_datas, pad_labels
        
        if flag:
            break