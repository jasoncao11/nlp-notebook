# -*- coding: utf-8 -*-
import csv
import random as rnd
import torch

idx2char = {}
idx2char[0] = "<PAD>"
with open("./data/char_vocabs.txt", "r", encoding="utf8") as rf:
    r = csv.reader(rf)
    for ind, line in enumerate(r,1):
        idx2char[ind] = line[0].strip()
    idx2char[ind+1] = "<UNK>"
char2idx = {char: idx for idx, char in idx2char.items()}

START_TAG, STOP_TAG = "<START>", "<STOP>"
label2idx = {START_TAG: 0, "O": 1, "B-PER": 2, "I-PER": 3, "B-LOC": 4, "I-LOC": 5, "B-ORG": 6, "I-ORG": 7, STOP_TAG: 8}
idx2label = {idx: label for label, idx in label2idx.items()}

def data_generator(corpus_path, char2idx, label2idx, batch_size, shuffle=False):
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
        buffer_lengths = []       
        
        max_len = 0
        for i in range(batch_size):
            if index >= num_lines:
                flag = True
                break
                        
            buffer_datas.append(datas[lines_index[index]])
            buffer_labels.append(labels[lines_index[index]])

            length = len(datas[lines_index[index]])
            buffer_lengths.append(length)            
            if length > max_len:
                max_len = length
            
            index += 1
            
        pad_datas = [x+["<PAD>"]*(max_len-len(x)) for x in buffer_datas]
        pad_labels = [x+['O']*(max_len-len(x)) for x in buffer_labels]
        
        pad_datas = torch.tensor([[char2idx[w] if w in char2idx else char2idx["<UNK>"] for w in seq] for seq in pad_datas], dtype=torch.long)
        pad_labels = torch.tensor([[label2idx[t] for t in tag] for tag in pad_labels], dtype=torch.long)
           
        yield pad_datas, pad_labels, buffer_lengths
        
        if flag:
            break