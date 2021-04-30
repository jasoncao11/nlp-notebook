# -*- coding: utf-8 -*-
import torch
import jieba
from torchtext.legacy import data

device = "cuda" if torch.cuda.is_available() else 'cpu'  

def tokenizer(text):
    res = []
    token = [tok for tok in jieba.cut(text)]
    if len(token) < 2:
        return token
    for n in [2, 3]:
        for x in range(len(token)):
            grams = token[x:x+n]
            if len(grams) < n:
                break            
            res.append('_'.join(grams))
    res.extend(token)
    return res

TEXT = data.Field(sequential=True, tokenize=tokenizer)
LABEL = data.Field(sequential=False, use_vocab=False)

train, val = data.TabularDataset.splits(
        path='../data/', 
        train='train.tsv',
        validation='dev.tsv',
        format='tsv',
        skip_header=True,
        fields=[('', None), ('label', LABEL), ('text', TEXT)])

TEXT.build_vocab(train, min_freq=5)
id2vocab = TEXT.vocab.itos
#print(TEXT.vocab.stoi)
#print(TEXT.vocab.itos)

train_iter, val_iter = data.BucketIterator.splits(
        (train, val), 
        sort_key=lambda x: len(x.text),
        batch_sizes=(256, 128), 
        device=device)