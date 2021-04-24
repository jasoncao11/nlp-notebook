# -*- coding: utf-8 -*-
import torch
import jieba
from torchtext import data

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

train, val, test = data.TabularDataset.splits(
        path='../data/', 
        train='train.tsv',
        validation='dev.tsv',
        test='test.tsv',
        format='tsv',
        skip_header=True,
        fields=[('', None), ('label', LABEL), ('text', TEXT)])

TEXT.build_vocab(train, min_freq=5)
id2vocab = TEXT.vocab.itos
#print(TEXT.vocab.stoi)
#print(TEXT.vocab.itos)

train_iter, val_iter, test_iter = data.BucketIterator.splits(
        (train, val, test), 
        sort_key=lambda x: len(x.text),
        batch_sizes=(256, 128, 64), 
        device=device,
        sort_within_batch=False,
        repeat=False)
#
#for k in train_iter:
#    text_idx_batch, label_idx_batch = k.text.t_(), k.label
#    print (text_idx_batch, label_idx_batch)
#    for x in text_idx_batch:
#        print(''.join([id2vocab[i.item()] for i in x]))
#        print('----------------------')
#    
#    
#for k in val_iter:
#    text_idx_batch, label_idx_batch = k.text.t_(), k.label
#    print (text_idx_batch, label_idx_batch)
#    for x in text_idx_batch:
#        print(''.join([id2vocab[i.item()] for i in x]))
#        print('----------------------')
#    print(label_idx_batch)