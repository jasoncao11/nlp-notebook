# -*- coding: utf-8 -*-
import torch
import jieba
from torchtext.legacy import data

device = "cuda" if torch.cuda.is_available() else 'cpu'

def tokenizer(text):
    token = [tok for tok in jieba.cut(text)]
    return token

TEXT = data.Field(tokenize=tokenizer,
                  init_token = '<sos>', 
                  eos_token = '<eos>', 
                  lower = True, 
                  batch_first = True)

train, val = data.TabularDataset.splits(
        path='./data/', 
        train='train.tsv',
        validation='dev.tsv',
        format='tsv',
        skip_header=True,
        fields=[('trg', TEXT), ('src', TEXT)])

TEXT.build_vocab(train, min_freq=2)
id2vocab = TEXT.vocab.itos
vocab2id = TEXT.vocab.stoi
PAD_IDX = vocab2id[TEXT.pad_token]
UNK_IDX = vocab2id[TEXT.unk_token]
SOS_IDX = vocab2id[TEXT.init_token]
EOS_IDX = vocab2id[TEXT.eos_token]
#train_iter 自动shuffle, val_iter 按照sort_key排序
train_iter, val_iter = data.BucketIterator.splits(
        (train, val),
        batch_sizes=(256, 128),
        sort_key=lambda x: len(x.src),
        device=device)

#for k in train_iter:
#    src_idx_batch, trg_idx_batch = k.src, k.trg
#    print (src_idx_batch, trg_idx_batch)
#    for x in trg_idx_batch:
#        print(''.join([id2vocab[i.item()] for i in x]))
#        print('----------------------')
#    print('\n')