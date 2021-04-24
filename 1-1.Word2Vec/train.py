# -*- coding: utf-8 -*-
import math
import numpy as np
import random
import torch
import torch.utils.data as tud
import torch.optim as optim
from tqdm import tqdm
from collections import Counter
from model import EmbeddingModel
from preprocess import WordEmbeddingDataset

device = "cuda" if torch.cuda.is_available() else 'cpu'
EPOCHS = 100
MAX_VOCAB_SIZE = 10000
EMBEDDING_SIZE = 200
BATCH_SIZE = 512
LR = 0.001
MAX_VOCAB_SIZE = 10000
TRAIN_DATA_PATH = 'text8_toy.txt'
OUT_DIR = './result_example'

with open(TRAIN_DATA_PATH) as f:
    text = f.read()

text = text.lower().split()

vocab_count_ = dict(Counter(text))
total_count = sum(vocab_count_.values())

p = {}
for k, v in vocab_count_.items():
    p[k] = (math.sqrt((v/total_count)/0.001)+1)*0.001/(v/total_count)

subsampling  = []
for word in text:
    if random.random() < p[word]:
        subsampling.append(word)    
    
vocab_count = dict(Counter(subsampling).most_common(MAX_VOCAB_SIZE - 1))
vocab_count['<UNK>'] = 1

idx2word = [word for word in vocab_count.keys()]
word2idx = {word:i for i, word in enumerate(idx2word)}

nc = np.array([count for count in vocab_count.values()], dtype=np.float32)** (3./4.)
word_freqs = nc / np.sum(nc)

dataset = WordEmbeddingDataset(subsampling, word2idx, word_freqs)
dataloader = tud.DataLoader(dataset, BATCH_SIZE, shuffle=True)

model = EmbeddingModel(len(idx2word), EMBEDDING_SIZE)
model.to(device)
model.train()
optimizer = optim.Adam(model.parameters(), lr=LR)

for epoch in range(EPOCHS):
    pbar = tqdm(dataloader)
    pbar.set_description("[Epoch {}]".format(epoch))    
    for i, (input_labels, pos_labels, neg_labels) in enumerate(pbar):
        input_labels = input_labels.to(device)
        pos_labels = pos_labels.to(device)
        neg_labels = neg_labels.to(device)      
        model.zero_grad()
        loss = model(input_labels, pos_labels, neg_labels).mean()
        loss.backward()
        optimizer.step()
        pbar.set_postfix(loss=loss.item())

model.save_embedding(OUT_DIR, idx2word) 