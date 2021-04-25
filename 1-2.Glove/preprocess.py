# -*- coding: utf-8 -*-
import torch
import torch.utils.data as tud
from collections import Counter, defaultdict

BATCH_SIZE = 512

class GloveDataset(tud.Dataset):
    def __init__(self, text, n_words=200000, window_size=5):
        super(GloveDataset, self).__init__()
        self.window_size = window_size
        self.tokens = text.split(" ")[:n_words]
        vocab = set(self.tokens)
        self.word2id = {w:i for i, w in enumerate(vocab)}
        self.id2word = {i:w for w, i in self.word2id.items()}
        self.vocab_size = len(vocab)
        self.id_tokens = [self.word2id[w] for w in self.tokens]
        
        cooc_mat = defaultdict(Counter)
        for i, w in enumerate(self.id_tokens):
            start_i = max(i - self.window_size, 0)
            end_i = min(i + self.window_size + 1, len(self.id_tokens))
            for j in range(start_i, end_i):
                if i != j:
                    c = self.id_tokens[j]
                    cooc_mat[w][c] += 1 / abs(j-i)

        self.i_idx = list()
        self.j_idx = list()
        self.xij = list()

        #Create indexes and x values tensors
        for w, cnt in cooc_mat.items():
            for c, v in cnt.items():
                self.i_idx.append(w)
                self.j_idx.append(c)
                self.xij.append(v)

        self.i_idx = torch.LongTensor(self.i_idx)
        self.j_idx = torch.LongTensor(self.j_idx)
        self.xij = torch.FloatTensor(self.xij)
        
    def __len__(self):
        return len(self.xij)

    def __getitem__(self, idx):
        return self.xij[idx], self.i_idx[idx], self.j_idx[idx]

traindataset = GloveDataset(open("text8_toy.txt").read())
id2word = traindataset.id2word
traindataloader = tud.DataLoader(traindataset, BATCH_SIZE, shuffle=True)

#for xij, i_idx, j_idx in traindataloader:
#    print(xij, i_idx, j_idx)
#    print('-----------')



            