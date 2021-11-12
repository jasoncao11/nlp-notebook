# -*- coding: utf-8 -*-
import os
import torch
import torch.nn as nn

class GloveModel(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        super(GloveModel, self).__init__()
        self.wi = nn.Embedding(num_embeddings, embedding_dim)
        self.wj = nn.Embedding(num_embeddings, embedding_dim)
        self.bi = nn.Embedding(num_embeddings, 1)
        self.bj = nn.Embedding(num_embeddings, 1)

        self.wi.weight.data.uniform_(-1, 1)
        self.wj.weight.data.uniform_(-1, 1)
        self.bi.weight.data.zero_()
        self.bj.weight.data.zero_()

    def forward(self, i_indices, j_indices):
        w_i = self.wi(i_indices) #[batch_size, embedding_dim]
        w_j = self.wj(j_indices) #[batch_size, embedding_dim]
        b_i = self.bi(i_indices).squeeze() #[batch_size]
        b_j = self.bj(j_indices).squeeze() #[batch_size]
        x = torch.sum(w_i * w_j, dim=1) + b_i + b_j #[batch_size]
        return x
    
    def save_embedding(self, outdir, idx2word):
        embeds = self.wi.weight.data.cpu().numpy() + self.wj.weight.data.cpu().numpy()        
        f1 = open(os.path.join(outdir, 'vec.tsv'), 'w')
        f2 = open(os.path.join(outdir, 'word.tsv'), 'w')        
        for idx in range(len(embeds)):
            word = idx2word[idx]
            embed = '\t'.join([str(x) for x in embeds[idx]])
            f1.write(embed+'\n')
            f2.write(word+'\n')
