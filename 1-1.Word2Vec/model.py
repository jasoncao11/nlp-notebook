# -*- coding: utf-8 -*-
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

class EmbeddingModel(nn.Module):
    def __init__(self, vocab_size, embed_size):
        super(EmbeddingModel, self).__init__()
        
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        
        self.in_embed = nn.Embedding(self.vocab_size, self.embed_size)
        self.out_embed = nn.Embedding(self.vocab_size, self.embed_size)
        self.init_embed()
    
    def init_embed(self):
        init = 0.5 / self.embed_size
        self.in_embed.weight.data.uniform_(-init, init)
        self.out_embed.weight.data.uniform_(-0, 0)
    
    def forward(self, input_labels, pos_labels, neg_labels):
        ''' input_labels: center words, [batch_size] which is one dimentional vector of batch size
            pos_labels: positive words, [batch_size, (window_size * 2)]
            neg_labels：negative words, [batch_size, K]            
            return: loss, [batch_size]
        '''
        input_embedding = self.in_embed(input_labels)# [batch_size, embed_size]
        pos_embedding = self.out_embed(pos_labels)# [batch_size, (window * 2), embed_size]
        neg_embedding = self.out_embed(neg_labels)# [batch_size,  K, embed_size]
        
        input_embedding = input_embedding.unsqueeze(2)# [batch_size, embed_size, 1]
        
        pos_dot = torch.bmm(pos_embedding, input_embedding)# [batch_size, (window * 2), 1]
        pos_dot = pos_dot.squeeze(2)# [batch_size, (window * 2)]
        
        neg_dot = torch.bmm(neg_embedding, -input_embedding)# [batch_size, K, 1]
        neg_dot = neg_dot.squeeze(2)# [batch_size, K]
        
        log_pos = F.logsigmoid(pos_dot).sum(1)# [batch_size]
        log_neg = F.logsigmoid(neg_dot).sum(1)# [batch_size]
        
        return -log_pos-log_neg # [batch_size]

    def save_embedding(self, outdir, idx2word):
        embeds = self.in_embed.weight.data.cpu().numpy()        
        f1 = open(os.path.join(outdir, 'vec.tsv'), 'w')
        f2 = open(os.path.join(outdir, 'word.tsv'), 'w')        
        for idx in range(len(embeds)):
            word = idx2word[idx]
            embed = '\t'.join([str(x) for x in embeds[idx]])
            f1.write(embed+'\n')
            f2.write(word+'\n')
