# -*- coding: utf-8 -*-
import torch.nn as nn
from torch.nn import functional as F

class FastText(nn.Module):
    def __init__(self, trial, vocab_size, class_num):
        super(FastText, self).__init__()
        
        self.embed_dim = trial.suggest_int("n_embedding", 200, 300, 50)
        self.hidden_size = trial.suggest_int("hidden_size", 64, 128, 2)
        self.dropout = nn.Dropout(0.5)

        self.embed = nn.Embedding(vocab_size, self.embed_dim, padding_idx=1)        
        self.fc1 = nn.Linear(self.embed_dim, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, class_num)

    def forward(self, x):
        embeds = self.embed(x)
        out = embeds.mean(dim=1)
        out = self.dropout(out)
        out = self.fc1(out)
        out = F.relu(out)
        out = self.fc2(out)
        logit = F.log_softmax(out, dim=1)
        return logit