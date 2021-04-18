# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch.nn import functional as F

class TextRCNN(nn.Module):
    def __init__(self, trial, vocab_size, class_num):
        super(TextRCNN, self).__init__()
        
        self.embed_dim = trial.suggest_int("n_embedding", 200, 300, 50)
        self.hidden_size = trial.suggest_int("hidden_size", 64, 128, 2)
        self.embed = nn.Embedding(vocab_size, self.embed_dim, padding_idx=1)        
        self.lstm = nn.LSTM(self.embed_dim, self.hidden_size, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(self.hidden_size * 2 + self.embed_dim, class_num)

    def forward(self, x):
        embeds = self.embed(x)  # [batch_size, seq_len, embed_dim]
        out, _ = self.lstm(embeds) # [batch_size, seq_len, hidden_size * 2]
        out = torch.cat((embeds, out), 2) # [batch_size, seq_len, hidden_size * 2 + embed_dim]
        out = F.relu(out) # [batch_size, seq_len, hidden_size * 2 + embed_dim]
        out = out.permute(0, 2, 1) # [batch_size, hidden_size * 2 + embed_dim, seq_len]
        out = F.max_pool1d(out, out.size(2)).squeeze(2) # [batch_size, hidden_size * 2 + embed_dim]        
        logit = F.log_softmax(self.fc(out), dim=1) # [batch_size, class_num]
        return logit        
        