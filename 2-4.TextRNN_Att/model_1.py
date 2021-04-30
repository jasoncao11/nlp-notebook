# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch.nn import functional as F

device = "cuda" if torch.cuda.is_available() else 'cpu'

class TextRNN_Att(nn.Module):
    def __init__(self, trial, vocab_size, class_num):
        super(TextRNN_Att, self).__init__()
        
        self.embed_dim = trial.suggest_int("n_embedding", 200, 300, 50)
        self.hidden_size = trial.suggest_int("hidden_size", 64, 128, 2)
        self.embed = nn.Embedding(vocab_size, self.embed_dim, padding_idx=1)        
        self.lstm = nn.LSTM(self.embed_dim, self.hidden_size, bidirectional=True, batch_first=True)
        self.tanh = nn.Tanh()
        self.w = nn.Parameter(torch.zeros(self.hidden_size * 2))
        self.fc = nn.Linear(self.hidden_size * 2, class_num)
 
    def init_state(self, bs):
        # [num_layers(=1) * num_directions(=2), batch_size, hidden_size]
        return (torch.randn(2, bs, self.hidden_size).to(device),
                torch.randn(2, bs, self.hidden_size).to(device))

    def forward(self, x):       
        init_state = self.init_state(len(x))
        embeds = self.embed(x) #[batch_size, seq_len, embed_dim]
        lstm_out, (final_hidden_state, final_cell_state) = self.lstm(embeds, init_state)
        lstm_out = self.tanh(lstm_out) # [batch_size, seq_len, hidden_size * num_directions(=2)]
        alpha = F.softmax(torch.matmul(lstm_out, self.w), dim=1).unsqueeze(-1) # [batch_size, seq_len,1]
        out = lstm_out * alpha  # [batch_size, seq_len, hidden_size * num_directions(=2)]
        out = torch.sum(out, 1)  # [batch_size, hidden_size * num_directions(=2)]
        out = F.relu(out)
        out = self.fc(out) #[batch_size, class_num]
        logit = F.log_softmax(out, dim=1)
        return logit