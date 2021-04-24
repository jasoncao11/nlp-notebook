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
        self.fc = nn.Linear(self.hidden_size * 2, class_num)
 
    def init_state(self, bs):
        # [num_layers(=1) * num_directions(=2), batch_size, hidden_size]
        return (torch.randn(2, bs, self.hidden_size).to(device),
                torch.randn(2, bs, self.hidden_size).to(device))

    # lstm_output : [batch_size, seq_len, hidden_size * num_directions(=2)]
    # final_state: [num_layers(=1) * num_directions(=2), batch_size, hidden_size]
    def attention_net(self, lstm_output, final_state):
        hidden = final_state.permute(1,0,2) #[batch_size, num_layers(=1) * num_directions(=2), hidden_size]
        hidden = hidden.reshape(-1, self.hidden_size * 2, 1) #[batch_size, hidden_size * num_directions(=2), 1]
        
        attn_weights = torch.bmm(lstm_output, hidden).squeeze(2) #[batch_size, seq_len]
        soft_attn_weights = F.softmax(attn_weights, 1).unsqueeze(-1) #[batch_size, seq_len,1]
        
        context = lstm_output * soft_attn_weights #[batch_size, seq_len, hidden_size * num_directions(=2)]
        context = torch.sum(context, dim=1) #[batch_size, hidden_size * num_directions(=2)]
    
        return context

    def forward(self, x):       
        init_state = self.init_state(len(x))
        embeds = self.embed(x) #[batch_size, seq_len, embed_dim]
        lstm_out, (final_hidden_state, final_cell_state) = self.lstm(embeds, init_state)
        attn_output = self.attention_net(lstm_out, final_hidden_state) #[batch_size, hidden_size * num_directions(=2)]
        out = self.fc(attn_output) #[batch_size, class_num]
        logit = F.log_softmax(out, dim=1)
        return logit

