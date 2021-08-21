# -*- coding: utf-8 -*-
import torch
import torch.nn as nn

class PromptEncoder(torch.nn.Module):
    def __init__(self, template, hidden_size, device):
        super().__init__()
        self.device = device
        self.hidden_size = hidden_size
        # ent embedding
        self.cloze_length = template
        self.cloze_mask = [
            [1] * self.cloze_length[0]  # first cloze
            + [1] * self.cloze_length[1]  # second cloze
            + [1] * self.cloze_length[2]  # third cloze
        ]
        self.cloze_mask = torch.LongTensor(self.cloze_mask).bool().to(self.device) #[[True, True, True, True, True, True, True, True, True]] =[1, sum(template)]

        self.seq_indices = torch.LongTensor(list(range(len(self.cloze_mask[0])))).to(self.device) #[0, 1, 2, 3, 4, 5, 6, 7, 8] = [sum(template)]
        # embedding
        self.embedding = torch.nn.Embedding(len(self.cloze_mask[0]), self.hidden_size).to(self.device)
        # LSTM
        self.lstm_head = torch.nn.LSTM(input_size=self.hidden_size,
                                       hidden_size=self.hidden_size // 2,
                                       num_layers=2,                                      
                                       bidirectional=True,
                                       batch_first=True)
        self.mlp_head = nn.Sequential(nn.Linear(self.hidden_size, self.hidden_size),
                                      nn.ReLU(),
                                      nn.Linear(self.hidden_size, self.hidden_size))
        print("init prompt encoder...")

    def forward(self):
        input_embeds = self.embedding(self.seq_indices).unsqueeze(0) #[1, 9(sum(template)), hidden_size]
        output_embeds = self.mlp_head(self.lstm_head(input_embeds)[0]).squeeze() #[9(sum(template)), hidden_size] 
        return output_embeds