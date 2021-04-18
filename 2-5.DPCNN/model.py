# -*- coding: utf-8 -*-
import torch.nn as nn
from torch.nn import functional as F

class DPCNN(nn.Module):
    def __init__(self, trial, vocab_size, class_num):
        super(DPCNN, self).__init__()
        ci = 1  # input chanel size
        kernel_num = 250 # output chanel size
        embed_dim = trial.suggest_int("n_embedding", 200, 300, 50)

        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=1)        
        self.conv_region = nn.Conv2d(ci, kernel_num, (3, embed_dim), stride=1)
        self.conv = nn.Conv2d(kernel_num, kernel_num, (3, 1), stride=1)
        
        self.max_pool = nn.MaxPool2d(kernel_size=(3, 1), stride=2)
        self.max_pool_2 = nn.MaxPool2d(kernel_size=(2, 1))
        
        self.padding = nn.ZeroPad2d((0, 0, 1, 1))  # top bottom
        self.relu = nn.ReLU()
        self.fc = nn.Linear(kernel_num, class_num)

    def forward(self, x):
        x = self.embed(x) # x: (batch, seq_len, embed_dim)
        x = x.unsqueeze(1) # x: (batch, 1, seq_len, embed_dim)
        m = self.conv_region(x)  # [batch_size, 250, seq_len-3+1, 1]
        
        x = self.padding(m)  # [batch_size, 250, seq_len, 1]
        x = self.relu(x) # [batch_size, 250, seq_len, 1]
        x = self.conv(x)  # [batch_size, 250, seq_len-3+1, 1]
        x = self.padding(x)  # [batch_size, 250, seq_len, 1]
        x = self.relu(x) # [batch_size, 250, seq_len, 1]
        x = self.conv(x)  # [batch_size, 250, seq_len-3+1, 1]

        x = x+m
        
        while x.size()[2] > 2:
            x = self._block(x)
        if x.size()[2] == 2:
            x = self.max_pool_2(x) # [batch_size, 250, 1, 1]  
        x = x.squeeze() # [batch_size, 250]
        logit = F.log_softmax(self.fc(x), dim=1)
        return logit

    def _block(self, x): # for example: [batch_size, 250, 4, 1]
        
        px = self.max_pool(x) # [batch_size, 250, 1, 1]

        x = self.padding(px) # [batch_size, 250, 3, 1]
        x = F.relu(x)
        x = self.conv(x) # [batch_size, 250, 1, 1]

        x = self.padding(x)
        x = F.relu(x)
        x = self.conv(x)

        # Short Cut
        x = x + px
        return x