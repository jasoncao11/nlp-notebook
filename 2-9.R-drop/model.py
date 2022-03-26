# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch.nn import functional as F

class TextCNN(nn.Module):
    def __init__(self, vocab_size, class_num):
        super(TextCNN, self).__init__()
        ci = 1  # input chanel size
        kernel_num = 300
        kernel_size = [2, 3, 4]
        embed_dim = 300
        dropout = 0.5

        self.embed = nn.Embedding(vocab_size, embed_dim)        
        self.convs = nn.ModuleList([nn.Conv2d(ci, kernel_num, (k, embed_dim)) for k in kernel_size])
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(len(kernel_size) * kernel_num, class_num)

        self.init_weight()
        
    def conv_and_pool(self, x, conv):
        # x: (batch, 1, sentence_length, embed_dim)
        x = conv(x)
        # x: (batch, kernel_num, H_out, 1)
        x = F.relu(x.squeeze(3))
        # x: (batch, kernel_num, H_out)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        #  (batch, kernel_num)
        return x

    def forward(self, x):
        # x: (batch, sentence_length)
        x = self.embed(x)
        # x: (batch, sentence_length, embed_dim)
        x = x.unsqueeze(1)
        # x: (batch, 1, sentence_length, embed_dim)       
        x = torch.cat([self.conv_and_pool(x, conv) for conv in self.convs], 1)  
        # x: (batch, len(kernel_size) * kernel_num)
        x = self.dropout(x)
        logit = self.fc1(x)
        return logit

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight)
                nn.init.constant_(m.bias, 0)