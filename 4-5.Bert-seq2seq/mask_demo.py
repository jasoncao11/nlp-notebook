# -*- coding: utf-8 -*-
import torch

token_type_id = torch.tensor([[1,1,1,1,1,0,0,0],[1,1,1,1,0,0,0,-1],[1,1,1,0,0,0,-1,-1]])
mask = torch.ones((1, 1, 8, 8), dtype=torch.float32).tril()

t1 = token_type_id.unsqueeze(1).unsqueeze(2).float()
t2 = (token_type_id != -1).unsqueeze(1).unsqueeze(3).float()

mask = ((mask+t1)*t2 > 0).float()
print(mask)