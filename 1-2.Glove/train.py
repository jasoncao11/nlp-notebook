# -*- coding: utf-8 -*-
import torch
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from model import GloveModel
from preprocess import traindataloader, id2word

device = "cuda" if torch.cuda.is_available() else 'cpu'
EPOCHS = 100
EMBEDDING_SIZE = 200
X_MAX = 100
ALPHA = 0.75
LR = 0.0001
OUT_DIR = './result_example'

model = GloveModel(len(id2word), EMBEDDING_SIZE)
model.to(device)
model.train()
optimizer = optim.Adam(model.parameters(), lr=LR)

def weight_func(x, x_max, alpha):
    wx = (x/x_max)**alpha
    wx = torch.min(wx, torch.ones_like(wx))
    return wx.to(device)

def wmse_loss(weights, inputs, targets):
    loss = weights * F.mse_loss(inputs, targets, reduction='none')
    return torch.mean(loss).to(device)

for epoch in range(EPOCHS):
    pbar = tqdm(traindataloader)
    pbar.set_description("[Epoch {}]".format(epoch))    
    for i, (xij, i_idx, j_idx) in enumerate(pbar):
        xij = xij.to(device)
        i_idx = i_idx.to(device)
        j_idx = j_idx.to(device)
        
        model.zero_grad()
        outputs = model(i_idx, j_idx)
        weights_x = weight_func(xij, X_MAX, ALPHA)
        loss = wmse_loss(weights_x, outputs, torch.log(xij))
        
        loss.backward()
        optimizer.step()
        pbar.set_postfix(loss=loss.item())

model.save_embedding(OUT_DIR, id2word) 